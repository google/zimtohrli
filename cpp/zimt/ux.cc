// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zimt/ux.h"

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "GL/gl.h"
#include "GL/glext.h"
#include "GLES3/gl32.h"
#include "GLFW/glfw3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "hwy/aligned_allocator.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "portaudio.h"
#include "zimt/audio.h"
#include "zimt/cam.h"
#include "zimt/elliptic.h"
#include "zimt/filterbank.h"
#include "zimt/zimtohrli.h"

// This file uses a lot of magic from the SIMD library Highway.
// In simplified terms, it will compile the code for multiple architectures
// using the "foreach_target.h" header file, and use the special namespace
// convention HWY_NAMESPACE to find the code to adapt to the SIMD functions,
// which are then called via HWY_DYNAMIC_DISPATCH. This leads to a lot of
// hard-to-explain Highway-related conventions being followed, like this here
// #define that makes this entire file be included by Highway in the process of
// building.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "zimt/ux.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
// Must come after foreach_target.h to avoid redefinition errors.
#include "hwy/highway.h"

// This is Highway magic conventions.
HWY_BEFORE_NAMESPACE();
namespace zimtohrli {

namespace HWY_NAMESPACE {

const hwy::HWY_NAMESPACE::ScalableTag<float> d;
using Vec = hwy::HWY_NAMESPACE::Vec<decltype(d)>;

void HwyMinMax(const hwy::AlignedNDArray<float, 2>& data, float& min,
               float& max) {
  for (size_t index_0 = 0; index_0 < data.shape()[0]; ++index_0) {
    for (size_t index_1 = 0; index_1 < data.shape()[1]; index_1 += Lanes(d)) {
      const Vec data_vec = Load(d, data[{index_0}].data() + index_1);
      min = std::min(min, ReduceMin(d, data_vec));
      max = std::max(max, ReduceMax(d, data_vec));
    }
  }
}

}  // namespace HWY_NAMESPACE

}  // namespace zimtohrli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zimtohrli {

HWY_EXPORT(HwyMinMax);

namespace {

void GLFWErrorCallback(int error, const char* description) {
  std::cerr << "GLFW error: " << error << ", " << description << std::endl;
}

void GLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                     GLsizei length, const GLchar* message,
                     const void* userParam) {
  std::cerr << "GLDebug callback: " << source << ", " << type << ", " << id
            << ", " << severity << ", " << length << ", " << message
            << std::endl;
}

enum ImageType { ImageTypeSpectrogram, ImageTypeDTW };

enum FileType { FileTypeA, FileTypeB, FileTypeDTW };

enum SpectrogramType {
  SpectrogramTypeEnergyChannelsDB,
  SpectrogramTypePartialEnergyChannelsDB,
  SpectrogramTypeSpectrogram
};

struct Image;

// Manages synchronizing crosshairs for all spectrograms.
using CrosshairManager =
    std::function<void(const std::optional<ImVec2>&, Image&)>;

// Manages synchronizing selections for all spectrograms.
using SelectManager = std::function<void(
    const std::optional<std::pair<ImVec2, ImVec2>>&, Image&)>;

// Manages texture allocation/deallocation.
class Texture {
 public:
  Texture() { glGenTextures(1, &id_); }

  Texture(Texture&& other) : id_(std::exchange(other.id_, 0)) {}
  Texture& operator=(Texture&& other) {
    id_ = std::exchange(other.id_, 0);
    return *this;
  }
  ~Texture() {
    if (id_ != 0) glDeleteTextures(1, &id_);
  }

  GLuint id() const { return id_; }

 private:
  GLuint id_;
};

// Keeps track of low/high bounds of arrays, and linearly scales array content
// to clamped values accordingly.
struct Clamp {
  Clamp() = default;
  explicit Clamp(const hwy::AlignedNDArray<float, 2>& array) { Span(array); }
  void Span(const hwy::AlignedNDArray<float, 2>& array) {
    HWY_DYNAMIC_DISPATCH(HwyMinMax)(array, low, high);
    range_reciprocal = 255 / (high - low);
  }
  uint8_t Uint8(float f) const {
    return static_cast<uint8_t>(std::clamp<float>(
        (f - low) * range_reciprocal, std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max()));
  }
  float low = std::numeric_limits<float>::max();
  float high = std::numeric_limits<float>::min();
  float range_reciprocal = 1.0;
};

// Contains a selection of time steps and CAM channels.
struct Selection {
  size_t left;
  size_t top;
  size_t right;
  size_t bottom;
};

// Computes and uploads an image based on a (width, height)-shaped array.
struct Image {
  Image(Image&& other) = default;
  Image(hwy::AlignedNDArray<float, 2>* array, const Clamp& clamp,
        CrosshairManager crosshair_manager, SelectManager select_manager,
        ImageType image_type, FileType file_type, size_t b_index,
        size_t channel_index, SpectrogramType spectrogram_type)
      : clamp(clamp),
        intensities(array),
        crosshair_manager(crosshair_manager),
        select_manager(select_manager),
        image_type(image_type),
        file_type(file_type),
        file_b_index(b_index),
        audio_channel_index(channel_index),
        spectrogram_type(spectrogram_type) {}
  void RedrawPixels() {
    pixels.reset(
        new uint8_t[static_cast<int>(render_size.x * render_size.y * 4)]);
    render_scale = {
        static_cast<float>(intensities->shape()[0]) / render_size.x,
        static_cast<float>(intensities->shape()[1]) / render_size.y};
    for (size_t render_x = 0; render_x < render_size.x; ++render_x) {
      for (size_t render_y = 0; render_y < render_size.y; ++render_y) {
        uint8_t value = clamp.Uint8(
            (*intensities)[{static_cast<size_t>(render_scale.x * render_x)}]
                          [static_cast<size_t>(render_scale.y * render_y)]);
        uint8_t* data_ptr =
            pixels.get() +
            (static_cast<size_t>(render_y * render_size.x) + render_x) * 4;
        data_ptr[0] = value;
        data_ptr[1] = value;
        data_ptr[2] = value;
        data_ptr[3] = 255;
      }
    }

    glBindTexture(GL_TEXTURE_2D, texture.id());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, static_cast<size_t>(render_size.x),
                 static_cast<size_t>(render_size.y), 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, pixels.get());
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  void SetVerticalLine(absl::string_view label, int x) {
    vertical_lines[label] = x;
    needs_redraw = true;
  }
  void DelVerticalLine(absl::string_view label) {
    vertical_lines.erase(label);
    needs_redraw = true;
  }
  void SetHorizontalLine(absl::string_view label, int y) {
    horizontal_lines[label] = y;
    needs_redraw = true;
  }
  void DelHorizontalLine(absl::string_view label) {
    horizontal_lines.erase(label);
    needs_redraw = true;
  }
  void Invert(uint8_t* in_data_ptr, uint8_t* out_data_ptr) {
    out_data_ptr[0] = 255 - in_data_ptr[0];
    out_data_ptr[1] = 255 - in_data_ptr[1];
    out_data_ptr[2] = 255 - in_data_ptr[2];
    out_data_ptr[3] = 255;
  }
  void Contrast(uint8_t* in_data_ptr, uint8_t* out_data_ptr) {
    if (in_data_ptr[0] * 0.299 + in_data_ptr[1] * 0.578 +
            in_data_ptr[2] * 0.114 >
        186) {
      out_data_ptr[0] = 0;
      out_data_ptr[1] = 0;
      out_data_ptr[2] = 0;
      out_data_ptr[3] = 255;
    } else {
      out_data_ptr[0] = 255;
      out_data_ptr[1] = 255;
      out_data_ptr[2] = 255;
      out_data_ptr[3] = 255;
    }
  }
  void RedrawExtras() {
    glBindTexture(GL_TEXTURE_2D, texture.id());
    if (vertical_lines.empty() && horizontal_lines.empty() &&
        !selected.has_value()) {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, render_size.x, render_size.y,
                      GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
    } else {
      std::vector<uint8_t> painted_data(
          pixels.get(), pixels.get() + static_cast<size_t>(render_size.x *
                                                           render_size.y * 4));
      if (selected.has_value()) {
        const std::pair<size_t, size_t> x_edges = std::minmax(
            selected->left / render_scale.x, selected->right / render_scale.x);
        const std::pair<size_t, size_t> y_edges = std::minmax(
            selected->top / render_scale.y, selected->bottom / render_scale.y);
        for (size_t y = y_edges.first; y < y_edges.second; ++y) {
          const size_t offset = y * render_size.x;
          for (size_t x = x_edges.first; x <= x_edges.second; ++x) {
            const size_t data_index = (offset + x) * 4;
            Invert(pixels.get() + data_index, painted_data.data() + data_index);
          }
        }
      }
      for (const auto& vertical_line : vertical_lines) {
        if (vertical_line.second < 0 || vertical_line.second >= render_size.x) {
          continue;
        }
        for (size_t y = 0; y < render_size.y; ++y) {
          const size_t data_index =
              (y * render_size.x + vertical_line.second) * 4;
          Contrast(pixels.get() + data_index, painted_data.data() + data_index);
        }
      }
      for (const auto& horizontal_line : horizontal_lines) {
        if (horizontal_line.second < 0 ||
            horizontal_line.second >= render_size.y) {
          continue;
        }
        const size_t offset = horizontal_line.second * render_size.x;
        for (size_t x = 0; x < render_size.x; ++x) {
          const size_t data_index = (offset + x) * 4;
          Contrast(pixels.get() + data_index, painted_data.data() + data_index);
        }
      }
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, render_size.x, render_size.y,
                      GL_RGBA, GL_UNSIGNED_BYTE, painted_data.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);
  }
  void SetCrosshair(const ImVec2& position) {
    const absl::string_view crosshair_label = "crosshair";
    SetVerticalLine(crosshair_label, position.x);
    SetHorizontalLine(crosshair_label, position.y);
  }
  void DelCrosshair() {
    const absl::string_view crosshair_label = "crosshair";
    DelVerticalLine(crosshair_label);
    DelHorizontalLine(crosshair_label);
  }
  void SetSelection(const std::optional<Selection>& s) {
    selected = s;
    needs_redraw = true;
  }
  void ProcessMouse(ImGuiIO* io, const ImVec2 negative_offset) {
    if (ImGui::IsItemClicked()) {
      select_manager(std::nullopt, *this);
    }
    const bool hovered = ImGui::IsItemHovered();
    const std::optional<ImVec2> pos =
        hovered ? std::optional<ImVec2>(
                      {std::round(io->MousePos.x - negative_offset.x),
                       std::round(io->MousePos.y - negative_offset.y)})
                : std::nullopt;
    if (pos.has_value() != last_mouse_pos.has_value() ||
        (pos.has_value() &&
         (pos->x != last_mouse_pos->x || pos->y != last_mouse_pos->y))) {
      crosshair_manager(pos, *this);
      last_mouse_pos = pos;
      if (pos.has_value() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        const ImVec2 drag_delta = ImGui::GetMouseDragDelta();
        const std::optional<std::pair<ImVec2, ImVec2>> simplified_selection =
            std::pair<ImVec2, ImVec2>{
                {std::min(std::max(0.0f, pos->x - drag_delta.x), render_size.x),
                 std::min(std::max(0.0f, pos->y - drag_delta.y),
                          render_size.y)},
                pos.value()};
        select_manager(simplified_selection, *this);
      }
    }
  }
  void Paint(ImGuiIO* io, const ImVec2& new_render_size) {
    const ImVec2 new_render_size_floor = {std::floor(new_render_size.x),
                                          std::floor(new_render_size.y)};
    if (new_render_size_floor.x != render_size.x ||
        new_render_size_floor.y != render_size.y) {
      render_size = new_render_size_floor;
      RedrawPixels();
      needs_redraw = true;
    }
    if (needs_redraw) {
      RedrawExtras();
      needs_redraw = false;
    }

    const ImVec2 image_p0 = ImGui::GetCursorScreenPos();
    ImGui::Image((void*)(uintptr_t)texture.id(),
                 {render_size.x, render_size.y});
    ProcessMouse(io, image_p0);
  }

  Clamp clamp;
  hwy::AlignedNDArray<float, 2>* intensities;
  CrosshairManager crosshair_manager;
  SelectManager select_manager;
  absl::flat_hash_map<std::string, int> vertical_lines;
  absl::flat_hash_map<std::string, int> horizontal_lines;
  std::optional<Selection> selected;
  std::optional<ImVec2> last_mouse_pos;
  std::optional<Selection> last_selection;
  bool needs_redraw = false;
  Texture texture;
  ImVec2 render_size = {0, 0};
  std::unique_ptr<uint8_t[]> pixels;
  ImVec2 render_scale = {0, 0};
  ImageType image_type;
  FileType file_type;
  size_t file_b_index;
  size_t audio_channel_index;
  SpectrogramType spectrogram_type;
};

// Something in the spectrogram images highlighted by the mouse crosshairs.
struct SpectrogramMouseHighlight {
  float value;
  std::pair<float, float> frequencies;
  float time;
};

// Contains clamps for each channel and spectrogram type.
// Reused for all files (both A and B) to ensure a uniform intensity scale of
// all images.
struct SpectrogramClamps {
  Clamp energy_channels_db;
  Clamp partial_energy_channels_db;
  Clamp spectrogram;
};

// Contains spectrogram images for a channel.
struct SpectrogramImages {
  SpectrogramImages(Analysis& analysis, const SpectrogramClamps& clamps,
                    CrosshairManager crosshair_manager,
                    SelectManager select_manager, FileType file_type,
                    size_t b_index, size_t channel_index)
      : energy_channels_db(
            &analysis.energy_channels_db, clamps.energy_channels_db,
            crosshair_manager, select_manager, ImageTypeSpectrogram, file_type,
            b_index, channel_index, SpectrogramTypeEnergyChannelsDB),
        partial_energy_channels_db(
            &analysis.partial_energy_channels_db,
            clamps.partial_energy_channels_db, crosshair_manager,
            select_manager, ImageTypeSpectrogram, file_type, b_index,
            channel_index, SpectrogramTypePartialEnergyChannelsDB),
        spectrogram(&analysis.spectrogram, clamps.spectrogram,
                    crosshair_manager, select_manager, ImageTypeSpectrogram,
                    file_type, b_index, channel_index,
                    SpectrogramTypeSpectrogram),
        file_type(file_type),
        file_b_index(b_index),
        audio_channel_index(channel_index) {}
  void Paint(ImGuiIO* io, const ImVec2& size) {
    const ImVec2 spectrogram_size = {size.x / 3, size.y};
    if (ImGui::BeginTable("A", 3)) {
      ImGui::TableNextColumn();
      ImGui::Text("Energy in dB SPL");
      if (highlighted_energy_channels_db.has_value()) {
        ImGui::SameLine();
        ImGui::Text(" (%.2fs, %.2f-%.2fHz, %.2f dB SPL)",
                    highlighted_energy_channels_db->time,
                    highlighted_energy_channels_db->frequencies.first,
                    highlighted_energy_channels_db->frequencies.second,
                    highlighted_energy_channels_db->value);
      }
      ImGui::TableNextColumn();
      ImGui::Text("Partial energy in dB SPL");
      if (highlighted_partial_energy_channels_db.has_value()) {
        ImGui::SameLine();
        ImGui::Text(" (%.2fs, %.2f-%.2fHz, %.2f dB SPL)",
                    highlighted_partial_energy_channels_db->time,
                    highlighted_partial_energy_channels_db->frequencies.first,
                    highlighted_partial_energy_channels_db->frequencies.second,
                    highlighted_partial_energy_channels_db->value);
      }
      ImGui::TableNextColumn();
      ImGui::Text("Partial energy in Phons");
      if (highlighted_spectrogram) {
        ImGui::SameLine();
        ImGui::Text(" (%.2fs, %.2f-%.2fHz, %.2f Phons)",
                    highlighted_spectrogram->time,
                    highlighted_spectrogram->frequencies.first,
                    highlighted_spectrogram->frequencies.second,
                    highlighted_spectrogram->value);
      }

      ImGui::TableNextColumn();
      energy_channels_db.Paint(io, spectrogram_size);
      ImGui::TableNextColumn();
      partial_energy_channels_db.Paint(io, spectrogram_size);
      ImGui::TableNextColumn();
      spectrogram.Paint(io, spectrogram_size);

      ImGui::EndTable();
    }
  }
  Image& GetImage(SpectrogramType type) {
    switch (type) {
      case SpectrogramTypeEnergyChannelsDB:
        return energy_channels_db;
      case SpectrogramTypePartialEnergyChannelsDB:
        return partial_energy_channels_db;
      default:
        return spectrogram;
    }
  }
  Image energy_channels_db;
  std::optional<SpectrogramMouseHighlight> highlighted_energy_channels_db;
  Image partial_energy_channels_db;
  std::optional<SpectrogramMouseHighlight>
      highlighted_partial_energy_channels_db;
  Image spectrogram;
  std::optional<SpectrogramMouseHighlight> highlighted_spectrogram;
  FileType file_type;
  size_t file_b_index;
  size_t audio_channel_index;
};

// Something in the DTW images highlighted by the mouse crosshairs.
struct DTWMouseHighlight {
  float time_a;
  float time_b;
};

// Generates a matrix of pixels for a dynamic time warp output, highlighting the
// optimal unwarped path between two sequences.
hwy::AlignedNDArray<float, 2> DTWPixels(
    const std::vector<std::pair<size_t, size_t>>& dtw) {
  const size_t first_a = dtw.front().first;
  const size_t first_b = dtw.front().second;
  const size_t last_a = dtw.back().first;
  const size_t last_b = dtw.back().second;
  hwy::AlignedNDArray<float, 2> pixels(
      {last_a - first_a + 1, last_b - first_b + 1});
  const size_t min_length = std::min(dtw.back().first, dtw.back().second);
  for (size_t p = 0; p < min_length; ++p) {
    pixels[{p}][p] = 0.5f;
  }
  for (const auto& [a, b] : dtw) {
    pixels[{a - first_a}][b - first_b] = 1.0f;
  }
  return pixels;
}

// Contains DTW images for a channel.
struct DTWImages {
  // Moves all fields, and then sets the intensities-pointers in the images to
  // the moved matrices to maintain coherence.
  DTWImages(DTWImages&& other)
      : energy_channels_db_pixels(std::move(other.energy_channels_db_pixels)),
        partial_energy_channels_db_pixels(
            std::move(other.partial_energy_channels_db_pixels)),
        spectrogram_pixels(std::move(other.spectrogram_pixels)),
        energy_channels_db(std::move(other.energy_channels_db)),
        partial_energy_channels_db(std::move(other.partial_energy_channels_db)),
        spectrogram(std::move(other.spectrogram)) {
    energy_channels_db.intensities = &energy_channels_db_pixels;
    partial_energy_channels_db.intensities = &partial_energy_channels_db_pixels;
    spectrogram.intensities = &spectrogram_pixels;
  }
  DTWImages(const AnalysisDTW& analysis, CrosshairManager crosshair_manager,
            SelectManager select_manager, size_t b_index, size_t channel_index)
      : energy_channels_db_pixels(DTWPixels(analysis.energy_channels_db)),
        partial_energy_channels_db_pixels(
            DTWPixels(analysis.partial_energy_channels_db)),
        spectrogram_pixels(DTWPixels(analysis.spectrogram)),
        energy_channels_db(&energy_channels_db_pixels,
                           Clamp(energy_channels_db_pixels), crosshair_manager,
                           select_manager, ImageTypeDTW, FileTypeDTW, b_index,
                           channel_index, SpectrogramTypeEnergyChannelsDB),
        partial_energy_channels_db(
            &partial_energy_channels_db_pixels,
            Clamp(partial_energy_channels_db_pixels), crosshair_manager,
            select_manager, ImageTypeDTW, FileTypeDTW, b_index, channel_index,
            SpectrogramTypePartialEnergyChannelsDB),
        spectrogram(&spectrogram_pixels, Clamp(spectrogram_pixels),
                    crosshair_manager, select_manager, ImageTypeDTW,
                    FileTypeDTW, b_index, channel_index,
                    SpectrogramTypeSpectrogram) {}
  void Paint(ImGuiIO* io, const ImVec2& size) {
    const ImVec2 dtw_size = {size.x / 3, size.y};
    if (ImGui::BeginTable("A", 3)) {
      ImGui::TableNextColumn();
      ImGui::Text("Energy in dB SPL");
      if (highlighted_energy_channels_db.has_value()) {
        ImGui::SameLine();
        ImGui::Text(" (A: %.2fs, B: %.2fs)",
                    highlighted_energy_channels_db->time_a,
                    highlighted_energy_channels_db->time_b);
      }
      ImGui::TableNextColumn();
      ImGui::Text("Partial energy in dB SPL");
      if (highlighted_partial_energy_channels_db.has_value()) {
        ImGui::SameLine();
        ImGui::Text(" (A: %.2fs, B: %.2fs)",
                    highlighted_partial_energy_channels_db->time_a,
                    highlighted_partial_energy_channels_db->time_b);
      }
      ImGui::TableNextColumn();
      ImGui::Text("Partial energy in Phons");
      if (highlighted_spectrogram) {
        ImGui::SameLine();
        ImGui::Text(" (A: %.2fs, B: %.2fs)", highlighted_spectrogram->time_a,
                    highlighted_spectrogram->time_b);
      }

      ImGui::TableNextColumn();
      energy_channels_db.Paint(io, dtw_size);
      ImGui::TableNextColumn();
      partial_energy_channels_db.Paint(io, dtw_size);
      ImGui::TableNextColumn();
      spectrogram.Paint(io, dtw_size);

      ImGui::EndTable();
    }
  }
  hwy::AlignedNDArray<float, 2> energy_channels_db_pixels;
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db_pixels;
  hwy::AlignedNDArray<float, 2> spectrogram_pixels;
  Image energy_channels_db;
  Image partial_energy_channels_db;
  Image spectrogram;
  std::optional<DTWMouseHighlight> highlighted_energy_channels_db;
  std::optional<DTWMouseHighlight> highlighted_partial_energy_channels_db;
  std::optional<DTWMouseHighlight> highlighted_spectrogram;
};

// Callback to run on the main thread.
using RenderCallback = std::function<void()>;

// Executor able to run render callbacks on the main thread.
using RenderCallbackExecutor = std::function<void(RenderCallback)>;

// An atomic bool that will be reset to zero if it's moved. Not used for any
// critically thread safe stuff, only to avoid undefined behavior with play
// buttons that disable while the sound is playing.
class MovableAtomicBool {
 public:
  MovableAtomicBool(bool other) : value_(other) {}
  MovableAtomicBool(MovableAtomicBool&& other) { value_ = false; }
  MovableAtomicBool& operator=(bool other) {
    value_ = other;
    return *this;
  }
  operator bool() const { return value_; }

 private:
  std::atomic<bool> value_ = false;
};

// Contains data and functions to present a file.
struct FilePresentation {
  FilePresentation(std::string name, std::vector<Analysis> analysis,
                   absl::Span<const SpectrogramClamps> clamps,
                   std::optional<AudioBuffer> audio,
                   RenderCallbackExecutor callback_executor,
                   CrosshairManager crosshair_manager,
                   SelectManager select_manager,
                   hwy::AlignedNDArray<float, 2>* thresholds_hz,
                   FileType file_type, size_t b_index)
      : name(std::move(name)),
        render_callback_executor(callback_executor),
        analysis(std::move(analysis)),
        audio(std::move(audio)),
        thresholds_hz(thresholds_hz),
        file_type(file_type),
        file_b_index(b_index) {
    for (size_t channel_index = 0; channel_index < this->analysis.size();
         ++channel_index) {
      images.emplace_back(this->analysis[channel_index], clamps[channel_index],
                          crosshair_manager, select_manager, file_type, b_index,
                          channel_index);
    }
  }
  void Paint(ImGuiIO* io, size_t channel_index, const ImVec2& size) {
    ImGui::Text("%.*s channel %zu", static_cast<int>(name.size()), name.data(),
                channel_index);

    images[channel_index].Paint(io, {size.x, size.y});
  }
  Filterbank CreateFilter(const std::pair<float, float> thresholds) {
    Cam cam;
    return Filterbank(
        {DigitalSOSBandPass(cam.filter_order, cam.filter_pass_band_ripple,
                            cam.filter_stop_band_ripple, thresholds.first,
                            thresholds.second, audio->SampleRate())});
  }
  // Computes the selected audio in this file presentation by converting
  // the selection to a frequency and time range, then computing a Zimtohrli
  // filter using those ranges, and then applying the filter to each audio
  // channel.
  void ComputeSelectedAudio() {
    const float frames_per_step =
        static_cast<float>(audio->Frames().shape()[1]) /
        images[0].energy_channels_db.intensities->shape()[0];
    const std::pair<size_t, size_t> threshold_frames =
        std::minmax(selected_coordinates->left, selected_coordinates->right);
    const size_t frame_offset = threshold_frames.first * frames_per_step;
    const size_t num_frames =
        (threshold_frames.second - threshold_frames.first) * frames_per_step;
    const size_t num_channels = audio->Frames().shape()[0];
    hwy::AlignedNDArray<float, 2> audio_slice({num_channels, num_frames});
    hwy::AlignedNDArray<float, 2> filtered_audio_channel({num_frames, 1});
    const std::pair<size_t, size_t> threshold_channels =
        std::minmax(selected_coordinates->top, selected_coordinates->bottom);
    const float left_hz = (*thresholds_hz)[{0}][threshold_channels.first];
    const float right_hz = (*thresholds_hz)[{2}][threshold_channels.second];
    Filterbank filter = CreateFilter({left_hz, right_hz});
    for (size_t channel_index = 0; channel_index < num_channels;
         ++channel_index) {
      std::memcpy(audio_slice[{channel_index}].data(),
                  audio->Frames()[{channel_index}].data() + frame_offset,
                  sizeof(float) * num_frames);
      filter.Filter(audio_slice[{channel_index}], filtered_audio_channel);
      for (size_t frame_index = 0; frame_index < num_frames; ++frame_index) {
        audio_slice[{channel_index}][frame_index] =
            filtered_audio_channel[{frame_index}][channel_index];
      }
    }
    selected_audio = AudioBuffer{.sample_rate = audio->SampleRate(),
                                 .frames = std::move(audio_slice)};
  }
  // Plays the audio (only the selection, if any) of this FilePresentation, and
  // creates a playback callback that moves a vertical line across the
  // spectrogram images in sync with the (internal, unfortunately not guarantee
  // to be in sync with actual audio output) playout.
  void Play() {
    if (!audio.has_value()) {
      return;
    }
    playing = true;
    AudioBuffer* to_play = &audio.value();
    int step_offset = 0;
    if (selected_coordinates.has_value()) {
      step_offset =
          std::min(selected_coordinates->left, selected_coordinates->right);
      if (!selected_audio.has_value()) {
        ComputeSelectedAudio();
      }
      to_play = &selected_audio.value();
    }
    const absl::StatusOr<PaStreamInfo> play_result =
        to_play->Play([&, step_offset](bool still_playing, size_t frame_index,
                                       const PaStreamInfo info) {
          render_callback_executor([this, step_offset, still_playing,
                                    frame_index, info]() {
            playing = still_playing;
            const int step_index =
                (frame_index -
                 static_cast<int>(info.outputLatency * audio->SampleRate())) *
                images[0].energy_channels_db.intensities->shape()[0] /
                audio->Frames().shape()[1];
            const int render_index =
                (step_offset + step_index) /
                images[0].energy_channels_db.render_scale.x;
            const absl::string_view line_label = "playing";
            const auto update_play_progress = [&](FilePresentation* file) {
              for (size_t channel_index = 0; channel_index < analysis.size();
                   ++channel_index) {
                if (still_playing) {
                  file->images[channel_index]
                      .energy_channels_db.SetVerticalLine(line_label,
                                                          render_index);
                  file->images[channel_index]
                      .partial_energy_channels_db.SetVerticalLine(line_label,
                                                                  render_index);
                  file->images[channel_index].spectrogram.SetVerticalLine(
                      line_label, render_index);
                } else {
                  file->images[channel_index]
                      .energy_channels_db.DelVerticalLine(line_label);
                  file->images[channel_index]
                      .partial_energy_channels_db.DelVerticalLine(line_label);
                  file->images[channel_index].spectrogram.DelVerticalLine(
                      line_label);
                }
              }
            };
            update_play_progress(this);
            for (FilePresentation* coupled : coupled_play_progress) {
              update_play_progress(coupled);
            }
          });
        });
    if (!play_result.ok()) {
      std::cerr << "Playing audio: " << play_result.status().message()
                << std::endl;
    }
  }
  void EachSpectrogram(std::function<void(SpectrogramImages&, Analysis&)> f) {
    for (size_t channel_index = 0; channel_index < images.size();
         ++channel_index) {
      f(images[channel_index], analysis[channel_index]);
    }
  }

  std::string name;
  RenderCallbackExecutor render_callback_executor;
  std::vector<Analysis> analysis;
  std::optional<AudioBuffer> audio;
  std::vector<SpectrogramImages> images;
  MovableAtomicBool playing = false;
  std::optional<Selection> selected_coordinates;
  std::optional<AudioBuffer> selected_audio;
  hwy::AlignedNDArray<float, 2>* thresholds_hz;
  std::vector<FilePresentation*> coupled_play_progress;
  FileType file_type;
  size_t file_b_index;
};

struct HighlightedDTW {
  float time_a;
  float time_b;
};

// Contains data and functions to present a dynamic time warp.
struct DTWPresentation {
  DTWPresentation(DTWPresentation&& other) = default;
  DTWPresentation(const std::string& file_b_name,
                  const std::vector<AnalysisDTW>& dtw,
                  float time_resolution_frequency,
                  CrosshairManager crosshair_manager,
                  SelectManager select_manager, size_t b_index,
                  std::optional<size_t> unwarp_window)
      : file_b_name(file_b_name),
        time_resolution_frequency(time_resolution_frequency),
        unwarp_window(unwarp_window) {
    for (size_t channel_index = 0; channel_index < dtw.size();
         ++channel_index) {
      images.emplace_back(dtw[channel_index], crosshair_manager, select_manager,
                          b_index, channel_index);
    }
  }
  void Paint(ImGuiIO* io, size_t channel_index, const ImVec2& size) {
    ImGui::Text("Dynamic time warp between A and %s channel %zu, %.2fs windows",
                file_b_name.c_str(), channel_index,
                static_cast<float>(unwarp_window.value_or(0) /
                                   time_resolution_frequency));
    images[channel_index].Paint(io, size);
  }
  std::string file_b_name;
  std::vector<hwy::AlignedNDArray<float, 2>> image_pixels;
  float time_resolution_frequency;
  std::vector<DTWImages> images;
  CrosshairManager crosshair_manager;
  SelectManager select_manager;
  std::optional<HighlightedDTW> highlighted_dtw;
  std::optional<size_t> unwarp_window;
};

// Creates and contains clamps for all channels.
struct ComparisonClamps {
  ComparisonClamps(const FileComparison& comparison) {
    for (size_t channel_index = 0;
         channel_index < comparison.comparison.analysis_a.size();
         ++channel_index) {
      SpectrogramClamps signal_clamps;
      signal_clamps.energy_channels_db.Span(
          comparison.comparison.analysis_a[channel_index].energy_channels_db);
      signal_clamps.partial_energy_channels_db.Span(
          comparison.comparison.analysis_a[channel_index]
              .partial_energy_channels_db);
      signal_clamps.spectrogram.Span(
          comparison.comparison.analysis_a[channel_index].spectrogram);
      for (const auto& constituent : comparison.comparison.analysis_b) {
        signal_clamps.energy_channels_db.Span(
            constituent[channel_index].energy_channels_db);
        signal_clamps.partial_energy_channels_db.Span(
            constituent[channel_index].partial_energy_channels_db);
        signal_clamps.spectrogram.Span(constituent[channel_index].spectrogram);
      }
      signal_channel_clamps.push_back(std::move(signal_clamps));
      SpectrogramClamps absolute_delta_clamps;
      for (const auto& constituent :
           comparison.comparison.analysis_absolute_delta) {
        absolute_delta_clamps.energy_channels_db.Span(
            constituent[channel_index].energy_channels_db);
        absolute_delta_clamps.partial_energy_channels_db.Span(
            constituent[channel_index].partial_energy_channels_db);
        absolute_delta_clamps.spectrogram.Span(
            constituent[channel_index].spectrogram);
      }
      absolute_delta_channel_clamps.push_back(std::move(absolute_delta_clamps));
      SpectrogramClamps relative_delta_clamps;
      for (const auto& constituent :
           comparison.comparison.analysis_relative_delta) {
        relative_delta_clamps.energy_channels_db.Span(
            constituent[channel_index].energy_channels_db);
        relative_delta_clamps.partial_energy_channels_db.Span(
            constituent[channel_index].partial_energy_channels_db);
        relative_delta_clamps.spectrogram.Span(
            constituent[channel_index].spectrogram);
      }
      relative_delta_channel_clamps.push_back(std::move(relative_delta_clamps));
    }
  }
  std::vector<SpectrogramClamps> signal_channel_clamps;
  std::vector<SpectrogramClamps> absolute_delta_channel_clamps;
  std::vector<SpectrogramClamps> relative_delta_channel_clamps;
};

std::vector<std::string> GetFilePaths(const std::vector<AudioFile>& files) {
  std::vector<std::string> result;
  result.reserve(files.size());
  for (size_t index = 0; index < files.size(); ++index) {
    result.push_back(files[index].Path());
  }
  return result;
}

std::vector<AudioBuffer> GetFileBuffers(std::vector<AudioFile> files) {
  std::vector<AudioBuffer> result;
  result.reserve(files.size());
  for (size_t index = 0; index < files.size(); ++index) {
    result.push_back(std::move(files[index]).ToBuffer());
  }
  return result;
}

std::vector<FilePresentation> GetFilePresentations(
    std::string title_prefix, std::vector<std::string> paths,
    std::vector<AudioBuffer> audio_buffers,
    std::vector<std::vector<Analysis>> analysis,
    const std::vector<SpectrogramClamps>& clamps,
    RenderCallbackExecutor callback_executor,
    CrosshairManager crosshair_manager, SelectManager select_manager,
    hwy::AlignedNDArray<float, 2>* thresholds_hz, FileType file_type) {
  std::vector<FilePresentation> results;
  for (size_t b_index = 0; b_index < analysis.size(); ++b_index) {
    std::optional<AudioBuffer> audio =
        audio_buffers.empty()
            ? std::nullopt
            : std::optional<AudioBuffer>(std::move(audio_buffers[b_index]));
    FilePresentation presentation(
        absl::StrCat(title_prefix, " (", paths[b_index], ")"),
        std::move(analysis[b_index]), clamps, std::move(audio),
        callback_executor, crosshair_manager, select_manager, thresholds_hz,
        file_type, b_index);
    results.push_back(std::move(presentation));
  }
  return results;
}

std::vector<AudioBuffer> GetFramesDeltaBuffers(
    std::vector<hwy::AlignedNDArray<float, 2>> frames_vector,
    float sample_rate) {
  std::vector<AudioBuffer> result;
  result.reserve(frames_vector.size());
  for (size_t index = 0; index < frames_vector.size(); ++index) {
    result.push_back(AudioBuffer{.sample_rate = sample_rate,
                                 .frames = std::move(frames_vector[index])});
  }
  return result;
}

std::vector<DTWPresentation> GetDTWPresentations(
    const std::vector<FilePresentation>& file_b_vector,
    const std::vector<std::vector<AnalysisDTW>>& dtw,
    float time_resolution_frequency, CrosshairManager crosshair_manager,
    SelectManager select_manager, std::optional<size_t> unwarp_window) {
  std::vector<DTWPresentation> result;
  result.reserve(dtw.size());
  for (size_t b_index = 0; b_index < dtw.size(); ++b_index) {
    result.emplace_back(file_b_vector[b_index].name, dtw[b_index],
                        time_resolution_frequency, crosshair_manager,
                        select_manager, b_index, unwarp_window);
  }
  return result;
}

// Contains positions in all spectrograms for a given time, according to the
// DTW.
struct Steps {
  size_t energy_channels_db;
  size_t partial_energy_channels_db;
  size_t spectrogram;
  size_t GetStep(SpectrogramType type) {
    switch (type) {
      case SpectrogramTypeEnergyChannelsDB:
        return energy_channels_db;
      case SpectrogramTypePartialEnergyChannelsDB:
        return partial_energy_channels_db;
      case SpectrogramTypeSpectrogram:
        return spectrogram;
      default:
        return -1;
    }
  }
};

// Maps steps that correspond to the same time between two sequences, in this
// case the spectrograms of two audio channels.
struct DTWMapping {
  // Creates a mapping for a given DTW (always from file A to file B) and stores
  // it as unordered maps in the right direction depending of whether we want a
  // mapping from A to B or from B to A.
  DTWMapping(const AnalysisDTW& dtw, bool a_to_b) {
    for (const auto& pair : dtw.energy_channels_db) {
      if (a_to_b) {
        energy_channels_db[pair.first] = pair.second;
      } else {
        energy_channels_db[pair.second] = pair.first;
      }
    }
    for (const auto& pair : dtw.partial_energy_channels_db) {
      if (a_to_b) {
        partial_energy_channels_db[pair.first] = pair.second;
      } else {
        partial_energy_channels_db[pair.second] = pair.first;
      }
    }
    for (const auto& pair : dtw.spectrogram) {
      if (a_to_b) {
        spectrogram[pair.first] = pair.second;
      } else {
        spectrogram[pair.second] = pair.first;
      }
    }
  }
  // Creates a "fake" mapping assuming both sequences are of identical length
  // and have no time warp. Used when no DTW was produced.
  Steps Map(size_t step_index) {
    return {
        .energy_channels_db = energy_channels_db.contains(step_index)
                                  ? energy_channels_db[step_index]
                                  : 0,
        .partial_energy_channels_db =
            partial_energy_channels_db.contains(step_index)
                ? partial_energy_channels_db[step_index]
                : 0,
        .spectrogram =
            spectrogram.contains(step_index) ? spectrogram[step_index] : 0,
    };
  }
  absl::flat_hash_map<size_t, size_t> energy_channels_db;
  absl::flat_hash_map<size_t, size_t> partial_energy_channels_db;
  absl::flat_hash_map<size_t, size_t> spectrogram;
};

std::vector<std::vector<DTWMapping>> GetDTWMappings(
    const std::vector<std::vector<AnalysisDTW>>& dtw_vector, bool a_to_b) {
  std::vector<std::vector<DTWMapping>> result;
  result.reserve(dtw_vector.size());
  for (const auto& file_dtw : dtw_vector) {
    std::vector<DTWMapping> file_mappings;
    file_mappings.reserve(file_dtw.size());
    for (const auto& channel_dtw : file_dtw) {
      file_mappings.push_back(DTWMapping(channel_dtw, a_to_b));
    }
    result.push_back(std::move(file_mappings));
  }
  return result;
}

// Contains positions for all spectorgrams for an audio channel.
struct Positions {
  ImVec2 energy_channels_db;
  ImVec2 partial_energy_channels_db;
  ImVec2 spectrogram;
};

namespace detail {

template <typename T, typename... Args>
auto tuple_append(T&& t, Args&&... args) {
  return std::tuple_cat(std::forward<T>(t), std::forward_as_tuple(args...));
}

}  // namespace detail

template <typename F, typename... FrontArgs>
decltype(auto) bind_front(F&& f, FrontArgs&&... frontArgs) {
  return [f = std::forward<F>(f),
          frontArgs = std::make_tuple(std::forward<FrontArgs>(frontArgs)...)](
             auto&&... backArgs) {
    return std::apply(
        f, detail::tuple_append(frontArgs,
                                std::forward<decltype(backArgs)>(backArgs)...));
  };
}

// Contains the graphical elements we want to render along with the logic to
// render them.
struct RenderContext {
  RenderContext(ImGuiIO* io, FileComparison comparison)
      : io(io),
        file_b_paths(GetFilePaths(comparison.file_b)),
        clamps(comparison),
        thresholds_hz(std::move(comparison.thresholds_hz)),
        file_a(absl::StrCat("A (", comparison.file_a.Path(), ")"),
               std::move(comparison.comparison.analysis_a),
               clamps.signal_channel_clamps,
               std::move(comparison.file_a).ToBuffer(),
               bind_front(&RenderContext::EnqueueCallback, this),
               bind_front(&RenderContext::ManageSpectrogramCrosshairs, this),
               bind_front(&RenderContext::ManageSpectrogramSelect, this),
               &this->thresholds_hz, FileTypeA, -1),
        file_b_vector(GetFilePresentations(
            "B", file_b_paths, GetFileBuffers(std::move(comparison.file_b)),
            std::move(comparison.comparison.analysis_b),
            clamps.signal_channel_clamps,
            bind_front(&RenderContext::EnqueueCallback, this),
            bind_front(&RenderContext::ManageSpectrogramCrosshairs, this),
            bind_front(&RenderContext::ManageSpectrogramSelect, this),
            &this->thresholds_hz, FileTypeB)),
        file_absolute_delta_vector(GetFilePresentations(
            "Absolute delta between A and B", file_b_paths, {},
            std::move(comparison.comparison.analysis_absolute_delta),
            clamps.absolute_delta_channel_clamps,
            bind_front(&RenderContext::EnqueueCallback, this),
            bind_front(&RenderContext::ManageSpectrogramCrosshairs, this),
            bind_front(&RenderContext::ManageSpectrogramSelect, this),
            &this->thresholds_hz, FileTypeA)),
        file_relative_delta_vector(GetFilePresentations(
            "Relative delta between A and B ", file_b_paths,
            GetFramesDeltaBuffers(std::move(comparison.comparison.frames_delta),
                                  file_a.audio->SampleRate()),
            std::move(comparison.comparison.analysis_relative_delta),
            clamps.relative_delta_channel_clamps,
            bind_front(&RenderContext::EnqueueCallback, this),
            bind_front(&RenderContext::ManageSpectrogramCrosshairs, this),
            bind_front(&RenderContext::ManageSpectrogramSelect, this),
            &this->thresholds_hz, FileTypeA)),
        time_resolution_frequency(comparison.time_resolution_frequency),
        dtw_vector(GetDTWPresentations(
            file_b_vector, comparison.comparison.dtw,
            comparison.time_resolution_frequency,
            bind_front(&RenderContext::ManageDTWCrosshairs, this),
            bind_front(&RenderContext::ManageDTWSelect, this),
            comparison.unwarp_window)),
        a_to_b(GetDTWMappings(comparison.comparison.dtw, /*a_to_b=*/true)),
        b_to_a(GetDTWMappings(comparison.comparison.dtw, /*a_to_b=*/false)),
        unwarp_window(comparison.unwarp_window) {
    for (size_t b_index = 0; b_index < file_b_vector.size(); ++b_index) {
      file_absolute_delta_vector[b_index].coupled_play_progress.push_back(
          &file_relative_delta_vector[b_index]);
    }
  }
  void Paint() {
    {
      const absl::MutexLock lock(&callback_mutex);
      RenderCallback callback = nullptr;
      for (; !callbacks.empty(); callbacks.pop()) {
        callbacks.front()();
      }
    }
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->Size);
    ImGui::Begin("Zimtohrli", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Play A") && !file_a.playing) {
      file_a.Play();
    }

    ImGui::SameLine();

    if (ImGui::Button("Play B") && !file_b_vector[selected_b_index].playing) {
      file_b_vector[selected_b_index].Play();
    }

    ImGui::SameLine();

    if (ImGui::Button("Play delta") &&
        !file_relative_delta_vector[selected_b_index].playing) {
      file_relative_delta_vector[selected_b_index].Play();
    }

    const ImVec2 display_size = io->DisplaySize;
    const ImVec2 size_per_row = {display_size.x - 30,
                                 (display_size.y - 250) / 3};
    if (ImGui::BeginTabBar("Channels")) {
      for (size_t channel_index = 0; channel_index < file_a.analysis.size();
           ++channel_index) {
        if (ImGui::BeginTabItem(
                absl::StrCat("Channel ", channel_index).c_str())) {
          file_a.Paint(io, channel_index, size_per_row);
          if (ImGui::BeginTabBar("B")) {
            for (size_t b_index = 0; b_index < file_b_vector.size();
                 ++b_index) {
              if (ImGui::BeginTabItem(
                      absl::StrCat("B (", b_index + 1, ")").c_str(), nullptr,
                      ImGui::IsKeyPressed(static_cast<ImGuiKey>(
                          static_cast<size_t>(ImGuiKey_1) + b_index))
                          ? ImGuiTabItemFlags_SetSelected
                          : 0)) {
                selected_b_index = b_index;
                if (ImGui::BeginTabBar("Delta")) {
                  if (ImGui::BeginTabItem("Relative delta (Q)", nullptr,
                                          ImGui::IsKeyPressed(ImGuiKey_Q)
                                              ? ImGuiTabItemFlags_SetSelected
                                              : 0)) {
                    file_relative_delta_vector[b_index].Paint(io, channel_index,
                                                              size_per_row);
                    ImGui::EndTabItem();
                  }
                  if (ImGui::BeginTabItem("Absolute delta (W)", nullptr,
                                          ImGui::IsKeyPressed(ImGuiKey_W)
                                              ? ImGuiTabItemFlags_SetSelected
                                              : 0)) {
                    file_absolute_delta_vector[b_index].Paint(io, channel_index,
                                                              size_per_row);
                    ImGui::EndTabItem();
                  }
                  if (unwarp_window.has_value()) {
                    if (ImGui::BeginTabItem("Time warp (E)", nullptr,
                                            ImGui::IsKeyPressed(ImGuiKey_E)
                                                ? ImGuiTabItemFlags_SetSelected
                                                : 0)) {
                      dtw_vector[b_index].Paint(io, channel_index,
                                                size_per_row);
                      ImGui::EndTabItem();
                    }
                  }
                  ImGui::EndTabBar();
                }
                file_b_vector[b_index].Paint(io, channel_index, size_per_row);
                ImGui::EndTabItem();
              }
            }
          }
          ImGui::EndTabBar();
          ImGui::EndTabItem();
        }
      }
      ImGui::EndTabBar();
    }

    ImGui::End();
  }
  // Enqueues a callback to be executed on the next paint, to ensure we do
  // everything paint-related in the same main thread.
  void EnqueueCallback(RenderCallback callback) {
    const absl::MutexLock lock(&callback_mutex);
    callbacks.push(callback);
  }
  // Handles mouse selections in the DTW images, currently unused.
  void ManageDTWSelect(const std::optional<std::pair<ImVec2, ImVec2>>& selected,
                       Image& image) {}
  void EachDTWImage(std::function<void(DTWImages&)> f) {
    for (auto& image : dtw_vector) {
      for (auto& channel_image : image.images) {
        f(channel_image);
      }
    }
  }
  // Manages mouse movements in the DTW images. Will create vertical lines in
  // the A and B spectrograms at the times corresponding to the position in the
  // DTW image.
  void ManageDTWCrosshairs(const std::optional<ImVec2>& position,
                           Image& crosshair_image) {
    if (position.has_value()) {
      const float step_a = position->x * crosshair_image.render_scale.x;
      const float time_a = step_a / time_resolution_frequency;
      const float step_b = position->y * crosshair_image.render_scale.y;
      const float time_b = step_b / time_resolution_frequency;
      EachSpectrogram([&](SpectrogramImages& image, const Analysis& analysis) {
        if (image.file_type == FileTypeA) {
          const ImVec2 pos(step_a / image.spectrogram.render_scale.x, -1);
          image.energy_channels_db.SetCrosshair(pos);
          image.partial_energy_channels_db.SetCrosshair(pos);
          image.spectrogram.SetCrosshair(pos);
        } else {
          const ImVec2 pos(step_b / image.spectrogram.render_scale.x, -1);
          image.energy_channels_db.SetCrosshair(pos);
          image.partial_energy_channels_db.SetCrosshair(pos);
          image.spectrogram.SetCrosshair(pos);
        }
      });
      EachDTWImage([&](DTWImages& image) {
        image.energy_channels_db.SetCrosshair(position.value());
        image.partial_energy_channels_db.SetCrosshair(position.value());
        image.spectrogram.SetCrosshair(position.value());
        image.highlighted_energy_channels_db = {.time_a = time_a,
                                                .time_b = time_b};
        image.highlighted_partial_energy_channels_db = {.time_a = time_a,
                                                        .time_b = time_b};
        image.highlighted_spectrogram = {.time_a = time_a, .time_b = time_b};
      });
    } else {
      EachDTWImage([&](DTWImages& image) {
        image.energy_channels_db.DelCrosshair();
        image.partial_energy_channels_db.DelCrosshair();
        image.spectrogram.DelCrosshair();
        image.highlighted_energy_channels_db = std::nullopt;
        image.highlighted_partial_energy_channels_db = std::nullopt;
        image.highlighted_spectrogram = std::nullopt;
      });
      EachSpectrogram([&](SpectrogramImages& image, const Analysis& analysis) {
        image.energy_channels_db.DelCrosshair();
        image.partial_energy_channels_db.DelCrosshair();
        image.spectrogram.DelCrosshair();
      });
    }
  }
  void EachFilePresentation(std::function<void(FilePresentation&)> f) {
    f(file_a);
    for (size_t b_index = 0; b_index < file_b_vector.size(); ++b_index) {
      f(file_b_vector[b_index]);
      f(file_absolute_delta_vector[b_index]);
      f(file_relative_delta_vector[b_index]);
    }
  }
  void EachSpectrogram(
      std::function<void(SpectrogramImages&, const Analysis&)> f) {
    EachFilePresentation(
        [&](FilePresentation& file) { file.EachSpectrogram(f); });
  }
  // Maps a selection in one image to a selection in another, using the DTW.
  Selection GetMappedSelection(const std::pair<ImVec2, ImVec2>& from_selection,
                               const Image& from_image,
                               SpectrogramImages& to_image) {
    const size_t mapped_start =
        GetMappedSteps(static_cast<size_t>(from_selection.first.x *
                                           from_image.render_scale.x),
                       from_image, to_image)
            .GetStep(from_image.spectrogram_type);
    size_t mapped_end =
        GetMappedSteps(static_cast<size_t>(from_selection.second.x *
                                           from_image.render_scale.x),
                       from_image, to_image)
            .GetStep(from_image.spectrogram_type);
    if (mapped_end > mapped_start !=
        from_selection.second.x > from_selection.first.x) {
      mapped_end = to_image.GetImage(from_image.spectrogram_type)
                       .intensities->shape()[0] -
                   1;
    }
    return {
        .left = mapped_start,
        .top = static_cast<size_t>(from_selection.first.y *
                                   from_image.render_scale.y),
        .right = mapped_end,
        .bottom = static_cast<size_t>(from_selection.second.y *
                                      from_image.render_scale.y),
    };
  }
  // Handles mouse selections in spectrograms. Will create selection graphics in
  // all spectrograms that match the same frequency range and DTW-mapped time
  // range.
  void ManageSpectrogramSelect(
      const std::optional<std::pair<ImVec2, ImVec2>>& selected, Image& image) {
    EachFilePresentation([&](FilePresentation& file) {
      std::optional<Selection> mapped_selection;
      if (selected.has_value()) {
        mapped_selection = GetMappedSelection(
            selected.value(), image, file.images[image.audio_channel_index]);
        file.selected_audio = std::nullopt;
        file.selected_coordinates = mapped_selection;
      } else {
        file.selected_audio = std::nullopt;
        file.selected_coordinates = std::nullopt;
      }
      file.EachSpectrogram(
          [&](SpectrogramImages& image, const Analysis& analysis) {
            image.energy_channels_db.SetSelection(mapped_selection);
            image.partial_energy_channels_db.SetSelection(mapped_selection);
            image.spectrogram.SetSelection(mapped_selection);
          });
    });
  }
  // Maps a time step in one image to a time step in another, according to the
  // DTW.
  Steps GetMappedSteps(size_t from_step_index, const Image& from_image,
                       const SpectrogramImages& to_image) {
    if (from_image.file_type == to_image.file_type) {
      if (from_image.file_type == FileTypeA) {
        return {
            .energy_channels_db = from_step_index,
            .partial_energy_channels_db = from_step_index,
            .spectrogram = from_step_index,
        };
      } else {
        size_t a_step_index =
            b_to_a[from_image.file_b_index][from_image.audio_channel_index]
                .Map(from_step_index)
                .GetStep(from_image.spectrogram_type);
        Steps result =
            a_to_b[to_image.file_b_index][from_image.audio_channel_index].Map(
                a_step_index);
        return result;
      }
    } else {
      if (from_image.file_type == FileTypeA) {
        return a_to_b[to_image.file_b_index][from_image.audio_channel_index]
            .Map(from_step_index);
      } else {
        return b_to_a[from_image.file_b_index][from_image.audio_channel_index]
            .Map(from_step_index);
      }
    }
  }
  // Computes positions in images based on time step, CAM channel (y-axis in the
  // images), and render scales.
  Positions ComputePositions(const Steps& steps, size_t cam_channel_index,
                             const ImVec2& render_scale) {
    if (render_scale.x == 0 || render_scale.y == 0) {
      return {};
    }
    return {
        .energy_channels_db = ImVec2(
            static_cast<float>(steps.energy_channels_db) / render_scale.x,
            static_cast<float>(cam_channel_index) / render_scale.y),
        .partial_energy_channels_db =
            ImVec2(static_cast<float>(steps.partial_energy_channels_db) /
                       render_scale.x,
                   static_cast<float>(cam_channel_index) / render_scale.y),
        .spectrogram =
            ImVec2(static_cast<float>(steps.spectrogram) / render_scale.x,
                   static_cast<float>(cam_channel_index) / render_scale.y),
    };
  }
  // Handles mouse crosshairs in spectrograms. Will create crosshairs in all
  // spectrograms in the places that match the CAM channel and the DTW-mapped
  // time.
  void ManageSpectrogramCrosshairs(
      const std::optional<ImVec2>& crosshair_position, Image& crosshair_image) {
    if (crosshair_position.has_value()) {
      const size_t crosshair_step_index = static_cast<size_t>(
          crosshair_position->x * crosshair_image.render_scale.x);
      const size_t cam_channel_index = static_cast<size_t>(
          crosshair_position->y * crosshair_image.render_scale.y);
      EachSpectrogram([&](SpectrogramImages& image, const Analysis& analysis) {
        const Steps mapped_step_indices =
            GetMappedSteps(crosshair_step_index, crosshair_image, image);

        image.energy_channels_db.SetCrosshair(
            ComputePositions(mapped_step_indices, cam_channel_index,
                             image.energy_channels_db.render_scale)
                .energy_channels_db);
        image.highlighted_energy_channels_db = {
            .value = analysis.energy_channels_db[{
                mapped_step_indices.energy_channels_db}][cam_channel_index],
            .frequencies = {thresholds_hz[{0}][cam_channel_index],
                            thresholds_hz[{2}][cam_channel_index]},
            .time = mapped_step_indices.energy_channels_db /
                    time_resolution_frequency};

        image.partial_energy_channels_db.SetCrosshair(
            ComputePositions(mapped_step_indices, cam_channel_index,
                             image.partial_energy_channels_db.render_scale)
                .partial_energy_channels_db);
        image.highlighted_partial_energy_channels_db = {
            .value = analysis.partial_energy_channels_db[{
                mapped_step_indices.partial_energy_channels_db}]
                                                        [cam_channel_index],
            .frequencies = {thresholds_hz[{0}][cam_channel_index],
                            thresholds_hz[{2}][cam_channel_index]},
            .time = mapped_step_indices.partial_energy_channels_db /
                    time_resolution_frequency};

        image.spectrogram.SetCrosshair(
            ComputePositions(mapped_step_indices, cam_channel_index,
                             image.spectrogram.render_scale)
                .spectrogram);
        image.highlighted_spectrogram = {
            .value = analysis.spectrogram[{mapped_step_indices.spectrogram}]
                                         [cam_channel_index],
            .frequencies = {thresholds_hz[{0}][cam_channel_index],
                            thresholds_hz[{2}][cam_channel_index]},
            .time =
                mapped_step_indices.spectrogram / time_resolution_frequency};
      });
    } else {
      EachSpectrogram([&](SpectrogramImages& image, const Analysis& analysis) {
        image.energy_channels_db.DelCrosshair();
        image.highlighted_energy_channels_db = std::nullopt;
        image.partial_energy_channels_db.DelCrosshair();
        image.highlighted_partial_energy_channels_db = std::nullopt;
        image.spectrogram.DelCrosshair();
        image.highlighted_spectrogram = std::nullopt;
      });
    }
  }

  ImGuiIO* io;
  std::vector<std::string> file_b_paths;
  ComparisonClamps clamps;
  hwy::AlignedNDArray<float, 2> thresholds_hz;
  absl::Mutex callback_mutex;
  FilePresentation file_a;
  std::vector<FilePresentation> file_b_vector;
  std::vector<FilePresentation> file_absolute_delta_vector;
  std::vector<FilePresentation> file_relative_delta_vector;
  float time_resolution_frequency;
  std::vector<DTWPresentation> dtw_vector;
  std::queue<RenderCallback> callbacks;
  std::optional<std::pair<ImVec2, ImVec2>> selected_coordinates;
  size_t selected_b_index = 0;
  std::vector<std::vector<DTWMapping>> a_to_b;
  std::vector<std::vector<DTWMapping>> b_to_a;
  std::optional<size_t> unwarp_window;
};

}  // namespace

UX::UX() {
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(GLDebugCallback, nullptr);
  glfwSetErrorCallback(GLFWErrorCallback);
  if (!glfwInit()) {
    std::cerr << "glfwInit() => false" << std::endl;
    return;
  }

  // GL 3.0 + GLSL 130
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // Create window with graphics context
  window_ = glfwCreateWindow(1600, 1200, "Zimtohrli", nullptr, nullptr);
  if (window_ == nullptr) {
    std::cerr << "window=nullptr" << std::endl;
    return;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // Enable vsync

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  io_ = &ImGui::GetIO();
  io_->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(
      window_, true);  // Second param install_callback=true will install
                       // GLFW callbacks and chain to existing ones.
  ImGui_ImplOpenGL3_Init();

  const absl::Status openResult = OpenAudio();
  if (!openResult.ok()) {
    std::cerr << openResult.message() << std::endl;
  }
}

void UX::Paint(FileComparison comparison) {
  RenderContext context(io_, std::move(comparison));

  const ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  while (!glfwWindowShouldClose(window_)) {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
    // tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to
    // your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input
    // data to your main application, or clear/overwrite your copy of the
    // keyboard data. Generally you may always pass all inputs to dear imgui,
    // and hide them from your application based on those two flags.
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    context.Paint();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window_, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
  }
}

UX::~UX() {
  const absl::Status closeResult = CloseAudio();
  if (!closeResult.ok()) {
    std::cerr << closeResult.message() << std::endl;
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

}  // namespace zimtohrli

#endif  // HWY_ONCE
