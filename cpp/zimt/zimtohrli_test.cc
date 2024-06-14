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

#include "zimt/zimtohrli.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "zimt/cam.h"
#include "zimt/filterbank.h"
#include "zimt/loudness.h"

namespace zimtohrli {

namespace {

Distance Dist(const Zimtohrli& z,
              const hwy::AlignedNDArray<float, 2>& spectrogram_a,
              const hwy::AlignedNDArray<float, 2>& spectrogram_b,
              float want_distance) {
  Distance result{.value = want_distance};
  for (size_t sample_index = 0; sample_index < spectrogram_a.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < spectrogram_a.shape()[1];
         ++channel_index) {
      const float spec_a_linear =
          pow(10, spectrogram_a[{sample_index}][channel_index] / 20);
      const float spec_b_linear =
          pow(10, spectrogram_b[{sample_index}][channel_index] / 20);
      const float noise_linear_amplitude = abs(spec_a_linear - spec_b_linear);
      const float noise_db =
          std::max(0.0, 20 * log10(noise_linear_amplitude + 1e-6));
      if (noise_db > result.max_absolute_delta.value) {
        result.max_absolute_delta.value = noise_db;
        result.max_absolute_delta.channel_index = channel_index;
        result.max_absolute_delta.sample_a_index = sample_index;
        result.max_absolute_delta.sample_b_index = sample_index;
        result.max_absolute_delta.spectrogram_a_value =
            spectrogram_a[{sample_index}][channel_index];
        result.max_absolute_delta.spectrogram_b_value =
            spectrogram_b[{sample_index}][channel_index];
      }
      const float delta_db = abs(spectrogram_a[{sample_index}][channel_index] -
                                 spectrogram_b[{sample_index}][channel_index]);
      if (delta_db > result.max_relative_delta.value) {
        result.max_relative_delta.value = delta_db;
        result.max_relative_delta.channel_index = channel_index;
        result.max_relative_delta.sample_a_index = sample_index;
        result.max_relative_delta.sample_b_index = sample_index;
        result.max_relative_delta.spectrogram_a_value =
            spectrogram_a[{sample_index}][channel_index];
        result.max_relative_delta.spectrogram_b_value =
            spectrogram_b[{sample_index}][channel_index];
      }
    }
  }
  return result;
}

void CheckDistanceNear(const Distance& distance_a, const Distance& distance_b) {
  const float tolerance = 1e-4;
  EXPECT_NEAR(distance_a.value, distance_b.value, tolerance);
  EXPECT_NEAR(distance_a.max_absolute_delta.value,
              distance_b.max_absolute_delta.value, tolerance);
  EXPECT_EQ(distance_a.max_absolute_delta.channel_index,
            distance_b.max_absolute_delta.channel_index);
  EXPECT_EQ(distance_a.max_absolute_delta.sample_a_index,
            distance_b.max_absolute_delta.sample_a_index);
  EXPECT_EQ(distance_a.max_absolute_delta.sample_b_index,
            distance_b.max_absolute_delta.sample_b_index);
  EXPECT_EQ(distance_a.max_absolute_delta.spectrogram_a_value,
            distance_b.max_absolute_delta.spectrogram_a_value);
  EXPECT_EQ(distance_a.max_absolute_delta.spectrogram_b_value,
            distance_b.max_absolute_delta.spectrogram_b_value);
  EXPECT_EQ(distance_a.max_relative_delta.value,
            distance_b.max_relative_delta.value);
  EXPECT_EQ(distance_a.max_relative_delta.channel_index,
            distance_b.max_relative_delta.channel_index);
  EXPECT_EQ(distance_a.max_relative_delta.sample_a_index,
            distance_b.max_relative_delta.sample_a_index);
  EXPECT_EQ(distance_a.max_relative_delta.sample_b_index,
            distance_b.max_relative_delta.sample_b_index);
  EXPECT_EQ(distance_a.max_relative_delta.spectrogram_a_value,
            distance_b.max_relative_delta.spectrogram_a_value);
  EXPECT_EQ(distance_a.max_relative_delta.spectrogram_b_value,
            distance_b.max_relative_delta.spectrogram_b_value);
}

TEST(Zimtohrli, DTWDistanceTest) {
  hwy::AlignedNDArray<float, 2> spectrogram_a({10, 1});
  const std::vector<float> a_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (size_t sample_index = 0; sample_index < a_values.size();
       ++sample_index) {
    spectrogram_a[{sample_index}][0] = a_values[sample_index];
  }
  hwy::AlignedNDArray<float, 2> spectrogram_b({10, 1});
  const std::vector<float> b_values = {0, 1, 2, 3, 3, 4, 5, 6, 8, 9};
  for (size_t sample_index = 0; sample_index < b_values.size();
       ++sample_index) {
    spectrogram_b[{sample_index}][0] = b_values[sample_index];
  }
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z = {.cam_filterbank = cam.CreateFilterbank(48000),
                 .unwarp_window_seconds = 0};
  EXPECT_NEAR(z.Distance(false, spectrogram_a, spectrogram_b).value,
              0.01090317964553833f, 1e-2f);
  z.unwarp_window_seconds = 4.0 / 48000.0;
  EXPECT_NEAR(z.Distance(false, spectrogram_a, spectrogram_b).value,
              0.0080544948577880859f, 1e-2f);
}

TEST(Zimtohrli, DistanceTest) {
  hwy::AlignedNDArray<float, 2> spectrogram_a({2, 2});
  hwy::AlignedNDArray<float, 2> spectrogram_b({2, 2});
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z = {.cam_filterbank = cam.CreateFilterbank(48000),
                 .unwarp_window_seconds = 0};

  spectrogram_b[{0}] = {1, 1};
  CheckDistanceNear(
      z.Distance(/* verbose */ true, spectrogram_a, spectrogram_b),
      Dist(z, spectrogram_a, spectrogram_b, 0.54945051670074463));

  spectrogram_b[{1}] = {4, 8};
  CheckDistanceNear(
      z.Distance(/* verbose */ true, spectrogram_a, spectrogram_b),
      Dist(z, spectrogram_a, spectrogram_b, 0.75766336917877197));

  spectrogram_b[{0}] = {-30, 0};
  CheckDistanceNear(
      z.Distance(/* verbose */ true, spectrogram_a, spectrogram_b),
      Dist(z, spectrogram_a, spectrogram_b, 0.99729388952255249));

  spectrogram_a = hwy::AlignedNDArray<float, 2>({64, 64});
  spectrogram_b = hwy::AlignedNDArray<float, 2>({64, 64});
  for (size_t sample_index = 0; sample_index < spectrogram_a.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < spectrogram_a.shape()[1];
         ++channel_index) {
      spectrogram_a[{sample_index}][channel_index] =
          sin(static_cast<float>(sample_index)) +
          cos(static_cast<float>(channel_index));
      spectrogram_b[{sample_index}][channel_index] =
          sin(spectrogram_a[{sample_index}][channel_index]);
    }
  }
  CheckDistanceNear(
      z.Distance(/* verbose */ true, spectrogram_a, spectrogram_b),
      Dist(z, spectrogram_a, spectrogram_b, 0.022919178009033203));
}

void CreateAudio(float sample_rate,
                 std::vector<std::vector<std::pair<float, float>>>
                     frequencies_and_amplitudes,
                 hwy::AlignedNDArray<float, 2>& audio) {
  for (size_t audio_channel_index = 0; audio_channel_index < audio.shape()[0];
       ++audio_channel_index) {
    for (size_t sample_index = 0; sample_index < audio.shape()[1];
         ++sample_index) {
      const float t = static_cast<float>(sample_index) / sample_rate;
      for (const auto& frequency_and_amplitude :
           frequencies_and_amplitudes[audio_channel_index]) {
        audio[{audio_channel_index}][sample_index] +=
            frequency_and_amplitude.second *
            sin(2 * M_PI * frequency_and_amplitude.first * t);
      }
    }
  }
}

hwy::AlignedNDArray<float, 2> CreateSignal(
    float sample_rate, float t,
    std::vector<std::pair<float, float>> frequencies_and_amplitudes) {
  hwy::AlignedNDArray<float, 2> signal(
      {1, static_cast<size_t>(sample_rate * t)});
  CreateAudio(sample_rate, {frequencies_and_amplitudes}, signal);
  return signal;
}

float MeanEnergy(const hwy::AlignedNDArray<float, 2>& channels,
                 size_t channel_index) {
  float sum = 0;
  for (size_t sample_index = 0; sample_index < channels.shape()[0];
       ++sample_index) {
    sum += channels[{sample_index}][channel_index];
  }
  return sum / static_cast<float>(channels.shape()[0]);
}

size_t PeakChannel(const hwy::AlignedNDArray<float, 2>& channels) {
  size_t result = 0;
  float peak_energy = -1;
  for (size_t channel_index = 0; channel_index < channels.shape()[1];
       ++channel_index) {
    const float energy = MeanEnergy(channels, channel_index);
    if (energy > peak_energy) {
      peak_energy = energy;
      result = channel_index;
    }
  }
  return result;
}

TEST(Zimtohrli, AnalysisTest) {
  const float full_scale_sine_db = 80;
  const float sample_rate = 48000;
  const float seconds_of_audio = 1;
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z = {.cam_filterbank = cam.CreateFilterbank(sample_rate),
                 .full_scale_sine_db = full_scale_sine_db};
  const size_t channel = 600;
  const float channel_hz = z.cam_filterbank->thresholds_hz[{1}][channel];
  const size_t last_energy_sample =
      static_cast<size_t>(z.perceptual_sample_rate * seconds_of_audio - 1);

  hwy::AlignedNDArray<float, 2> signal =
      CreateSignal(sample_rate, seconds_of_audio, {{channel_hz, 1.0}});
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[1], z.cam_filterbank->filter.Size()});

  Analysis result = z.Analyze(signal[{0}], channels);

  EXPECT_EQ(PeakChannel(result.spectrogram), channel);
  EXPECT_NEAR(result.energy_channels_db[{last_energy_sample}][channel],
              full_scale_sine_db, 4.5);
}

void CheckNear(const hwy::AlignedNDArray<float, 2>& array_a,
               const hwy::AlignedNDArray<float, 2>& array_b) {
  for (size_t sample_index = 0; sample_index < array_a.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < array_a.shape()[1];
         ++channel_index) {
      EXPECT_NEAR(array_a[{sample_index}][channel_index],
                  array_b[{sample_index}][channel_index], 1e-9);
    }
  }
}

void CheckIsLinearDelta(hwy::Span<const float> signal_a,
                        hwy::Span<const float> signal_b,
                        hwy::Span<const float> signal_delta) {
  for (size_t sample_index = 0; sample_index < signal_a.size();
       ++sample_index) {
    EXPECT_NEAR(signal_a[sample_index] - signal_b[sample_index],
                signal_delta[sample_index], 1e-9);
  }
}

void CheckIsDbDelta(const Zimtohrli& z,
                    const hwy::AlignedNDArray<float, 2>& array_a,
                    const hwy::AlignedNDArray<float, 2>& array_b,
                    const hwy::AlignedNDArray<float, 2>& delta) {
  for (size_t sample_index = 0; sample_index < array_a.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < array_a.shape()[1];
         ++channel_index) {
      EXPECT_NEAR(
          20 * log10(z.epsilon +
                     abs(pow(10, array_a[{sample_index}][channel_index] / 20) -
                         pow(10, array_b[{sample_index}][channel_index] / 20))),
          delta[{sample_index}][channel_index], 1.5)
          << "array_a[{sample_index}][channel_index]="
          << array_a[{sample_index}][channel_index] << "\n"
          << "array_b[{sample_index}][channel_index]="
          << array_b[{sample_index}][channel_index] << "\n"
          << "delta[{sample_index}][channel_index]="
          << delta[{sample_index}][channel_index] << "\n";
    }
  }
}

void CheckIsAbsDiff(const Zimtohrli& z,
                    const hwy::AlignedNDArray<float, 2>& array_a,
                    const hwy::AlignedNDArray<float, 2>& array_b,
                    const hwy::AlignedNDArray<float, 2>& delta) {
  for (size_t sample_index = 0; sample_index < array_a.shape()[0];
       ++sample_index) {
    for (size_t channel_index = 0; channel_index < array_a.shape()[1];
         ++channel_index) {
      EXPECT_NEAR(abs(array_a[{sample_index}][channel_index] -
                      array_b[{sample_index}][channel_index]),
                  delta[{sample_index}][channel_index], 1)
          << "array_a[{sample_index}][channel_index]="
          << array_a[{sample_index}][channel_index] << "\n"
          << "array_b[{sample_index}][channel_index]="
          << array_b[{sample_index}][channel_index] << "\n"
          << "delta[{sample_index}][channel_index]="
          << delta[{sample_index}][channel_index] << "\n";
    }
  }
}

TEST(Zimtohrli, ComparisonTest) {
  const float full_scale_sine_db = 80;
  const float sample_rate = 48000;
  const float seconds_of_audio = 1;
  const size_t num_samples =
      static_cast<size_t>(sample_rate * seconds_of_audio);
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z = {.cam_filterbank = cam.CreateFilterbank(sample_rate),
                 .unwarp_window_seconds = 0,
                 .full_scale_sine_db = full_scale_sine_db};
  const size_t channel_0 = 600;
  const float channel_0_hz = z.cam_filterbank->thresholds_hz[{1}][channel_0];
  const size_t channel_1 = 700;
  const float channel_1_hz = z.cam_filterbank->thresholds_hz[{1}][channel_1];

  hwy::AlignedNDArray<float, 2> audio_a({1, num_samples});
  CreateAudio(sample_rate, {{{channel_0_hz, 1.0}}}, audio_a);
  std::vector<hwy::AlignedNDArray<float, 2>> audio_b;
  std::vector<hwy::AlignedNDArray<float, 2>*> audio_b_pointers;
  hwy::AlignedNDArray<float, 2> audio_b_frames({1, num_samples});
  audio_b.push_back(std::move(audio_b_frames));
  audio_b_pointers.push_back(&audio_b[0]);
  CreateAudio(sample_rate, {{{channel_1_hz, 1.0}}}, audio_b[0]);
  hwy::AlignedNDArray<float, 2> channels(
      {audio_a.shape()[1], z.cam_filterbank->filter.Size()});

  Analysis analysis_a = z.Analyze(audio_a[{0}], channels);
  Analysis analysis_b = z.Analyze(audio_b[0][{0}], channels);

  Comparison comparison = z.Compare(audio_a, audio_b_pointers);

  CheckNear(analysis_a.energy_channels_db,
            comparison.analysis_a[0].energy_channels_db);
  CheckNear(analysis_a.partial_energy_channels_db,
            comparison.analysis_a[0].partial_energy_channels_db);
  CheckNear(analysis_a.spectrogram, comparison.analysis_a[0].spectrogram);
  CheckNear(analysis_b.energy_channels_db,
            comparison.analysis_b[0][0].energy_channels_db);
  CheckNear(analysis_b.partial_energy_channels_db,
            comparison.analysis_b[0][0].partial_energy_channels_db);
  CheckNear(analysis_b.spectrogram, comparison.analysis_b[0][0].spectrogram);

  CheckIsLinearDelta(audio_a[{0}], audio_b[0][{0}],
                     comparison.frames_delta[0][{0}]);

  CheckIsDbDelta(z, analysis_a.energy_channels_db,
                 analysis_b.energy_channels_db,
                 comparison.analysis_absolute_delta[0][0].energy_channels_db);
  CheckIsDbDelta(
      z, analysis_a.partial_energy_channels_db,
      analysis_b.partial_energy_channels_db,
      comparison.analysis_absolute_delta[0][0].partial_energy_channels_db);
  CheckIsDbDelta(z, analysis_a.spectrogram, analysis_b.spectrogram,
                 comparison.analysis_absolute_delta[0][0].spectrogram);
  CheckIsAbsDiff(z, analysis_a.energy_channels_db,
                 analysis_b.energy_channels_db,
                 comparison.analysis_relative_delta[0][0].energy_channels_db);
  CheckIsAbsDiff(
      z, analysis_a.partial_energy_channels_db,
      analysis_b.partial_energy_channels_db,
      comparison.analysis_relative_delta[0][0].partial_energy_channels_db);
  CheckIsAbsDiff(z, analysis_a.spectrogram, analysis_b.spectrogram,
                 comparison.analysis_relative_delta[0][0].spectrogram);
}

TEST(Zimtohrli, NormalizeAmplitudeTest) {
  hwy::AlignedNDArray<float, 1> reference({8});
  reference[{}] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  hwy::AlignedNDArray<float, 1> signal({5});
  signal[{}] = {0.25, 0.25, 0.25, 0.25, 0.25};
  const EnergyAndMaxAbsAmplitude reference_measurements =
      Measure(reference[{}]);
  EXPECT_NEAR(reference_measurements.energy_db_fs, 20 * std::log10(0.5 * 0.5),
              1e-4);
  EXPECT_EQ(reference_measurements.max_abs_amplitude, 0.5);
  const EnergyAndMaxAbsAmplitude signal_measurements =
      NormalizeAmplitude(reference_measurements.max_abs_amplitude, signal[{}]);
  for (size_t index = 0; index < signal.shape()[0]; ++index) {
    EXPECT_EQ(signal[{}][index], 0.5);
  }
  EXPECT_NEAR(signal_measurements.energy_db_fs, 20 * std::log10(0.5 * 0.5),
              1e-4);
  EXPECT_EQ(signal_measurements.max_abs_amplitude, 0.5);
}

TEST(Zimtohrli, SpectrogramTest) {
  const float full_scale_sine_db = 80;
  const float sample_rate = 48000;
  const float seconds_of_audio = 1;
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z = {.cam_filterbank = cam.CreateFilterbank(sample_rate),
                 .full_scale_sine_db = full_scale_sine_db};
  const size_t low_channel = 600;
  const size_t high_channel = 620;

  hwy::AlignedNDArray<float, 2> channels(
      {static_cast<size_t>(sample_rate * seconds_of_audio),
       z.cam_filterbank->filter.Size()});
  const size_t last_energy_sample =
      static_cast<size_t>(100 * seconds_of_audio - 1);
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> loudness(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});

  const float low_channel_hz =
      z.cam_filterbank->thresholds_hz[{1}][low_channel];
  const float low_channel_amplitude = 1.0;
  z.Spectrogram(CreateSignal(sample_rate, seconds_of_audio,
                             {{low_channel_hz, low_channel_amplitude}})[{0}],
                channels, energy_channels_db, partial_energy_channels_db,
                spectrogram);
  EXPECT_EQ(PeakChannel(spectrogram), low_channel);
  EXPECT_NEAR(energy_channels_db[{last_energy_sample}][low_channel],
              full_scale_sine_db, 4.5);
  Loudness().PhonsFromSPL(energy_channels_db, z.cam_filterbank->thresholds_hz,
                          loudness);
  EXPECT_NEAR(loudness[{last_energy_sample}][low_channel],
              spectrogram[{last_energy_sample}][low_channel], 1);
  const float energy_in_low_channel_solo =
      spectrogram[{last_energy_sample}][low_channel];

  const float high_channel_hz =
      z.cam_filterbank->thresholds_hz[{1}][high_channel];
  const float high_channel_amplitude = 0.1;
  z.Spectrogram(CreateSignal(sample_rate, seconds_of_audio,
                             {{high_channel_hz, high_channel_amplitude}})[{0}],
                channels, energy_channels_db, partial_energy_channels_db,
                spectrogram);
  EXPECT_EQ(PeakChannel(spectrogram), high_channel);
  EXPECT_NEAR(energy_channels_db[{last_energy_sample}][high_channel],
              full_scale_sine_db + 20 * log10(high_channel_amplitude), 4);
  Loudness().PhonsFromSPL(energy_channels_db, z.cam_filterbank->thresholds_hz,
                          loudness);
  EXPECT_NEAR(loudness[{last_energy_sample}][high_channel],
              spectrogram[{last_energy_sample}][high_channel], 1);

  z.Spectrogram(CreateSignal(sample_rate, seconds_of_audio,
                             {{low_channel_hz, low_channel_amplitude},
                              {high_channel_hz, high_channel_amplitude}})[{0}],
                channels, energy_channels_db, partial_energy_channels_db,
                spectrogram);
  EXPECT_EQ(PeakChannel(spectrogram), low_channel);
  EXPECT_NEAR(spectrogram[{last_energy_sample}][low_channel],
              energy_in_low_channel_solo, 0.5);
}

void BM_SpectrogramDistanceVsSeconds(benchmark::State& state) {
  const size_t sample_rate = 48000;
  const float seconds_of_audio = static_cast<float>(state.range(0));
  const Cam cam;
  Zimtohrli z1 = {.cam_filterbank = cam.CreateFilterbank(sample_rate),
                  .unwarp_window_seconds = 0};
  Zimtohrli z2 = {.cam_filterbank = cam.CreateFilterbank(sample_rate),
                  .unwarp_window_seconds = 0};
  hwy::AlignedNDArray<float, 1> signal(
      {static_cast<size_t>(sample_rate * seconds_of_audio)});
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[1], z1.cam_filterbank->filter.Size()});
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram1(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram2(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});

  for (auto s : state) {
    z1.Spectrogram(signal[{}], channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram1);
    z2.Spectrogram(signal[{}], channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram2);
    z1.Distance(false, spectrogram1, spectrogram2);
  }
  state.SetItemsProcessed(signal.size() * state.iterations());
}
BENCHMARK_RANGE(BM_SpectrogramDistanceVsSeconds, 1, 8);

void BM_StreamingSpectrogram(benchmark::State& state) {
  const size_t sample_rate = 48000;
  Zimtohrli z{.cam_filterbank = Cam{}.CreateFilterbank(sample_rate)};
  hwy::AlignedNDArray<float, 1> signal(
      {static_cast<size_t>(sample_rate * state.range(0))});
  for (auto s : state) {
    z.StreamingSpectrogram(signal[{}]);
  }
  state.SetItemsProcessed(signal.size() * state.iterations());
}
BENCHMARK_RANGE(BM_StreamingSpectrogram, 1, 32);

void BM_PreallocSpectrogram(benchmark::State& state) {
  const size_t sample_rate = 48000;
  Zimtohrli z{.cam_filterbank = Cam{}.CreateFilterbank(sample_rate)};
  hwy::AlignedNDArray<float, 1> signal(
      {static_cast<size_t>(sample_rate * state.range(0))});
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[1], z.cam_filterbank->filter.Size()});
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {static_cast<size_t>(100 * state.range(0)), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {static_cast<size_t>(100 * state.range(0)), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
  for (auto s : state) {
    z.Spectrogram(signal[{}], channels, energy_channels_db,
                  partial_energy_channels_db, spectrogram);
  }
  state.SetItemsProcessed(signal.size() * state.iterations());
}
BENCHMARK_RANGE(BM_PreallocSpectrogram, 1, 32);

void BM_RepeatedAllocSpectrogram(benchmark::State& state) {
  const size_t sample_rate = 48000;
  Zimtohrli z{.cam_filterbank = Cam{}.CreateFilterbank(sample_rate)};
  hwy::AlignedNDArray<float, 1> signal(
      {static_cast<size_t>(sample_rate * state.range(0))});
  for (auto s : state) {
    hwy::AlignedNDArray<float, 2> channels(
        {signal.shape()[1], z.cam_filterbank->filter.Size()});
    hwy::AlignedNDArray<float, 2> energy_channels_db(
        {static_cast<size_t>(100 * state.range(0)), channels.shape()[1]});
    hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
        {static_cast<size_t>(100 * state.range(0)), channels.shape()[1]});
    hwy::AlignedNDArray<float, 2> spectrogram(
        {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
    z.Spectrogram(signal[{}], channels, energy_channels_db,
                  partial_energy_channels_db, spectrogram);
  }
  state.SetItemsProcessed(signal.size() * state.iterations());
}
BENCHMARK_RANGE(BM_RepeatedAllocSpectrogram, 1, 32);

void BM_SpectrogramDistanceVsResolution(benchmark::State& state) {
  const size_t sample_rate = 48000;
  const float seconds_of_audio = 1;
  Cam cam;
  cam.minimum_bandwidth_hz = static_cast<float>(state.range(0));
  Zimtohrli z1{.cam_filterbank = cam.CreateFilterbank(sample_rate),
               .unwarp_window_seconds = 0};
  Zimtohrli z2{.cam_filterbank = cam.CreateFilterbank(sample_rate),
               .unwarp_window_seconds = 0};
  hwy::AlignedNDArray<float, 1> signal(
      {static_cast<size_t>(sample_rate * seconds_of_audio)});
  hwy::AlignedNDArray<float, 2> channels(
      {signal.shape()[1], z1.cam_filterbank->filter.Size()});
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {static_cast<size_t>(100 * seconds_of_audio), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram1(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram2(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});

  for (auto s : state) {
    z1.Spectrogram(signal[{}], channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram1);
    z2.Spectrogram(signal[{}], channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram2);
    z1.Distance(false, spectrogram1, spectrogram2);
  }
  state.SetItemsProcessed(signal.size() * state.iterations());
}
BENCHMARK_RANGE(BM_SpectrogramDistanceVsResolution, 1, 64);

TEST(Zimtohrli, FindMaxDistortionTest) {
  // Reference sound pressure of a sine signal with amplitude 1.
  const float full_scale_sine_db = 80;
  const float sample_rate = 48000;
  const float seconds_of_audio = 1;
  const float noise_start = 0.4;
  const float noise_end = 0.5;

  // Create a signal with a combination of 400 and 800 Hz.
  const hwy::AlignedNDArray<float, 2> signal =
      CreateSignal(sample_rate, seconds_of_audio, {{400, 0.8}, {800, 0.5}});

  // Create a distortion with some noise. Add more noise between 0.5s and
  // 0.6s.
  hwy::AlignedNDArray<float, 2> distortion =
      CreateSignal(sample_rate, seconds_of_audio, {{400, 0.8}, {800, 0.5}});
  uint32_t seed = 0;
  for (size_t sample_index = 0; sample_index < distortion.shape()[1];
       ++sample_index) {
    float t = static_cast<float>(sample_index) / sample_rate;
    float noise_amplitude = 0.01;
    if (t >= noise_start && t <= noise_end) {
      noise_amplitude = 0.1;
    }
    distortion[{}][sample_index] +=
        noise_amplitude * 2 *
        (static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) -
         0.5);
  }

  // Initialize two Zimtohrli instances with default values.
  const Cam cam{.minimum_bandwidth_hz = 1};
  Zimtohrli z1{.cam_filterbank = cam.CreateFilterbank(sample_rate),
               .unwarp_window_seconds = 0,
               .full_scale_sine_db = full_scale_sine_db};
  FilterbankState z1_state = z1.cam_filterbank->filter.NewState();
  Zimtohrli z2{.cam_filterbank = cam.CreateFilterbank(sample_rate),
               .unwarp_window_seconds = 0,
               .full_scale_sine_db = full_scale_sine_db};
  FilterbankState z2_state = z2.cam_filterbank->filter.NewState();

  // Seconds of audio to compare per step.
  const float chunk_seconds = 0.2;
  // Chunk to feed the Zimtohrli instances.
  hwy::AlignedNDArray<float, 2> chunk(
      {1, static_cast<size_t>(sample_rate * chunk_seconds)});

  // Working memory.
  hwy::AlignedNDArray<float, 2> channels(
      {chunk.shape()[1], z1.cam_filterbank->filter.Size()});
  hwy::AlignedNDArray<float, 2> energy_channels_db(
      {static_cast<size_t>(100 * chunk_seconds), channels.shape()[1]});
  hwy::AlignedNDArray<float, 2> partial_energy_channels_db(
      {static_cast<size_t>(100 * chunk_seconds), channels.shape()[1]});

  hwy::AlignedNDArray<float, 2> spectrogram1(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});
  hwy::AlignedNDArray<float, 2> spectrogram2(
      {energy_channels_db.shape()[0], energy_channels_db.shape()[1]});

  float max_distance = 0;
  float max_distance_time = 0;
  // Step through the audio and compare the distance for each chunk.
  for (size_t offset = 0; offset + chunk.shape()[1] < signal.shape()[1];
       offset += chunk.shape()[1]) {
    hwy::CopyBytes(signal[{0}].data() + offset, chunk[{0}].data(),
                   sizeof(float) * chunk.shape()[1]);
    z1.Spectrogram(chunk[{0}], z1_state, channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram1);
    hwy::CopyBytes(distortion.data() + offset, chunk.data(),
                   sizeof(float) * chunk.shape()[1]);
    z2.Spectrogram(chunk[{0}], z2_state, channels, energy_channels_db,
                   partial_energy_channels_db, spectrogram2);
    const float distance = z1.Distance(false, spectrogram1, spectrogram2).value;
    if (distance > max_distance) {
      max_distance = distance;
      max_distance_time = static_cast<float>(offset) / sample_rate;
    }
  }
  EXPECT_NEAR(max_distance_time, 0.4, 1e-4);
}

}  // namespace

}  // namespace zimtohrli