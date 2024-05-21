#include "fourier_bank.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_split.h"
#include "sndfile.hh"

namespace tabuli {

float GetRotatorGains(int i) {
  static const float kRotatorGains[kNumRotators] = {
      1.050645, 1.948438, 3.050339, 3.967913, 4.818584, 5.303335, 5.560281,
      5.490826, 5.156689, 4.547374, 3.691308, 2.666868, 1.539254, 0.656948,
      0.345893, 0.327111, 0.985318, 1.223506, 0.447645, 0.830961, 1.075181,
      0.613335, 0.902695, 0.855391, 0.817774, 0.823359, 0.841483, 0.838562,
      0.831912, 0.808731, 0.865214, 0.808036, 0.850837, 0.821305, 0.839458,
      0.829195, 0.836373, 0.827271, 0.836018, 0.834514, 0.825624, 0.836999,
      0.833990, 0.832992, 0.830897, 0.832593, 0.846116, 0.824796, 0.829331,
      0.844509, 0.838830, 0.821733, 0.840738, 0.841735, 0.827570, 0.838581,
      0.837742, 0.834965, 0.842970, 0.832145, 0.847596, 0.840942, 0.830891,
      0.850632, 0.841468, 0.838383, 0.841493, 0.855118, 0.826750, 0.848000,
      0.874356, 0.812177, 0.849037, 0.893550, 0.832527, 0.827986, 0.877198,
      0.851760, 0.846317, 0.883044, 0.843178, 0.856925, 0.857045, 0.860695,
      0.894345, 0.870391, 0.839519, 0.870541, 0.870573, 0.902951, 0.871798,
      0.818328, 0.871413, 0.921101, 0.863915, 0.793014, 0.936519, 0.888107,
      0.856968, 0.821018, 0.987345, 0.904846, 0.783447, 0.973613, 0.903628,
      0.875688, 0.931024, 0.992087, 0.806914, 1.050332, 0.942569, 0.800870,
      1.210426, 0.916555, 0.817352, 1.126946, 0.985119, 0.922530, 0.994633,
      0.959602, 0.381419, 1.879201, 2.078451, 0.475196, 0.952731, 1.709305,
      1.383894, 1.557669,
  };
  return kRotatorGains[i];
}

int Rotators::FindMedian3xLeaker(float window) {
  // Approximate filter delay. TODO: optimize this value along with gain values.
  // Recordings can sound better with -2.32 as it pushes the bass signals a bit
  // earlier and likely compensates human hearing's deficiency for temporal
  // separation.
  const float kMagic = -2.2028003503591482;
  const float kAlmostHalfForRounding = 0.4687;
  return static_cast<int>(kMagic / log(window) + kAlmostHalfForRounding);
}

void Rotators::Filter(hwy::Span<const float> signal,
                      hwy::AlignedNDArray<float, 2>& channels) {
  const int audio_channel = 0;

  size_t out_ix = 0;
  OccasionallyRenormalize();
  for (int64_t i = 0; i < signal.size(); ++i) {
    for (int k = 0; k < kNumRotators; ++k) {
      int64_t delayed_ix = i - advance[k];
      float sample = 0;
      if (delayed_ix > 0) {
        sample = signal[delayed_ix];
      }
      AddAudio(audio_channel, k, sample);
    }
    IncrementAll();
    if (i >= max_delay_) {
      for (int k = 0; k < kNumRotators; ++k) {
        float amplitude =
            std::sqrt(rot[2][k] * rot[2][k] + rot[3][k] * rot[3][k]);
        channels[{out_ix}][k] = HardClip(amplitude);
      }
      ++out_ix;
    }
  }
}

Rotators::Rotators(int num_channels, std::vector<float> frequency,
                   std::vector<float> filter_gains, const float sample_rate,
                   float global_gain) {
  channel.resize(num_channels);
  for (int i = 0; i < kNumRotators; ++i) {
    // The parameter relates to the frequency shape overlap and window length
    // of triple leaking integrator.
    float kWindow = 0.9996;
    float w40Hz = std::pow(kWindow, 128.0 / kNumRotators);  // at 40 Hz.
    window[i] = pow(w40Hz, std::max(1.0, frequency[i] / 40.0));
    delay[i] = FindMedian3xLeaker(window[i]);
    float windowM1 = 1.0f - window[i];
    max_delay_ = std::max(max_delay_, delay[i]);
    float f = frequency[i] * 2.0f * M_PI / sample_rate;
    gain[i] = filter_gains[i] * global_gain * pow(windowM1, 3.0);
    rot[0][i] = float(std::cos(f));
    rot[1][i] = float(-std::sin(f));
    rot[2][i] = sqrt(gain[i]);
    rot[3][i] = 0.0f;
  }
  for (size_t i = 0; i < kNumRotators; ++i) {
    advance[i] = max_delay_ - delay[i];
  }
  rotator_frequency = frequency;
}

void Rotators::Increment(int c, int i, float audio) {
  if (c == 0) {
    float tr = rot[0][i] * rot[2][i] - rot[1][i] * rot[3][i];
    float tc = rot[0][i] * rot[3][i] + rot[1][i] * rot[2][i];
    rot[2][i] = tr;
    rot[3][i] = tc;
  }
  channel[c].accu[0][i] *= window[i];
  channel[c].accu[1][i] *= window[i];
  channel[c].accu[2][i] *= window[i];
  channel[c].accu[3][i] *= window[i];
  channel[c].accu[4][i] *= window[i];
  channel[c].accu[5][i] *= window[i];
  channel[c].accu[0][i] += rot[2][i] * audio;
  channel[c].accu[1][i] += rot[3][i] * audio;
  channel[c].accu[2][i] += channel[c].accu[0][i];
  channel[c].accu[3][i] += channel[c].accu[1][i];
  channel[c].accu[4][i] += channel[c].accu[2][i];
  channel[c].accu[5][i] += channel[c].accu[3][i];
}

void Rotators::AddAudio(int c, int i, float audio) {
  channel[c].accu[0][i] += rot[2][i] * audio;
  channel[c].accu[1][i] += rot[3][i] * audio;
}
void Rotators::OccasionallyRenormalize() {
  for (int i = 0; i < kNumRotators; ++i) {
    float norm =
        sqrt(gain[i] / (rot[2][i] * rot[2][i] + rot[3][i] * rot[3][i]));
    rot[2][i] *= norm;
    rot[3][i] *= norm;
  }
}
void Rotators::IncrementAll() {
  for (int i = 0; i < kNumRotators; i++) {
    const float tr = rot[0][i] * rot[2][i] - rot[1][i] * rot[3][i];
    const float tc = rot[0][i] * rot[3][i] + rot[1][i] * rot[2][i];
    rot[2][i] = tr;
    rot[3][i] = tc;
  }
  for (int c = 0; c < channel.size(); ++c) {
    for (int i = 0; i < kNumRotators; i++) {
      const float w = window[i];
      channel[c].accu[0][i] *= w;
      channel[c].accu[1][i] *= w;
      channel[c].accu[2][i] *= w;
      channel[c].accu[3][i] *= w;
      channel[c].accu[4][i] *= w;
      channel[c].accu[5][i] *= w;
      channel[c].accu[2][i] += channel[c].accu[0][i];
      channel[c].accu[3][i] += channel[c].accu[1][i];
      channel[c].accu[4][i] += channel[c].accu[2][i];
      channel[c].accu[5][i] += channel[c].accu[3][i];
    }
  }
}
float Rotators::GetSampleAll(int c) {
  float retval = 0;
  for (int i = 0; i < kNumRotators; ++i) {
    retval +=
        (rot[2][i] * channel[c].accu[4][i] + rot[3][i] * channel[c].accu[5][i]);
  }
  return retval;
}
float Rotators::GetSample(int c, int i, FilterMode mode) const {
  return (
      mode == IDENTITY ? (rot[2][i] * channel[c].accu[4][i] +
                          rot[3][i] * channel[c].accu[5][i])
      : mode == AMPLITUDE
          ? std::sqrt(gain[i] * (channel[c].accu[4][i] * channel[c].accu[4][i] +
                                 channel[c].accu[5][i] * channel[c].accu[5][i]))
          : std::atan2(channel[c].accu[4][i], channel[c].accu[5][i]));
}

float BarkFreq(float v) {
  constexpr float linlogsplit = 0.1;
  if (v < linlogsplit) {
    return 20.0 + (v / linlogsplit) * 20.0;  // Linear 20-40 Hz.
  } else {
    float normalized_v = (v - linlogsplit) * (1.0 / (1.0 - linlogsplit));
    return 40.0 * pow(500.0, normalized_v);  // Logarithmic 40-20000 Hz.
  }
}

float HardClip(float v) { return std::max(-1.0f, std::min(1.0f, v)); }

RotatorFilterBank::RotatorFilterBank(size_t num_rotators, size_t num_channels,
                                     size_t samplerate, size_t num_threads,
                                     const std::vector<float>& filter_gains,
                                     float global_gain) {
  num_rotators_ = num_rotators;
  num_channels_ = num_channels;
  num_threads_ = num_threads;
  std::vector<float> freqs(num_rotators);
  for (size_t i = 0; i < num_rotators_; ++i) {
    freqs[i] = BarkFreq(static_cast<float>(i) / (num_rotators_ - 1));
    // printf("%d %g\n", i, freqs[i]);
  }
  rotators_.reset(
      new Rotators(num_channels, freqs, filter_gains, samplerate, global_gain));

  max_delay_ = rotators_->max_delay_;
  QCHECK_LE(max_delay_, kBlockSize);
  fprintf(stderr, "Rotator bank output delay: %zu\n", max_delay_);
  filter_outputs_.resize(num_rotators);
  for (std::vector<float>& output : filter_outputs_) {
    output.resize(num_channels_ * kBlockSize, 0.f);
  }
}

// TODO(jyrki): filter all at once in the generic case, filtering one
// is not memory friendly in this memory tabulation.
void RotatorFilterBank::FilterOne(size_t f_ix, const float* history,
                                  int64_t total_in, int64_t len,
                                  FilterMode mode, float* output) {
  size_t out_ix = 0;
  for (int64_t i = 0; i < len; ++i) {
    int64_t delayed_ix = total_in + i - rotators_->advance[f_ix];
    size_t histo_ix = num_channels_ * (delayed_ix & kHistoryMask);
    for (size_t c = 0; c < num_channels_; ++c) {
      float delayed = history[histo_ix + c];
      rotators_->Increment(c, f_ix, delayed);
    }
    if (total_in + i >= max_delay_) {
      for (size_t c = 0; c < num_channels_; ++c) {
        output[out_ix * num_channels_ + c] =
            rotators_->GetSample(c, f_ix, mode);
      }
      ++out_ix;
    }
  }
}

int64_t RotatorFilterBank::FilterAllSingleThreaded(const float* history,
                                                   int64_t total_in,
                                                   int64_t len, FilterMode mode,
                                                   float* output,
                                                   size_t output_size) {
  size_t out_ix = 0;
  for (size_t c = 0; c < num_channels_; ++c) {
    rotators_->OccasionallyRenormalize();
  }
  for (int64_t i = 0; i < len; ++i) {
    for (size_t c = 0; c < num_channels_; ++c) {
      for (int k = 0; k < kNumRotators; ++k) {
        int64_t delayed_ix = total_in + i - rotators_->advance[k];
        size_t histo_ix = num_channels_ * (delayed_ix & kHistoryMask);
        float delayed = history[histo_ix + c];
        rotators_->AddAudio(c, k, delayed);
      }
    }
    rotators_->IncrementAll();
    if (total_in + i >= max_delay_) {
      for (size_t c = 0; c < num_channels_; ++c) {
        output[out_ix * num_channels_ + c] =
            HardClip(rotators_->GetSampleAll(c));
      }
      ++out_ix;
    }
  }
  size_t out_len = total_in < max_delay_
                       ? std::max<int64_t>(0, len - (max_delay_ - total_in))
                       : len;
  return out_len;
}

int64_t RotatorFilterBank::FilterAll(const float* history, int64_t total_in,
                                     int64_t len, FilterMode mode,
                                     float* output, size_t output_size) {
  auto run = [&](size_t thread) {
    while (true) {
      size_t my_task = next_task_++;
      if (my_task >= num_rotators_) return;
      FilterOne(my_task, history, total_in, len, mode,
                filter_outputs_[my_task].data());
    }
  };
  next_task_ = 0;
  std::vector<std::future<void>> futures;
  futures.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    futures.push_back(std::async(std::launch::async, run, i));
  }
  for (size_t i = 0; i < num_threads_; ++i) {
    futures[i].get();
  }
  size_t out_len = total_in < max_delay_
                       ? std::max<int64_t>(0, len - (max_delay_ - total_in))
                       : len;
  for (size_t i = 0; i < out_len; ++i) {
    for (size_t j = 0; j < num_rotators_; ++j) {
      for (size_t c = 0; c < num_channels_; ++c) {
        size_t out_idx = (i * num_rotators_ + j) * num_channels_ + c;
        output[out_idx] = filter_outputs_[j][i * num_channels_ + c];
      }
    }
  }
  return out_len;
}

}  // namespace tabuli