[![Tests](https://github.com/google/zimtohrli/workflows/Test%20Zimtohrli/badge.svg)](https://github.com/google/zimtohrli/actions)

# Zimtohrli: A New Psychoacoustic Perceptual Metric for Audio Compression

Zimtohrli is a psychoacoustic perceptual metric that quantifies the human
observable difference in two audio signals in the proximity of
just-noticeable-differences.

In this project we study the psychological and physiological responses
associated with sound, to create a new more accurate model for measuring
human-subjective similarity between sounds.
The main focus will be on just-noticeable-difference to get most significant
benefits in high quality audio compression.
The main goals of the project is to further both existing and new practical
audio (and video containing audio) compression, and also be able to plug in the
resulting psychoacoustic similarity measure into audio related machine learning
models.

## Design

Zimtohrli implements a perceptually-motivated audio similarity metric that
models the human auditory system through a multi-stage signal processing
pipeline. The metric operates on audio signals sampled at 48 kHz and produces a
scalar distance value that correlates with human perception of audio quality
differences.

### Signal Processing Pipeline

The algorithm consists of four main stages:

1. **Auditory Filterbank Analysis**: The input signal is processed through a
   bank of 128 filters with center frequencies logarithmically spaced from
   17.858 Hz to 20,352.7 Hz. These filters are implemented using a
   computationally efficient rotating phasor algorithm that computes spectral
   energy at each frequency band. The filterbank incorporates
   bandwidth-dependent exponential windowing to model frequency selectivity of
   the basilar membrane.

2. **Physiological Modeling**: The filtered signals undergo several transformations 
   inspired by auditory physiology:
   - A resonator model simulating the mechanical response of the ear drum and
     middle ear structures, implemented as a second-order IIR filter with
     physiologically-motivated coefficients
   - Energy computation through a cascade of three leaky integrators, modeling
     temporal integration in the auditory system
   - Loudness transformation using a logarithmic function with
     frequency-dependent gains calibrated to equal-loudness contours

3. **Temporal Alignment**: To handle temporal misalignments between reference
   and test signals, the algorithm employs Dynamic Time Warping (DTW) with a
   perceptually-motivated cost function. The warping path minimizes a weighted
   combination of spectral distance (raised to power 0.233) and temporal
   distortion penalties.

4. **Perceptual Similarity Computation**: The aligned spectrograms are compared
   using a modified Neurogram Similarity Index Measure (NSIM). This metric
   computes windowed statistics (mean, variance, covariance) over 6 temporal
   frames and 5 frequency channels, combining intensity and structure components
   through empirically-optimized non-linear functions inspired by SSIM.

### Key Parameters

- **Perceptual sampling rate**: 84 Hz (derived from [high gamma band](https://doi.org/10.1523/JNEUROSCI.5297-10.2011) frequency)
- **NSIM temporal window**: 6 frames (~71 ms)
- **NSIM frequency window**: 5 channels
- **Reference level**: 78.3 dB SPL for unity amplitude sine wave

The final distance metric is computed as 1 - NSIM, providing a value between 0
(identical) and 1 (maximally different) that correlates with subjective quality
assessments.

## Performance

For correlation performance with a few datasets see [CORRELATION.md](CORRELATION.md).

The datasets can be acquired using the tools [coresvnet](go/bin/coresvnet),
[perceptual_audio](go/bin/perceptual_audio), [sebass_db](go/bin/sebass_db),
[odaq](go/bin/odaq), and [tcd_voip](go/bin/tcd_voip).

Zimtohrli can compare ~70 seconds of audio per second on a single 2.5GHz core.

## Correlation Testing

Zimtohrli includes a comprehensive correlation testing framework to validate how
well audio quality metrics correlate with human perception. The system evaluates
metrics against multiple listening test datasets containing either Mean Opinion
Scores (MOS) or Just Noticeable Difference (JND) ratings.

### How Correlation Scoring Works

The system uses two different evaluation methods depending on the dataset type:

- **For MOS datasets**: Calculates Spearman rank correlation coefficient between
  predicted scores and human ratings. Higher correlation values (closer to 1.0)
  indicate better alignment with human perception.
- **For JND datasets**: Determines classification accuracy by finding an optimal
  threshold that maximizes correct predictions of whether differences are
  audible. The score represents the percentage of correct classifications.

### Running Correlation Tests

1. **Install external metrics** (optional):
   ```bash
   ./install_external_metrics.sh /path/to/destination
   ```

2. **Acquire datasets** using the provided tools in `go/bin/`

3. **Calculate metrics**:
   ```bash
   go run go/bin/score/score.go -calculate "/path/to/datasets/*" -calculate_zimtohrli -calculate_visqol
   ```

4. **Generate correlation report**:
   ```bash
   go run go/bin/score/score.go -report "/path/to/datasets/*" > correlation_report.md
   ```

The report includes correlation tables for each dataset and a global leaderboard
showing mean squared error across all studies, where lower values indicate
better overall performance.

## Compatibility

Zimtohrli is a project under development, and is built and tested in a Debian-like
environment. It's built to work with C++17.

## Minimal simple usage

The very simplest way to use Zimtohrli is to just include the `zimtohrli.h` header.

This allows you to

```
#include "zimtohrli.h"

const Zimtohrli z();
const Spectrogram spec_a = z.Analyze(Span(samples_a, size_a));
Spectrogram spec_b = z.Analyze(Span(samples_b, size_b));
const float distance = z.Distance(spec_a, spec_b);
```

The samples have to be floats between -1 and 1 at 48kHz sample rate.

## Build

Some dependencies for Zimtohrli are downloaded and managed by the build script,
but others need to be installed before building.

- cmake
- ninja-build

To build the compare tool, a few more dependencies are necessary:

- libogg-dev
- libvorbis-dev
- libflac-dev
- libopus-dev
- libasound2-dev
- libglfw3-dev
- libsoxr-dev

Finally, to build and test the Python and Go wrappers, the following dependencies
are necessary:

- golang-go
- python3
- xxd
- zlib1g-dev
- ffmpeg

To install these in a Debian-like system:

```
sudo apt install -y cmake ninja-build clang clang-tidy libogg-dev libvorbis-dev libflac-dev libopus-dev libasound2-dev libglfw3-dev libsoxr-dev golang-go python3 xxd zlib1g-dev ffmpeg
```

Once they are installed, configure the project:

```
./configure.sh
```

Build the project:
```
(cd build && ninja)
```

### Address sanitizer build

To build with address sanitizer, configure a new build directory with asan configured:


```
./configure.sh asan
```

Build the project:
```
(cd asan_build && ninja)
```

### Debug build

To build with debug symbols, configure a new build directory with debugging configured:


```
./configure.sh debug
```

Build the project:
```
(cd debug_build && ninja)
```

### Testing

```
(cd build && ninja && ninja test)
```
