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

TODO(describe the tabuli filters)

After the basic spectrogram is created, it can be compared to other spectrograms.

To compare two spectrograms, they are first analyzed using a
[dynamic time warp](https://doi.org/10.1007/BF01074755) process tuned to mimic
the experienced distance between unsynchronized audio.

They are then compared using a
[neurogram similarity index measure](https://doi.org/10.1016/j.specom.2011.09.004)
process tuned to mimic the experienced distance between audio distortions.

## Performance

For correlation performance with a few datasets see [COMPARISON.md](COMPARISON.md).

Most of those datasets can be acquired using the tools [coresvnet](go/bin/coresvnet),
[perceptual_audio](go/bin/perceptual_audio), [sebass_db](go/bin/sebass_db),
[odaq](go/bin/odaq), and [tcd_voip](go/bin/tcd_voip).
A couple of them are unpublished and can't be downloaded.

Zimtohrli can compare ~70 seconds of audio per second on a single 2.5GHz core.

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

## Quirks

- When building with ninja, the Go wrapper glue file go/goohrli/goohrli.a is built.
  Currently there's a known bug: ninja sometimes doesn't detect that this file needs to
  be rebuilt when the C++ files it depends on are changed.
  Therefore, sometimes `rm go/goohrli/goorhli.a` before running ninja is needed.
  E.g. `( rm go/goohrli/goohrli.a ; cd build && ninja )`.

