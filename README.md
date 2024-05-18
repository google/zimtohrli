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

For more details about how Zimtohrli works, see [zimtohrli.ipynb](zimtohrli.ipynb).

## Performance

Zimtohrli is able to compare sounds at ~10x real time for 48kHz sample rate using
a single modern CPU core, but exact times may vary based on the settings used.

For correlation performance with a few datasets see [COMPARISON.md](COMPARISON.md).

Most of those datasets can be acquired using the tools [coresvnet](go/bin/coresvnet), [perceptual_audio](go/bin/perceptual_audio), [sebass_db](go/bin/sebass_db), and [tcd_voip](go/bin/tcd_voip).
A couple of them are unpublished and can't be downloaded.

## Compatibility

Zimtohrli is a project under development, and is built and tested in a Debian-like environment.

## Build

Some dependencies for Zimtohrli are downloaded and managed by the build script, but others need to be installed before building.

- cmake
- ninja-build

To build the compare tool, a few more dependencies are necessary:

- libogg-dev
- libvorbis-dev
- libflac-dev
- libopus-dev
- libasound2-dev
- libglfw3-dev

Finally, to build and test the Python and Go wrappers, the following dependencies are necessary:

- golang-go
- python3
- xxd
- zlib1g-dev
- ffmpeg

To install these in a Debian-like system:

```
sudo apt install -y cmake ninja-build clang clang-tidy libogg-dev libvorbis-dev libflac-dev libopus-dev libasound2-dev libglfw3-dev golang-go python3 xxd zlib1g-dev ffmpeg
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

