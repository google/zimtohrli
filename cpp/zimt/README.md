# Zimtohrli

Zimtohrli is a perceptual audio metric.

In simplified terms, it performs a frequency analysis of a sound using a filter
bank where each filter has a bandwidth corresponding to the frequency resolution
of human hearing at that sub spectrum.

This frequency analysis is then passed to a masking model, that attenuates the
energy in channels that are predicted to be masked by more energetic channels.

The result of the masking model is then passed to a loudness model, that uses
the process described in ISO 226 to convert the energy values the Phons scale,
which is linear to human perception of sound energy.

## Example

To find the most distorted window between two signals:

- Create two Zimtohrli instances.
- Walk through both signals one window at a time.
- Create a Zimtohrli spectrogram for each window.
- Compute the distance between the spectrograms.

See the `FindMaxDistortionTest` in `zimtohrli_test.cc` for details.

## Command line comparison tool

The compare tool (cpp/zimt/compare.cc) runs in three modes:

- Default, where it will simply output a Zimtohrli distance score between each
  channel in a file A compared to an arbitrary number of file B.

  Output will be:
  [Distance between file A channel 0 and file B_0 channel 0]
  ...
  [Distance between file A channel 0 and file B_n channel 0]
  ...
  [Distance between file A channel m and file B_0 channel m]
  ...
  [Distance between file A channel m and file B_n channel m]

  To run compare in default mode, just provide --path_a and --path_b arguments.

- Verbose, where it will output some metadata about how the distance evolves
  for each psychoacoustic model that is applied, and which timesteps and
  frequencies contribute the most to the distance.

  To run compare in verbose mode, provide the --verbose=true argument.

- UX, where it will also open an ImGui window displaying spectrograms of the
  compared files and their relative and absolute differences. The UX also allows
  playback of frequency filtered and time delimited segments of the files and
  the difference noise.