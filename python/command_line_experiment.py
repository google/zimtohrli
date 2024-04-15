"""Just-Noticeable-Difference (JND) Command line listening experiment.

The experiment is doing side by side comparison of two sound loops. One loop
consists of 8 repetitions of the audio signal, undistorted. The second loop
consists of 8 repetitions of the audio signal, with the 5th being distorted by
a, specified via command line argument, distortion.
The listener tries to identify which of the two loops contained the distortion.
The search for the JND point starts from high amounts of distortions and while
the distortions are easily detectable, halves them, in order to trim down the
search space for the JND.
After trimming we sample data from the remaining search space and fit a sigmoid
curve to them. The point where the sigmoid starts rising is our estimation of
the JND point.
"""

from collections.abc import Sequence
import itertools
import json
import math
import random
import subprocess
import sys
import time
from typing import Callable

from absl import app
import numpy as np
from numpy import typing as np_typing
import scipy
from scipy import stats
import scipy.io.wavfile

from google3.third_party.zimtohrli.python import distorted_loops as dl

# Arbitrary scale for the signal to be audible when written in .wav format.
_SCALING = 10000
# Reference audio volume in dB for conducting the listening tests. The listener
# will be prompted to tune their device's audio so that a full scale sine at
# 1kHz, will be at _REFERENCE_VOLUME_DB dB. That way we can reproduce the volume
# in which the listening experiment was conducted.
_REFERENCE_VOLUME_DB = 90
# Filename for JSON output file that stores the filenames of the .wav files
# containing the distorted samples used for listening tests, and the
# coresponding results.
_JSON_FILENAME = 'listening_experiment_results.json'
# Dictionary to store lists experimental data. We will keep a list of the .wav
# filenames, the listener's responses and the distorion amounts, for every
# iteration of the experiment.
_experimental_data = {}


def oracle(amount: float, signal: np.ndarray,
           distortion: Callable[[np.ndarray, float],
                                np.ndarray], x_train: list[float],
           y_train: list[int], distortion_type: str) -> bool:
  """Oracle for the binary search.

    Returns True when the `amount` of `distortion` is detectable by the listener
    when applied on `signal`.
    To decide if the detection is detectable, we are using a criterion inspired
    by binomial testing and p-values. In each trial of the experiment we
    calculate a `conviction that we are above 50% detection rate`, which is
    defined as `1 - p_value_for_detection_rate_being_above_0.5`, as if we were
    looking at the so far data for the first time and a p-value could be
    calculated.
    If the observed results, at any point during the first 10 trials, give a
    `conviction that we are above 50% detection rate`, above or equal to 0.97,
    we consider the detection detectable and return True.
    Otherwise, we consider the distortion undetectable and return False.

  Args:
    amount: Amount of distortion to apply.
    signal: Original signal.
    distortion: Function that applies `amount` distortion on signal
    x_train: X-axis training data for fitting a sigmoid. Corresponds to
      distortion amounts.
    y_train: Y-axis training data for fitting a sigmoid. Corresponds to 1/0
      based on whether the listener detected the distortion or not.
    distortion_type: Type of distortion as string.

  Returns:
    True/False based on whether the listener detected the distortion or not.
  """
  successes = 0

  for trials in itertools.count(start=1):
    x_train.append(amount)

    successes += listening_test(signal, distortion, amount, y_train,
                                distortion_type)

    # Calculate conviction of the detection rate being above 50%.
    conviction_above_50 = sum(
        stats.binom.pmf(m, trials, 0.5) for m in range(successes + 1))

    if trials >= 4 and conviction_above_50 >= 0.97:
      return True
    if trials >= 10:
      return False


def _sigm(k: np_typing.ArrayLike,
          x: np_typing.ArrayLike) -> np_typing.ArrayLike:
  k1, k2 = k
  return 0.5 + 0.5 * (1 / (1 + np.exp(k1 + k2 * -x)))


def _sigm_inverse(k: np_typing.ArrayLike,
                  y: np_typing.ArrayLike) -> np_typing.ArrayLike:
  k1, k2 = k
  return (k1 - np.log((2 - 2 * y) / (2 * y - 1))) / k2


def gen_loss_f(
    f: Callable[[np_typing.ArrayLike, np_typing.ArrayLike],
                np_typing.ArrayLike], x: np_typing.ArrayLike,
    y: np_typing.ArrayLike
) -> Callable[[np_typing.ArrayLike], np_typing.ArrayLike]:
  return lambda k: y - f(k, x)


def solve(f: Callable[[np_typing.ArrayLike, np_typing.ArrayLike],
                      np_typing.ArrayLike], x: np_typing.ArrayLike,
          y: np_typing.ArrayLike) -> np.ndarray:
  sol = scipy.optimize.least_squares(gen_loss_f(f, x, y), (1., 1.))
  k = sol['x']
  return k


def binary_search(max_distortion: int, x_train: list[int], y_train: list[int],
                  distortion: Callable[[np.ndarray, float], np.ndarray],
                  signal: np.ndarray, search_oracle: Callable[..., bool],
                  distortion_type: str) -> tuple[float, int]:
  """Conducts binary search to prune the interval of distortion amounts.

  As long as the `mid` distortion is not detectable, according to `oracle`,
  the function halves the high bound of the interval.

  Args:
    max_distortion: Max distortion amount we are going to consider.
    x_train: X-axis training data (corresponds to distortion amount).
    y_train: Y-axis training data (corresponds to distortion amount being/not
    being detectable)
    distortion: Distortion applying function to be passed to `oracle`.
    signal: Signal to apply the distortion.
    search_oracle: Function to use as oracle for the iterations of the binary
    search. Making this an argument allows us to mock user input for testing.
    distortion_type: Type of distortion as string.

  Returns:
    Tuple of high bound for which the search stopped and possibly modified max
    distortion.
  """
  high = max_distortion
  low = 0
  while True:
    if high <= low + 0.01:
      break

    mid = low + (high - low) / 2
    print('Executing test with distortion strength ', mid)
    result = search_oracle(mid, signal, distortion, x_train, y_train,
                           distortion_type)

    if result:
      high = mid
    # If oracle failed without reducing `high`, double the `max_distortion` and
    # retry.
    elif high == max_distortion:
      max_distortion *= 2
      high = max_distortion
    else:
      break

  return high, max_distortion


def initial_messages() -> None:
  """Prints the initial messages to the user.

  Prompts to tune the device's sound level and gives instructions.
  """
  # Prompt the user to tune the device's volume.
  prompt_message = (
      'You will first hear a 10 sec duration sine beep. Take off '
      'your headphones and insert a decibelometer between them. '
      "Tune the device's volume so that the sine beep is at %ddB." %
      _REFERENCE_VOLUME_DB)
  print(prompt_message)

  tuning_signal = dl.signal_beep(1000., 10, 48000)
  tuning_signal_filename = 'tuning_signal.wav'
  scipy.io.wavfile.write(tuning_signal_filename, 48000,
                         (tuning_signal / np.max(np.abs(tuning_signal)) *
                          32767).astype(np.int16))
  _experimental_data['tuning_signal'] = tuning_signal_filename

  char = input('Respond with [y]es when you are ready to tune.')
  while char != 'y':
    char = input()

  tuning_command_list = ['aplay', '-q', 'tuning_signal.wav']
  subprocess.run(tuning_command_list, check=True)
  time.sleep(1)

  start_message = (
      'Listen to the two samples being played.\nDecide whether the '
      'distorted one is the [f]irst or the [s]econd.')
  print(start_message)

  while True:
    char = input('Type [y]es to start.\n')
    if char == 'y':
      break


def initialize_experimental_data() -> None:
  """Helper to initialize the experimental data dictionary.
  """
  _experimental_data['filenames'] = []
  _experimental_data['listener_responses'] = []
  _experimental_data['amounts'] = []


def gen_distorted_loop_filename(distortion_type: str, amount: float) -> str:
  """Helper to create filenames for the distorted loop wav files.

  Generates a filename that contains the distortion type and the distortion
  amount, with the integer and fractional parts separated by underscore.
  We keep three decimal digits in the fractional part.

  Args:
    distortion_type: Type of distortion.
    amount: Amount of distortion applied.

  Returns:
    Filename of .wav file containing the distortion type and amount.
  """
  fractional_part, integer_part = math.modf(amount)
  integer_part = int(integer_part)
  fractional_part = int(round(fractional_part, 3) * 1000)
  return 'distorted_loop_{type}_{int_part}_{frac_part}.wav'.format(
      type=distortion_type,
      int_part=integer_part,
      frac_part=fractional_part)


def listening_test(signal: np.ndarray, distortion: Callable[[np.ndarray, float],
                                                            np.ndarray],
                   amount: float, y_train: list[int],
                   distortion_type: str) -> bool:
  """Plays a distorted audio and collects the listener's response.

  Plays a loop of 8 repetitions of the signal, with the 5th being distorted by
  `amount` amount of distortion, side by side with a loop of 8 repetitions of
  the original signal. Asks the listener to identify the distorted loop
  and records their response in y_train (1 if correct, 0 if incorrect).

  Args:
    signal: Signal to distort.
    distortion: Distortion applying function.
    amount: Amount of distortion to apply.
    y_train: Training data in which we append our result.
    distortion_type: Type of distortion as string.

  Returns:
    True if listener detected correctly, False otherwise.
  """
  distorted_loop_filename = gen_distorted_loop_filename(distortion_type, amount)
  _experimental_data['filenames'].append(distorted_loop_filename)
  _experimental_data['amounts'].append(amount)

  no_dist_loop, dist_loop = dl.distorted_loops(signal, distortion, amount)
  scipy.io.wavfile.write(distorted_loop_filename, 48000,
                         (dist_loop * _SCALING).astype(np.int16))
  scipy.io.wavfile.write('undistorted_loop.wav', 48000,
                         (no_dist_loop * _SCALING).astype(np.int16))

  # Play the distorted and the undistorted samples, deciding randomly which
  # is first and which is second.
  char = 't'
  while char == 't':
    rand = random.getrandbits(1)

    play_distorted_list = ['aplay', '-q', distorted_loop_filename]
    play_undistorted_list = ['aplay', '-q', 'undistorted_loop.wav']

    first_command_list = play_distorted_list if rand else play_undistorted_list
    second_command_list = play_undistorted_list if rand else play_distorted_list

    subprocess.run(first_command_list, check=True)
    time.sleep(1)
    subprocess.run(second_command_list, check=True)

    char = input('[f]irst or [s]econd or [t]ry again...\n')
    while char != 'f' and char != 's' and char != 't':
      char = input()

    if char == 't':
      continue

    if (char == 'f' and rand) or (char == 's' and not rand):
      print('The listener detected correctly for distortion amount ', amount)
      y_train.append(1)
      _experimental_data['listener_responses'].append(1)
      return True
    else:
      y_train.append(0)
      _experimental_data['listener_responses'].append(0)
      return False


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    sys.exit('Usage: python3 command_line_experiment.py <distortion_type>')

  initial_messages()
  initialize_experimental_data()

  # Original signal
  sine = dl.signal_beep(1000.)

  # Parameters for creating noise maskers
  width_hz = 200.
  freq_hz = 1100.
  sample_rate = 48000
  stdev_white = -50.
  stdev_bandlimited = -20.
  amplitude = 0.5
  rng = np.random.default_rng()

  # Distortion functions.
  white_noise_distortion = dl.as_distorter(
      dl.gen_white_noise_masker, stdev=stdev_white, rng=rng)
  bandlimited_noise_distortion = dl.as_distorter(
      dl.gen_bandlimited_noise_masker,
      width_hz=width_hz,
      freq_hz=freq_hz,
      stdev=stdev_bandlimited,
      rng=rng)
  sine_distortion = dl.as_distorter(
      dl.gen_sine_masker,
      amount=amplitude,
      freq_hz=freq_hz,
      sample_rate=sample_rate)
  intensity_distortion = dl.intensity_distortion
  frequency_distortion = dl.frequency_distortion
  timing_distortion = dl.timing_distortion

  # Parse the 'distortion type' command line argument.
  distortion_type = argv[1]
  if distortion_type == 'intensity':
    distortion = intensity_distortion
  elif distortion_type == 'frequency':
    distortion = frequency_distortion
  elif distortion_type == 'timing':
    distortion = timing_distortion
  elif distortion_type == 'white_noise':
    distortion = white_noise_distortion
  elif distortion_type == 'bandlimited_noise':
    distortion = bandlimited_noise_distortion
  elif distortion_type == 'sine':
    distortion = sine_distortion
  else:
    sys.exit('Supported types of distortions are: '
             "'intensity', 'frequency', 'timing','white_noise', "
             "'bandlimited_noise', 'sine'.")

  # Initial max distortion value for defining our sampling interval of
  # distortion amounts as [0, max_distortion].
  max_distortion = 20

  # Training data for fitting the sigmoid.
  x_train = []
  y_train = []

  high, max_distortion = binary_search(max_distortion, x_train, y_train,
                                       distortion, sine, oracle,
                                       distortion_type)

  # Boundaries of the sampling interval.
  sampling_interv_right = high
  sampling_interv_left = 0

  reps = 3

  # We perform `reps` repetitions of sampling data points, fitting the sigmoid
  # and then repeating with more samples in the interesting region of the
  # sigmoid.
  for rep in range(reps):
    # Create `extra_data_count` extra training data, in the sampling interval.
    # For the first fitting, collect more data than for the refittings.
    extra_data_count = 30 if rep == 0 else 10
    x_extra = np.linspace(
        sampling_interv_left,
        sampling_interv_right,
        extra_data_count,
        endpoint=True)
    y_extra = []

    for x in x_extra:
      listening_test(sine, distortion, x, y_extra, distortion_type)

    # Concatenate the x and y results to create arrays with the training data.
    x_train = np.concatenate((np.array(x_train), np.array(x_extra)))
    y_train = np.concatenate((np.array(y_train), np.array(y_extra)))

    # Fit the sigmoid.
    k_sigm = solve(_sigm, x_train, y_train)

    # Lets find the interval where the sigmoid is between 0.51 and 0.99.
    # This is the part of the sigmoid which has non zero derivative, and we are
    # going to collect new training data in this interval to refit.
    lower_bound = 0.51
    upper_bound = 0.99

    sampling_interv_left = max(_sigm_inverse(k_sigm, lower_bound), 0)
    sampling_interv_right = min(
        _sigm_inverse(k_sigm, upper_bound), max_distortion)

  # After fitting the sigmoid, resampling training data in the non-zero
  # derivative part, and refitting, we get the JND point as the point where
  # the sigmoid just starts rising, which is `sampling_interv_left`.
  print('JND point is ', sampling_interv_left)

  # Apply the sigmoid to the amounts of distortion we used during the
  # experiment.
  _experimental_data['sigmoid_values'] = _sigm(
      k_sigm, np.asarray(_experimental_data['amounts'])).tolist()
  # Save module constants and sigmoid constants
  _experimental_data['sigmoid_constants'] = k_sigm.tolist()
  _experimental_data['reference_volume_db'] = _REFERENCE_VOLUME_DB
  _experimental_data['scaling'] = _SCALING

  # Write the data to the JSON file.
  json_object = json.dumps(_experimental_data, indent=4)
  with open(_JSON_FILENAME, 'w') as outfile:
    outfile.write(json_object)

if __name__ == '__main__':
  app.run(main)
