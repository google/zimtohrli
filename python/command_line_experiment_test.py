"""Tests for command_line_experiment."""

from typing import Callable

import numpy as np

from google3.testing.pybase import googletest
from google3.third_party.zimtohrli.python import command_line_experiment as cle
from google3.third_party.zimtohrli.python import distorted_loops as dl


class CommandLineExperimentTest(googletest.TestCase):

  def test_sigm(self):
    k = [-1, 1]

    np.testing.assert_almost_equal(cle._sigm(k, 0), 0.86552928931)
    np.testing.assert_almost_equal(cle._sigm(k, 1000), 1.0)
    np.testing.assert_almost_equal(cle._sigm(k, -1000), 0.5)

  def test_sigm_inverse(self):
    k = [-1, 1]
    np.testing.assert_almost_equal(cle._sigm_inverse(k, cle._sigm(k, 0)), 0.0)
    np.testing.assert_almost_equal(cle._sigm_inverse(k, cle._sigm(k, 0.1)), 0.1)
    np.testing.assert_almost_equal(
        cle._sigm_inverse(k, cle._sigm(k, -0.1)), -0.1)

  def test_gen_loss_f(self):
    k = (-1, 1)
    x = np.array([0, 1])
    y = np.array([0, 1])
    loss_f = cle.gen_loss_f(cle._sigm, x, y)
    np.testing.assert_allclose(loss_f(k), y-cle._sigm(k, x))

  def mock_oracle(self, amount: float, signal: np.ndarray,
                  distortion: Callable[[np.ndarray, float],
                                       np.ndarray], x_train: list[float],
                  y_train: list[int], distortion_type: str) -> bool:
    """Mock oracle to test the `binary_search` function.

      Mocks the responses that the listener would provide through the keyboard.

    Args:
      amount: Based on the amount value we return a mock true or false response.
      signal: Unused.
      distortion: Unused.
      x_train: Unused.
      y_train: Unused.
      distortion_type: Unused.

    Returns:
      Mock true or false respone, based on amount provided.
    """
    # Unused arguments.
    del signal, distortion, x_train, y_train, distortion_type

    # Return some hardcoded boolean values, just to test a specific scenario
    # in the folowing unit test
    if amount == 20.0:
      return False
    elif amount == 40.0:
      return True
    else:  # We don't care about other values.
      return True

  def test_binary_search(self):
    max_distortion = 40
    signal = np.empty(1)
    distortion = dl.intensity_distortion  # Dummy distortion for argument.
    distortion_type = 'intensity'
    x_train = []
    y_train = []

    high, max_distortion = cle.binary_search(max_distortion, x_train, y_train,
                                             distortion, signal,
                                             self.mock_oracle, distortion_type)
    np.testing.assert_almost_equal(high, 40.0)
    np.testing.assert_almost_equal(max_distortion, 80.0)

  def test_gen_distorted_loop_filename(self):
    np.testing.assert_string_equal(
        cle.gen_distorted_loop_filename('intensity', 33.3333),
        'distorted_loop_intensity_33_333.wav')
    np.testing.assert_string_equal(
        cle.gen_distorted_loop_filename('intensity', 33.33),
        'distorted_loop_intensity_33_330.wav')
    np.testing.assert_string_equal(
        cle.gen_distorted_loop_filename('timing', 10),
        'distorted_loop_timing_10_0.wav')


if __name__ == '__main__':
  googletest.main()
