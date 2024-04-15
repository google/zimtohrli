# Copyright 2022 The Zimtohrli Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for zimtohrli.python.listening_experiments.

We test basic functionality of the ListeningExperiments class with synthetic
data.
"""

import numpy as np
from google3.testing.pybase import googletest

# copybara:strip_begin
from google3.third_party.zimtohrli.python import listening_experiments
# copybara:strip_end_and_replace_begin
# from . import listening_experiments
# copybara:replace_end

class ListeningExperimentsTest(googletest.TestCase):

  def test_basic_sequence(self):
    exp = listening_experiments.ListeningExperiment(
        np.ones((3,)), 7 * np.ones((3,)), 5)
    np.testing.assert_almost_equal(
        exp.place_in_sequence(1000), [
            1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1.,
            1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 7., 7.,
            7., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
            0., 0., 0., 0., 0., 1., 1., 1.
        ])

if __name__ == '__main__':
  googletest.main()
