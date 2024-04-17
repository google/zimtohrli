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
"""Tests for google3.third_party.zimtohrli.python.elliptic."""

import jax.numpy as jnp
import numpy as np
import scipy
import unittest
import parameterized
import elliptic


class EllipticTest(unittest.TestCase):

    @parameterized.parameterize(
        dict(order=4, rp=1, rs=80, passband=(100, 200)),
        dict(order=3, rp=3, rs=30, passband=(100, 200)),
        dict(order=2, rp=10, rs=60, passband=(20, 21)),
        dict(order=4, rp=0.1, rs=30, passband=(20, 21)),
        dict(order=1, rp=3, rs=0, passband=(1000, 1200)),
        dict(order=2, rp=3, rs=3, passband=(10000, 10200)),
    )
    def test_filter(self, order, rp, rs, passband):
        fs = 48000
        sig = scipy.signal.chirp(np.linspace(0, 1, fs), 20, 1, 20000, method="linear")
        coeffs = scipy.signal.ellip(
            N=order,
            rp=rp,
            rs=rs,
            Wn=passband,
            btype="bandpass",
            output="sos",
            fs=fs,
        )
        got_filt = elliptic.iirfilter(
            jnp.asarray(
                [coeffs],
            ),
            sig,
        )[0, :]
        want_filt = scipy.signal.sosfilt(coeffs, sig)
        np.testing.assert_allclose(got_filt, want_filt, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
