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
"""Handles parameterizing functions."""

import functools


def parameterize(*kwargs):
    """Parameterizes a function with key-value args."""

    def decorator(func):
        @functools.wraps(func)
        def call_with_parameters(self, **inner_kwargs):
            for kwarg in kwargs:
                func(self, **kwarg)

        return call_with_parameters

    return decorator
