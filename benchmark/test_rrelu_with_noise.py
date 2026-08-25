# Copyright 2026 FlagOS Contributors
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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 1.0 / 3.0


class RreluWithNoiseBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            noise = torch.empty_like(inp)
            yield inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, True, None


class RreluWithNoiseInplaceBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            noise = torch.empty_like(inp)
            # The benchmark framework reuses positional arguments for repeated
            # calls. Both implementations therefore observe the same mutated
            # in-place buffer, without charging the benchmark for a reset copy.
            yield inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, True, None


@pytest.mark.rrelu_with_noise
def test_rrelu_with_noise():
    bench = RreluWithNoiseBenchmark(
        op_name="rrelu_with_noise",
        torch_op=torch.ops.aten.rrelu_with_noise,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rrelu_with_noise_
def test_rrelu_with_noise_inplace():
    bench = RreluWithNoiseInplaceBenchmark(
        op_name="rrelu_with_noise_",
        torch_op=torch.ops.aten.rrelu_with_noise_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
