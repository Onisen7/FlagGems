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
            noise = torch.zeros_like(inp)
            yield inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, self.training, None


class RreluWithNoiseInplaceBenchmark(base.UnaryPointwiseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshot_key = None
        self._snapshot = None

    def get_latency(self, op, *args, **kwargs):
        # Benchmark.run measures the reference implementation first and then
        # FlagGems with the same positional tensors. Restore both mutable
        # buffers before the second measurement so the two implementations
        # start from the same distribution. The shared benchmark base still
        # repeats the callable on that buffer during warmup/repetition.
        if len(args) >= 2 and torch.is_tensor(args[0]) and torch.is_tensor(args[1]):
            key = (tuple(args[0].shape), args[0].dtype, args[0].device)
            if self._snapshot_key == key:
                args[0].copy_(self._snapshot[0])
                args[1].copy_(self._snapshot[1])
            else:
                self._snapshot_key = key
                self._snapshot = (args[0].clone(), args[1].clone())
        return super().get_latency(op, *args, **kwargs)

    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            noise = torch.zeros_like(inp)
            yield inp, noise, DEFAULT_LOWER, DEFAULT_UPPER, self.training, None


@pytest.mark.rrelu_with_noise
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise(training):
    bench = RreluWithNoiseBenchmark(
        op_name="rrelu_with_noise",
        torch_op=torch.ops.aten.rrelu_with_noise,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=False,
    )
    bench.training = training
    bench.run()


@pytest.mark.rrelu_with_noise_
@pytest.mark.parametrize("training", [False, True])
def test_rrelu_with_noise_inplace(training):
    bench = RreluWithNoiseInplaceBenchmark(
        op_name="rrelu_with_noise_",
        torch_op=torch.ops.aten.rrelu_with_noise_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.training = training
    bench.run()
