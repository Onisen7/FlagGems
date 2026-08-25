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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


SHAPES = [(2, 19, 7), (1024, 1024), (16, 128, 64, 60)]
DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 1.0 / 3.0


def _bounds(training):
    # Equal bounds remove random-number differences between reference and
    # FlagGems while still exercising the training branch.
    return (0.25, 0.25) if training else (DEFAULT_LOWER, DEFAULT_UPPER)


def _run(op_name, self, noise, lower, upper, training):
    op = getattr(torch.ops.aten, op_name)
    return op(self, noise, lower, upper, training, None)


@pytest.mark.parametrize("op_name", ["rrelu_with_noise", "rrelu_with_noise_"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rrelu_with_noise(op_name, training, shape, dtype):
    lower, upper = _bounds(training)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if not training:
        noise.uniform_(lower, upper)

    ref_inp = utils.to_reference(inp.clone())
    ref_noise = utils.to_reference(noise.clone())
    ref_result = _run(op_name, ref_inp, ref_noise, lower, upper, training)

    with flag_gems.use_gems():
        result = _run(op_name, inp, noise, lower, upper, training)

    if op_name.endswith("_"):
        assert result.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(result, ref_result, dtype)
    utils.gems_assert_close(noise, ref_noise, dtype)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rrelu_with_noise_inplace_alias(training, dtype):
    lower, upper = _bounds(training)
    inp = torch.randn((37,), dtype=dtype, device=flag_gems.device)
    noise = torch.zeros_like(inp)
    if not training:
        noise.uniform_(lower, upper)
    input_ptr = inp.data_ptr()

    with flag_gems.use_gems():
        result = _run("rrelu_with_noise_", inp, noise, lower, upper, training)

    assert result.data_ptr() == input_ptr


def test_rrelu_with_noise_autograd():
    dtype = torch.float32
    lower = upper = 0.25
    inp = torch.randn((257,), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone()).requires_grad_()
    ref_noise = torch.zeros_like(ref_inp)
    ref_out = _run("rrelu_with_noise", ref_inp, ref_noise, lower, upper, True)
    ref_out.sum().backward()

    gems_inp = inp.clone().requires_grad_()
    gems_noise = torch.zeros_like(gems_inp)
    with flag_gems.use_gems():
        gems_out = _run(
            "rrelu_with_noise", gems_inp, gems_noise, lower, upper, True
        )
        gems_out.sum().backward()

    utils.gems_assert_close(gems_inp.grad, ref_inp.grad, dtype)
