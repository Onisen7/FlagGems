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

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333
_ASCEND_RRELU_BLOCK = 1024
_ASCEND_RRELU_UNROLL = 4


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_train(self, noise):
    # PyTorch's CPU/CUDA path samples for self <= 0 (including signed zero)
    # and records one for positive/NaN elements.
    not_positive = self <= 0
    effective_noise = tl.where(not_positive, noise, 1.0)
    output = tl.where(not_positive, self * effective_noise, self)
    return output, effective_noise


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_train_ascend(self, noise):
    # torch_npu records a unit slope for both +0.0 and -0.0. Match the native
    # Ascend aten reference without changing the CPU/CUDA zero-point behavior.
    negative = self < 0
    effective_noise = tl.where(negative, noise, 1.0)
    output = tl.where(negative, self * effective_noise, self)
    return output, effective_noise


@pointwise_dynamic(
    is_tensor=[True, False],
    num_outputs=1,
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_eval(self, slope):
    return tl.where(self > 0, self, self * slope)


@triton.jit(do_not_specialize=["lower", "upper", "philox_seed", "philox_offset"])
def _rrelu_with_noise_ascend_train_fused_kernel(
    self_ptr,
    output_ptr,
    noise_ptr,
    n_elements,
    lower,
    upper,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    # Generate the random slope and write both outputs in one kernel. The
    # four-way unroll follows FlagGems' Philox convention: one random value
    # maps to one element in each of four adjacent blocks.
    pid = tl.program_id(0)
    base = pid * BLOCK * _ASCEND_RRELU_UNROLL
    lane = tl.arange(0, BLOCK)

    seed = philox_seed.to(tl.int64)
    offset = philox_offset.to(tl.int64)
    c0 = (offset & 0xFFFFFFFF).to(tl.uint32) + pid * BLOCK + lane
    c1 = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    zeros = c0 * 0
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, zeros, zeros)
    r0 = uint_to_uniform_float(r0) * (upper - lower) + lower
    r1 = uint_to_uniform_float(r1) * (upper - lower) + lower
    r2 = uint_to_uniform_float(r2) * (upper - lower) + lower
    r3 = uint_to_uniform_float(r3) * (upper - lower) + lower

    off0 = base + lane
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    mask0 = off0 < n_elements
    value0 = tl.load(self_ptr + off0, mask=mask0, other=0.0)
    negative0 = value0 < 0
    noise0 = tl.where(negative0, r0, 1.0)
    tl.store(noise_ptr + off0, noise0, mask=mask0)
    tl.store(
        output_ptr + off0, tl.where(negative0, value0 * noise0, value0), mask=mask0
    )

    mask1 = off1 < n_elements
    value1 = tl.load(self_ptr + off1, mask=mask1, other=0.0)
    negative1 = value1 < 0
    noise1 = tl.where(negative1, r1, 1.0)
    tl.store(noise_ptr + off1, noise1, mask=mask1)
    tl.store(
        output_ptr + off1, tl.where(negative1, value1 * noise1, value1), mask=mask1
    )

    mask2 = off2 < n_elements
    value2 = tl.load(self_ptr + off2, mask=mask2, other=0.0)
    negative2 = value2 < 0
    noise2 = tl.where(negative2, r2, 1.0)
    tl.store(noise_ptr + off2, noise2, mask=mask2)
    tl.store(
        output_ptr + off2, tl.where(negative2, value2 * noise2, value2), mask=mask2
    )

    mask3 = off3 < n_elements
    value3 = tl.load(self_ptr + off3, mask=mask3, other=0.0)
    negative3 = value3 < 0
    noise3 = tl.where(negative3, r3, 1.0)
    tl.store(noise_ptr + off3, noise3, mask=mask3)
    tl.store(
        output_ptr + off3, tl.where(negative3, value3 * noise3, value3), mask=mask3
    )


@triton.jit(do_not_specialize=["slope"])
def _rrelu_with_noise_ascend_eval_kernel(
    self_ptr,
    output_ptr,
    n_elements,
    slope,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * _ASCEND_RRELU_UNROLL
    lane = tl.arange(0, BLOCK)
    off0 = base + lane
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK

    mask0 = off0 < n_elements
    value0 = tl.load(self_ptr + off0, mask=mask0, other=0.0)
    tl.store(
        output_ptr + off0, tl.where(value0 > 0, value0, value0 * slope), mask=mask0
    )

    mask1 = off1 < n_elements
    value1 = tl.load(self_ptr + off1, mask=mask1, other=0.0)
    tl.store(
        output_ptr + off1, tl.where(value1 > 0, value1, value1 * slope), mask=mask1
    )

    mask2 = off2 < n_elements
    value2 = tl.load(self_ptr + off2, mask=mask2, other=0.0)
    tl.store(
        output_ptr + off2, tl.where(value2 > 0, value2, value2 * slope), mask=mask2
    )

    mask3 = off3 < n_elements
    value3 = tl.load(self_ptr + off3, mask=mask3, other=0.0)
    tl.store(
        output_ptr + off3, tl.where(value3 > 0, value3, value3 * slope), mask=mask3
    )


def _check_rrelu_with_noise_args(self, noise, lower, upper):
    if self.shape != noise.shape:
        raise RuntimeError(
            "noise tensor must have the same shape as self. "
            f"Got self.shape = {tuple(self.shape)} "
            f"and noise.shape = {tuple(noise.shape)}"
        )
    if self.device != noise.device:
        raise RuntimeError(
            f"self and noise must be on the same device, got "
            f"{self.device} and {noise.device}"
        )
    if self.dtype != noise.dtype:
        raise RuntimeError(
            f"self and noise must have the same dtype, got "
            f"{self.dtype} and {noise.dtype}"
        )
    if not self.is_floating_point():
        raise RuntimeError(
            f"rrelu_with_noise is not implemented for dtype {self.dtype}"
        )
    if not math.isfinite(float(lower)):
        raise RuntimeError(f"rrelu: lower bound must be finite, got {lower}")
    if not math.isfinite(float(upper)):
        raise RuntimeError(f"rrelu: upper bound must be finite, got {upper}")
    if float(lower) > float(upper):
        raise RuntimeError(
            f"Lower bound should be less than or equal to the upper bound, "
            f"got lower={lower} and upper={upper}"
        )


def _fill_training_noise(noise, lower, upper, generator):
    # For a strided workspace, sample contiguously and let the training kernel
    # scatter effective noise into the caller's layout while producing output.
    if noise.is_contiguous():
        noise.uniform_(float(lower), float(upper), generator=generator)
        return noise

    sampled = torch.empty_like(noise, memory_format=torch.contiguous_format)
    sampled.uniform_(float(lower), float(upper), generator=generator)
    return sampled


def _is_ascend_device(tensor):
    # Ascend's Triton backend does not reliably support a pointwise output
    # aliasing an input or arbitrary-rank strided output. Keep this fallback
    # local to private-use backends so CUDA and other vendors retain the
    # direct out0/out1 path.
    return tensor.device.type in ("npu", "privateuseone")


def _rrelu_with_noise_ascend_fused_impl(
    self, noise, lower, upper, training, generator, out
):
    """Fast Ascend path for inference/benchmark calls without autograd."""
    self_work = self if self.is_contiguous() else self.contiguous()
    n_elements = self_work.numel()
    output_work = (
        self_work
        if out is self and self.is_contiguous()
        else torch.empty_like(self_work)
    )

    if training:
        noise_work = (
            noise
            if noise.is_contiguous()
            else torch.empty_like(noise, memory_format=torch.contiguous_format)
        )
        increment = triton.cdiv(n_elements, _ASCEND_RRELU_UNROLL)
        philox_seed, philox_offset = philox_backend_seed_offset(
            increment, generator=generator
        )
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK"] * _ASCEND_RRELU_UNROLL),
        )
        with torch_device_fn.device(self_work.device):
            _rrelu_with_noise_ascend_train_fused_kernel[grid](
                self_work.reshape(-1),
                output_work.reshape(-1),
                noise_work.reshape(-1),
                n_elements,
                float(lower),
                float(upper),
                philox_seed,
                philox_offset,
                BLOCK=_ASCEND_RRELU_BLOCK,
            )
        if noise_work is not noise:
            noise.copy_(noise_work.reshape(noise.shape))
    else:
        grid = lambda meta: (
            triton.cdiv(n_elements, meta["BLOCK"] * _ASCEND_RRELU_UNROLL),
        )
        with torch_device_fn.device(self_work.device):
            _rrelu_with_noise_ascend_eval_kernel[grid](
                self_work.reshape(-1),
                output_work.reshape(-1),
                n_elements,
                (float(lower) + float(upper)) * 0.5,
                BLOCK=_ASCEND_RRELU_BLOCK,
            )

    if out is not None:
        if output_work is not out:
            with torch.no_grad():
                out.copy_(output_work.reshape(out.shape))
        return out
    return output_work.reshape(self.shape)


def _rrelu_with_noise_ascend_safe_impl(
    self, noise, lower, upper, training, generator, out
):
    self_work = self if self.is_contiguous() else self.contiguous()
    self_flat = self_work.reshape(-1)

    if training:
        noise_work = (
            noise
            if noise.is_contiguous()
            else torch.empty_like(noise, memory_format=torch.contiguous_format)
        )
        sampled_noise = _fill_training_noise(noise_work, lower, upper, generator)
        # Keep the sampled input and effective-noise output disjoint. Ascend's
        # pointwise compiler does not reliably preserve values when an input
        # tensor is also supplied as a multi-output out1 buffer.
        effective_noise = torch.empty_like(
            noise_work, memory_format=torch.contiguous_format
        )
        result_flat, _ = _rrelu_with_noise_train_ascend(
            self_flat,
            sampled_noise.reshape(-1),
            out1=effective_noise.reshape(-1),
        )
        noise.copy_(effective_noise.reshape(noise.shape))
    else:
        slope = (float(lower) + float(upper)) * 0.5
        result_flat = _rrelu_with_noise_eval(self_flat, slope)

    result = result_flat.reshape(self.shape)
    if out is not None:
        # Avoid out0=self aliasing in the Ascend-generated pointwise kernel.
        # no_grad prevents an internal copy_ from adding CopyBackwards to the
        # user-visible autograd graph; the enclosing aten op owns autograd.
        with torch.no_grad():
            out.copy_(result)
        return out
    return result


def _rrelu_with_noise_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
    out=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper)

    if self.numel() == 0:
        return torch.empty_like(self) if out is None else out

    if _is_ascend_device(self):
        if not self.requires_grad and not noise.requires_grad:
            return _rrelu_with_noise_ascend_fused_impl(
                self, noise, lower, upper, training, generator, out
            )
        return _rrelu_with_noise_ascend_safe_impl(
            self, noise, lower, upper, training, generator, out
        )

    if training:
        sampled_noise = _fill_training_noise(noise, lower, upper, generator)
        if out is None:
            output, _ = _rrelu_with_noise_train(self, sampled_noise, out1=noise)
            return output
        _rrelu_with_noise_train(self, sampled_noise, out0=out, out1=noise)
        return out
    else:
        slope = (float(lower) + float(upper)) * 0.5
        if out is None:
            return _rrelu_with_noise_eval(self, slope)
        return _rrelu_with_noise_eval(self, slope, out0=out)


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise."""
    logger.debug("GEMS RRELU_WITH_NOISE")
    return _rrelu_with_noise_impl(self, noise, lower, upper, training, generator)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    """FlagGems implementation of aten.rrelu_with_noise_."""
    logger.debug("GEMS RRELU_WITH_NOISE_")
    _rrelu_with_noise_impl(self, noise, lower, upper, training, generator, out=self)
    return self


__all__ = ["rrelu_with_noise", "rrelu_with_noise_"]
