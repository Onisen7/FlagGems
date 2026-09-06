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

"""Ascend implementation of aten.rrelu_with_noise and aten.rrelu_with_noise_."""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.random_utils import philox_backend_seed_offset


logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333
_UNROLL = 4


def _uniform_block(args):
    # Keep the same conservative choices as Ascend exponential.py without
    # importing the backend heuristic module at import/decorator time.
    return 512 if args["N"] <= 512 else 1024


def _uniform_num_warps(args):
    if args["N"] <= 512:
        return 4
    if args["N"] <= 1024:
        return 8
    return 16


_PHILOX_SA = tl.constexpr(0xD2511F53)
_PHILOX_SB = tl.constexpr(0xCD9E8D57)
_PHILOX_KEY_A = tl.constexpr(0x9E3779B9)
_PHILOX_KEY_B = tl.constexpr(0xBB67AE85)


@triton.jit
def _mulhi_u32_limb(a, b):
    a0 = a & 0xFFFF
    a1 = (a >> 16) & 0xFFFF
    b0 = b & 0xFFFF
    b1 = (b >> 16) & 0xFFFF
    w0 = a0 * b0
    t = a1 * b0 + ((w0 >> 16) & 0xFFFF)
    w1 = a0 * b1 + (t & 0xFFFF)
    return a1 * b1 + ((t >> 16) & 0xFFFF) + ((w1 >> 16) & 0xFFFF)


@triton.jit
def _philox4x32_10(seed, c0, c1, c2, c3):
    seed64 = seed.to(tl.uint64)
    k0 = (seed64 & 0xFFFFFFFF).to(tl.uint32)
    k1 = ((seed64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
    for _ in tl.static_range(10):
        hi0 = _mulhi_u32_limb(c0, _PHILOX_SA)
        lo0 = c0 * _PHILOX_SA
        hi1 = _mulhi_u32_limb(c2, _PHILOX_SB)
        lo1 = c2 * _PHILOX_SB
        n0 = hi1 ^ c1 ^ k0
        n1 = lo1
        n2 = hi0 ^ c3 ^ k1
        n3 = lo0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = k0 + _PHILOX_KEY_A
        k1 = k1 + _PHILOX_KEY_B
    return c0, c1, c2, c3


@triton.jit
def _uint32_to_uniform_float(r):
    x = r.to(tl.int32, bitcast=True)
    xa = x ^ (x >> 31)
    return xa.to(tl.float32) * 4.6566127342e-10


@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N"])
def _rrelu_uniform_kernel(
    out_ptr,
    N,
    from_,
    to,
    philox_seed,
    philox_offset,
    UNROLL,
    BLOCK: tl.constexpr,
):
    # Use the same worker/task decomposition as Ascend exponential.py. The
    # manual Philox implementation is supported by the Ascend Triton backend,
    # unlike tl.philox on some CANN releases.
    n_workers = tl.num_programs(0)
    pid = tl.program_id(0)
    n_tasks = tl.cdiv(N, BLOCK * UNROLL)
    tasks_per_worker = tl.cdiv(n_tasks, n_workers)

    for task_index in range(tasks_per_worker):
        task_id = pid + task_index * n_workers
        seed64 = philox_seed.to(tl.int64)
        offset64 = philox_offset.to(tl.int64)
        c0 = (offset64 & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((offset64 >> 32) & 0xFFFFFFFF).to(tl.uint32)
        i4 = task_id * BLOCK + tl.arange(0, BLOCK)
        c0 += i4
        zeros = c0 * 0
        r0, r1, r2, r3 = _philox4x32_10(seed64, c0, c1, zeros, zeros)
        scale = to - from_
        r0 = _uint32_to_uniform_float(r0) * scale + from_
        r1 = _uint32_to_uniform_float(r1) * scale + from_
        r2 = _uint32_to_uniform_float(r2) * scale + from_
        r3 = _uint32_to_uniform_float(r3) * scale + from_

        start = task_id.to(tl.int64) * BLOCK * 4
        off0 = start + tl.arange(0, BLOCK)
        off1 = off0 + BLOCK
        off2 = off1 + BLOCK
        off3 = off2 + BLOCK
        tl.store(out_ptr + off0, r0, mask=off0 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off1, r1, mask=off1 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off2, r2, mask=off2 < N, eviction_policy="evict_first")
        tl.store(out_ptr + off3, r3, mask=off3 < N, eviction_policy="evict_first")


def _fill_training_noise(noise, lower, upper, generator=None):
    """Fill a contiguous tensor with Ascend-supported Philox uniform values."""
    N = noise.numel()
    if N == 0:
        return noise

    # RReLU consumes one Philox 4-value block per four output elements, which
    # preserves generator advancement and supports an explicit torch.Generator.
    increment = triton.cdiv(N, _UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    block = _uniform_block({"N": N})
    num_warps = _uniform_num_warps({"N": N})

    def grid_fn(meta):
        grid = triton.cdiv(N, block * _UNROLL)
        return (min(grid, 240),)

    with torch_device_fn.device(noise.device):
        _rrelu_uniform_kernel[grid_fn](
            noise,
            N,
            float(lower),
            float(upper),
            philox_seed,
            philox_offset,
            _UNROLL,
            BLOCK=block,
            num_warps=num_warps,
        )
    return noise


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=2,
    promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_train_ascend(self, noise):
    # torch_npu uses unit slope at both signed zeros; sample only x < 0.
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


def _check_args(self, noise, lower, upper):
    if self.shape != noise.shape:
        raise RuntimeError("noise tensor must have the same shape as self")
    if self.device != noise.device:
        raise RuntimeError("self and noise must be on the same device")
    if self.dtype != noise.dtype:
        raise RuntimeError("self and noise must have the same dtype")
    if not self.is_floating_point():
        raise RuntimeError(f"rrelu_with_noise is not implemented for {self.dtype}")
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        raise RuntimeError("rrelu bounds must be finite")
    if float(lower) > float(upper):
        raise RuntimeError(
            f"Lower bound should be less than or equal to upper bound, "
            f"got lower={lower}, upper={upper}"
        )


def _impl(self, noise, lower, upper, training, generator, out):
    _check_args(self, noise, lower, upper)
    if self.numel() == 0:
        return self if out is not None else torch.empty_like(self)

    self_work = self if self.is_contiguous() else self.contiguous()
    self_flat = self_work.reshape(-1)

    if training:
        noise_work = noise
        if not noise_work.is_contiguous():
            noise_work = torch.empty_like(
                noise_work, memory_format=torch.contiguous_format
            )
        _fill_training_noise(noise_work, lower, upper, generator)

        # Keep the sampled input and effective-noise output disjoint because
        # Ascend pointwise multi-output aliasing is not reliable on all CANNs.
        effective_noise = torch.empty_like(
            noise_work, memory_format=torch.contiguous_format
        )
        result_flat, _ = _rrelu_with_noise_train_ascend(
            self_flat,
            noise_work.reshape(-1),
            out1=effective_noise.reshape(-1),
        )
        noise.copy_(effective_noise.reshape(noise_work.shape))
    else:
        slope = (float(lower) + float(upper)) * 0.5
        result_flat = _rrelu_with_noise_eval(self_flat, slope)

    result = result_flat.reshape(self.shape)
    if out is None:
        return result
    with torch.no_grad():
        out.copy_(result)
    return out


def rrelu_with_noise(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    logger.debug("GEMS ASCEND RRELU_WITH_NOISE")
    return _impl(self, noise, lower, upper, training, generator, None)


def rrelu_with_noise_(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    logger.debug("GEMS ASCEND RRELU_WITH_NOISE_")
    _impl(self, noise, lower, upper, training, generator, self)
    return self


__all__ = ["rrelu_with_noise", "rrelu_with_noise_"]
