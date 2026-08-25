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

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

DEFAULT_LOWER = 0.125
DEFAULT_UPPER = 0.3333333333333333


@pointwise_dynamic(
    is_tensor=[True, True],
    num_outputs=1,
    promotion_methods=[(0, 1, "DEFAULT")],
)
@triton.jit
def _rrelu_with_noise_forward(self, noise):
    # The noise tensor contains the per-element slope for the non-positive
    # branch. Positive elements are copied unchanged.
    return tl.where(self > 0, self, self * noise)


def _check_rrelu_with_noise_args(self, noise, lower, upper, generator):
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
    if generator is not None:
        raise AssertionError("generator is not supported in FlagGems")


def _fill_training_noise(self, noise, lower, upper):
    # The ATen operator writes the sampled slope into the caller-provided
    # noise buffer. FlagGems currently uses the default device RNG only.
    sampled = torch.rand_like(self) * (float(upper) - float(lower)) + float(lower)
    effective_noise = torch.where(self > 0, torch.ones_like(self), sampled)
    noise.copy_(effective_noise)


def _rrelu_with_noise_impl(
    self,
    noise,
    lower=DEFAULT_LOWER,
    upper=DEFAULT_UPPER,
    training=False,
    generator=None,
):
    _check_rrelu_with_noise_args(self, noise, lower, upper, generator)

    if training:
        _fill_training_noise(self, noise, lower, upper)
        slope = noise
    else:
        slope = torch.full_like(self, (float(lower) + float(upper)) * 0.5)

    return _rrelu_with_noise_forward(self, slope)


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
    return _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator
    )


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
    output = _rrelu_with_noise_impl(
        self, noise, lower, upper, training, generator
    )
    self.copy_(output)
    return self


__all__ = ["rrelu_with_noise", "rrelu_with_noise_"]
