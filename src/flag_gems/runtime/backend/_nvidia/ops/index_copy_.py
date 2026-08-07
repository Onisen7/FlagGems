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

import triton
import triton.language as tl

from flag_gems.utils import libentry
import importlib
import logging
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch

from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer

logger = logging.getLogger(__name__)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.writeline("from flag_gems.utils import libentry")

    code.newline()
    code.newline()

    return code
    

def generate_index_copy_kernel(
    rank: int,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
     #the decorators
    code.writeline("@libentry()")
    code.writeline("@triton.jit")

     #signature
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        if rank > 0:
            code.writeline("index,")
            code.writeline("src,")
            code.writeline("out,")
            code.writeline("N,")
            code.writeline("inp_numel: tl.constexpr,")
            code.writeline("inp_stride_dim: tl.constexpr,")
            code.writeline("inp_shape_dim: tl.constexpr,")
            code.writeline("src_shape_dim: tl.constexpr,")
            code.writeline("delta: tl.constexpr,")

            stride_args = ", ".join(f"src_stride_{i}: tl.constexpr" for i in range(rank))
            code.writeline(f"{stride_args}, # stride for src")

            shape_args = ", ".join(f"src_shape_{i}: tl.constexpr" for i in range(rank))
            code.writeline(f"{shape_args}, # shape for src")
            #code.writeline("N,")
            #code.writeline("inp_numel,")
            #code.writeline("inp_stride_dim,")
            #code.writeline("inp_shape_dim,")
            #code.writeline("src_shape_dim,")
            #code.writeline("delta,")

            #stride_args = ", ".join(f"src_stride_{i}: int" for i in range(rank))
            #code.writeline(f"{stride_args}, # stride for src")

            #shape_args = ", ".join(f"src_shape_{i}: int" for i in range(rank))
            #code.writeline(f"{shape_args}, # shape for src")

            code.writeline("BLOCK_SIZE: tl.constexpr,")

        code.writeline("):")

         #Kernel Code
        with code.indent():
            code.writeline("pid = tl.program_id(axis=0)")
            code.writeline("offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            code.writeline("mask = offsets < N")

            for i in range(rank - 1, -1, -1):
                code.writeline(f"src_offset{i} = offsets % src_shape_{i}")
                code.writeline(f"offsets = offsets // src_shape_{i}")
            code.newline()
            comp = [f"src_offset{i} * src_stride_{i}" for i in range(rank)]
            code.writeline(f"src_offset = {' + '.join(comp)}")

            code.writeline("pre_cal = (inp_stride_dim * src_shape_dim)")

             #index copy
            code.writeline("pre_idx = (src_offset // pre_cal).to(tl.int64)")
            code.writeline(
                "dim_idx = (src_offset % pre_cal // inp_stride_dim).to(tl.int64)"
            )
            code.writeline(
                "src_dim_idx = (tl.load(index + dim_idx, mask=mask, other=0)).to(tl.int64)"
            )
            #code.writeline(
            #    'assert src_dim_idx >= 0 and src_dim_idx < inp_shape_dim, "0 <= index < self.size(dim)"'
            #)
            #code.writeline(
            #    "src_dim_idx = tl.load(index + dim_idx, mask=mask, other=0)"
            #)
            code.writeline(
                'tl.device_assert((src_dim_idx >= 0) & (src_dim_idx < inp_shape_dim), '
                '"index value out of bounds: 0 <= index < self.size(dim)")'
            )

            code.writeline(
                "input_idx = (src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim).to(tl.int64)"
            )

            code.writeline("input_mask = (input_idx >= 0) & (input_idx < inp_numel)")
            code.writeline("store_mask = mask & input_mask")
            code.writeline("src_val = tl.load(src + src_offset, mask=mask, other=0)")
            code.writeline("tl.store(out + input_idx, src_val, mask=store_mask)")

        code.newline()
        code.newline()
        return code


def parameter_for_wrapper() -> str:
    # out, index, src, dim, inp_stride_dim, src_shape_dim, delta, N, inp.numel()
    parameters: List[str] = []
    parameters.append("out")
    parameters.append("index")
    parameters.append("src")
    parameters.append("dim")
    parameters.append("inp_stride_dim")
    parameters.append("inp_shape_dim")
    parameters.append("src_shape_dim")
    parameters.append("delta")
    parameters.append("N")
    parameters.append("inp_numel")

    return ", ".join(parameters)


def generate_destination_passing_wrapper(
    rank: int,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    parameters: str = parameter_for_wrapper()
    wrapper_signature: str = f"def {wrapper_name} ({parameters}):"
    code.writeline(wrapper_signature)

    with code.indent():
        code.writeline("src_strides = list(src.stride())")
        code.writeline("src_shapes = list(src.shape)")

        # kernel launch
        code.writeline("if N <= 4096:")
        code.writeline("    BLOCK_SIZE = 64")
        code.writeline("elif N <= 65536:")
        code.writeline("    BLOCK_SIZE = 128")
        code.writeline("elif N <= 524288:")
        code.writeline("    BLOCK_SIZE = 256")
        code.writeline("else:")
        code.writeline("    BLOCK_SIZE = 512")
        code.writeline("grid = (triton.cdiv(N, BLOCK_SIZE),)")
        kernel_launch: str = f"{kernel_name}[grid]("
        code.writeline(kernel_launch)
        with code.indent():
            code.writeline(
                "index, src, out, N, inp_numel, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, "
            )
            if rank > 0:
                s = ", ".join(f"src_strides[{i}]" for i in range(rank))
                code.writeline(f"{s},")

                s = ", ".join(f"src_shapes[{i}]" for i in range(rank))
                code.writeline(f"{s},")
            code.writeline("BLOCK_SIZE=BLOCK_SIZE")
        code.writeline(")")
        code.writeline("return out")

    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
) -> IndentedBuffer:
    # inputs: [out, index, src, dim, inp_stride_dim, inp_shape_dim, src_shape_dim, delta, N, inp.numel()]
    shape = inputs[2].shape
    rank = len(shape)

    code = generate_imports(code)
    code = generate_index_copy_kernel(rank, kernel_name, code)
    code = generate_destination_passing_wrapper(rank, wrapper_name, kernel_name, code)
    return code


class IndexCopyFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        key = f"{self.arg_key(*args)}"
        if key in self.overloads:
            return self.overloads[key](*args, **kwargs)

        code = IndentedBuffer()
        code = generate_code(
            args,
            "_index_copy_wrapper",
            "_index_copy_jit_function",
            code,
        )

        file_name = f"index_copy_rank_{key}_pid_{self.pid}.py"

        try:
            with open(code_cache_dir() / file_name, "wt", encoding="utf-8") as f:
                f.write(code.getvalue())

            # load
            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}_pid_{self.pid}",
                f.name,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_copy_wrapper")
            self.overloads[key] = overload
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate or load index_copy kernel: {e}"
            ) from e

        return overload(*args, **kwargs)

    def arg_key(self, *args):
    # args 顺序固定：out, index, src, dim, inp_stride_dim, ...
        out = args[0]
        src = args[2]
        dim = args[3]
    # 使用 src 的完整形状和步幅，以及 dim、inp 在 dim 上的大小和步幅，确保唯一性
        key = f"{src.shape}_{src.stride()}_{dim}_{out.shape[dim]}_{out.stride(dim)}"
        return key
    #def arg_key(self, *args):
        #tensors = [item for item in args if torch.is_tensor(item)]
        #max_rank = max(item.ndim for item in tensors)
        #return max_rank


_index_copy_func = IndexCopyFunction()


_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

@libentry()
@triton.jit
def _index_copy_clone_kernel(
    inp,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, value, mask=mask)


def _clone_without_copy_dispatch(inp):
    # 扁平 kernel 只处理标准连续张量；其他布局保留原生语义。
    if not inp.is_contiguous():
        return torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)

    out = torch.empty_like(inp)
    n_elements = inp.numel()
    if n_elements == 0:
        return out

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _index_copy_clone_kernel[grid](
        inp,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return out

def index_copy(inp, dim, index, src):
    logger.debug("GEMS INDEX_COPY")
    #assert ((0 <= index) * (index < inp.size(dim))).equal(
    #    torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    #), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"

    # Use native clone to avoid potential issues with FlagGems copy_ dispatch
    #out = torch.ops.aten.clone.default.redispatch(_FALLBACK_KEYSET, inp)
    out = _clone_without_copy_dispatch(inp)

    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_copy_func(
        out,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
    )
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS INDEX_COPY_")
    #assert ((0 <= index) * (index < inp.size(dim))).equal(
    #    torch.ones(tuple(index.shape), dtype=torch.bool, device=inp.device)
    #), "0 <= index < self.size(dim)"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.numel() == src.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert all(
        (inp.size(i) == src.size(i)) or i == dim for i in range(0, inp.ndim)
    ), "src.size(d) == self.size(d) for all dimensions d != dim"


    dim %= inp.ndim
    inp_stride_dim = inp.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = inp.size(dim)
    delta = inp.size(dim) - src_shape_dim
    N = src.numel()

    _index_copy_func(
        inp,
        index,
        src,
        dim,
        inp_stride_dim,
        inp_shape_dim,
        src_shape_dim,
        delta,
        N,
        inp.numel(),
    )
    return inp
