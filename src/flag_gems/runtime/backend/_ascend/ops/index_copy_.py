import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "num_index_blocks",
    ]
)
def index_copy_ascend_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    num_index_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(axis=0)
    pid_n = tle.program_id(axis=1)

    outer_id = pid_m // num_index_blocks
    index_block = pid_m - outer_id * num_index_blocks

    index_offset = (
        index_block * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    )
    inner_offset = (
        pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    )

    index_mask = index_offset < index_len
    dst_index = tl.load(
        index_ptr + index_offset,
        mask=index_mask,
        other=0,
    )

    # Ascend 当前不支持带 mask 的 tl.device_assert。
    # 这里先保证越界索引不会造成非法写内存。
    valid_index = (dst_index >= 0) & (dst_index < input_dim)
    mask = (
        index_mask
        & (inner_offset < inner_size)
        & valid_index
    )

    src_offset = (
        (outer_id * index_len + index_offset) * inner_size
        + inner_offset
    )
    dst_offset = (
        (outer_id * input_dim + dst_index) * inner_size
        + inner_offset
    )

    value = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(out_ptr + dst_offset, value, mask=mask)

@libentry()
@triton.jit(
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "numel",
    ]
)
def index_copy_ascend_flat_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tle.program_id(axis=0) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    mask = offsets < numel

    offsets_i64 = offsets.to(tl.int64)

    inner_offset = offsets_i64 % inner_size
    src_row = offsets_i64 // inner_size
    index_offset = src_row % index_len
    outer_offset = src_row // index_len

    dst_index = tl.load(
        index_ptr + index_offset,
        mask=mask,
        other=0,
    )

    valid_index = (dst_index >= 0) & (dst_index < input_dim)
    store_mask = mask & valid_index

    value = tl.load(
        src_ptr + offsets_i64,
        mask=mask,
        other=0.0,
    )

    dst_offset = (
        (outer_offset * input_dim + dst_index) * inner_size
        + inner_offset
    )

    tl.store(
        out_ptr + dst_offset,
        value,
        mask=store_mask,
    )


@triton.jit
def index_copy_ascend_large_1d_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    program_start = pid * ELEMENTS_PER_PROGRAM
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)

    for block_start in tl.range(
        0,
        ELEMENTS_PER_PROGRAM,
        BLOCK_SIZE,
    ):
        src_offsets = program_start + block_start + offsets
        src_mask = src_offsets < index_len

        dst_offsets = tl.load(
            index_ptr + src_offsets,
            mask=src_mask,
            other=0,
        ).to(tl.int64)

        valid_index = (
            (dst_offsets >= 0)
            & (dst_offsets < input_dim)
        )
        mask = src_mask & valid_index

        values = tl.load(
            src_ptr + src_offsets,
            mask=src_mask,
        )
        tl.store(
            out_ptr + dst_offsets,
            values,
            mask=mask,
        )

@libentry()
@triton.jit(
    do_not_specialize=[
        "input_dim",
        "index_len",
        "inner_size",
        "row_count",
    ]
)
def index_copy_ascend_row_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    input_dim,
    index_len,
    inner_size,
    row_count,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tle.program_id(axis=0)
    row_id_i64 = row_id.to(tl.int64)

    inner_offsets = (
        tle.program_id(axis=1) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    inner_offsets_i64 = inner_offsets.to(tl.int64)

    row_mask = row_id < row_count
    inner_mask = inner_offsets < inner_size

    index_offset = row_id_i64 % index_len
    outer_offset = row_id_i64 // index_len

    dst_index = tl.load(
        index_ptr + index_offset,
        mask=row_mask,
        other=0,
    )

    valid_index = (
        (dst_index >= 0)
        & (dst_index < input_dim)
    )

    mask = row_mask & inner_mask & valid_index

    src_offsets = (
        row_id_i64 * inner_size
        + inner_offsets_i64
    )

    dst_offsets = (
        (outer_offset * input_dim + dst_index)
        * inner_size
        + inner_offsets
    )

    value = tl.load(
        src_ptr + src_offsets,
        mask=mask,
        other=0.0,
    )

    tl.store(
        out_ptr + dst_offsets,
        value,
        mask=mask,
    )


def _select_tile(inner_size):
    if inner_size == 1:
        return 128, 1
    if inner_size <= 4:
        return 64, 4
    if inner_size <= 64:
        return 16, 64
    return 8, 128
    
def _select_flat_block_size(numel, inner_size):
    if inner_size == 1:
        if numel <= 4096:
            return 256
        return 1024


    return 1024

def _select_row_block_size(inner_size):
    if inner_size <= 128:
        return 128
    if inner_size <= 512:
        return 256
    return 1024

def _launch_index_copy(out, dim, index, src):
    #index_len = index.numel()
    #inner_size = math.prod(out.shape[dim + 1 :])
    #outer_size = math.prod(out.shape[:dim])
    dim = dim % out.ndim

    outer_size = math.prod(out.shape[:dim])
    dim_size = out.shape[dim]
    inner_size = math.prod(out.shape[dim + 1 :])
    index_len = index.numel()

    if index_len == 0 or inner_size == 0:
        return

    input_dim = out.size(dim)
    numel = outer_size * index_len * inner_size

    with torch_device_fn.device(out.device):
        if (
            out.ndim == 1
            and dim == 0
            and inner_size == 1
            and numel >= _LARGE_1D_THRESHOLD
        ):
            block_size = _LARGE_1D_BLOCK_SIZE
            elements_per_program = _LARGE_1D_ELEMENTS_PER_PROGRAM
            grid = (
                triton.cdiv(index_len, elements_per_program),
            )

            index_copy_ascend_large_1d_kernel[grid](
                out,
                src,
                index,
                input_dim,
                index_len,
                BLOCK_SIZE=block_size,
                ELEMENTS_PER_PROGRAM=elements_per_program,
            )
        else:
            row_count = outer_size * index_len
            block_size = 256
            grid = (
                row_count,
                triton.cdiv(inner_size, block_size),
            )

            index_copy_ascend_row_kernel[grid](
                out,
                src,
                index,
                input_dim,
                index_len,
                inner_size,
                row_count,
                BLOCK_SIZE=block_size,
            )
      

def index_copy(inp, dim, index, src):
    logger.debug("GEMS ASCEND INDEX_COPY")
    dim = dim % inp.ndim
    out = inp.clone()
    _launch_index_copy(out, dim, index, src)
    return out


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS ASCEND INDEX_COPY_")
    dim = dim % inp.ndim
    _launch_index_copy(inp, dim, index, src)
    return inp
