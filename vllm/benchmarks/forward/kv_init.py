# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fill the KV blocks a benchmark case uses.

KV contents can change performance: sparse attention picks which blocks to
read from them, and quantized caches need valid values. Only the case's blocks
are written, group by group: groups can alias one allocation with different
layouts, and the batch builder keeps their block IDs disjoint.

Floating-point caches get Gaussian values. Packed (integer) caches need their
layer to define `init_kv_cache_random(kv_cache, block_ranges, generator)`,
which writes valid records into the given blocks with the model's own ops.
"""

from collections.abc import Iterator
from typing import Literal

import torch

from vllm.benchmarks.forward.batch import BenchBatch
from vllm.model_executor.layers.mamba.abstract import MambaBase

# Fill in chunks to bound temporaries.
_CHUNK_NUMEL = 1 << 26


def _tensors(kv_cache) -> Iterator[torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.numel():
            yield kv_cache
    elif isinstance(kv_cache, (list, tuple)):
        for item in kv_cache:
            yield from _tensors(item)


def _block_dim(tensor: torch.Tensor, num_blocks: int) -> tuple[int, int]:
    """Return (block dim, kernel blocks per allocator block)."""
    for dim in range(min(2, tensor.dim())):
        if tensor.shape[dim] >= num_blocks and tensor.shape[dim] % num_blocks == 0:
            return dim, tensor.shape[dim] // num_blocks
    raise ValueError(f"No block dim of size {num_blocks} in {tuple(tensor.shape)}")


def _fill_random(tensor: torch.Tensor, std: float, generator: torch.Generator) -> None:
    if tensor.numel() > _CHUNK_NUMEL and tensor.shape[0] > 1:
        step = max(1, tensor.shape[0] * _CHUNK_NUMEL // tensor.numel())
        for start in range(0, tensor.shape[0], step):
            _fill_random(tensor[start : start + step], std, generator)
        return
    values = torch.randn(tensor.shape, generator=generator, device=tensor.device)
    tensor.copy_(values.mul_(std))


@torch.inference_mode()
def init_kv_blocks(
    static_forward_context: dict,
    kv_cache_groups,
    batch: BenchBatch,
    num_blocks: int,
    mode: Literal["random", "zeros"],
    generator: torch.Generator,
) -> None:
    for group, ranges in zip(kv_cache_groups, batch.group_block_ranges):
        for name in group.layer_names:
            layer = static_forward_context[name]
            for tensor in _tensors(getattr(layer, "kv_cache", None)):
                dim, factor = _block_dim(tensor, num_blocks)
                kernel_ranges = [(b * factor, e * factor) for b, e in ranges]
                for begin, end in kernel_ranges:
                    tensor.narrow(dim, begin, end - begin).zero_()
                if mode == "zeros":
                    continue
                if not tensor.dtype.is_floating_point:
                    if not hasattr(layer, "init_kv_cache_random"):
                        raise ValueError(
                            f"{name}: random init of a {tensor.dtype} KV cache "
                            "needs init_kv_cache_random. Use --kv-init zeros."
                        )
                    assert dim == 0, (name, tuple(tensor.shape))
                    layer.init_kv_cache_random(tensor, kernel_ranges, generator)
                    continue
                # Recurrent state accumulates over tokens: keep it small.
                std = 0.1 if isinstance(layer, MambaBase) else 1.0
                for begin, end in kernel_ranges:
                    _fill_random(tensor.narrow(dim, begin, end - begin), std, generator)


def touched_kv_views(
    static_forward_context: dict,
    kv_cache_groups,
    batch: BenchBatch,
    num_blocks: int,
) -> list[torch.Tensor]:
    """Views of the KV blocks a step may write: each request's last
    ceil(scheduled / block_size) + 2 blocks, in every group."""
    scheduled_tokens = [batch.expected[req_id][0] for req_id in batch.req_ids]
    views = []
    for group, ranges in zip(kv_cache_groups, batch.group_block_ranges):
        spec = getattr(group.kv_cache_spec, "first_spec", group.kv_cache_spec)
        for name in group.layer_names:
            layer = static_forward_context[name]
            for tensor in _tensors(getattr(layer, "kv_cache", None)):
                dim, factor = _block_dim(tensor, num_blocks)
                for (begin, end), tokens in zip(ranges, scheduled_tokens):
                    first = max(begin, end - (-(-tokens // spec.block_size) + 2))
                    views.append(
                        tensor.narrow(dim, first * factor, (end - first) * factor)
                    )
    return views
