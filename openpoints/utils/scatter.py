from __future__ import annotations

import torch

try:  # pragma: no cover - prefer the optimized extension when available.
    from torch_scatter import scatter as _torch_scatter
    from torch_scatter import scatter_softmax as _torch_scatter_softmax
except Exception:  # pragma: no cover - Colab may not have torch_scatter preinstalled.
    _torch_scatter = None
    _torch_scatter_softmax = None


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dim={dim} is out of range for tensor with {ndim} dimensions")
    return dim


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()

    if index.dim() == 0:
        index = index.reshape(1)

    if index.dim() == 1:
        view_shape = [1] * src.dim()
        view_shape[dim] = index.shape[0]
        index = index.view(*view_shape)

    while index.dim() < src.dim():
        index = index.unsqueeze(-1)

    expand_shape = []
    for idx_size, src_size in zip(index.shape, src.shape):
        if idx_size == src_size or idx_size == 1:
            expand_shape.append(src_size)
            continue
        raise ValueError(
            f"Index shape {tuple(index.shape)} is not broadcastable to source shape {tuple(src.shape)}"
        )

    return index.expand(*expand_shape)


def _infer_dim_size(index: torch.Tensor, dim_size: int | None) -> int:
    if dim_size is not None:
        return dim_size
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def _output_shape(src: torch.Tensor, dim: int, dim_size: int) -> list[int]:
    shape = list(src.shape)
    shape[dim] = dim_size
    return shape


def _prepare_output(
    src: torch.Tensor,
    dim: int,
    dim_size: int,
    out: torch.Tensor | None,
    fill_value: int | float = 0,
) -> torch.Tensor:
    shape = _output_shape(src, dim, dim_size)
    if out is None:
        return torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
    out.resize_(shape)
    out.fill_(fill_value)
    return out


def _empty_group_fill_value(src: torch.Tensor, reduce: str) -> int | float | bool:
    if src.dtype.is_floating_point:
        return float("-inf") if reduce == "max" else float("inf")
    if src.dtype == torch.bool:
        return False if reduce == "max" else True
    info = torch.iinfo(src.dtype)
    return info.min if reduce == "max" else info.max


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if _torch_scatter is not None:
        return _torch_scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)

    reduce = {"add": "sum", "mul": "prod"}.get(reduce, reduce)
    if reduce not in {"sum", "mean", "max", "min"}:
        raise NotImplementedError(f"Fallback scatter does not support reduce={reduce!r}")

    dim = _normalize_dim(dim, src.dim())
    expanded_index = _expand_index(index, src, dim)
    dim_size = _infer_dim_size(expanded_index, dim_size)

    if reduce == "sum":
        output = _prepare_output(src, dim, dim_size, out, fill_value=0)
        output.scatter_add_(dim, expanded_index, src)
        return output

    if reduce == "mean":
        output = _prepare_output(src, dim, dim_size, out, fill_value=0)
        output.scatter_add_(dim, expanded_index, src)
        counts = torch.zeros_like(output)
        counts.scatter_add_(dim, expanded_index, torch.ones_like(src))
        return output / counts.clamp_min(1)

    fill_value = _empty_group_fill_value(src, reduce)
    output = _prepare_output(src, dim, dim_size, out, fill_value=fill_value)
    counts = torch.zeros(_output_shape(src, dim, dim_size), dtype=torch.long, device=src.device)
    counts.scatter_add_(dim, expanded_index, torch.ones_like(expanded_index, dtype=torch.long))

    if hasattr(output, "scatter_reduce_"):
        torch_reduce = "amax" if reduce == "max" else "amin"
        output.scatter_reduce_(dim, expanded_index, src, reduce=torch_reduce, include_self=True)
    else:  # pragma: no cover - modern Colab/PyTorch should use scatter_reduce_.
        perm = list(range(src.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        src_perm = src.permute(*perm).reshape(-1, src.shape[dim])
        idx_perm = expanded_index.permute(*perm).reshape(-1, expanded_index.shape[dim])
        out_perm = output.permute(*perm).reshape(-1, output.shape[dim])
        for row_idx in range(src_perm.shape[0]):
            values = src_perm[row_idx]
            targets = idx_perm[row_idx]
            for value_idx in range(values.shape[0]):
                target = targets[value_idx].item()
                current = out_perm[row_idx, target]
                candidate = values[value_idx]
                out_perm[row_idx, target] = torch.maximum(current, candidate) if reduce == "max" else torch.minimum(current, candidate)

    return output.masked_fill(counts == 0, 0)


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    if _torch_scatter_softmax is not None:
        return _torch_scatter_softmax(src=src, index=index, dim=dim, dim_size=dim_size)

    dim = _normalize_dim(dim, src.dim())
    expanded_index = _expand_index(index, src, dim)
    dim_size = _infer_dim_size(expanded_index, dim_size)

    grouped_max = scatter(src, expanded_index, dim=dim, dim_size=dim_size, reduce="max")
    centered = src - torch.gather(grouped_max, dim, expanded_index)
    exp_src = centered.exp()
    grouped_sum = scatter(exp_src, expanded_index, dim=dim, dim_size=dim_size, reduce="sum")
    normalizer = torch.gather(grouped_sum, dim, expanded_index).clamp_min(eps)
    return exp_src / normalizer
