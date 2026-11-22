"""Normalization and cross-attention layers used by the fusion network."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _floating_tensor(name: str, value: Tensor, ndim: int | tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if value.ndim not in allowed:
        raise ValueError(f"{name} must have {allowed} dimensions; got {tuple(value.shape)}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}.")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must not contain an empty dimension.")


def _same_storage_type(name: str, value: Tensor, reference: Tensor) -> None:
    if value.device != reference.device:
        raise ValueError(f"{name} is on {value.device}, but expected {reference.device}.")
    if value.dtype != reference.dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, but expected {reference.dtype}.")


def _autocast_enabled(device_type: str) -> bool:
    # The device argument is unavailable in some supported PyTorch versions.
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:
        if device_type == "cpu":
            return torch.is_autocast_cpu_enabled()
        return torch.is_autocast_enabled()


def _matches_parameter(name: str, value: Tensor, parameter: Tensor) -> None:
    if value.device != parameter.device:
        raise ValueError(
            f"{name} is on {value.device}, but the layer parameters are on {parameter.device}."
        )
    if value.dtype != parameter.dtype and not _autocast_enabled(value.device.type):
        raise TypeError(
            f"{name} has dtype {value.dtype}, but the layer parameters use {parameter.dtype}; "
            "move the model and inputs to the same dtype or use autocast."
        )


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned channel-wise weight.

    The normalization calculation is performed in float32 and then converted
    back to the input dtype before applying the weight, as in the research model.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        _positive_int("dim", dim)
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError(f"x must have final dimension {self.weight.numel()}.")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype.")
        if x.device != self.weight.device:
            raise ValueError("x and RMSNorm weights must be on the same device.")
        norm_input = x.float()
        output = norm_input * torch.rsqrt(norm_input.pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class CrossAttentionBlock(nn.Module):
    """Cross-attention followed by a gated feed-forward residual block.

    Queries may be token sequences ``(B, Q, D)`` or image features
    ``(B, D, H, W)``. Keys and values share the sequence ``(B, K, D)``.
    Attention logits are divided by ``sqrt(D)``, not ``sqrt(head_dim)``:
    this nonstandard scaling is part of the original model definition.

    No attention mask is introduced here. Callers must preserve the intended
    token sequence; changing padding or masking changes the model computation.
    Parameter attribute names retain compatibility with the original fusion
    block state dictionaries.
    """

    def __init__(self, *, feature_dim: int = 256, head_dim: int = 32) -> None:
        super().__init__()
        _positive_int("feature_dim", feature_dim)
        _positive_int("head_dim", head_dim)
        if feature_dim % head_dim:
            raise ValueError("feature_dim must be divisible by head_dim.")
        self.head_dim = head_dim
        self.input_dim = feature_dim

        # Keep module creation order stable for seeded initialization.
        self.attention_norm = RMSNorm(self.input_dim)
        self.wq = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.wk = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.wv = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.wo = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.ffn_norm = RMSNorm(self.input_dim)
        self.w1 = nn.Linear(self.input_dim, self.input_dim * 4, bias=False)
        self.w2 = nn.Linear(self.input_dim * 4, self.input_dim, bias=False)
        self.w3 = nn.Linear(self.input_dim, self.input_dim * 4, bias=False)

    def forward(
        self, queryd: Tensor, keyd: Tensor, ifreturn_scores: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return the query-aligned sequence and, optionally, attention scores."""
        _floating_tensor("queryd", queryd, (3, 4))
        _floating_tensor("keyd", keyd, 3)
        _same_storage_type("keyd", keyd, queryd)
        _matches_parameter("queryd", queryd, self.wq.weight)
        if queryd.shape[0] != keyd.shape[0] or keyd.shape[-1] != self.input_dim:
            raise ValueError("Query/key batches and feature dimensions must agree with the layer.")

        if queryd.ndim == 4:
            batch, channels, height, width = queryd.shape
            if channels != self.input_dim:
                raise ValueError(f"queryd must have {self.input_dim} image channels.")
            queryd = queryd.reshape(batch, channels, height * width).transpose(1, 2)
        elif queryd.shape[-1] != self.input_dim:
            raise ValueError(f"queryd must have final dimension {self.input_dim}.")

        batch, query_length, channels = queryd.shape
        key_length = keyd.shape[1]
        heads = self.input_dim // self.head_dim
        queries = self.wq(self.attention_norm(queryd)).reshape(
            batch, query_length, heads, self.head_dim
        ).transpose(1, 2)
        keys = self.wk(keyd).reshape(batch, key_length, heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.input_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        values = self.wv(keyd).reshape(batch, key_length, heads, self.head_dim).transpose(1, 2)
        output = self.wo(
            torch.matmul(scores, values).transpose(1, 2).reshape(batch, query_length, channels)
        )
        output = self.ffn_norm(queryd + output)
        output = output + self.w2(F.silu(self.w1(output)) * self.w3(output))
        return (output, scores) if ifreturn_scores else output
