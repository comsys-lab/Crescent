"""Shared NVFP4/FP6 fake-quantization math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

NVFP4_BLOCK_SIZE = 16


@dataclass(frozen=True)
class FPFormat:
    key: str
    exp_bits: int
    mant_bits: int
    bias: int
    finite_exp_max: int

    @property
    def max_power_exp(self) -> int:
        return self.finite_exp_max - self.bias


FP4_E2M1 = FPFormat("fp4_e2m1", 2, 1, 1, 3)
FP6_E2M3 = FPFormat("fp6_e2m3", 2, 3, 1, 3)
_LEVEL_CACHE: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def build_positive_fp_levels_cpu(fmt: FPFormat) -> Tuple[torch.Tensor, torch.Tensor]:
    values, lsb_even = [], []
    max_mant = 2 ** fmt.mant_bits - 1
    mant_scale = 2.0 ** (-fmt.mant_bits)
    for m in range(max_mant + 1):
        values.append((2.0 ** (1 - fmt.bias)) * (m * mant_scale)); lsb_even.append((m & 1) == 0)
    for e in range(1, fmt.finite_exp_max + 1):
        for m in range(max_mant + 1):
            enc = (e << fmt.mant_bits) | m
            values.append((2.0 ** (e - fmt.bias)) * (1.0 + m * mant_scale)); lsb_even.append((enc & 1) == 0)
    pairs = sorted(zip(values, lsb_even), key=lambda x: x[0])
    return torch.tensor([p[0] for p in pairs], dtype=torch.float32), torch.tensor([p[1] for p in pairs], dtype=torch.bool)


def get_positive_fp_levels(fmt: FPFormat, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (fmt.key, str(device))
    if key not in _LEVEL_CACHE:
        lv, ev = build_positive_fp_levels_cpu(fmt)
        _LEVEL_CACHE[key] = (lv.to(device), ev.to(device))
    return _LEVEL_CACHE[key]


def quantize_to_fp_levels(x: torch.Tensor, levels: torch.Tensor, lsb_even: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    abs_x = torch.nan_to_num(x.abs(), nan=0.0, posinf=float(levels[-1].item()), neginf=0.0)
    flat = abs_x.reshape(-1)
    hi = torch.clamp(torch.searchsorted(levels, flat, right=False), 0, levels.numel() - 1)
    lo = torch.clamp(hi - 1, 0, levels.numel() - 1)
    lo_v, hi_v = levels[lo], levels[hi]
    dist_lo, dist_hi = torch.abs(flat - lo_v), torch.abs(hi_v - flat)
    choose_hi = (dist_hi < dist_lo) | ((dist_hi == dist_lo) & lsb_even[hi])
    return torch.where(choose_hi, hi_v, lo_v).reshape_as(abs_x) * sign


def fake_fp8_e4m3_scale(t: torch.Tensor) -> torch.Tensor:
    x = t.to(torch.float32)
    max_finite = float(torch.finfo(torch.float8_e4m3fn).max)
    x2 = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=max_finite, neginf=-max_finite), -max_finite, max_finite)
    y = x2.to(torch.float8_e4m3fn).to(torch.float32)
    min_positive = 2.0 ** -9
    return torch.where((y == 0) & (x2 != 0), torch.sign(x2) * min_positive, y)


def block_last_dimension(tensor: torch.Tensor, block_size: int, quantize_blocks) -> torch.Tensor:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return tensor
    orig_dtype, orig_shape = tensor.dtype, tensor.shape
    last_dim = orig_shape[-1]
    work_2d = tensor.reshape(-1, last_dim)
    rows = work_2d.shape[0]
    pad_len = (block_size - last_dim % block_size) % block_size
    padded = last_dim + pad_len
    if pad_len:
        work = torch.zeros((rows, padded), device=tensor.device, dtype=orig_dtype)
        work[:, :last_dim].copy_(work_2d)
    else:
        work = work_2d
    blocks = work.contiguous().view(-1, block_size).to(torch.float32)
    out = quantize_blocks(blocks).view(rows, padded)
    if pad_len:
        out = out[:, :last_dim]
    return out.reshape(orig_shape).to(orig_dtype)


def quantize_nvfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / fp4_max
    scale = fake_fp8_e4m3_scale(ideal_scale)
    scale = torch.where(max_abs > 0, torch.clamp(scale, min=2.0 ** -24), torch.ones_like(scale))
    return quantize_to_fp_levels(blocks / scale, levels, lsb_even) * scale


def nvfp4_block_scale(blocks: torch.Tensor) -> torch.Tensor:
    """The per-block NVFP4 scale (FP8 E4M3, based on FP4 max) alone, with no
    quantization applied -- the same scale quantize_nvfp4_blocks and
    quantize_nvfp4_scale_fp6_blocks use internally."""
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    fp4_levels, _ = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = fp4_levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / fp4_max
    scale = fake_fp8_e4m3_scale(ideal_scale)
    return torch.where(max_abs > 0, torch.clamp(scale, min=2.0 ** -24), torch.ones_like(scale))


def quantize_nvfp4_scale_fp6_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """NVFP4 block scale (FP8 E4M3, based on FP4 max), FP6 E2M3 quantization levels."""
    scale = nvfp4_block_scale(blocks)
    fp6_levels, fp6_lsb = get_positive_fp_levels(FP6_E2M3, blocks.device)
    return quantize_to_fp_levels(blocks / scale, fp6_levels, fp6_lsb) * scale


def block_scale_2d(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Same block layout as block_last_dimension, but returns one NVFP4 scale
    value per block (shape [rows, num_blocks]) instead of quantized elements."""
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    work_2d = tensor.reshape(-1, last_dim).to(torch.float32)
    rows = work_2d.shape[0]
    pad_len = (block_size - last_dim % block_size) % block_size
    padded = last_dim + pad_len
    if pad_len:
        work = torch.zeros((rows, padded), device=tensor.device, dtype=torch.float32)
        work[:, :last_dim].copy_(work_2d)
    else:
        work = work_2d
    blocks = work.contiguous().view(-1, block_size)
    scale = nvfp4_block_scale(blocks).view(rows, padded // block_size)
    return scale.contiguous()


@torch.no_grad()
def quantize_tensor(tensor: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> torch.Tensor:
    return block_last_dimension(tensor, block_size, quantize_nvfp4_blocks)


@torch.no_grad()
def quantize_tensor_fp6_same_scale(tensor: torch.Tensor, block_size: int = NVFP4_BLOCK_SIZE) -> torch.Tensor:
    """FP6 E2M3 quantization using the same block scale as the FP4 baseline."""
    return block_last_dimension(tensor, block_size, quantize_nvfp4_scale_fp6_blocks)
