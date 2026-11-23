#!/usr/bin/env python3
from __future__ import annotations

"""Rotation vs no-rotation, weight+GPTQ, boost method (Llama).

No product-aware correction, no CUDA extension — just two GPTQ weight-quantization
pipelines compared head to head:
  NVFP4_ONLY          : plain block GPTQ (no rotation), QK quantized as-is.
                        Reuses ptq_cache/no_rotation/.
  NVFP4_ROTATION_ONLY : group-wise Hadamard rotation of linear activations/weights
                        (Rotated GPTQ rotated-Hessian PTQ) + QK Hadamard rotation, then
                        NVFP4 quantize. Reuses ptq_cache/mrgptq/.

Both caches already exist on disk, so this script only recomputes GPTQ if a cache
is missing. Runs on CPU or GPU (set DEVICE_OVERRIDE=cpu to force CPU when the GPUs
are occupied by other experiments).
"""

import csv
import gc
import inspect
import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_REVISION = "rotation-vs-none-weight-gptq-boost-llama-v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESULT_CSV = os.environ.get("RESULT_CSV", "llama_rotation_weight_gptq_results.csv")

SEED = int(os.environ.get("SEED", "1111"))
PPL_SEQ_LEN = int(os.environ.get("PPL_SEQ_LEN", "2048"))
PPL_STRIDE = int(os.environ.get("PPL_STRIDE", "512"))
PPL_MAX_STEPS = int(os.environ.get("PPL_MAX_STEPS", "64"))
PPL_DECIMAL_PLACES = 15

TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") != "0"
ROUNDING_MODE = os.environ.get("ROUNDING_MODE", "determ")
ATTENTION_QUERY_CHUNK = int(os.environ.get("ATTENTION_QUERY_CHUNK", "128"))

_device_override = os.environ.get("DEVICE_OVERRIDE", "").strip().lower()
if _device_override in ("cpu", "cuda"):
    DEVICE = torch.device(_device_override)
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SKIP_LM_HEAD = True
EXPECTED_LINEAR_LAYERS = 224
EXPECTED_ATTENTION_LAYERS = 32

EPSILON = float(os.environ.get("EPSILON", "1e-8"))
NVFP4_BLOCK_SIZE = 16
MXFP4_BLOCK_SIZE = 32

# Rotated GPTQ paper (Appendix K sweep): Hadamard rotation matching the quantization
# group size gives the best results on average; for NVFP4 that group size is 16.
# (QK rotation size is our own extension, not covered by the paper; matched to 16
# for the same reason.) Previous default of 1024/128 did not match this and is why
# the mrgptq cache dir below is versioned separately.
LINEAR_ROT_GROUP_SIZE = int(os.environ.get("LINEAR_ROT_GROUP_SIZE", "16"))
QK_GROUP_SIZE = int(os.environ.get("QK_GROUP_SIZE", "16"))

GPTQ_CALIB_SEQ_LEN = int(os.environ.get("GPTQ_CALIB_SEQ_LEN", "512"))
GPTQ_CALIB_STRIDE = int(os.environ.get("GPTQ_CALIB_STRIDE", "512"))
GPTQ_CALIB_STEPS = int(os.environ.get("GPTQ_CALIB_STEPS", "8"))
GPTQ_DAMP_PERCENT = float(os.environ.get("GPTQ_DAMP_PERCENT", "0.01"))
GPTQ_CPU = os.environ.get("GPTQ_CPU", "1") != "0"

NO_ROTATION_CACHE_DIR = os.environ.get(
    "NO_ROTATION_CACHE_DIR",
    os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "no_rotation"),
)
MRGPTQ_CACHE_DIR = os.environ.get(
    "MRGPTQ_CACHE_DIR",
    os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "mrgptq_g16"),
)

CSV_FIELDS = [
    "script_revision", "model", "experiment", "name", "format", "rotation", "qk_quantize",
    "ppl", "status", "elapsed_sec", "ptq_elapsed_sec", "block_size",
    "linear_rot_group_size", "qk_rot_group_size",
    "ppl_seq_len", "ppl_stride", "ppl_max_steps",
    "gptq_calib_seq_len", "gptq_calib_stride", "gptq_calib_steps", "gptq_damp_percent",
    "rounding_mode", "skip_lm_head", "device", "error",
]


def format_ppl(ppl: float) -> str:
    return f"{float(ppl):.{PPL_DECIMAL_PLACES}f}"


# =============================================================================
# Utilities
# =============================================================================

def same_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_model_name(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").replace(".", "_").replace(" ", "_")


def short_error(exc: BaseException) -> str:
    return repr(exc).replace("\n", "\\n")[:4000]


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_group_size(last_dim: int, group_size: int, label: str) -> None:
    if not is_power_of_two(group_size):
        raise ValueError(f"{label}: group_size must be power-of-two, got {group_size}")
    if last_dim % group_size != 0:
        raise ValueError(f"{label}: last_dim={last_dim} not divisible by group_size={group_size}")


def load_wikitext2_text(split: str) -> str:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(row["text"] for row in dataset if row.get("text") and row["text"].strip())


def load_tokenizer_and_encodings(full_text: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer(full_text, return_tensors="pt")


def load_model() -> nn.Module:
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.to(DEVICE)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


@torch.no_grad()
def measure_ppl(model: nn.Module, encodings, desc: str) -> float:
    input_ids_cpu = encodings.input_ids
    total_len = input_ids_cpu.size(1)
    nll_sum, nll_tokens, steps, prev_end_loc = 0.0, 0, 0, 0
    for begin_loc in range(0, total_len, PPL_STRIDE):
        end_loc = min(begin_loc + PPL_SEQ_LEN, total_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break
        input_ids = input_ids_cpu[:, begin_loc:end_loc].to(DEVICE)
        labels = input_ids.clone()
        labels[:, :-trg_len] = -100
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss.float()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        loss_tokens = int((labels[:, 1:] != -100).sum().item())
        nll_sum += float(loss.item()) * loss_tokens
        nll_tokens += loss_tokens
        steps += 1
        prev_end_loc = end_loc
        if steps == 1 or steps % 8 == 0:
            print(f"[{desc}] step={steps} running_ppl={math.exp(nll_sum / nll_tokens):.6f}")
        del input_ids, labels, outputs, loss
        if PPL_MAX_STEPS > 0 and steps >= PPL_MAX_STEPS:
            break
        if end_loc >= total_len:
            break
    if nll_tokens == 0:
        raise RuntimeError("No PPL tokens evaluated")
    return float(math.exp(nll_sum / nll_tokens))


# =============================================================================
# FP4 fake quantization
# =============================================================================

@dataclass(frozen=True)
class FPFormat:
    key: str
    display_name: str
    exp_bits: int
    mant_bits: int
    bias: int
    finite_exp_max: int

    @property
    def max_power_exp(self) -> int:
        return self.finite_exp_max - self.bias


FP4_E2M1 = FPFormat("fp4_e2m1", "FP4_E2M1", 2, 1, 1, 3)
_LEVEL_CACHE: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def build_positive_fp_levels_cpu(fmt: FPFormat) -> Tuple[torch.Tensor, torch.Tensor]:
    values: List[float] = []
    lsb_even: List[bool] = []
    max_mant = 2 ** fmt.mant_bits - 1
    mant_scale = 2.0 ** (-fmt.mant_bits)
    for m in range(max_mant + 1):
        values.append((2.0 ** (1 - fmt.bias)) * (m * mant_scale))
        lsb_even.append((m & 1) == 0)
    for e in range(1, fmt.finite_exp_max + 1):
        for m in range(max_mant + 1):
            enc = (e << fmt.mant_bits) | m
            values.append((2.0 ** (e - fmt.bias)) * (1.0 + m * mant_scale))
            lsb_even.append((enc & 1) == 0)
    pairs = sorted(zip(values, lsb_even), key=lambda p: p[0])
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.float32),
        torch.tensor([p[1] for p in pairs], dtype=torch.bool),
    )


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
    if ROUNDING_MODE in ("determ", "nearest_even", "roundTiesToEven"):
        choose_hi = (hi_v - flat) < (flat - lo_v)
        tie = (hi_v - flat) == (flat - lo_v)
        choose_hi = choose_hi | (tie & lsb_even[hi])
    elif ROUNDING_MODE in ("stoc", "stochastic"):
        denom = torch.clamp(hi_v - lo_v, min=1e-30)
        choose_hi = torch.rand_like(flat) < torch.clamp((flat - lo_v) / denom, 0.0, 1.0)
    else:
        raise NotImplementedError(f"Unsupported rounding mode: {ROUNDING_MODE}")
    return (torch.where(choose_hi, hi_v, lo_v).reshape_as(abs_x)) * sign


def fake_fp8_e4m3_scale(t: torch.Tensor) -> torch.Tensor:
    x = t.to(torch.float32)
    max_f = float(torch.finfo(torch.float8_e4m3fn).max)
    x2 = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=max_f, neginf=-max_f), -max_f, max_f)
    y = x2.to(torch.float8_e4m3fn).to(torch.float32)
    # FP8 E4M3 underflows small nonzero magnitudes to exactly 0; floor those up to
    # the smallest representable positive subnormal instead of collapsing to zero,
    # so the NVFP4 block scale never becomes degenerate (was: scale -> 0 -> clamped
    # to 2**-24, blowing up block/scale and saturating the whole block to garbage).
    min_positive = 2.0 ** -9
    return torch.where((y == 0) & (x2 != 0), torch.sign(x2) * min_positive, y)


def format_block_size(format_name: str) -> int:
    if format_name == "NVFP4":
        return NVFP4_BLOCK_SIZE
    if format_name == "MXFP4":
        return MXFP4_BLOCK_SIZE
    raise ValueError(f"Unknown format: {format_name}")


UE8M0_MIN_EXP, UE8M0_MAX_EXP = -127, 127


def round_scale_ue8m0(scale: torch.Tensor) -> torch.Tensor:
    """MXFP4's own scale format (OCP Microscaling spec): pure power-of-two,
    no mantissa bits, rounded UP (ceiling in log2 space) so the block's own
    max element never overflows -- MXFP4 has no tensor-level correction to
    fall back on if it rounded down (unlike NVFP4's two-level scheme)."""
    scale = torch.clamp(scale, min=2.0 ** UE8M0_MIN_EXP)
    exp_up = torch.ceil(torch.log2(scale))
    exp_up = torch.clamp(exp_up, UE8M0_MIN_EXP, UE8M0_MAX_EXP)
    return torch.pow(2.0, exp_up)


def block_last_dimension(tensor: torch.Tensor, block_size: int, block_fn) -> torch.Tensor:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return tensor
    orig_dtype = tensor.dtype
    was_1d = tensor.dim() == 1
    work = tensor.unsqueeze(0) if was_1d else tensor
    orig_shape = list(work.shape)
    pad = (-orig_shape[-1]) % block_size
    if pad:
        work = F.pad(work, (0, pad), "constant", 0.0)
    blocks = work.contiguous().view(-1, block_size).to(torch.float32)
    out = block_fn(blocks).view(list(work.shape))
    out = out.narrow(-1, 0, orig_shape[-1])
    if was_1d:
        out = out.squeeze(0)
    return out.to(orig_dtype)


def quantize_nvfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / fp4_max
    scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(ideal_scale), min=2.0 ** -24), torch.ones_like(max_abs))
    return quantize_to_fp_levels(blocks / scale, levels, lsb_even) * scale


def quantize_mxfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """MXFP4: same E2M1 levels as NVFP4, but block_size=32 and a single-level
    UE8M0 (pure power-of-two, round-up) scale -- no tensor-level correction."""
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / fp4_max
    scale = torch.where(max_abs > 0, round_scale_ue8m0(ideal_scale), torch.ones_like(max_abs))
    return quantize_to_fp_levels(blocks / scale, levels, lsb_even) * scale


def quantize_blocks_for_format(format_name: str, blocks: torch.Tensor) -> torch.Tensor:
    if format_name == "MXFP4":
        return quantize_mxfp4_blocks(blocks)
    return quantize_nvfp4_blocks(blocks)


@torch.no_grad()
def quantize_tensor(tensor: torch.Tensor, format_name: str) -> torch.Tensor:
    block_size = format_block_size(format_name)
    return block_last_dimension(tensor, block_size, lambda blocks: quantize_blocks_for_format(format_name, blocks))


# =============================================================================
# This project's boost method (hypothesis_v4 H3-H5): base E2M1 everywhere in
# a block except the top-k positions with the largest base-quantization
# error, which get requantized at E2M7 (wide-mantissa) grade. Activation-only
# -- the weight side keeps whatever GPTQ scheme produced the cache (plain
# hessiandiag_sT or Rotated GPTQ), unaffected by this. Ported from hypothesis_v4/
# common/formats.py's quantize_tensor_topk_activation_e2m7 to reuse this
# file's own FP format machinery so no cross-project import is needed.
# =============================================================================

FP4_E2M7 = FPFormat("fp4_e2m7", "FP4_E2M7", 2, 7, 1, 3)  # same exponent range as FP4_E2M1, wider mantissa
BOOST_TOPK_K = {"NVFP4": int(os.environ.get("BOOST_TOPK_K", "2")), "MXFP4": int(os.environ.get("BOOST_TOPK_K_MXFP4", "4"))}  # matches H3/H5's k per format (block_size 16 vs 32)


def quantize_e2m7_blocks(blocks: torch.Tensor) -> torch.Tensor:
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M7, blocks.device)
    e2m7_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / e2m7_max
    scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(ideal_scale), min=2.0 ** -24), torch.ones_like(max_abs))
    return quantize_to_fp_levels(blocks / scale, levels, lsb_even) * scale


def quantize_e2m7_mxfp4_blocks(blocks: torch.Tensor) -> torch.Tensor:
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M7, blocks.device)
    e2m7_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / e2m7_max
    scale = torch.where(max_abs > 0, round_scale_ue8m0(ideal_scale), torch.ones_like(max_abs))
    return quantize_to_fp_levels(blocks / scale, levels, lsb_even) * scale


def e2m7_blocks_for_format(format_name: str, blocks: torch.Tensor) -> torch.Tensor:
    if format_name == "MXFP4":
        return quantize_e2m7_mxfp4_blocks(blocks)
    return quantize_e2m7_blocks(blocks)


@torch.no_grad()
def quantize_tensor_topk_boost(tensor: torch.Tensor, format_name: str, k: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return tensor
    if k is None:
        k = BOOST_TOPK_K[format_name]
    block_size = format_block_size(format_name)
    orig_dtype = tensor.dtype
    was_1d = tensor.dim() == 1
    work = tensor.unsqueeze(0) if was_1d else tensor
    orig_shape = list(work.shape)
    last_dim = orig_shape[-1]
    pad = (-last_dim) % block_size
    if pad:
        work = F.pad(work, (0, pad), "constant", 0.0)
    n_blocks = work.shape[-1] // block_size
    flat = work.contiguous().view(-1, n_blocks, block_size).to(torch.float32)

    blocks2d = flat.reshape(-1, block_size)
    x_base = quantize_blocks_for_format(format_name, blocks2d).view(-1, n_blocks, block_size)
    x_boost = e2m7_blocks_for_format(format_name, blocks2d).view(-1, n_blocks, block_size)

    diff = (flat - x_base).abs()
    kk = min(k, block_size)
    topk_idx = diff.argsort(dim=-1, descending=True)[..., :kk]
    result = x_base.scatter(-1, topk_idx, x_boost.gather(-1, topk_idx))

    out = result.view(list(work.shape)).narrow(-1, 0, last_dim)
    if was_1d:
        out = out.squeeze(0)
    return out.to(orig_dtype)


# =============================================================================
# Hadamard rotation
# =============================================================================

def hadamard_transform_last_dim(x: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size <= 1:
        return x
    validate_group_size(x.shape[-1], group_size, "Hadamard")
    orig_dtype = x.dtype
    orig_shape = tuple(x.shape)
    n_groups = x.shape[-1] // group_size
    y = x.to(torch.float32).contiguous().reshape(-1, n_groups, group_size)
    h = 1
    while h < group_size:
        y = y.reshape(-1, n_groups, group_size // (2 * h), 2, h)
        a, b = y[:, :, :, 0, :], y[:, :, :, 1, :]
        y = torch.cat((a + b, a - b), dim=-1).reshape(-1, n_groups, group_size)
        h *= 2
    return (y.reshape(orig_shape) / math.sqrt(group_size)).to(orig_dtype)


def rotate_weight_columns(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    return hadamard_transform_last_dim(weight, group_size)


# =============================================================================
# GPTQ: non-rotated (NVFP4_ONLY) and rotated (NVFP4_ROTATION_ONLY)
# =============================================================================

def should_skip_linear(name: str) -> bool:
    return SKIP_LM_HEAD and (name == "lm_head" or name.endswith(".lm_head"))


def should_quantize_module(name: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) and not should_skip_linear(name)


def iter_target_linear_names(model: nn.Module) -> List[str]:
    return [n for n, m in model.named_modules() if should_quantize_module(n, m)]


@torch.no_grad()
def collect_hessian_plain(model: nn.Module, module: nn.Linear, encodings) -> torch.Tensor:
    h_cpu: Optional[torch.Tensor] = None
    count = 0

    def hook(_mod, inputs) -> None:
        nonlocal h_cpu, count
        x = inputs[0].detach()
        if not torch.is_floating_point(x):
            return
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        h = x2.cpu().t().matmul(x2.cpu()) if GPTQ_CPU else x2.t().matmul(x2).cpu()
        h_cpu = h if h_cpu is None else h_cpu.add_(h)
        count += int(x2.shape[0])
        del x2

    handle = module.register_forward_pre_hook(hook)
    input_ids_cpu = encodings.input_ids
    total_len = input_ids_cpu.size(1)
    try:
        steps = 0
        for begin in range(0, total_len, GPTQ_CALIB_STRIDE):
            end = min(begin + GPTQ_CALIB_SEQ_LEN, total_len)
            if end <= begin:
                break
            input_ids = input_ids_cpu[:, begin:end].to(DEVICE)
            model(input_ids=input_ids, use_cache=False)
            del input_ids
            steps += 1
            if steps >= GPTQ_CALIB_STEPS:
                break
    finally:
        handle.remove()
    if h_cpu is None or count <= 0:
        raise RuntimeError("No calibration activations collected")
    return (h_cpu / float(count)).contiguous()


@torch.no_grad()
def collect_hessian_rotated(model: nn.Module, module: nn.Linear, encodings) -> torch.Tensor:
    h_cpu: Optional[torch.Tensor] = None
    count = 0

    def hook(_mod, inputs) -> None:
        nonlocal h_cpu, count
        x = inputs[0].detach()
        if not torch.is_floating_point(x):
            return
        x_rot = hadamard_transform_last_dim(x, LINEAR_ROT_GROUP_SIZE)
        x2 = x_rot.reshape(-1, x_rot.shape[-1]).to(torch.float32)
        h = x2.cpu().t().matmul(x2.cpu()) if GPTQ_CPU else x2.t().matmul(x2).cpu()
        h_cpu = h if h_cpu is None else h_cpu.add_(h)
        count += int(x2.shape[0])
        del x_rot, x2

    handle = module.register_forward_pre_hook(hook)
    input_ids_cpu = encodings.input_ids
    total_len = input_ids_cpu.size(1)
    try:
        steps = 0
        for begin in range(0, total_len, GPTQ_CALIB_STRIDE):
            end = min(begin + GPTQ_CALIB_SEQ_LEN, total_len)
            if end <= begin:
                break
            input_ids = input_ids_cpu[:, begin:end].to(DEVICE)
            model(input_ids=input_ids, use_cache=False)
            del input_ids
            steps += 1
            if steps >= GPTQ_CALIB_STEPS:
                break
    finally:
        handle.remove()
    if h_cpu is None or count <= 0:
        raise RuntimeError("No calibration activations collected")
    return (h_cpu / float(count)).contiguous()


def quantize_nvfp4_blocks_mse(blocks: torch.Tensor, n_candidates: int = 25) -> torch.Tensor:
    """Rotated GPTQ Ingredient 1: MSE-optimized grid. Searches a grid of scale multipliers
    (relative to the absmax-derived scale) and keeps whichever minimizes per-block MSE,
    instead of always mapping the block maximum to the FP4 max (absmax scaling)."""
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    base_ideal_scale = max_abs / fp4_max

    best_scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(base_ideal_scale), min=2.0 ** -24), torch.ones_like(max_abs))
    best_dequant = quantize_to_fp_levels(blocks / best_scale, levels, lsb_even) * best_scale
    best_mse = ((best_dequant - blocks) ** 2).mean(dim=1, keepdim=True)

    for factor in torch.linspace(0.3, 1.0, n_candidates - 1)[:-1].tolist():
        cand_scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(base_ideal_scale * factor), min=2.0 ** -24), torch.ones_like(max_abs))
        cand_dequant = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
        cand_mse = ((cand_dequant - blocks) ** 2).mean(dim=1, keepdim=True)
        better = cand_mse < best_mse
        best_mse = torch.where(better, cand_mse, best_mse)
        best_scale = torch.where(better, cand_scale, best_scale)
        best_dequant = torch.where(better, cand_dequant, best_dequant)

    return best_dequant


@torch.no_grad()
def quantize_tensor_mse(tensor: torch.Tensor, format_name: str) -> torch.Tensor:
    block_size = format_block_size(format_name)
    return block_last_dimension(tensor, block_size, quantize_nvfp4_blocks_mse)


# Ported as-is from the proven H0/nvfp4_hessiandiag_sT_ppl.py recipe (the one
# that actually built the real ptq_cache/nvfp4/*/ptq_cache_NVFP4.pt weights) --
# NOT the simpler single-level 25-candidate search this replaces, which
# under-performed the real cache (missing the tensor-level s_T search is why).
# Two levels, both Hessian-diagonal-weighted:
#   1. s_T: ONE global scale multiplier for the whole weight tensor, searched
#      once via find_best_s_T_hessiandiag over a wide octave range.
#   2. per-block E4M3 scale: searched per 16-column block via
#      quantize_nvfp4_blocks_hessiandiag, on top of the chosen s_T.
_HESSIANDIAG_E4M3_MULTS = [2.0 ** (k / 8.0) for k in range(-32, 17)]
_HESSIANDIAG_S_T_LOG2_OCTAVES = range(-24, 25)
_HESSIANDIAG_S_T_SAMPLE_BLOCKS = 100_000


def _hessiandiag_s_T_reference(global_amax: torch.Tensor, fp4_max: torch.Tensor) -> torch.Tensor:
    s_T = torch.clamp(global_amax / (float(torch.finfo(torch.float8_e4m3fn).max) * fp4_max), min=2.0 ** -24)
    return torch.where(global_amax > 0, s_T, torch.ones_like(s_T))


def quantize_nvfp4_blocks_hessiandiag(blocks: torch.Tensor, h_diag_block: torch.Tensor, s_T: float) -> torch.Tensor:
    """Per-block E4M3 scale search (49 candidates, s_T already fixed),
    objective = Hessian-diagonal-weighted sum of squared error."""
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = (max_abs / fp4_max) / s_T
    h_row = h_diag_block.to(device=blocks.device, dtype=blocks.dtype).view(1, -1)

    best_err, best_q = None, None
    for mult in _HESSIANDIAG_E4M3_MULTS:
        cand_scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(ideal_scale * mult) * s_T, min=2.0 ** -24), torch.ones_like(max_abs))
        q = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
        err = ((blocks - q).pow(2) * h_row).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
    return best_q


def find_best_s_T_hessiandiag(weight: torch.Tensor, hessian: torch.Tensor, block_size: int) -> float:
    """Search the ONE tensor-level s_T that minimizes total Hessian-weighted
    error on a sample of this weight's blocks (each already scored at ITS
    OWN best per-block E4M3 candidate for that s_T)."""
    device = weight.device
    in_f = weight.shape[1]
    n_blocks_per_row = in_f // block_size
    h_diag = torch.diag(hessian).to(device, torch.float32)
    h_diag_blocks = h_diag[:n_blocks_per_row * block_size].reshape(n_blocks_per_row, block_size)

    weight_blocks_all = weight[:, :n_blocks_per_row * block_size].reshape(-1, n_blocks_per_row, block_size).to(device, torch.float32)
    out_features = weight_blocks_all.shape[0]
    h_tiled = h_diag_blocks.unsqueeze(0).expand(out_features, -1, -1)

    blocks_all = weight_blocks_all.reshape(-1, block_size)
    h_all = h_tiled.reshape(-1, block_size)
    n_total = blocks_all.shape[0]
    if n_total > _HESSIANDIAG_S_T_SAMPLE_BLOCKS:
        idx = torch.randperm(n_total, device=device)[:_HESSIANDIAG_S_T_SAMPLE_BLOCKS]
        blocks, h_sample = blocks_all[idx], h_all[idx]
    else:
        blocks, h_sample = blocks_all, h_all

    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    global_amax = blocks.abs().amax()
    s_T_ref = _hessiandiag_s_T_reference(global_amax, fp4_max).item()
    s_T_candidates = [s_T_ref * (2.0 ** k) for k in _HESSIANDIAG_S_T_LOG2_OCTAVES] + [1.0]

    best_total_err, best_s_T = None, 1.0
    for s_T in s_T_candidates:
        ideal_scale = (max_abs / fp4_max) / s_T
        best_err_this = None
        for mult in _HESSIANDIAG_E4M3_MULTS:
            cand_scale = torch.where(max_abs > 0, torch.clamp(fake_fp8_e4m3_scale(ideal_scale * mult) * s_T, min=2.0 ** -24), torch.ones_like(max_abs))
            q = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
            err_sum = ((blocks - q).pow(2) * h_sample).sum(dim=1, keepdim=True)
            best_err_this = err_sum if best_err_this is None else torch.minimum(best_err_this, err_sum)
        total = best_err_this.sum().item()
        if best_total_err is None or total < best_total_err - 1e-9:
            best_total_err, best_s_T = total, s_T
    return best_s_T


# MXFP4 has no tensor-level scale (see round_scale_ue8m0's docstring), so its
# Hessian-weighted search is single-level: just a small set of octave shifts
# around the round-up UE8M0 scale, weighted by the Hessian diagonal. No
# find_best_s_T equivalent needed -- there is nothing tensor-wide to search.
_HESSIANDIAG_UE8M0_OCTAVE_SHIFTS = [-3, -2, -1, 0, 1]


def quantize_mxfp4_blocks_hessiandiag(blocks: torch.Tensor, h_diag_block: torch.Tensor) -> torch.Tensor:
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    levels, lsb_even = get_positive_fp_levels(FP4_E2M1, blocks.device)
    fp4_max = levels[-1].to(device=blocks.device, dtype=blocks.dtype)
    ideal_scale = max_abs / fp4_max
    h_row = h_diag_block.to(device=blocks.device, dtype=blocks.dtype).view(1, -1)
    base_scale = torch.where(max_abs > 0, round_scale_ue8m0(ideal_scale), torch.ones_like(max_abs))

    best_err, best_q = None, None
    for shift in _HESSIANDIAG_UE8M0_OCTAVE_SHIFTS:
        cand_scale = torch.clamp(base_scale * (2.0 ** shift), min=2.0 ** UE8M0_MIN_EXP)
        q = quantize_to_fp_levels(blocks / cand_scale, levels, lsb_even) * cand_scale
        err = ((blocks - q).pow(2) * h_row).sum(dim=1, keepdim=True)
        if best_err is None:
            best_err, best_q = err, q
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
    return best_q


@torch.no_grad()
def block_gptq_quantize_weight(
    weight: torch.Tensor, hessian: torch.Tensor, format_name: str,
    use_mse_grid: bool = False, use_hessiandiag_sT: bool = False, use_act_order: bool = False,
) -> torch.Tensor:
    """Shared OBS-style block GPTQ update; caller decides whether `weight` is pre-rotated.

    use_mse_grid: Rotated GPTQ Ingredient 1 (MSE-optimized per-block scale, unweighted).
    use_hessiandiag_sT: proven two-level scale search ported from
      H0/nvfp4_hessiandiag_sT_ppl.py -- one tensor-level s_T (found once via
      find_best_s_T_hessiandiag) times a per-block E4M3 scale, both chosen to
      minimize Hessian-diagonal-weighted squared error (Danila Mishin's
      optscale objective, but with the s_T level the earlier single-level
      quantize_nvfp4_blocks_hessian_mse attempt was missing). Takes priority
      over use_mse_grid if both are set.
    use_act_order: Rotated GPTQ Ingredient 2, dynamic form (permute columns by descending
      Hessian diagonal before quantizing). The paper's "static" variant exists only to
      avoid a runtime re-shuffle cost and gives the same accuracy as dynamic act-order,
      so for offline PPL evaluation the dynamic form is equivalent and simpler.
    """
    _, in_features = weight.shape
    block_size = format_block_size(format_name)
    work_device = torch.device("cpu") if GPTQ_CPU else weight.device
    w = weight.detach().to(work_device, torch.float32).clone()
    h = hessian.to(work_device, torch.float32).clone()

    perm = inv_perm = None
    if use_act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        inv_perm = torch.argsort(perm)
        w = w[:, perm].contiguous()
        h = h[perm][:, perm].contiguous()

    diag = torch.diag(h)
    s_T = 1.0
    if use_hessiandiag_sT and format_name == "NVFP4":
        s_T = find_best_s_T_hessiandiag(w, h, block_size)  # MXFP4 has no tensor-level scale to search (see quantize_mxfp4_blocks_hessiandiag)
    damp = GPTQ_DAMP_PERCENT * torch.clamp(diag.mean(), min=1e-12)
    idx = torch.arange(in_features, device=work_device)
    h[idx, idx] += damp
    hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    quantize_fn = quantize_tensor_mse if use_mse_grid else quantize_tensor
    q = torch.empty_like(w)
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        w_block = w[:, start:end]
        if use_hessiandiag_sT and format_name == "NVFP4":
            q_block = quantize_nvfp4_blocks_hessiandiag(w_block, diag[start:end], s_T).to(work_device, torch.float32)
        elif use_hessiandiag_sT and format_name == "MXFP4":
            # 2026-09-08: an isolated Linear-only (no QK) test suggested this search hurt
            # (naive=8.07, GPTQ+naive-scale=8.02, GPTQ+this search=11.00), so it was
            # briefly disabled -- but the FULL production pipeline (QK included, all 4
            # rotation-table arms) came back uniformly WORSE without it (Baseline
            # 11.00->12.10, Ours 7.76->7.82, Rotation 10.71->11.01, Ours+Rotation
            # 7.79->7.88), so the isolated test was misleading and this is restored.
            q_block = quantize_mxfp4_blocks_hessiandiag(w_block, diag[start:end]).to(work_device, torch.float32)
        else:
            q_block = quantize_fn(w_block, format_name).to(work_device, torch.float32)
        q[:, start:end] = q_block
        if end < in_features:
            error = w_block - q_block
            h_bb = hessian_inv[start:end, start:end]
            h_br = hessian_inv[start:end, end:]
            w[:, end:] -= error.matmul(torch.linalg.solve(h_bb, h_br))
        del w_block, q_block

    if use_act_order:
        q = q[:, inv_perm]
    return q.to(weight.device, weight.dtype)


class PlainQuantLinear(nn.Module):
    """Linear with runtime activation quantization but no rotation (GPTQ weight is fixed)."""

    def __init__(self, q_weight: torch.Tensor, bias: Optional[torch.Tensor], format_name: str) -> None:
        super().__init__()
        self.format_name = format_name
        self.weight = nn.Parameter(q_weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_q = quantize_tensor(input, self.format_name)
        return F.linear(x_q, self.weight, self.bias)


@torch.no_grad()
def build_plain_model(format_name: str, encodings) -> Tuple[nn.Module, float]:
    """NVFP4_ONLY: no rotation. Reuses ptq_cache/no_rotation/. Wraps every target
    Linear with PlainQuantLinear so activations are quantized too, not just weights."""
    model = load_model()
    target_names = iter_target_linear_names(model)
    cache_path = os.path.join(NO_ROTATION_CACHE_DIR, f"ptq_cache_{format_name}.pt")
    start = time.time()
    if os.path.exists(cache_path):
        print(f"[plain GPTQ {format_name}] Loading cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        for name in target_names:
            module = get_module_by_name(model, name)
            bias = module.bias.data if module.bias is not None else None
            set_module_by_name(model, name, PlainQuantLinear(cache[name].to(model.device), bias, format_name))
        print(f"[plain GPTQ {format_name}] Loaded {len(target_names)} layers from cache.")
        return model, time.time() - start

    os.makedirs(NO_ROTATION_CACHE_DIR, exist_ok=True)
    cache: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(target_names, 1):
        module = get_module_by_name(model, name)
        if idx % 32 == 0 or idx == len(target_names):
            print(f"[plain GPTQ {format_name}] {idx}/{len(target_names)} {name}")
        hessian = collect_hessian_plain(model, module, encodings)
        q_weight = block_gptq_quantize_weight(module.weight.data, hessian, format_name, use_hessiandiag_sT=True)
        bias = module.bias.data if module.bias is not None else None
        set_module_by_name(model, name, PlainQuantLinear(q_weight, bias, format_name))
        cache[name] = q_weight.cpu()
        del hessian, q_weight
        cleanup()
    torch.save(cache, cache_path)
    print(f"[plain GPTQ {format_name}] cache saved -> {cache_path}")
    return model, time.time() - start


class BoostQuantLinear(nn.Module):
    """Ours (no rotation): same GPTQ weight as PlainQuantLinear, but the
    activation is quantized with this project's topk E2M7 boost method
    instead of plain quantize_tensor."""

    def __init__(self, q_weight: torch.Tensor, bias: Optional[torch.Tensor], format_name: str, k: Optional[int] = None) -> None:
        super().__init__()
        self.format_name = format_name
        self.k = k
        self.weight = nn.Parameter(q_weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_q = quantize_tensor_topk_boost(input, self.format_name, self.k)
        return F.linear(x_q, self.weight, self.bias)


@torch.no_grad()
def build_boost_model(format_name: str, encodings) -> Tuple[nn.Module, float]:
    """NVFP4_OURS: reuses the SAME GPTQ weight cache as NVFP4_ONLY (weight
    quantization is unaffected by the boost method, which is activation-only)
    -- requires that cache already exist (run NVFP4_ONLY first)."""
    model = load_model()
    target_names = iter_target_linear_names(model)
    cache_path = os.path.join(NO_ROTATION_CACHE_DIR, f"ptq_cache_{format_name}.pt")
    start = time.time()
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Expected existing GPTQ cache at {cache_path}; run NVFP4_ONLY first to build it.")
    print(f"[boost, no rotation {format_name}] Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    for name in target_names:
        module = get_module_by_name(model, name)
        bias = module.bias.data if module.bias is not None else None
        set_module_by_name(model, name, BoostQuantLinear(cache[name].to(model.device), bias, format_name))
    print(f"[boost, no rotation {format_name}] Loaded {len(target_names)} layers from cache.")
    return model, time.time() - start


class RotatedQuantLinear(nn.Module):
    """Linear with online group-wise Hadamard rotation of the input activation."""

    def __init__(self, q_weight: torch.Tensor, bias: Optional[torch.Tensor], format_name: str) -> None:
        super().__init__()
        self.format_name = format_name
        self.group_size = LINEAR_ROT_GROUP_SIZE
        self.weight = nn.Parameter(q_weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_rot = hadamard_transform_last_dim(input, self.group_size)
        x_q = quantize_tensor(x_rot, self.format_name)
        return F.linear(x_q, self.weight, self.bias)


@torch.no_grad()
def build_rotated_model(format_name: str, encodings) -> Tuple[nn.Module, float]:
    """NVFP4_ROTATION_ONLY: Rotated GPTQ rotated-Hessian PTQ. Reuses ptq_cache/mrgptq/."""
    model = load_model()
    target_names = iter_target_linear_names(model)
    cache_path = os.path.join(MRGPTQ_CACHE_DIR, f"ptq_cache_{format_name}.pt")
    start = time.time()

    if os.path.exists(cache_path):
        print(f"[Rotated GPTQ {format_name}] Loading cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        for name in target_names:
            module = get_module_by_name(model, name)
            bias = module.bias.data if module.bias is not None else None
            set_module_by_name(model, name, RotatedQuantLinear(cache[name].to(model.device), bias, format_name))
        print(f"[Rotated GPTQ {format_name}] Loaded {len(target_names)} layers from cache.")
        return model, time.time() - start

    os.makedirs(MRGPTQ_CACHE_DIR, exist_ok=True)
    cache: Dict[str, torch.Tensor] = {}
    for idx, name in enumerate(target_names, 1):
        module = get_module_by_name(model, name)
        validate_group_size(module.in_features, LINEAR_ROT_GROUP_SIZE, f"Linear {name}")
        if idx % 32 == 0 or idx == len(target_names):
            print(f"[Rotated GPTQ {format_name}] {idx}/{len(target_names)} {name}")
        hessian = collect_hessian_rotated(model, module, encodings)
        w_rot = rotate_weight_columns(module.weight.data, LINEAR_ROT_GROUP_SIZE)
        q_weight = block_gptq_quantize_weight(w_rot, hessian, format_name, use_hessiandiag_sT=True)  # same proven two-level scale search, computed in the rotated coordinate space of w_rot/hessian, matching the non-rotated arm for a fair comparison
        bias = module.bias.data if module.bias is not None else None
        set_module_by_name(model, name, RotatedQuantLinear(q_weight, bias, format_name))
        cache[name] = q_weight.cpu()
        del hessian, w_rot, q_weight
        cleanup()
    torch.save(cache, cache_path)
    print(f"[Rotated GPTQ {format_name}] cache saved -> {cache_path}")
    return model, time.time() - start


class RotatedBoostQuantLinear(nn.Module):
    """Ours + Rotation: rotate the input, boost-quantize it (topk E2M7),
    then matmul with the already-rotated Rotated GPTQ weight from the cache."""

    def __init__(self, q_weight: torch.Tensor, bias: Optional[torch.Tensor], format_name: str, k: Optional[int] = None) -> None:
        super().__init__()
        self.format_name = format_name
        self.group_size = LINEAR_ROT_GROUP_SIZE
        self.k = k
        self.weight = nn.Parameter(q_weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_rot = hadamard_transform_last_dim(input, self.group_size)
        x_q = quantize_tensor_topk_boost(x_rot, self.format_name, self.k)
        return F.linear(x_q, self.weight, self.bias)


@torch.no_grad()
def build_rotated_boost_model(format_name: str, encodings) -> Tuple[nn.Module, float]:
    """NVFP4_OURS_ROTATION: reuses the SAME Rotated GPTQ weight cache as
    NVFP4_ROTATION_ONLY -- requires that cache already exist."""
    model = load_model()
    target_names = iter_target_linear_names(model)
    cache_path = os.path.join(MRGPTQ_CACHE_DIR, f"ptq_cache_{format_name}.pt")
    start = time.time()
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Expected existing Rotated GPTQ cache at {cache_path}; run NVFP4_ROTATION_ONLY first to build it.")
    print(f"[boost, rotated {format_name}] Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    for name in target_names:
        module = get_module_by_name(model, name)
        bias = module.bias.data if module.bias is not None else None
        set_module_by_name(model, name, RotatedBoostQuantLinear(cache[name].to(model.device), bias, format_name))
    print(f"[boost, rotated {format_name}] Loaded {len(target_names)} layers from cache.")
    return model, time.time() - start


# =============================================================================
# Attention: plain QK quantize vs QK Hadamard-rotate-then-quantize
# =============================================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_compat(q, k, cos, sin):
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    cos, sin = cos.to(q.device, q.dtype), sin.to(q.device, q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def is_llama_attention_module(module: nn.Module) -> bool:
    return all(hasattr(module, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")) and (
        module.__class__.__name__.startswith("Llama") or hasattr(module, "num_key_value_groups")
    )


class QKAttention(nn.Module):
    """Q/K after RoPE. qk_quantize=False leaves Q/K at BF16 (LINEAR_ONLY-style).
    qk_rotate=True Hadamard-rotates before quantizing (only meaningful when qk_quantize=True)."""

    def __init__(self, attention: nn.Module, format_name: str, qk_rotate: bool, qk_quantize: bool = True, qk_boost: bool = False) -> None:
        super().__init__()
        self.format_name = format_name
        self.qk_rotate = qk_rotate
        self.qk_quantize = qk_quantize
        self.qk_boost = qk_boost
        self.qk_group_size = QK_GROUP_SIZE
        self.query_chunk = ATTENTION_QUERY_CHUNK

        self.config = attention.config
        self.layer_idx = getattr(attention, "layer_idx", 0)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj
        self.rotary_emb = getattr(attention, "rotary_emb", None)
        self.q_norm, self.k_norm = getattr(attention, "q_norm", None), getattr(attention, "k_norm", None)

        self.num_heads = int(getattr(attention, "num_heads", self.config.num_attention_heads))
        self.num_key_value_heads = int(getattr(attention, "num_key_value_heads", self.config.num_key_value_heads))
        self.head_dim = int(getattr(attention, "head_dim", self.config.hidden_size // self.num_heads))
        self.num_key_value_groups = int(getattr(attention, "num_key_value_groups", self.num_heads // self.num_key_value_heads))
        self.scaling = float(getattr(attention, "scaling", self.head_dim ** -0.5))
        self.attention_dropout = float(getattr(attention, "attention_dropout", getattr(self.config, "attention_dropout", 0.0)))
        self.is_causal = bool(getattr(attention, "is_causal", True))

        fwd_params = inspect.signature(attention.forward).parameters
        self.legacy_three_outputs = "output_attentions" in fwd_params and "use_cache" in fwd_params

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.qk_quantize:
            return tensor
        if self.qk_rotate:
            validate_group_size(self.head_dim, self.qk_group_size, "QK head_dim")
            tensor = hadamard_transform_last_dim(tensor, self.qk_group_size)
        if self.qk_boost:
            return quantize_tensor_topk_boost(tensor, self.format_name)
        return quantize_tensor(tensor, self.format_name)

    def _position_embeddings(self, value_states, position_ids, position_embeddings):
        if position_embeddings is not None:
            return position_embeddings
        try:
            return self.rotary_emb(value_states, position_ids)
        except TypeError:
            return self.rotary_emb(value_states, position_ids=position_ids)

    def _slice_mask(self, attention_mask, q0, q1, q_len, kv_len, device):
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                return attention_mask[..., q0:q1, :kv_len]
            if attention_mask.dim() == 2:
                return attention_mask[:, None, None, :kv_len]
            return attention_mask
        if not self.is_causal or kv_len <= 1:
            return None
        qpos = (torch.arange(q0, q1, device=device) + (kv_len - q_len)).view(-1, 1)
        kpos = torch.arange(kv_len, device=device).view(1, -1)
        mask = torch.zeros((1, 1, q1 - q0, kv_len), device=device, dtype=torch.float32)
        return mask.masked_fill((kpos > qpos).view(1, 1, q1 - q0, kv_len), -torch.inf)

    @torch.no_grad()
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None,
                past_key_values=None, **kwargs):
        if past_key_value is None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        cos, sin = self._position_embeddings(v, position_ids, position_embeddings)
        q, k = apply_rope_compat(q, k, cos, sin)
        if past_key_value is not None:
            kwargs2 = {"sin": sin, "cos": cos, "cache_position": cache_position}
            try:
                k, v = past_key_value.update(k, v, self.layer_idx, kwargs2)
            except TypeError:
                k, v = past_key_value.update(k, v, self.layer_idx)

        q_q = self._quantize(q)
        k_q = self._quantize(k)
        k_q_rep = repeat_kv(k_q, self.num_key_value_groups)
        v_rep = repeat_kv(v, self.num_key_value_groups)
        kv_len = k_q_rep.shape[-2]

        out_bh = torch.empty((bsz, self.num_heads, q_len, self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        chunk = q_len if self.query_chunk <= 0 else self.query_chunk
        for q0 in range(0, q_len, chunk):
            q1 = min(q0 + chunk, q_len)
            scores = torch.matmul(q_q[:, :, q0:q1, :].to(torch.float32), k_q_rep.to(torch.float32).transpose(-2, -1)) * self.scaling
            mask = self._slice_mask(attention_mask, q0, q1, q_len, kv_len, scores.device)
            if mask is not None:
                scores = scores + mask.to(scores.device, scores.dtype)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
            probs = F.dropout(probs, p=self.attention_dropout, training=self.training)
            out_bh[:, :, q0:q1, :].copy_(torch.matmul(probs, v_rep))

        attn_out = self.o_proj(out_bh.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim))
        if self.legacy_three_outputs:
            return attn_out, None, past_key_value
        return attn_out, None


def replace_attention_modules(module: nn.Module, format_name: str, qk_rotate: bool, qk_quantize: bool = True, qk_boost: bool = False, prefix: str = "") -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if child_name == "self_attn" and is_llama_attention_module(child):
            setattr(module, child_name, QKAttention(child, format_name, qk_rotate, qk_quantize, qk_boost))
            replaced += 1
        else:
            replaced += replace_attention_modules(child, format_name, qk_rotate, qk_quantize, qk_boost, full_name)
    return replaced


# =============================================================================
# Results and execution
# =============================================================================

def make_row(experiment: str, format_name: Optional[str], rotation: bool, qk_quantize: bool = True, ptq_elapsed: float = 0.0) -> Dict[str, Any]:
    is_baseline = experiment == "BF16_BASELINE"
    return {
        "script_revision": SCRIPT_REVISION,
        "model": MODEL_NAME,
        "experiment": experiment,
        "name": f"{safe_model_name(MODEL_NAME)}__{experiment}",
        "format": format_name or "BF16",
        "rotation": rotation,
        "qk_quantize": "" if is_baseline else qk_quantize,
        "ppl": "",
        "status": "",
        "elapsed_sec": "",
        "ptq_elapsed_sec": "" if is_baseline else round(ptq_elapsed, 3),
        "block_size": "" if is_baseline else format_block_size(format_name),
        "linear_rot_group_size": LINEAR_ROT_GROUP_SIZE if rotation else "",
        "qk_rot_group_size": QK_GROUP_SIZE if rotation else "",
        "ppl_seq_len": PPL_SEQ_LEN,
        "ppl_stride": PPL_STRIDE,
        "ppl_max_steps": PPL_MAX_STEPS,
        "gptq_calib_seq_len": "" if is_baseline else GPTQ_CALIB_SEQ_LEN,
        "gptq_calib_stride": "" if is_baseline else GPTQ_CALIB_STRIDE,
        "gptq_calib_steps": "" if is_baseline else GPTQ_CALIB_STEPS,
        "gptq_damp_percent": "" if is_baseline else GPTQ_DAMP_PERCENT,
        "rounding_mode": ROUNDING_MODE,
        "skip_lm_head": SKIP_LM_HEAD,
        "device": str(DEVICE),
        "error": "",
    }


def run_one(experiment: str, model_builder, format_name: str, rotation: bool, encodings, rows: List[Dict[str, Any]],
            qk_rotate: Optional[bool] = None, qk_quantize: bool = True, qk_boost: bool = False) -> None:
    if qk_rotate is None:
        qk_rotate = rotation
    model: Optional[nn.Module] = None
    try:
        same_seeds(SEED)
        ptq_start = time.time()
        model, ptq_elapsed = model_builder(format_name, encodings)
        replaced_attn = replace_attention_modules(model, format_name, qk_rotate, qk_quantize, qk_boost)
        if replaced_attn != EXPECTED_ATTENTION_LAYERS:
            raise RuntimeError(f"Expected {EXPECTED_ATTENTION_LAYERS} attention layers, got {replaced_attn}")
        print(f"[{experiment}] weights ready in {ptq_elapsed:.1f}s, attention wrappers={replaced_attn}, qk_rotate={qk_rotate}, qk_quantize={qk_quantize}")

        row = make_row(experiment, format_name, rotation, qk_quantize, ptq_elapsed)
        start = time.time()
        try:
            ppl = measure_ppl(model, encodings, desc=row["name"])
            row.update({"ppl": format_ppl(ppl), "status": "OK", "elapsed_sec": round(time.time() - start, 3)})
            print(f"{row['name']} PPL = {row['ppl']}")
        except Exception as exc:
            traceback.print_exc()
            row.update({"ppl": "FAILED", "status": "FAILED", "elapsed_sec": round(time.time() - start, 3), "error": short_error(exc)})
        rows.append(row)
        save_csv(rows, RESULT_CSV)
    finally:
        if model is not None:
            del model
        cleanup()


def run_baseline(encodings, rows: List[Dict[str, Any]]) -> None:
    row = make_row("BF16_BASELINE", None, False)
    model: Optional[nn.Module] = None
    start = time.time()
    try:
        same_seeds(SEED)
        model = load_model()
        ppl = measure_ppl(model, encodings, desc=row["name"])
        row.update({"ppl": format_ppl(ppl), "status": "OK", "elapsed_sec": round(time.time() - start, 3)})
        print(f"{row['name']} PPL = {row['ppl']}")
    finally:
        if model is not None:
            del model
        cleanup()
    rows.append(row)
    save_csv(rows, RESULT_CSV)


def print_run_contract() -> None:
    print("=" * 110)
    print("Rotation vs no-rotation, weight+GPTQ, boost method (Llama)")
    print("=" * 110)
    print(f"Revision           : {SCRIPT_REVISION}")
    print(f"Model              : {MODEL_NAME}")
    print(f"Device             : {DEVICE}")
    print(f"No-rotation cache  : {NO_ROTATION_CACHE_DIR}")
    print(f"Rotated GPTQ cache      : {MRGPTQ_CACHE_DIR}")
    print(f"Methods to run: {[m[0] for m in METHOD_SPECS]}")
    print("=" * 110)


# (experiment, model_builder, format_name, rotation_label, qk_rotate, qk_quantize, qk_boost)
METHOD_SPECS: List[Tuple[str, Any, str, bool, bool, bool, bool]] = [
    ("NVFP4_ONLY", build_plain_model, "NVFP4", False, False, True, False),          # Baseline
    ("NVFP4_OURS", build_boost_model, "NVFP4", False, False, True, True),           # Ours: same GPTQ weight cache as NVFP4_ONLY, boost activation quant
    ("NVFP4_ROTATION_ONLY", build_rotated_model, "NVFP4", True, False, True, False),  # Rotation. QK quantized but never rotated (Rotated GPTQ rotation+GPTQ correction only applies to linear weights)
    ("NVFP4_OURS_ROTATION", build_rotated_boost_model, "NVFP4", True, False, True, True),  # Ours + Rotation: same Rotated GPTQ weight cache as NVFP4_ROTATION_ONLY, boost activation quant
    ("MXFP4_ONLY", build_plain_model, "MXFP4", False, False, True, False),          # Baseline
    ("MXFP4_OURS", build_boost_model, "MXFP4", False, False, True, True),           # Ours: same GPTQ weight cache as MXFP4_ONLY, boost activation quant
    ("MXFP4_ROTATION_ONLY", build_rotated_model, "MXFP4", True, False, True, False),  # Rotation. QK quantized but never rotated
    ("MXFP4_OURS_ROTATION", build_rotated_boost_model, "MXFP4", True, False, True, True),  # Ours + Rotation: same Rotated GPTQ weight cache as MXFP4_ROTATION_ONLY, boost activation quant
]
_methods_env = os.environ.get("METHODS", "").strip()
if _methods_env:
    _wanted = [m.strip() for m in _methods_env.split(",") if m.strip()]
    METHOD_SPECS = [spec for spec in METHOD_SPECS if spec[0] in _wanted]
    if not METHOD_SPECS:
        raise ValueError(f"METHODS={_methods_env!r} matched no known method")

RUN_BASELINE = os.environ.get("RUN_BASELINE", "1") != "0"


def main() -> None:
    same_seeds(SEED)
    print_run_contract()

    print("\nLoading WikiText-2 test text...")
    full_text = load_wikitext2_text("test")
    tokenizer, encodings = load_tokenizer_and_encodings(full_text)
    del tokenizer, full_text
    print(f"Encoding shape: {tuple(encodings.input_ids.shape)}")

    rows: List[Dict[str, Any]] = []
    if RUN_BASELINE:
        run_baseline(encodings, rows)
    for experiment, model_builder, format_name, rotation, qk_rotate, qk_quantize, qk_boost in METHOD_SPECS:
        run_one(experiment, model_builder, format_name, rotation, encodings, rows, qk_rotate=qk_rotate, qk_quantize=qk_quantize, qk_boost=qk_boost)

    print("\n" + "=" * 110)
    print("FINAL RESULTS")
    for row in rows:
        print(f"{row['name']:<90} PPL = {row['ppl']}")
    print(f"Saved CSV -> {RESULT_CSV}")


if __name__ == "__main__":
    main()
