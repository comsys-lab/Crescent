"""Shared GEMM-output Relative MSE machinery for the Crescent GEMM Output Error experiment.

Reuses common/formats.py's quantization primitives directly (no reimplementation):
  quantize_tensor_config                   -> plain single-precision block quantization (Native FP4 / All E2M3)
  quantize_tensor_grouped_boost            -> Crescent: group-local boost selected by squared quantization error
  quantize_tensor_grouped_boost_by_magnitude -> Magnitude: group-local boost selected by raw |activation|

Mirrors common/ppl_w4a4.py's FORMAT_CONFIG / LINEAR_DIV / QK_DIV exactly, so
"same quantization/scaling settings as the existing PPL/zero-shot experiments" holds by construction.

GEMM output definitions (Y = GEMM output, FP32 accumulation throughout):
  Linear : Y = quant_fn(x) @ W^T          (W = BF16 original weight for the "bf16" setting,
                                            GPTQ-cache weight, identical across all FP4 settings, otherwise)
  Attention (Q @ K^T / sqrt(head_dim)):
    bf16/native/all_e2m3 : plain, same quantizer applied to both Q and K (or none, for bf16)
    magnitude/crescent     : error-compensation decomposition, EXACTLY ppl_w4a4.py's arm_setup
                             formula (q_base@k_boost^T + q_boost@k_base^T - q_base@k_base^T) --
                             copied verbatim from quant_targets.patch_eager_ropegqa_decomposed's
                             attn_weights line, not re-derived.
"""
import math
import os
import sys

import torch

HYP_COMMON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common")
sys.path.insert(0, HYP_COMMON)
from formats import quantize_tensor_config, quantize_tensor_group_topk_boost  # noqa: E402

# Mirrors ppl_w4a4.py's FORMAT_CONFIG / LINEAR_DIV / QK_DIV verbatim.
FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost="E2M3_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost="E2M3_mxfp4_ue8m0"),
}
# Final config (2026-11-07): Linear = exact top-2 per 16-element group; Q/K = top-1 per 16.
LINEAR_DIV = 16
LINEAR_K = 2
QK_DIV = 16
METHODS = ["native", "magnitude", "crescent", "all_e2m3"]
METHOD_LABELS = {"native": "Native FP4", "magnitude": "Magnitude", "crescent": "Crescent", "all_e2m3": "All E2M3"}


def _plain(x, fmt, config_key):
    c = FORMAT_CONFIG[fmt]
    cfg_name = c["base_config"] if config_key == "base" else c["boost"]
    return quantize_tensor_config(x.detach().to(torch.float32), cfg_name, c["block_size"])


def quantize_activation(x: torch.Tensor, fmt: str, method: str, group: int, k: int = 1) -> torch.Tensor:
    """x: real BF16 (or fp32) activation tensor, same tensor object reused across every method
    call for a given (layer, fmt) pair -- the harness is responsible for that reuse, this
    function never mutates or caches x. Returns an FP32 tensor (the quantized/boosted value,
    still representable in bf16 -- fp32 here only avoids repeated dtype casts upstream)."""
    c = FORMAT_CONFIG[fmt]
    x32 = x.detach().to(torch.float32)
    if method == "native":
        return quantize_tensor_config(x32, c["base_config"], c["block_size"])
    if method == "all_e2m3":
        return quantize_tensor_config(x32, c["boost"], c["block_size"])
    if method in ("magnitude", "crescent"):
        return quantize_tensor_group_topk_boost(x32, group, k, block_size=c["block_size"], base_config=c["base_config"],
                                                boost_config=c["boost"], by="error" if method == "crescent" else "magnitude")
    raise ValueError(method)


@torch.no_grad()
def linear_gemm_outputs(x_bf16: torch.Tensor, w_bf16: torch.Tensor, w_gptq: torch.Tensor, fmt: str):
    """x_bf16: real captured activation [..., in_features]. w_bf16: original unquantized weight
    [out_features, in_features]. w_gptq: GPTQ-cache weight, same shape, used for every FP4 setting.
    Returns dict setting -> Y (FP32 tensor [..., out_features])."""
    x32 = x_bf16.detach().to(torch.float32)
    out = {"bf16": x32 @ w_bf16.detach().to(torch.float32).T}
    wg = w_gptq.detach().to(torch.float32)
    for m in METHODS:
        xq = quantize_activation(x_bf16, fmt, m, LINEAR_DIV, LINEAR_K)
        out[m] = xq @ wg.T
    return out


@torch.no_grad()
def attn_gemm_outputs(q_bf16: torch.Tensor, k_bf16: torch.Tensor, head_dim: int, fmt: str):
    """q_bf16/k_bf16: real captured Q/K AFTER RoPE (and, for GQA, after repeat_kv so both
    operands share head count) -- [bsz, heads, seq, head_dim]. Returns dict setting -> Y
    (FP32 attn_weights, pre-softmax, [bsz, heads, seq_q, seq_k])."""
    scale = 1.0 / math.sqrt(head_dim)
    qf, kf = q_bf16.detach().to(torch.float32), k_bf16.detach().to(torch.float32)
    out = {"bf16": (qf @ kf.transpose(-2, -1)) * scale}
    for m in ("native", "all_e2m3"):
        q2 = quantize_activation(q_bf16, fmt, m, QK_DIV)
        k2 = quantize_activation(k_bf16, fmt, m, QK_DIV)
        out[m] = (q2 @ k2.transpose(-2, -1)) * scale
    for m in ("magnitude", "crescent"):
        q_base = quantize_activation(q_bf16, fmt, "native", QK_DIV)
        q_boost = quantize_activation(q_bf16, fmt, m, QK_DIV)
        k_base = quantize_activation(k_bf16, fmt, "native", QK_DIV)
        k_boost = quantize_activation(k_bf16, fmt, m, QK_DIV)
        # EXACT copy of quant_targets.patch_eager_ropegqa_decomposed's attn_weights formula
        # (delta x delta term dropped, matching the real correction datapath).
        out[m] = (q_base @ k_boost.transpose(-2, -1) + q_boost @ k_base.transpose(-2, -1)
                  - q_base @ k_base.transpose(-2, -1)) * scale
    return out


def relative_mse(y_method: torch.Tensor, y_ref: torch.Tensor) -> tuple:
    """Returns (relative_mse, sum_sq_error, sum_sq_ref, n_elements), all FP32/Python float."""
    diff = (y_method.to(torch.float32) - y_ref.to(torch.float32))
    sse = diff.pow(2).sum().item()
    ssr = y_ref.to(torch.float32).pow(2).sum().item()
    n = y_ref.numel()
    return (sse / ssr if ssr > 0 else float("nan")), sse, ssr, n


def reduction_rate(rel_mse_native: float, rel_mse_method: float) -> float:
    if rel_mse_native == 0:
        return float("nan")
    return (rel_mse_native - rel_mse_method) / rel_mse_native * 100.0
