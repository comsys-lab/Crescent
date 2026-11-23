"""Block quantization configs: element format x scale policy, named "<elem>_<scale>".

  E2M1_amax_E4M3      NVFP4: E2M1 elements, block 16, scale = amax / 6 rounded to E4M3, with the tensor-level
                      FP32 scale of the two-level scheme (quantize_tensor_config(two_level=True)).
  E2M1_mxfp4_ue8m0    MXFP4: E2M1 elements, block 32, scale rounded up to a power of two (E8M0).
  E2M{m}_amax_E4M3,   same scale policy, E2M{m} elements (m = 1..7); E2M3 is Crescent's extended format.
  E2M{m}_mxfp4_ue8m0
  E8M1_amax_E4M3,     Figure 2 exponent relaxation: E8M1 elements, original scale policy.
  E8M1_mxfp4_ue8m0
  E2M1_optimal_none   Figure 2 scaling-factor relaxation: E2M1 elements, unrounded FP32 scale chosen per block
                      from SCALE_MULT_CANDIDATES to minimize that block's MSE.

Relaxed element formats (E8M1, E2M{m}) keep the scale computed against E2M1's max (scale_ref_fmt), so only the
element format changes. Without it, E8M1's max (~5.1e38) would give a scale far below E4M3's range.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant import (
    FPFormat, get_positive_fp_levels, quantize_to_fp_levels, fake_fp8_e4m3_scale,
    block_last_dimension,
)

BLOCK_SIZE = 16
SCALE_MULT_CANDIDATES = [2.0 ** (k / 8.0) for k in range(-16, 17)]


def make_fp_format(key: str, exp_bits: int, mant_bits: int) -> FPFormat:
    bias = 2 ** (exp_bits - 1) - 1
    finite_exp_max = 2 ** exp_bits - 1
    return FPFormat(key, exp_bits, mant_bits, bias, finite_exp_max)


FMT_E2M1 = make_fp_format("e2m1", 2, 1)  # NVFP4 standard
FMT_E2M7 = make_fp_format("e2m7", 2, 7)  # mantissa axis (BF16 mantissa width)
FMT_E8M1 = make_fp_format("e8m1", 8, 1)  # exponent axis (BF16 exponent width)

# E8M1's own max representable value is 2**128*1.5 ~= 5.1e38, which exceeds
# float32's max finite value (~3.4e38) -- quant.py's level
# cache is float32, so that top level (and fmt_max derived from it) silently
# becomes +inf there, collapsing amax/optimal scale search to a single
# degenerate clamped value for every block. Build a float64 levels tensor
# locally (same construction as build_positive_fp_levels_cpu) so wide-
# exponent formats stay numerically valid; narrow formats (E2M1, E2M7) are
# unaffected either way since their values never approach float32's range.
_LEVEL_CACHE_F64 = {}


def _positive_fp_levels_f64(fmt: FPFormat, device: torch.device):
    key = (fmt.key, str(device))
    if key not in _LEVEL_CACHE_F64:
        max_mant = 2 ** fmt.mant_bits - 1
        mant_scale = 2.0 ** (-fmt.mant_bits)
        values, lsb_even = [], []
        for m in range(max_mant + 1):
            values.append((2.0 ** (1 - fmt.bias)) * (m * mant_scale)); lsb_even.append((m & 1) == 0)
        for e in range(1, fmt.finite_exp_max + 1):
            for m in range(max_mant + 1):
                enc = (e << fmt.mant_bits) | m
                values.append((2.0 ** (e - fmt.bias)) * (1.0 + m * mant_scale)); lsb_even.append((enc & 1) == 0)
        pairs = sorted(zip(values, lsb_even), key=lambda p: p[0])
        lv = torch.tensor([p[0] for p in pairs], dtype=torch.float64, device=device)
        ev = torch.tensor([p[1] for p in pairs], dtype=torch.bool, device=device)
        _LEVEL_CACHE_F64[key] = (lv, ev)
    return _LEVEL_CACHE_F64[key]


def round_scale(scale: torch.Tensor, scale_format: str) -> torch.Tensor:
    if scale_format == "none":
        return scale
    if scale_format == "e4m3":
        # Round-to-nearest-even, matching real NVFP4's spec (NOT round-up --
        # that's MXFP4's convention, which NVFP4 explicitly moved away from
        # since E4M3's finer granularity makes the round-up-to-avoid-
        # saturation workaround unnecessary; see arXiv:2509.25149 Appendix
        # B.2/B.4). Confirmed against the primary source after an earlier,
        # wrong "round up" attempt here based on a slide's Q_E4M3^RU
        # notation and an independently-written script that both turned out
        # to describe MXFP4's rule, not NVFP4's.
        return fake_fp8_e4m3_scale(scale)
    if scale_format == "ue8m0":
        # MXFP4's own scale format (OCP Microscaling spec): pure power-of-
        # two, no mantissa bits, rounded UP (ceiling in log2 space) so the
        # block's own max element never overflows -- MXFP4 has no tensor-
        # level correction to fall back on if it rounded down (unlike
        # NVFP4's two-level scheme). Mirrors common/mxfp4.py's
        # round_up_ue8m0 exactly; duplicated here (not imported) so this
        # module has no import-time dependency on mxfp4.py.
        UE8M0_MIN_EXP, UE8M0_MAX_EXP = -127, 127
        scale = torch.clamp(scale, min=2.0 ** UE8M0_MIN_EXP)
        if os.environ.get("MXFP4_SCALE_RULE") == "floor":
            # OCP floor rule E = floor(log2(amax / 4)); scale arrives as amax / 6 (E2M1 ref), so x1.5
            exp_up = torch.floor(torch.log2(scale * 1.5))
        else:
            exp_up = torch.ceil(torch.log2(scale))
        exp_up = torch.clamp(exp_up, UE8M0_MIN_EXP, UE8M0_MAX_EXP)
        return torch.pow(2.0, exp_up)
    raise ValueError(f"Unknown scale_format: {scale_format}")


E4M3_MAX = 448.0  # FP8 E4M3's own largest finite representable magnitude


def quantize_blocks_config(blocks: torch.Tensor, fmt: FPFormat, scale_selection: str, scale_format: str, scale_ref_fmt: FPFormat = None, s_enc: torch.Tensor = None) -> torch.Tensor:
    orig_dtype = blocks.dtype
    levels, lsb_even = _positive_fp_levels_f64(fmt, blocks.device)
    ref_levels, _ = _positive_fp_levels_f64(scale_ref_fmt or fmt, blocks.device)
    fmt_max = ref_levels[-1]
    blocks64 = blocks.to(torch.float64)
    max_abs = blocks64.abs().amax(dim=-1, keepdim=True)
    ideal_scale = max_abs / fmt_max  # s_dec,b: local per-block decode scale, unrounded

    if scale_selection == "amax":
        if scale_format == "e4m3" and s_enc is not None:
            # Real NVFP4's two-level scheme (arXiv:2509.25149 Appendix B.2):
            # the LOCAL block scale is not rounded to E4M3 directly -- it's
            # first multiplied by the TENSOR-level global encode scale
            # s_enc = (fmt_max*448)/amax(whole tensor), rounded to E4M3 in
            # that renormalized range, then divided back out by s_enc to
            # recover the usable per-block scale. s_enc=None (below) keeps
            # the simpler single-level round(s_dec,b) behavior for configs
            # that don't have a well-defined "whole tensor" (e.g. isolated
            # synthetic blocks with no surrounding tensor).
            s_dec_b_e4m3 = round_scale(ideal_scale * s_enc, scale_format)
            s_dec_b_e4m3 = torch.clamp(s_dec_b_e4m3, min=2.0 ** -170)
            scale = s_dec_b_e4m3 / s_enc
        else:
            scale = round_scale(ideal_scale, scale_format)
        scale = torch.where(max_abs > 0, torch.clamp(scale, min=2.0 ** -170), torch.ones_like(scale))
        q = quantize_to_fp_levels(blocks64 / scale, levels, lsb_even) * scale
        return q.to(orig_dtype)

    if scale_selection == "optimal":
        best_err, best_q = None, None
        for mult in SCALE_MULT_CANDIDATES:
            cand = round_scale(ideal_scale * mult, scale_format)
            cand = torch.where(max_abs > 0, torch.clamp(cand, min=2.0 ** -170), torch.ones_like(cand))
            q = quantize_to_fp_levels(blocks64 / cand, levels, lsb_even) * cand
            err = (blocks64 - q).pow(2).sum(dim=-1, keepdim=True)
            if best_err is None:
                best_err, best_q = err, q
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_q = torch.where(better, q, best_q)
        return best_q.to(orig_dtype)

    raise ValueError(f"Unknown scale_selection: {scale_selection}")


CONFIGS = {
    "E2M1_amax_E4M3": dict(fmt=FMT_E2M1, scale_selection="amax", scale_format="e4m3"),
    "E2M1_optimal_none": dict(fmt=FMT_E2M1, scale_selection="optimal", scale_format="none"),
    "E8M1_amax_E4M3": dict(fmt=FMT_E8M1, scale_selection="amax", scale_format="e4m3", scale_ref_fmt=FMT_E2M1),
    "E2M7_amax_E4M3": dict(fmt=FMT_E2M7, scale_selection="amax", scale_format="e4m3", scale_ref_fmt=FMT_E2M1),
    # MXFP4 (block 32): "ue8m0" uses the single-level power-of-two round-up scale. E2M1_optimal_none
    # serves both formats (its scale is not tied to a scale format).
    "E2M1_mxfp4_ue8m0": dict(fmt=FMT_E2M1, scale_selection="amax", scale_format="ue8m0"),
    "E8M1_mxfp4_ue8m0": dict(fmt=FMT_E8M1, scale_selection="amax", scale_format="ue8m0", scale_ref_fmt=FMT_E2M1),
    "E2M7_mxfp4_ue8m0": dict(fmt=FMT_E2M7, scale_selection="amax", scale_format="ue8m0", scale_ref_fmt=FMT_E2M1),
}

# E2M2..E2M6 (Figure 5 mantissa sweep; E2M3 = Crescent's extended format). m=1 and m=7 are defined above.
for _m in range(2, 7):
    _fmt = make_fp_format(f"e2m{_m}", 2, _m)
    CONFIGS[f"E2M{_m}_amax_E4M3"] = dict(fmt=_fmt, scale_selection="amax", scale_format="e4m3", scale_ref_fmt=FMT_E2M1)
    CONFIGS[f"E2M{_m}_mxfp4_ue8m0"] = dict(fmt=_fmt, scale_selection="amax", scale_format="ue8m0", scale_ref_fmt=FMT_E2M1)


@torch.no_grad()
def quantize_tensor_config(tensor: torch.Tensor, config_name: str, block_size: int = BLOCK_SIZE, two_level: bool = True) -> torch.Tensor:
    """two_level=True (default) implements NVFP4's real tensor-level +
    block-level scaling (s_enc computed from THIS tensor's own amax -- see
    quantize_blocks_config's docstring comment). Pass False to fall back to
    the simpler single-level round(amax_block/fmt_max) behavior, e.g. for
    synthetic per-block data with no natural "whole tensor" to define
    s_enc against."""
    cfg = CONFIGS[config_name]
    ref_fmt = cfg.get("scale_ref_fmt") or cfg["fmt"]
    ref_levels, _ = _positive_fp_levels_f64(ref_fmt, tensor.device)
    ref_fmt_max = ref_levels[-1]

    def qfn(blocks):
        s_enc = None
        if two_level and cfg["scale_selection"] == "amax" and cfg["scale_format"] == "e4m3":
            # s_enc = (fmt_max * E4M3_MAX) / amax(tensor) -- NOT amax/(fmt_max*E4M3_MAX).
            # Confirmed against arXiv:2509.25149 Eq. 1 after an earlier,
            # inverted version of this formula caused catastrophic failure
            # (100% error) on real model layers with small-to-moderate amax
            # (the common case) -- s_enc must be LARGE (>>1) for typical
            # activation magnitudes, remapping the tensor up into E4M3's
            # representable range, not tiny.
            tensor_amax = blocks.to(torch.float64).abs().amax()
            tensor_amax = torch.clamp(tensor_amax, min=2.0 ** -170)
            s_enc = (ref_fmt_max * E4M3_MAX) / tensor_amax
        return quantize_blocks_config(blocks, cfg["fmt"], cfg["scale_selection"], cfg["scale_format"], cfg.get("scale_ref_fmt"), s_enc)

    return block_last_dimension(tensor, block_size, qfn)


@torch.no_grad()
def quantize_tensor_grouped_boost(tensor: torch.Tensor, group_size: int, block_size: int = BLOCK_SIZE,
                                   base_config: str = "E2M1_amax_E4M3", boost_config: str = "E2M3_amax_E4M3") -> torch.Tensor:
    """In each group of `group_size` elements, the one element with the largest squared quantization error
    takes its boost_config value; the rest keep base_config. Used by Figure 11's correctness checks."""
    x_base = quantize_tensor_config(tensor, base_config, block_size)
    x_boost = quantize_tensor_config(tensor, boost_config, block_size)
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    n_blocks = last_dim // block_size
    if n_blocks == 0 or block_size % group_size != 0:
        return x_base
    usable = n_blocks * block_size
    num_groups = block_size // group_size

    t = tensor.reshape(-1, last_dim)
    xb = x_base.reshape(-1, last_dim)
    xu = x_boost.reshape(-1, last_dim)

    t_blocks = t[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xb_blocks = xb[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xu_blocks = xu[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)

    diff_sq = (t_blocks - xb_blocks).pow(2)
    diff_grouped = diff_sq.reshape(*diff_sq.shape[:-1], num_groups, group_size)
    local_idx = diff_grouped.argmax(dim=-1, keepdim=True)
    group_mask = torch.zeros_like(diff_grouped, dtype=torch.bool).scatter_(-1, local_idx, True)
    group_mask = group_mask.reshape(diff_sq.shape)
    result_blocks = torch.where(group_mask, xu_blocks, xb_blocks)

    result = xb.clone()
    result[:, :usable] = result_blocks.reshape(-1, usable).to(result.dtype)
    return result.reshape(orig_shape)


@torch.no_grad()
def quantize_tensor_topk_activation_e2m7(tensor: torch.Tensor, k: int, block_size: int = BLOCK_SIZE,
                                          base_config: str = "E2M1_amax_E4M3", boost_config: str = "E2M7_amax_E4M3") -> torch.Tensor:
    """In each block, the k elements with the largest quantization error |a_i - Q(a)_i| take their
    boost_config value; the rest keep base_config (Figure 5)."""
    x_base = quantize_tensor_config(tensor, base_config, block_size)
    x_boost = quantize_tensor_config(tensor, boost_config, block_size)
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    n_blocks = last_dim // block_size
    if n_blocks == 0 or k <= 0:
        return x_base
    usable = n_blocks * block_size

    t = tensor.reshape(-1, last_dim)
    xb = x_base.reshape(-1, last_dim)
    xu = x_boost.reshape(-1, last_dim)

    t_blocks = t[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xb_blocks = xb[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xu_blocks = xu[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)

    diff = (t_blocks - xb_blocks).abs()
    k = min(k, block_size)
    topk_idx = diff.argsort(dim=-1, descending=True)[..., :k]
    result_blocks = xb_blocks.scatter(-1, topk_idx, xu_blocks.gather(-1, topk_idx))

    result = xb.clone()
    result[:, :usable] = result_blocks.reshape(-1, usable).to(result.dtype)
    return result.reshape(orig_shape)


@torch.no_grad()
def quantize_tensor_group_topk_boost(tensor: torch.Tensor, group_size: int, k: int, block_size: int = BLOCK_SIZE,
                                     base_config: str = "E2M1_amax_E4M3", boost_config: str = "E2M3_amax_E4M3",
                                     by: str = "error") -> torch.Tensor:
    """Exact top-k per contiguous group of `group_size` (group_size must divide block_size): the k positions
    with the largest squared base-quantization error (by='error', Crescent) or the largest |a| (by='magnitude')
    get requantized at boost_config. k=1 equals quantize_tensor_grouped_boost(_by_magnitude)."""
    x_base = quantize_tensor_config(tensor, base_config, block_size)
    x_boost = quantize_tensor_config(tensor, boost_config, block_size)
    orig_shape = tensor.shape
    last_dim = orig_shape[-1]
    n_blocks = last_dim // block_size
    if n_blocks == 0 or block_size % group_size != 0 or k <= 0:
        return x_base
    usable = n_blocks * block_size
    num_groups = block_size // group_size

    t = tensor.reshape(-1, last_dim)
    xb = x_base.reshape(-1, last_dim)
    xu = x_boost.reshape(-1, last_dim)
    t_blocks = t[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xb_blocks = xb[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)
    xu_blocks = xu[:, :usable].reshape(-1, n_blocks, block_size).to(torch.float32)

    score = (t_blocks - xb_blocks).pow(2) if by == "error" else t_blocks.abs()
    sg = score.reshape(*score.shape[:-1], num_groups, group_size)
    idx = sg.topk(min(k, group_size), dim=-1).indices
    mask = torch.zeros_like(sg, dtype=torch.bool).scatter_(-1, idx, True).reshape(score.shape)
    result_blocks = torch.where(mask, xu_blocks, xb_blocks)

    result = xb.clone()
    result[:, :usable] = result_blocks.reshape(-1, usable).to(result.dtype)
    return result.reshape(orig_shape)
