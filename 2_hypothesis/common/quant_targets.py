"""Shared Linear+QK+PV quantization machinery for hypothesis_v4: every
nn.Linear input (except lm_head) + Q/K (before Q@K^T) + P/V (before P@V)
are ALL quantized together as the baseline -- this project's "everything
quantized" scope, unlike hypothesis_v3's Linear-only scope. Established
via ad-hoc verification that this baseline (7.2887 for llama/NVFP4)
matches the old motivation slide's Linear+Q+K number (7.25) once Linear
is included, and that adding P+V on top costs almost nothing extra
(H0's "P alone is cheap" finding generalizes to P+V together).

Each surface (Linear, Q, K, V, P) can independently be left at plain
base-format quantization, or corrected with H5's grouped-local-max boost
(1 position per group of H5_GROUP_SIZE boosted to a higher-mantissa
format, selected by that surface's OWN squared quantization error) --
see quantize_tensor_grouped_boost in formats.py. There is no cross-
surface/joint selection: this matches how the original hypothesis_v4
corrected Linear/Q/K/V independently (each with its own top-k), just
with grouped selection instead of exact top-k, and P added to the
target set now that H0 showed it's cheap to include.

Attention patching (patch_eager_ropegqa/patch_eager_opt) takes plain
per-surface callables (q_fn, k_fn, v_fn, p_fn), not a fixed mode enum --
H1 needs arbitrary CONFIG_NAMEs (E2M1_optimal_none, E8M1_amax_E4M3, ...)
for its scale/exponent/mantissa axis decomposition, not just plain/H5.
The Quantizer class is a convenience for the common plain/H5 case used
by H2-H5's PPL scripts; build ad-hoc callables directly (with
quantize_p_causal for P) for anything else.

llama/mistral share one eager-attention code shape (RoPE + repeat_kv);
OPT uses a different bmm-based shape (no RoPE, tuple past_key_value).
Requires attn_implementation="eager" -- SDPA never materializes P.
"""

import os
import sys
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formats import quantize_tensor_config, quantize_tensor_grouped_boost, quantize_tensor_topk_activation_e2m7

BLOCK_SIZE = {"nvfp4": 16, "mxfp4": 32}
BASE_CONFIG = {"nvfp4": "E2M1_amax_E4M3", "mxfp4": "E2M1_mxfp4_ue8m0"}
BOOST_CONFIG = {"nvfp4": "E2M3_amax_E4M3", "mxfp4": "E2M3_mxfp4_ue8m0"}
H5_GROUP_SIZE = 8


def should_skip_linear(name: str) -> bool:
    return name == "lm_head" or name.endswith(".lm_head")


# ---------------------------------------------------------------- weight ---
#
# "Weight Quantization" always means loading this real GPTQ-calibrated
# cache (Cholesky sequential error compensation), never naive round-to-
# nearest -- established convention from hypothesis_v2's
# measure_weight_vs_activation.py / P_target_selection/quant_targets.py.
# Caveat inherited from there: these caches were built with an older
# single-level (pre-tensor-scale-s_enc) NVFP4 quantizer.

CACHE_PATH_TEMPLATE = os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "{fmt}/{model_key}/ptq_cache_NVFP4.pt")


@torch.no_grad()
def load_weights_from_cache(model: nn.Module, model_key: str, fmt: str) -> int:
    cache_path = CACHE_PATH_TEMPLATE.format(fmt=fmt, model_key=model_key)
    cache = torch.load(cache_path, map_location="cpu")
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    count = 0
    for name, w in cache.items():
        if name in modules:
            modules[name].weight.data.copy_(w.to(device, dtype))
            count += 1
    return count


# ------------------------------------------------------------------- P ---
#
# All three functions below are fully vectorized (no Python loop over
# blocks) -- an earlier per-block-loop version (128 blocks x 32 layers x
# 64 PPL steps = 262k+ iterations) took ~5 minutes per model per arm,
# impractical across 3 models x 2 formats x many arms. The key insight:
# quantize_tensor_config's per-block scale (amax over just that block's
# own elements) is already computed independently per row/block
# regardless of what else is in the same call, so quantizing the WHOLE
# [q_len, k_len] tensor in one call and then masking to the causal-valid
# positions afterward gives bit-identical per-block results to the old
# loop -- only the two-level scheme's tensor-wide s_enc scope changes
# (now the true whole-P-matrix amax in one shot, arguably more faithful
# to NVFP4's real per-tensor global scale than a tiny per-slice s_enc).

QUERY_CHUNK = 512  # bounds peak memory (a full [heads,2048,2048] fp32/fp64
                    # P matrix OOM'd an 8B model at seq_len=2048); still
                    # 4x fewer iterations than 2048/QUERY_CHUNK would need
                    # if it matched the old block_size=16 loop's 128 steps.


def _causal_block_mask(q_len: int, k_len: int, block_size: int, row_offset: int, device) -> torch.Tensor:
    """mask[r, c] = True iff key-block containing column c is one a query
    row (row_offset + r) has FULLY seen causally. Matches the original
    per-block loop exactly: the trailing partial causal block for every
    row is excluded (mask False)."""
    rows = (torch.arange(q_len, device=device) + row_offset).view(q_len, 1)
    cols = torch.arange(k_len, device=device).view(1, k_len)
    n_blocks = k_len // block_size
    block_of_col = cols // block_size
    valid = (block_of_col + 1) * block_size - 1 <= rows
    valid = valid & (cols < n_blocks * block_size)
    return valid.view(1, 1, q_len, k_len)


def _chunked_query(attn_weights: torch.Tensor, block_size: int, fn) -> torch.Tensor:
    """Applies fn(chunk, row_offset) over the query dimension in pieces
    of QUERY_CHUNK rows, concatenating results -- bounds peak memory
    while keeping the vectorized (no per-16-block loop) speedup."""
    q_len = attn_weights.shape[2]
    if q_len <= QUERY_CHUNK:
        return fn(attn_weights, 0)
    parts = []
    for start in range(0, q_len, QUERY_CHUNK):
        end = min(start + QUERY_CHUNK, q_len)
        parts.append(fn(attn_weights[:, :, start:end, :], start))
    return torch.cat(parts, dim=2)


@torch.no_grad()
def quantize_p_causal(attn_weights: torch.Tensor, config_name: str, block_size: int) -> torch.Tensor:
    """attn_weights: [bsz, heads, q_len, k_len], post-softmax. For each
    key-block b (width block_size), only query rows that have fully seen
    that block causally get it quantized; every row's trailing partial
    causal block stays BF16 (deliberately, matching the original H0
    P-only isolation work). Generic single-config version -- no boost."""
    k_len = attn_weights.shape[3]

    def fn(chunk, row_offset):
        full = quantize_tensor_config(chunk.detach().to(torch.float32), config_name, block_size).to(chunk.dtype)
        mask = _causal_block_mask(chunk.shape[2], k_len, block_size, row_offset, chunk.device)
        return torch.where(mask, full, chunk)

    return _chunked_query(attn_weights, block_size, fn)


@torch.no_grad()
def quantize_p_causal_full(attn_weights: torch.Tensor, config_name: str, block_size: int) -> torch.Tensor:
    """Like quantize_p_causal, but quantizes every causal-valid element (col <= row),
    including the trailing partial block that quantize_p_causal leaves at BF16. No
    special per-row scale handling is needed for the partial block: quantize_tensor_config's
    per-block amax (formats.py's quantize_blocks_config, max over dim=-1) already only
    reflects the causal-valid entries, since the future entries in that same block are
    already exactly 0 (post-softmax causal masking) and can't exceed a nonzero valid
    amax -- and 0 is exactly representable in every FP format here, so those entries
    quantize back to exactly 0. Only the MASK criterion changes vs quantize_p_causal
    (col <= row instead of "whole block already in the past")."""
    k_len = attn_weights.shape[3]

    def fn(chunk, row_offset):
        full = quantize_tensor_config(chunk.detach().to(torch.float32), config_name, block_size).to(chunk.dtype)
        q_len = chunk.shape[2]
        rows = (torch.arange(q_len, device=chunk.device) + row_offset).view(1, 1, q_len, 1)
        cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
        mask = cols <= rows
        return torch.where(mask, full, chunk)

    return _chunked_query(attn_weights, block_size, fn)


@torch.no_grad()
def quantize_p_causal_grouped_boost(attn_weights: torch.Tensor, base_config: str, boost_config: str,
                                     block_size: int, group_size: int) -> torch.Tensor:
    """Same causal-block gating as quantize_p_causal, but each qualifying
    block gets the grouped-local-max boost (1 position per group_size
    boosted to boost_config, selected by group-local argmax squared
    quantization error) instead of plain base_config."""
    k_len = attn_weights.shape[3]
    n_blocks = k_len // block_size
    usable = n_blocks * block_size
    num_groups = block_size // group_size

    def fn(chunk, row_offset):
        chunk32 = chunk.detach().to(torch.float32)
        base = quantize_tensor_config(chunk32, base_config, block_size).to(chunk.dtype)
        boosted = quantize_tensor_config(chunk32, boost_config, block_size).to(chunk.dtype)
        shape = chunk.shape

        diff_sq = (chunk[..., :usable].to(torch.float32) - base[..., :usable].to(torch.float32)).pow(2)
        diff_grouped = diff_sq.reshape(*shape[:-1], n_blocks, num_groups, group_size)
        local_idx = diff_grouped.argmax(dim=-1, keepdim=True)
        group_mask = torch.zeros_like(diff_grouped, dtype=torch.bool).scatter_(-1, local_idx, True)
        group_mask = group_mask.reshape(*shape[:-1], usable)

        full_group_mask = torch.zeros(shape, dtype=torch.bool, device=chunk.device)
        full_group_mask[..., :usable] = group_mask
        selected = torch.where(full_group_mask, boosted, base)

        causal_mask = _causal_block_mask(shape[2], k_len, block_size, row_offset, chunk.device)
        return torch.where(causal_mask, selected, chunk)

    return _chunked_query(attn_weights, block_size, fn)


@torch.no_grad()
def quantize_p_causal_topk_boost(attn_weights: torch.Tensor, base_config: str, boost_config: str,
                                  block_size: int, k: int) -> torch.Tensor:
    """H3's EXACT top-k (global sort within the block, not group-local),
    same causal-block gating as quantize_p_causal."""
    k_len = attn_weights.shape[3]
    n_blocks = k_len // block_size
    usable = n_blocks * block_size

    def fn(chunk, row_offset):
        chunk32 = chunk.detach().to(torch.float32)
        base = quantize_tensor_config(chunk32, base_config, block_size).to(chunk.dtype)
        boosted = quantize_tensor_config(chunk32, boost_config, block_size).to(chunk.dtype)
        shape = chunk.shape

        diff_sq = (chunk[..., :usable].to(torch.float32) - base[..., :usable].to(torch.float32)).pow(2)
        diff_blocks = diff_sq.reshape(*shape[:-1], n_blocks, block_size)
        order = diff_blocks.argsort(dim=-1, descending=True)
        topk_mask = torch.zeros_like(diff_blocks, dtype=torch.bool)
        topk_mask.scatter_(-1, order[..., :k], True)
        topk_mask = topk_mask.reshape(*shape[:-1], usable)

        full_topk_mask = torch.zeros(shape, dtype=torch.bool, device=chunk.device)
        full_topk_mask[..., :usable] = topk_mask
        selected = torch.where(full_topk_mask, boosted, base)

        causal_mask = _causal_block_mask(shape[2], k_len, block_size, row_offset, chunk.device)
        return torch.where(causal_mask, selected, chunk)

    return _chunked_query(attn_weights, block_size, fn)


# --------------------------------------------------------- plain/H5 fmt ---

class Quantizer:
    """Convenience bundle of plain/H5 quant functions for one fmt
    ("nvfp4"/"mxfp4"), used by H2-H5's PPL scripts. H1 (arbitrary
    CONFIG_NAME per axis) builds its own callables directly instead."""

    def __init__(self, fmt: str):
        assert fmt in ("nvfp4", "mxfp4")
        self.fmt = fmt
        self.block_size = BLOCK_SIZE[fmt]
        self.base_config = BASE_CONFIG[fmt]
        self.boost_config = BOOST_CONFIG[fmt]

    def fn(self, mode: str):
        """Returns a tensor->tensor callable for Linear input, Q, K or V."""
        return {"plain": self.plain, "h5": self.h5}[mode]

    def plain(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_tensor_config(x.detach().to(torch.float32), self.base_config, self.block_size).to(x.dtype)

    def h5(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_tensor_grouped_boost(x.detach().to(torch.float32), H5_GROUP_SIZE,
                                              block_size=self.block_size, base_config=self.base_config,
                                              boost_config=self.boost_config).to(x.dtype)

    def p_fn(self, mode: str):
        """Returns an attn_weights->attn_weights callable for P (causal)."""
        if mode == "plain":
            return lambda aw: quantize_p_causal(aw, self.base_config, self.block_size)
        return lambda aw: quantize_p_causal_grouped_boost(aw, self.base_config, self.boost_config,
                                                           self.block_size, H5_GROUP_SIZE)


# --------------------------------------------------------------- Linear ---

class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, quant_fn) -> None:
        super().__init__()
        self.linear = linear
        self.quant_fn = quant_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.quant_fn(x))


def replace_linear_modules(module: nn.Module, quant_fn) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not should_skip_linear(child_name):
            setattr(module, child_name, QuantLinear(child, quant_fn))
            replaced += 1
        else:
            replaced += replace_linear_modules(child, quant_fn)
    return replaced


# -------------------------------------------------- attention (llama/mistral) ---

def patch_eager_ropegqa(mod, q_fn, k_fn, v_fn, p_fn):
    """mod: transformers.models.llama.modeling_llama or ...mistral.modeling_mistral.
    q_fn/k_fn/v_fn: tensor -> tensor. p_fn: attn_weights -> attn_weights
    (post-softmax, already causal-aware)."""
    attn_cls = mod.LlamaAttention if hasattr(mod, "LlamaAttention") else (
        mod.MistralAttention if hasattr(mod, "MistralAttention") else mod.MinistralAttention)
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        # transformers >= ~4.49 dropped self.num_heads/self.num_key_value_heads from
        # the attention module (only self.head_dim/self.num_key_value_groups remain) --
        # read the head counts from self.config instead. Found when bringing this
        # project up on a fresh "super" server with a much newer transformers than the
        # rtx4090 box (4.44.2) this was written against; see HANDOFF_MOE_UPGRADE.md.
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

        qs, ks = q_fn(query_states), k_fn(key_states)
        attn_weights = torch.matmul(qs, ks.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        ps = p_fn(attn_weights)
        vs = v_fn(value_states)
        attn_weights = nn.functional.dropout(ps, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, vs)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # transformers >= ~4.49's decoder layer does `hidden_states, _ = self.self_attn(...)`
        # -- a 2-tuple, not the old 3-tuple with past_key_value (caching now goes through
        # the Cache object's own .update(), never the return value). Returning 3 here makes
        # that unpack raise ValueError: too many values to unpack.
        return attn_output, None

    attn_cls.forward = patched_forward
    return attn_cls, orig_forward


# ------------------------------------------------------------- attention (opt) ---

def patch_eager_opt(mod, q_fn, k_fn, v_fn, p_fn):
    """OPT: Q@K^T -> softmax -> P -> P@V exists exactly like llama/mistral --
    it's a different CODE SHAPE (torch.bmm on [bsz*heads, ...]-flattened
    tensors, tuple-based past_key_value, no RoPE), not a different
    mechanism."""
    orig_forward = mod.OPTAttention.forward

    def patched_forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None,
                         layer_head_mask=None, output_attentions=False, **kwargs):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        qs = q_fn(query_states.view(bsz, self.num_heads, tgt_len, self.head_dim)).view(*proj_shape)
        ks = k_fn(key_states.view(bsz, self.num_heads, src_len, self.head_dim)).view(*proj_shape)
        attn_weights = torch.bmm(qs, ks.transpose(1, 2))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_4d = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        ps = p_fn(attn_weights_4d).view(bsz * self.num_heads, tgt_len, src_len)
        vs = v_fn(value_states.view(bsz, self.num_heads, src_len, self.head_dim)).view(*proj_shape)

        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None
        attn_probs = nn.functional.dropout(ps, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, vs)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    mod.OPTAttention.forward = patched_forward
    return mod.OPTAttention, orig_forward


def patch_eager(arch: str, mod, q_fn, k_fn, v_fn, p_fn):
    if arch == "opt":
        return patch_eager_opt(mod, q_fn, k_fn, v_fn, p_fn)
    return patch_eager_ropegqa(mod, q_fn, k_fn, v_fn, p_fn)


# ------------------------------------------------- decomposed (no delta x delta) ---
#
# QK^T and P@V are activation x activation matmuls -- when BOTH operands
# are boosted at the same element position (which happens routinely:
# Q's and K's top-k/grouped selections are independent, so ~k^2/block_size
# positions overlap on average), naively multiplying the two boosted
# tensors computes (a_base+da)(b_base+db) = a_base*b_base + a_base*db +
# da*b_base + da*db. Real hardware's correction datapath only ever
# multiplies a base-precision value against a delta (never delta x
# delta -- there is no boosted x boosted multiplier), so da*db must be
# dropped from the simulation too. Algebraically, dropping da*db:
#   a_base@b_boost^T + a_boost@b_base^T - a_base@b_base^T
# reproduces exactly a_base*b_base + a_base*db + da*b_base (the base@base
# term appears in both of the first two products, so it's subtracted
# back out once). Linear's activation x WEIGHT matmul never needs this
# -- weight is exact BF16 (da_weight = 0 always), so a_boost@W is
# already exact with no da*db term to drop.

def patch_eager_ropegqa_decomposed(mod, base_fns: dict, boost_fns: dict):
    """base_fns/boost_fns: {"q":fn, "k":fn, "v":fn, "p":fn} -- q/k/v are
    tensor->tensor, p is attn_weights->attn_weights (post-softmax,
    causal-aware). See module-level comment above for the decomposition."""
    attn_cls = mod.LlamaAttention if hasattr(mod, "LlamaAttention") else (
        mod.MistralAttention if hasattr(mod, "MistralAttention") else mod.MinistralAttention)
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        # see patch_eager_ropegqa's comment -- self.num_heads/num_key_value_heads no
        # longer exist on newer transformers' attention modules, read from config.
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

        q_base, q_boost = base_fns["q"](query_states), boost_fns["q"](query_states)
        k_base, k_boost = base_fns["k"](key_states), boost_fns["k"](key_states)
        attn_weights = (torch.matmul(q_base, k_boost.transpose(2, 3))
                         + torch.matmul(q_boost, k_base.transpose(2, 3))
                         - torch.matmul(q_base, k_base.transpose(2, 3))) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        p_base, p_boost = base_fns["p"](attn_weights), boost_fns["p"](attn_weights)
        v_base, v_boost = base_fns["v"](value_states), boost_fns["v"](value_states)
        attn_output = (torch.matmul(p_base, v_boost) + torch.matmul(p_boost, v_base)
                       - torch.matmul(p_base, v_base))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # transformers >= ~4.49's decoder layer does `hidden_states, _ = self.self_attn(...)`
        # -- a 2-tuple, not the old 3-tuple with past_key_value (caching now goes through
        # the Cache object's own .update(), never the return value). Returning 3 here makes
        # that unpack raise ValueError: too many values to unpack.
        return attn_output, None

    attn_cls.forward = patched_forward
    return attn_cls, orig_forward


def patch_eager_opt_decomposed(mod, base_fns: dict, boost_fns: dict):
    orig_forward = mod.OPTAttention.forward

    def patched_forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None,
                         layer_head_mask=None, output_attentions=False, **kwargs):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states_4d = self._shape(query_states, tgt_len, bsz)
        key_states_4d = key_states.view(bsz, self.num_heads, -1, self.head_dim)
        src_len = key_states_4d.shape[2]

        q_base = base_fns["q"](query_states_4d).view(*proj_shape)
        q_boost = boost_fns["q"](query_states_4d).view(*proj_shape)
        k_base = base_fns["k"](key_states_4d).view(*proj_shape)
        k_boost = boost_fns["k"](key_states_4d).view(*proj_shape)
        attn_weights = (torch.bmm(q_base, k_boost.transpose(1, 2))
                         + torch.bmm(q_boost, k_base.transpose(1, 2))
                         - torch.bmm(q_base, k_base.transpose(1, 2)))

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_probs_4d = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)

        p_proj_shape = (bsz * self.num_heads, tgt_len, src_len)
        p_base = base_fns["p"](attn_probs_4d).view(*p_proj_shape)
        p_boost = boost_fns["p"](attn_probs_4d).view(*p_proj_shape)
        value_states_4d = value_states.view(bsz, self.num_heads, src_len, self.head_dim)
        v_base = base_fns["v"](value_states_4d).view(*proj_shape)
        v_boost = boost_fns["v"](value_states_4d).view(*proj_shape)
        attn_output = (torch.bmm(p_base, v_boost) + torch.bmm(p_boost, v_base) - torch.bmm(p_base, v_base))

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    mod.OPTAttention.forward = patched_forward
    return mod.OPTAttention, orig_forward


def patch_eager_decomposed(arch: str, mod, base_fns: dict, boost_fns: dict):
    if arch == "opt":
        return patch_eager_opt_decomposed(mod, base_fns, boost_fns)
    return patch_eager_ropegqa_decomposed(mod, base_fns, boost_fns)


MODEL_CONFIG = {
    "llama": dict(model_name="meta-llama/Llama-3.1-8B-Instruct", arch="ropegqa"),
    "llama_base": dict(model_name="meta-llama/Llama-3.1-8B", arch="ropegqa"),
    "opt": dict(model_name="facebook/opt-6.7b", arch="opt"),  # dropped from active lineup, kept for regression reference
    "mistral": dict(model_name="mistralai/Mistral-7B-v0.3", arch="ropegqa"),  # dropped from active lineup, kept for regression reference
    # dropped from active lineup in favor of a second MoE data point (Phi-3.5-MoE-instruct,
    # ~6.6B active vs Llama's 8B dense -- same-active-compute dense-vs-MoE comparison alongside
    # Qwen3-30B-A3B's larger/sparser MoE) -- NOT because of a bug; its NVFP4 down_proj catastrophic-
    # block issue was found and fixed (see weight_gptq.py's GPTQ_COND_THRESHOLD docstring), and its
    # per-layer weight-quant error distribution is statistically indistinguishable from Llama's.
    # Code and existing results kept for regression reference, no new experiments planned.
    "ministral": dict(model_name="mistralai/Ministral-8B-Instruct-2410", arch="ropegqa"),
    # final 3-model lineup's dense slot (alongside qwen3_moe and olmoe, both MoE) -- confirmed
    # text-only (MistralForCausalLM, no vision_config/text_config wrapper) unlike Ministral-3-8B/
    # Ministral-3-14B-Base-2512 and Gemma-3-27B, which turned out to be VLM wrappers requiring
    # extra .language_model submodule extraction. "mistral" in the name routes attn_mod_for to
    # transformers.models.mistral.modeling_mistral automatically -- no new patch file needed.
    "mistral_small_24b": dict(model_name="mistralai/Mistral-Small-24B-Base-2501", arch="ropegqa"),
}
