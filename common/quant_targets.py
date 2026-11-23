"""Activation-quantization hooks for the dense model (Llama-3.1-8B, eager attention).

  replace_linear_modules : wraps every nn.Linear except lm_head so its input goes through quant_fn.
  patch_eager            : patches LlamaAttention.forward with per-operand callables q_fn, k_fn, v_fn
                           (tensor -> tensor) and p_fn (post-softmax attention probabilities).
  patch_eager_decomposed : same, but QK^T and P@V are computed as base x selected + selected x base -
                           base x base, i.e. the selected x selected delta term is dropped, as in
                           Crescent's correction datapath.
  quantize_p_causal_full : block quantization of P on causal positions only.

Requires attn_implementation="eager" (SDPA never materializes P). The MoE models have their own patches
in qwen3_moe_targets.py and deepseek_moe_model.py.
"""

import os
import sys
import math

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formats import quantize_tensor_config, quantize_tensor_grouped_boost, quantize_tensor_topk_activation_e2m7



def should_skip_linear(name: str) -> bool:
    return name == "lm_head" or name.endswith(".lm_head")




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
    causal block stays BF16. Single config, no selection."""
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


# ------------------------------------------------------------ attention ---

def patch_eager_ropegqa(mod, q_fn, k_fn, v_fn, p_fn):
    """mod: transformers.models.llama.modeling_llama.
    q_fn/k_fn/v_fn: tensor -> tensor. p_fn: attn_weights -> attn_weights
    (post-softmax, already causal-aware)."""
    attn_cls = mod.LlamaAttention
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        # transformers >= ~4.49 dropped self.num_heads/self.num_key_value_heads from
        # the attention module (only self.head_dim/self.num_key_value_groups remain) --
        # read the head counts from self.config instead. Found when bringing this
        # project up on a fresh "super" server with a much newer transformers than the
        # rtx4090 box (4.44.2) this was written against.
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

def patch_eager(arch: str, mod, q_fn, k_fn, v_fn, p_fn):
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
    attn_cls = mod.LlamaAttention
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


def patch_eager_decomposed(arch: str, mod, base_fns: dict, boost_fns: dict):
    return patch_eager_ropegqa_decomposed(mod, base_fns, boost_fns)


MODEL_CONFIG = {  # dense model; the MoE models are loaded by qwen3_moe_targets.py / deepseek_moe_model.py
    "llama_base": dict(model_name="meta-llama/Llama-3.1-8B", arch="ropegqa"),
}
