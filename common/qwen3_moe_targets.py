"""Shared Qwen3-30B-A3B (qwen3_moe) monkeypatch infra -- the MoE analog of
quant_targets.py's patch_eager_ropegqa/patch_eager_ropegqa_decomposed, used
by every Hn hypothesis script's *_moe_qwen3.py variant (H2's real_moe_qwen3.py,
H3/H4/H5's real_moe_qwen3.py + ppl_moe_qwen3.py).

Two real differences from quant_targets.py's dense-model patches (found by
reading the actual transformers source -- see HANDOFF_MOE_UPGRADE.md):

1. Qwen3MoeAttention has QK-Norm (self.q_norm/self.k_norm, a per-head
   RMSNorm) applied to Q/K right after the reshape, BEFORE RoPE -- no
   analog in Llama/Mistral/Ministral.
2. The FFN experts (Qwen3MoeExperts) are NOT nn.Linear modules -- gate_up_proj/
   down_proj are raw nn.Parameter tensors of shape [num_experts, ...], matmul'd
   per active expert via nn.functional.linear(x, self.gate_up_proj[expert_idx])
   inside a Python loop. There is nothing to register_forward_pre_hook on or
   wrap via quant_targets.replace_linear_modules -- patch_qwen3_moe_experts
   reimplements that loop, routing the two tensors that feed each expert's
   matmuls (current_state for gateup_proj, current_hidden_states for
   down_proj) through caller-supplied functions. Since gate_up_proj/down_proj
   are exact BF16 weights (this project only ever quantizes activations),
   this is a single-sided quantize -- exactly like quant_targets.
   replace_linear_modules's QuantLinear wrapping for dense models' Linear
   categories, no error-compensation decomposition needed (that's only for
   activation x activation matmuls -- QK^T here, not activation x weight).

Q/K/V/O_proj ARE plain nn.Linear on Qwen3MoeAttention (confirmed via
inspect.getsource), so quant_targets.replace_linear_modules already handles
the qkv_proj/o_proj Linear-input categories correctly for this model with NO
changes -- only attention's Q/K tensors (post-projection) and the expert FFN
need the patches in this file.
"""
import torch
import torch.nn as nn

MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def patch_eager_qwen3_moe(mod, q_fn, k_fn, v_fn, p_fn):
    """Plain (non-decomposed) Qwen3MoeAttention patch -- q_fn/k_fn/v_fn:
    tensor->tensor, p_fn: attn_weights->attn_weights (post-softmax). Use for
    probe-only (H2 style, q_fn/k_fn are identity-returning side-effect
    probes) or single-precision-level arms (e.g. the *_ALL plain arm, no
    base/boost split needed since there's no delta x delta cross term to
    drop)."""
    attn_cls = mod.Qwen3MoeAttention
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                         past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        assert query_states.shape[1] == num_heads and key_states.shape[1] == num_key_value_heads

        cos, sin = position_embeddings
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

        qs, ks = q_fn(query_states), k_fn(key_states)
        attn_weights = torch.matmul(qs, ks.transpose(2, 3)) * self.scaling
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

        return attn_output, None

    attn_cls.forward = patched_forward
    return attn_cls, orig_forward


def patch_eager_qwen3_moe_decomposed(mod, base_fns: dict, boost_fns: dict):
    """Decomposed (error-compensation) Qwen3MoeAttention patch -- base_fns/
    boost_fns: {"q":fn, "k":fn, "v":fn, "p":fn}. QK^T (activation x
    activation) drops the delta x delta cross term exactly like
    quant_targets.patch_eager_ropegqa_decomposed -- see that function's
    module-level comment in quant_targets.py for the derivation. Adds
    QK-Norm (before RoPE) on top of that, which plain ropegqa doesn't have."""
    attn_cls = mod.Qwen3MoeAttention
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                         past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        assert query_states.shape[1] == num_heads and key_states.shape[1] == num_key_value_heads

        cos, sin = position_embeddings
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

        q_base, q_boost = base_fns["q"](query_states), boost_fns["q"](query_states)
        k_base, k_boost = base_fns["k"](key_states), boost_fns["k"](key_states)
        attn_weights = (torch.matmul(q_base, k_boost.transpose(2, 3))
                         + torch.matmul(q_boost, k_base.transpose(2, 3))
                         - torch.matmul(q_base, k_base.transpose(2, 3))) * self.scaling
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

        return attn_output, None

    attn_cls.forward = patched_forward
    return attn_cls, orig_forward


def patch_qwen3_moe_experts(mod, gateup_fn, down_fn):
    """Reimplements Qwen3MoeExperts.forward's per-active-expert loop
    verbatim, routing the two tensors that feed each expert's matmuls
    through gateup_fn/down_fn (identity-returning probes for MSE-only
    measurement scripts, or an actual quantize/boost function for ppl.py
    scripts -- both use cases share this one function, see module
    docstring). No error-compensation decomposition: gate_up_proj/
    down_proj are exact BF16 weights, never quantized, so this is a
    single-sided quantize exactly like a dense model's Linear category."""
    cls = mod.Qwen3MoeExperts
    orig_forward = cls.forward

    def patched_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(gateup_fn(current_state), self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(down_fn(current_hidden_states), self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    cls.forward = patched_forward
    return cls, orig_forward
