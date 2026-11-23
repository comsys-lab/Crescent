"""H3 (v4), Qwen3 MoE variant -- can the H2 top-k error ratio methodology
run on Qwen3-30B-A3B (qwen3_moe architecture, 128 routed experts, top-8,
NO shared expert -- unlike Qwen1.5-MoE/qwen2_moe's 60 experts + 1 shared
expert, see real_moe.py)? 30.5B total params (~3.3B active/token) -- does
NOT fit this box's 24GB-per-GPU budget in BF16 (needs ~61GB); loaded with
device_map="auto" across the free GPUs only (CUDA_VISIBLE_DEVICES must be
set by the caller to exclude any GPU in use by someone else -- see
HANDOFF_MOE_UPGRADE.md).

Two structural differences from real_moe.py (Qwen1.5-MoE), both found by
reading the actual transformers source for this model before writing this
file (per HANDOFF_MOE_UPGRADE.md's standing gotcha about not assuming a
calling convention carries over):

1. Qwen3MoeAttention has QK-Norm: q_proj/k_proj outputs are each passed
   through a per-head RMSNorm (self.q_norm/self.k_norm) BEFORE RoPE is
   applied. Llama/Mistral/Ministral have no such step. Naively reusing
   quant_targets.patch_eager_ropegqa's reshape+RoPE logic here would probe
   Q/K quantization error on the PRE-norm projection, which never appears
   anywhere in the real model's actual attention math (self.q_norm output
   is a completely different tensor) -- a silent correctness bug, not a
   crash. patch_eager_qwen3_moe below applies q_norm/k_norm in the same
   place the real Qwen3MoeAttention.forward does.

2. The FFN experts are NOT individual nn.Linear modules in a ModuleList
   (the old qwen2_moe shape real_moe.py's iter_target_linears assumes).
   Qwen3MoeExperts stores gate_up_proj/down_proj as raw batched
   nn.Parameter tensors (shape [num_experts, ...]) and computes each
   active expert's matmul via nn.functional.linear(x, self.gate_up_proj
   [expert_idx]) inside a Python loop over only the experts that got any
   tokens routed to them this batch. There is no nn.Linear instance to
   register_forward_pre_hook on -- iter_target_linears would silently find
   zero Linears for the gateup_proj/down_proj categories (Q/K/V/O are
   still plain nn.Linear on Qwen3MoeAttention, those categories are
   unaffected). patch_qwen3_moe_experts below reimplements
   Qwen3MoeExperts.forward with the same per-expert-hit loop, probing
   "current_state" (the fused gate+up matmul's input -- already the
   correct single input for a "gateup_proj" category, since Qwen3's
   gate_up_proj is pre-fused into one matmul rather than two separate
   gate_proj/up_proj Linears sharing an input) and "current_hidden_states"
   (down_proj's input, post-SwiGLU).

The router (Qwen3MoeTopKRouter) is also a raw nn.Parameter + F.linear, not
an nn.Linear -- unlike qwen2_moe's mlp.gate/shared_expert_gate (which WERE
nn.Linear and needed explicit exclusion in should_skip_linear), it is
automatically invisible to iter_target_linears; no exclusion list needed
here.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs only> python3 real_moe_qwen3.py <nvfp4|mxfp4>
"""
import csv
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config

torch.manual_seed(1111)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def should_skip_linear(name: str) -> bool:
    return name == "lm_head" or name.endswith(".lm_head")


CATEGORY_SUFFIXES = [
    ("q_proj", "qkv_proj"), ("k_proj", "qkv_proj"), ("v_proj", "qkv_proj"),
    ("o_proj", "o_proj"),
]


def categorize(name: str) -> str:
    for suffix, category in CATEGORY_SUFFIXES:
        if name.endswith(suffix):
            return category
    return "other"


def iter_target_linears(model: nn.Module):
    """Only Q/K/V/O -- gateup_proj/down_proj are NOT nn.Linear here (see
    module docstring), probed inline from inside patch_qwen3_moe_experts
    instead."""
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        category = categorize(n)
        if category == "other":
            continue
        targets.append((n, m, category))
    return targets


@torch.no_grad()
def topk_error_ratio_of(x: torch.Tensor, block_size: int, base_config: str, k_values: list) -> dict:
    K = x.shape[-1]
    num_blocks = K // block_size
    if num_blocks == 0:
        return None
    x = x.reshape(-1, K)[:, : num_blocks * block_size].to(torch.float32)
    xq = quantize_tensor_config(x, base_config, block_size)
    eps = (x - xq).reshape(-1, block_size)
    eps_sq = eps.pow(2)
    eps_sq_sorted, _ = eps_sq.sort(dim=-1, descending=True)
    cum = eps_sq_sorted.cumsum(dim=-1)
    total_raw = cum[..., -1:]
    valid = (total_raw.squeeze(-1) > 0)
    if valid.sum() == 0:
        return None
    total = total_raw.clamp(min=1e-30)
    ratio_topk = (100 * cum / total)[valid].mean(dim=0)
    return {k: ratio_topk[k - 1].item() for k in k_values}


def accumulate(cat_sum: dict, cat_n: dict, category: str, res: dict, k_values: list):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {k: 0.0 for k in [0] + k_values}
        cat_n[category] = 0
    for k in k_values:
        cat_sum[category][k] += res[k]
    cat_n[category] += 1


def identity(x):
    return x


def patch_eager_qwen3_moe(mod, q_fn, k_fn, v_fn, p_fn):
    """Qwen3MoeAttention: config-based head counts (self.num_heads doesn't
    exist on this transformers version's attention modules -- see
    quant_targets.patch_eager_ropegqa's comment), QK-Norm applied to Q/K
    right after projection+reshape and BEFORE RoPE (the real forward's
    order: view -> q_norm/k_norm -> transpose -> RoPE), 2-tuple return
    (this transformers version's decoder layer does `hidden_states, _ =
    self.self_attn(...)`, not the old 3-tuple with past_key_value)."""
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


def patch_qwen3_moe_experts(mod, gateup_probe, down_probe):
    """Qwen3MoeExperts has no nn.Linear submodules (gate_up_proj/down_proj
    are batched nn.Parameter tensors, matmul'd via F.linear inside a
    Python loop over active experts only) -- reimplements that loop
    verbatim from the real forward (see module docstring), calling the
    probes on exactly the tensors that feed each expert's two matmuls.
    Purely a probe (identity passthrough); unlike the attention patches
    this project also uses for real PPL boosting, this one never needs a
    boosted/quantized variant since H2 only measures topk-error-ratio."""
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
            gateup_probe(current_state)
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            down_probe(current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    cls.forward = patched_forward
    return cls, orig_forward


@torch.no_grad()
def measure_model(fmt: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config = cfg["block_size"], cfg["base_config"]
    k_values = list(range(1, block_size + 1))

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(MODEL_NAME, full_text)

    cat_sum, cat_n = {}, {}

    def q_probe(x):
        accumulate(cat_sum, cat_n, "Q", topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)
        return x

    def k_probe(x):
        accumulate(cat_sum, cat_n, "K", topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)
        return x

    def gateup_probe(x):
        accumulate(cat_sum, cat_n, "gateup_proj", topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)

    def down_probe(x):
        accumulate(cat_sum, cat_n, "down_proj", topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)

    # Patch BEFORE from_pretrained(device_map="auto") -- accelerate's
    # dispatch_model captures each module's original forward at load time;
    # patching the class afterward never takes effect (see HANDOFF_MOE_
    # UPGRADE.md's second gotcha, already hit once by real_moe.py).
    import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
    attn_cls, orig_attn_forward = patch_eager_qwen3_moe(attn_mod, q_probe, k_probe, identity, identity)
    experts_cls, orig_experts_forward = patch_qwen3_moe_experts(attn_mod, gateup_probe, down_probe)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager",
                                                   device_map="auto")
    model.eval()

    targets = iter_target_linears(model)
    print(f"Hooking {len(targets)} Linear modules (Q/K/V/O only; "
          f"gateup_proj/down_proj probed inline from Qwen3MoeExperts.forward)", flush=True)

    def make_hook(category):
        def hook(_mod, inputs):
            accumulate(cat_sum, cat_n, category, topk_error_ratio_of(inputs[0].detach(), block_size, base_config, k_values), k_values)
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    attn_cls.forward = orig_attn_forward
    experts_cls.forward = orig_experts_forward
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {0: 0.0}
        avg.update({k: cat_sum[category][k] / cat_n[category] for k in k_values})
        print(f"[{fmt}][qwen3_moe] {category} ({cat_n[category]}): " + " ".join(f"top{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    by_category = measure_model(fmt)

    csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}_qwen3_moe.csv")
    categories = sorted(by_category.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "k"] + categories)
        w.writeheader()
        for k in [0] + k_values:
            row = {"model": "qwen3_moe", "k": k}
            for category in categories:
                row[category] = by_category[category].get(k, "")
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H2_QWEN3_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
