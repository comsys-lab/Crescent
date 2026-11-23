"""H2 (v4), MoE variant -- exploratory: can the H2 top-k error ratio
methodology run on a Qwen MoE model? Qwen/Qwen1.5-MoE-A2.7B-Chat (qwen2_moe
architecture, 60 routed experts + 1 shared expert per layer, transformers
4.44.2's Qwen2MoeAttention -- an OLDER rotary-embedding calling convention
than llama/mistral's, so this needs its own attention patch, not
quant_targets.patch_eager_ropegqa). 14.3B total params (2.7B active/token)
-- doesn't fit one 24GB GPU in BF16, loaded with device_map="auto" across
both GPUs instead (so this script cannot run alongside another model on the
other GPU).

Differences from real.py (dense models):
  - iter_target_linears does NOT dedup by (layer, category) -- H2's dense
    dedup assumes one Linear per category per layer, but MoE has 60
    experts + 1 shared expert all sharing the same category per layer.
    Every individual expert Linear gets its own hook here (each expert
    only sees the subset of tokens actually routed to it, so no
    double-counting).
  - the router gate Linears (mlp.gate: hidden->num_experts,
    mlp.shared_expert_gate: hidden->1) are excluded -- they're routing
    logits, not part of the FFN matmul pathway this project quantizes.

Usage: python3 real_moe.py <nvfp4|mxfp4>
"""
import csv
import math
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config

torch.manual_seed(1111)

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048

CATEGORY_SUFFIXES = [
    ("q_proj", "qkv_proj"), ("k_proj", "qkv_proj"), ("v_proj", "qkv_proj"),
    ("o_proj", "o_proj"),
    ("gate_proj", "gateup_proj"), ("up_proj", "gateup_proj"),
    ("down_proj", "down_proj"),
]
LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def should_skip_linear(name: str) -> bool:
    return (name == "lm_head" or name.endswith(".lm_head")
            or name.endswith(".mlp.gate") or name.endswith(".shared_expert_gate"))


def categorize(name: str) -> str:
    for suffix, category in CATEGORY_SUFFIXES:
        if name.endswith(suffix):
            return category
    return "other"


def layer_index_of(name: str) -> int:
    m = LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else -1


def iter_target_linears(model: nn.Module):
    """No (layer, category) dedup -- MoE has many Linears sharing a
    category per layer (60 experts + 1 shared expert), unlike dense
    models where dedup just skips accidental duplicate references."""
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        targets.append((n, m, categorize(n)))
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


def patch_eager_qwen2_moe(mod, q_fn, k_fn, v_fn, p_fn):
    """Qwen2MoeAttention (transformers 4.44.2) uses an OLDER rotary calling
    convention than llama/mistral in this same transformers version:
    self.rotary_emb(value_states, seq_len=...) + apply_rotary_pos_emb(...,
    position_ids) (5 args, not the newer cos/sin-tuple style) -- copied
    from Qwen2MoeAttention.forward with q/k routed through q_fn/k_fn and
    the real cache-update path stripped (matches quant_targets.
    patch_eager_ropegqa's convention: single forward pass, no generation)."""
    attn_cls = mod.Qwen2MoeAttention
    orig_forward = attn_cls.forward

    def patched_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    attn_cls.forward = patched_forward
    return attn_cls, orig_forward


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

    # Patch the attention class's forward BEFORE loading with device_map=
    # "auto" -- accelerate's dispatch_model wraps each module's forward with
    # a device-placement hook and captures "old_forward" = module.forward
    # AT LOAD TIME; patching the class afterward never reaches that
    # already-captured reference, so Q/K probes silently never fire (found
    # by the first run of this script: only qkv_proj/o_proj -- the plain
    # nn.Linear pre-hooks -- showed up, Q/K never did, no error either).
    import transformers.models.qwen2_moe.modeling_qwen2_moe as attn_mod
    attn_cls, orig_forward = patch_eager_qwen2_moe(attn_mod, q_probe, k_probe, identity, identity)

    from transformers import AutoModelForCausalLM
    # 14.3B total params in BF16 (~28.6GB) doesn't fit one 24GB GPU --
    # shard across both with device_map="auto" instead of .to(DEVICE).
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager",
                                                   device_map="auto")
    model.eval()

    targets = iter_target_linears(model)
    print(f"Hooking {len(targets)} Linear modules (MoE: 60 experts + 1 shared expert x 3 Linears x 24 layers expected)", flush=True)

    def make_hook(category):
        def hook(_mod, inputs):
            accumulate(cat_sum, cat_n, category, topk_error_ratio_of(inputs[0].detach(), block_size, base_config, k_values), k_values)
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device)
    model(input_ids=ids)
    attn_cls.forward = orig_forward
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {0: 0.0}
        avg.update({k: cat_sum[category][k] / cat_n[category] for k in k_values})
        print(f"[{fmt}][qwen_moe] {category} ({cat_n[category]}): " + " ".join(f"top{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    by_category = measure_model(fmt)

    csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}_qwen_moe.csv")
    categories = sorted(by_category.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "k"] + categories)
        w.writeheader()
        for k in [0] + k_values:
            row = {"model": "qwen_moe", "k": k}
            for category in categories:
                row[category] = by_category[category].get(k, "")
            w.writerow(row)
    print(f"Saved -> {csv_path}", flush=True)
    print("H2_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
