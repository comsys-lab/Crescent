"""Figure 3 (dense model, Llama-3.1-8B): how concentrated is the squared quantization error within a block?

For every NVFP4 / MXFP4 block of an activation tensor,
  ratio(k) = sum_{i in S_k} eps_i^2 / sum_i eps_i^2,  eps_i = a_i - Q(a)_i,
  S_k = the k elements with the largest eps_i^2, k = 1..block size,
averaged over blocks, then over calls of each activation category.

Formats: NVFP4 = E2M1_amax_E4M3 (block 16), MXFP4 = E2M1_mxfp4_ue8m0 (block 32).
Categories: Linear inputs grouped by shared input (qkv_proj, o_proj, gateup_proj, down_proj; one Linear
per (layer, category)), and Q, K after RoPE. Input: one 2048-token WikiText-2 sequence.
MoE models (Qwen3-30B-A3B, DeepSeek-MoE-16B): moe.py. Laplace model: laplace.py.

Usage: python3 dense.py <nvfp4|mxfp4> [model_key ...]   (default: llama_base)
Output: results/real_summary_{fmt}_{model}.csv (one row per k, one column per category)
"""

import csv
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config
from quant_targets import should_skip_linear, patch_eager, MODEL_CONFIG

torch.manual_seed(1111)

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def categorize(name: str) -> str:
    for suffix, category in CATEGORY_SUFFIXES:
        if name.endswith(suffix):
            return category
    return "other"


def layer_index_of(name: str) -> int:
    m = LAYER_INDEX_RE.search(name)
    return int(m.group(1)) if m else -1


def iter_target_linears(model: nn.Module):
    """One Linear per (layer, category): q/k/v_proj and gate/up_proj share an input."""
    seen = set()
    targets = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        category = categorize(n)
        key = (layer_index_of(n), category)
        if key in seen:
            continue
        seen.add(key)
        targets.append((n, m, category))
    return targets


@torch.no_grad()
def topk_error_ratio_of(x: torch.Tensor, block_size: int, base_config: str, k_values: list) -> dict:
    """x: [..., K]. Returns {k: mean % of block error captured by top-k}."""
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
        cat_sum[category] = {k: 0.0 for k in [0] + k_values}  # k=0: 0% by definition
        cat_n[category] = 0
    for k in k_values:
        cat_sum[category][k] += res[k]
    cat_n[category] += 1


def average(fmt: str, model_key: str, cat_sum: dict, cat_n: dict, k_values: list) -> dict:
    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {0: 0.0}
        avg.update({k: cat_sum[category][k] / cat_n[category] for k in k_values})
        print(f"[{fmt}][{model_key}] {category} ({cat_n[category]}): " + " ".join(f"top{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg
    return by_category


def write_summary(path: str, per_model: dict, k_values: list):
    categories = sorted({c for by_cat in per_model.values() for c in by_cat})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "k"] + categories)
        w.writeheader()
        for m, by_cat in per_model.items():
            for k in [0] + k_values:
                row = {"model": m, "k": k}
                for category in categories:
                    row[category] = by_cat.get(category, {}).get(k, "")
                w.writerow(row)
    print(f"Saved -> {path}", flush=True)


def identity(x):
    return x


@torch.no_grad()
def measure_model(fmt: str, model_key: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config = cfg["block_size"], cfg["base_config"]
    k_values = list(range(1, block_size + 1))

    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]
    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    from transformers import AutoModelForCausalLM
    import transformers.models.llama.modeling_llama as llama_mod
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()

    cat_sum, cat_n = {}, {}

    def probe(category):
        def fn(x):
            accumulate(cat_sum, cat_n, category, topk_error_ratio_of(x.detach(), block_size, base_config, k_values), k_values)
            return x
        return fn

    def make_hook(category):
        p = probe(category)

        def hook(_mod, inputs):  # returns None: the Linear input is left unchanged
            p(inputs[0])
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in iter_target_linears(model)]
    attn_cls, orig_forward = patch_eager(arch, llama_mod, probe("Q"), probe("K"), identity, identity)

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(DEVICE)
    model(input_ids=ids)
    attn_cls.forward = orig_forward
    for h in handles:
        h.remove()

    by_category = average(fmt, model_key, cat_sum, cat_n, k_values)
    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    suffix = os.environ.get("RESULT_SUFFIX", "")
    if len(model_keys) == 1:  # keep the model in the file name
        suffix = f"_{model_keys[0]}{suffix}"
    summary_csv = os.path.join(RESULTS_DIR, f"real_summary_{fmt}{suffix}.csv")
    per_model = {}
    for mk in model_keys:
        per_model[mk] = measure_model(fmt, mk)
        write_summary(summary_csv, per_model, k_values)


if __name__ == "__main__":
    main()
