"""H3 (v4): Relative MSE Reduction Rate -- upgrading the top-k largest-
|eps_i^2| positions per block to E2M7 (exact, not approximated),
sweeping k=0..block_size. Covers Linear categories plus Q, K (this
project's baseline quantizes Linear+Q+K together -- V and P were
dropped, H0 isolation showed negligible PPL impact everywhere).

Usage: python3 real.py <nvfp4|mxfp4> [model ...]
"""

import csv
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config
from quant_targets import should_skip_linear, patch_eager, MODEL_CONFIG

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", boost_config="E2M7_amax_E4M3"),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", boost_config="E2M7_mxfp4_ue8m0"),
}
PROBE_SEQ_LEN = 2048

CATEGORY_SUFFIXES = [
    ("q_proj", "qkv_proj"), ("k_proj", "qkv_proj"), ("v_proj", "qkv_proj"),
    ("o_proj", "o_proj"), ("out_proj", "o_proj"),
    ("gate_proj", "gateup_proj"), ("up_proj", "gateup_proj"),
    ("down_proj", "down_proj"),
    ("fc1", "fc1"), ("fc2", "fc2"),
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
def reduction_rate_of(x: torch.Tensor, block_size: int, base_config: str, boost_config: str, k_values: list) -> dict:
    K = x.shape[-1]
    num_blocks = K // block_size
    if num_blocks == 0:
        return None
    x = x.reshape(-1, K)[:, : num_blocks * block_size].to(torch.float32)
    xq = quantize_tensor_config(x, base_config, block_size)
    xq_boost = quantize_tensor_config(x, boost_config, block_size)

    x_b = x.reshape(-1, block_size)
    eps = x_b - xq.reshape(-1, block_size)
    eps_boost = x_b - xq_boost.reshape(-1, block_size)
    eps_sq = eps.pow(2)
    sig = x_b.pow(2).sum(dim=-1).clamp(min=1e-30)

    r_base = (eps_sq.sum(dim=-1) / sig).mean().item()
    order = eps_sq.argsort(dim=-1, descending=True)

    result = {0: 0.0}
    for k in k_values:
        topk_idx = order[:, :k]
        mask = torch.zeros_like(eps_sq, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        r_k_num = torch.where(mask, eps_boost.pow(2), eps_sq).sum(dim=-1)
        r_k = (r_k_num / sig).mean().item()
        result[k] = 100 * (1 - r_k / r_base) if r_base > 0 else 0.0
    return result


def identity(x):
    return x


def attn_mod_for(model_name: str, arch: str):
    if arch == "opt":
        import transformers.models.opt.modeling_opt as opt_mod
        return opt_mod
    if "ministral" in model_name.lower():
        import transformers.models.ministral.modeling_ministral as ministral_mod
        return ministral_mod
    if "mistral" in model_name.lower():
        import transformers.models.mistral.modeling_mistral as mistral_mod
        return mistral_mod
    import transformers.models.llama.modeling_llama as llama_mod
    return llama_mod


def accumulate(cat_sum, cat_n, category, res, k_values):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {k: 0.0 for k in [0] + k_values}
        cat_n[category] = 0
    for k in [0] + k_values:
        cat_sum[category][k] += res[k]
    cat_n[category] += 1


@torch.no_grad()
def measure_model(fmt: str, model_key: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    block_size, base_config, boost_config = cfg["block_size"], cfg["base_config"], cfg["boost_config"]
    k_values = list(range(1, block_size + 1))

    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]
    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()

    cat_sum, cat_n = {}, {}
    targets = iter_target_linears(model)

    def make_hook(category):
        def hook(_mod, inputs):
            accumulate(cat_sum, cat_n, category,
                       reduction_rate_of(inputs[0].detach(), block_size, base_config, boost_config, k_values), k_values)
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    def q_probe(x):
        accumulate(cat_sum, cat_n, "Q", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    def k_probe(x):
        accumulate(cat_sum, cat_n, "K", reduction_rate_of(x.detach(), block_size, base_config, boost_config, k_values), k_values)
        return x

    attn_mod = attn_mod_for(model_name, arch)
    attn_cls, orig_forward = patch_eager(arch, attn_mod, q_probe, k_probe, identity, identity)

    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(DEVICE)
    model(input_ids=ids)
    attn_cls.forward = orig_forward
    for h in handles:
        h.remove()

    by_category = {}
    for category in sorted(cat_sum.keys()):
        avg = {k: cat_sum[category][k] / cat_n[category] for k in [0] + k_values}
        print(f"[{fmt}][{model_key}] {category} ({cat_n[category]}): " + " ".join(f"k{k}={avg[k]:.2f}%" for k in k_values), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]
    k_values = list(range(1, FORMAT_CONFIG[fmt]["block_size"] + 1))
    suffix = os.environ.get("RESULT_SUFFIX", "")
    if len(model_keys) == 1:  # keep the model in the file name (e.g. ppl_nvfp4_llama_base.csv)
        suffix = f"_{model_keys[0]}{suffix}"
    summary_csv = os.path.join(RESULTS_DIR, f"real_{fmt}{suffix}.csv")
    per_model = {}
    all_categories = set()
    for mk in model_keys:
        per_model[mk] = measure_model(fmt, mk)
        all_categories.update(per_model[mk].keys())
        categories = sorted(all_categories)

        fieldnames = ["model", "k"] + categories
        rows = []
        for m in per_model:
            for k in [0] + k_values:
                row = {"model": m, "k": k}
                for category in categories:
                    row[category] = per_model[m].get(category, {}).get(k, "")
                rows.append(row)

        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Saved -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
