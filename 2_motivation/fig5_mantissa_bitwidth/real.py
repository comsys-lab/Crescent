"""H4 (v4): boost-mantissa necessity sweep (E2M1..E2M7, exponent fixed
at 2) -- how much of the topk-boost recovery does each mantissa depth
capture, per surface (Linear categories + Q/K)? MSE only (real model
activations), one forward pass covers all M values per surface. k fixed
at 12.5% of block (matching H2/H5), uniformly across every surface.

Usage: python3 real.py <nvfp4|mxfp4> [model ...]
"""

import csv
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings
from formats import quantize_tensor_config, quantize_tensor_topk_activation_e2m7
from quant_targets import should_skip_linear, patch_eager, MODEL_CONFIG

torch.manual_seed(1111)

MANTISSA_VALUES = [1, 2, 3, 4, 5, 6, 7]
PROBE_SEQ_LEN = 2048

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3", k=2),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0", k=4),
}

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


def boost_config_name(fmt: str, mant_bits: int) -> str:
    return f"E2M{mant_bits}_amax_E4M3" if fmt == "nvfp4" else f"E2M{mant_bits}_mxfp4_ue8m0"


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
def reduction_pct_all_m(x: torch.Tensor, fmt: str, k: int, block_size: int, base_config: str) -> dict:
    """{m: % reduction in per-block RELATIVE MSE (eps^2 summed per block,
    divided by that block's signal energy, then averaged over blocks --
    matching H2/H3/H5's convention) when the top-k largest-base-error
    positions are boosted to mantissa=m}, for m in MANTISSA_VALUES."""
    x32 = x.detach().to(torch.float32)
    xb = quantize_tensor_config(x32, base_config, block_size)
    K = x32.shape[-1]
    n_blocks = K // block_size
    if n_blocks == 0:
        return None
    usable = n_blocks * block_size
    x_b = x32[..., :usable].reshape(-1, block_size)
    base_b = xb[..., :usable].reshape(-1, block_size)
    sig = x_b.pow(2).sum(dim=-1).clamp(min=1e-30)
    r_base = ((x_b - base_b).pow(2).sum(dim=-1) / sig).mean().item()
    if r_base <= 0:
        return None
    out = {}
    for m in MANTISSA_VALUES:
        xu = quantize_tensor_topk_activation_e2m7(x32, k, block_size=block_size,
                                                    base_config=base_config, boost_config=boost_config_name(fmt, m))
        xu_b = xu[..., :usable].reshape(-1, block_size)
        r_m = ((x_b - xu_b).pow(2).sum(dim=-1) / sig).mean().item()
        out[m] = 100 * (1 - r_m / r_base)
    return out


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


def accumulate(cat_sum, cat_n, category, res):
    if res is None:
        return
    if category not in cat_sum:
        cat_sum[category] = {m: 0.0 for m in MANTISSA_VALUES}
        cat_n[category] = 0
    for m in MANTISSA_VALUES:
        cat_sum[category][m] += res[m]
    cat_n[category] += 1


@torch.no_grad()
def measure_model(fmt: str, model_key: str) -> dict:
    cfg = FORMAT_CONFIG[fmt]
    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]
    k, block_size, base_config = cfg["k"], cfg["block_size"], cfg["base_config"]

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
            accumulate(cat_sum, cat_n, category, reduction_pct_all_m(inputs[0].detach(), fmt, k, block_size, base_config))
        return hook

    handles = [m.register_forward_pre_hook(make_hook(category)) for _n, m, category in targets]

    def q_probe(x):
        accumulate(cat_sum, cat_n, "Q", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
        return x

    def k_probe(x):
        accumulate(cat_sum, cat_n, "K", reduction_pct_all_m(x.detach(), fmt, k, block_size, base_config))
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
        avg = {m: cat_sum[category][m] / cat_n[category] for m in MANTISSA_VALUES}
        print(f"[{fmt}][{model_key}] {category}: " + " ".join(f"m{m}={avg[m]:.2f}%" for m in MANTISSA_VALUES), flush=True)
        by_category[category] = avg

    del model
    cleanup_cuda()
    return by_category


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]
    suffix = os.environ.get("RESULT_SUFFIX", "")
    if len(model_keys) == 1:  # keep the model in the file name (e.g. ppl_nvfp4_llama_base.csv)
        suffix = f"_{model_keys[0]}{suffix}"
    summary_csv = os.path.join(RESULTS_DIR, f"real_summary_{fmt}{suffix}.csv")
    per_model = {}
    all_categories = set()
    for mk in model_keys:
        per_model[mk] = measure_model(fmt, mk)
        all_categories.update(per_model[mk].keys())
        categories = sorted(all_categories)

        fieldnames = ["model", "m"] + categories
        rows = []
        for m_model in per_model:
            for m in MANTISSA_VALUES:
                row = {"model": m_model, "m": m}
                for category in categories:
                    row[category] = per_model[m_model].get(category, {}).get(m, "")
                rows.append(row)

        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Saved -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
