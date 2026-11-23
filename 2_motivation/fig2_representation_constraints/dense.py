"""Figure 2 (dense model, Llama-3.1-8B): which representation constraint -- scaling factor, exponent
bitwidth, or mantissa precision -- dominates activation quantization error?

Each constraint is relaxed separately while the other two stay as in the original format:
  NVFP4 (block 16, E4M3 scale)          MXFP4 (block 32, E8M0 scale)
  E2M1_amax_E4M3    original format      E2M1_mxfp4_ue8m0   original format
  E2M1_optimal_none scaling factor       E2M1_optimal_none  scaling factor (unrounded optimal scale)
  E8M1_amax_E4M3    exponent bitwidth    E8M1_mxfp4_ue8m0   exponent bitwidth
  E2M7_amax_E4M3    mantissa precision   E2M7_mxfp4_ue8m0   mantissa precision

Activations: Linear inputs, Q and K (after RoPE) from one 2048-token WikiText-2 sequence.
Relative MSE is measured in a diagnostic pass that returns the original tensors; a separate pass applies
the same config to Linear/Q/K and reports WikiText-2 PPL (weights, V and P stay BF16).
MoE models (Qwen3-30B-A3B, DeepSeek-MoE-16B): moe.py.

Usage: python3 dense.py <nvfp4|mxfp4> [model_key ...]   (default: llama_base)
Output: results/real_{fmt}_{model}.csv, real_per_layer_*.csv, real_summary_*.csv
"""

import csv
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config
from quant_targets import should_skip_linear, replace_linear_modules, patch_eager, MODEL_CONFIG

torch.manual_seed(1111)

FORMAT_CONFIG = {
    "nvfp4": dict(block_size=16, base_config="E2M1_amax_E4M3",
                  config_names=["E2M1_amax_E4M3", "E2M1_optimal_none", "E8M1_amax_E4M3", "E2M7_amax_E4M3"]),
    "mxfp4": dict(block_size=32, base_config="E2M1_mxfp4_ue8m0",
                  config_names=["E2M1_mxfp4_ue8m0", "E2M1_optimal_none", "E8M1_mxfp4_ue8m0", "E2M7_mxfp4_ue8m0"]),
}
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
PROBE_SEQ_LEN = 2048


def identity(x):
    return x


def categorize(name: str) -> str:
    if name.endswith((".q_proj", ".k_proj", ".v_proj")):
        return "qkv_proj"
    if name.endswith(".o_proj"):
        return "o_proj"
    if name.endswith((".gate_proj", ".up_proj")):
        return "gateup_proj"
    if name.endswith(".down_proj"):
        return "down_proj"
    return "other"


def layer_index_of(name: str) -> int:
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def iter_target_linears(model: nn.Module):
    """One representative Linear per (layer_index, category) -- q/k/v_proj
    (and gate/up_proj) share the same input, so they'd otherwise produce
    byte-identical MSE rows."""
    seen = set()
    out = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        key = (layer_index_of(n), categorize(n))
        if key in seen:
            continue
        seen.add(key)
        out.append((n, m))
    return out


def rel_mse(x32: torch.Tensor, xq32: torch.Tensor, block_size: int) -> float:
    K = x32.shape[-1]
    num_blocks = K // block_size
    if num_blocks == 0:
        return 0.0
    xb = x32[..., : num_blocks * block_size].reshape(-1, block_size)
    xqb = xq32[..., : num_blocks * block_size].reshape(-1, block_size)
    err = (xb - xqb).pow(2).sum(dim=-1)
    sig = xb.pow(2).sum(dim=-1)
    valid = sig > 0
    if not valid.any():
        return 0.0
    return (err[valid] / sig[valid]).mean().item()


def attn_mod_for(model_name: str, arch: str):
    import transformers.models.llama.modeling_llama as llama_mod
    return llama_mod


@torch.no_grad()
def measure_axis_mse(model_name: str, arch: str, config_name: str, block_size: int, encodings) -> dict:
    """Diagnostic-only pass: instruments Linear (forward_pre_hook) and
    Q/K (via patch_eager with pass-through instrumented fns) to record
    MSE^rel(G) per surface, WITHOUT changing the forward pass's actual
    values (quantized value computed but discarded). V/P stay untouched
    (identity, not even probed)."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()

    per_layer = {}

    def make_linear_hook(name):
        def hook(_mod, inputs):
            x32 = inputs[0].detach().to(torch.float32)
            xq32 = quantize_tensor_config(x32, config_name, block_size)
            per_layer[name] = rel_mse(x32, xq32, block_size)
        return hook

    targets = iter_target_linears(model)
    handles = [m.register_forward_pre_hook(make_linear_hook(n)) for n, m in targets]

    surface_mse = {"Q": [], "K": []}

    def q_probe(x):
        x32 = x.detach().to(torch.float32)
        xq32 = quantize_tensor_config(x32, config_name, block_size)
        surface_mse["Q"].append(rel_mse(x32, xq32, block_size))
        return x

    def k_probe(x):
        x32 = x.detach().to(torch.float32)
        xq32 = quantize_tensor_config(x32, config_name, block_size)
        surface_mse["K"].append(rel_mse(x32, xq32, block_size))
        return x

    attn_cls, orig_forward = patch_eager(arch, attn_mod_for(model_name, arch), q_probe, k_probe, identity, identity)
    ids = encodings.input_ids[:, :PROBE_SEQ_LEN].to(DEVICE)
    model(input_ids=ids)
    attn_cls.forward = orig_forward
    for h in handles:
        h.remove()

    del model
    cleanup_cuda()

    result = dict(per_layer)
    for surf in ("Q", "K"):
        vals = surface_mse[surf]
        result[surf] = sum(vals) / len(vals) if vals else 0.0
    return result


def measure_ppl_all_quantized(model_name: str, arch: str, config_name: str, block_size: int, encodings) -> float:
    """PPL with config_name applied to Linear inputs, Q and K. V, P and weights stay BF16."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(DEVICE)
    model.eval()

    def qfn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), config_name, block_size).to(x.dtype)

    n_linear = replace_linear_modules(model, qfn)
    attn_mod = attn_mod_for(model_name, arch)
    attn_cls, orig_forward = patch_eager(arch, attn_mod, qfn, qfn, identity, identity)
    try:
        ppl = measure_ppl(model, encodings, f"{model_name}__{config_name}", DEVICE)
    finally:
        attn_cls.forward = orig_forward
    del model
    cleanup_cuda()
    return ppl


def measure(fmt: str, model_key: str, per_layer_rows: list) -> list:
    cfg = FORMAT_CONFIG[fmt]
    block_size, config_names = cfg["block_size"], cfg["config_names"]
    model_cfg = MODEL_CONFIG[model_key]
    model_name, arch = model_cfg["model_name"], model_cfg["arch"]

    same_seeds()
    full_text = load_wikitext2_text()
    _, encodings = load_tokenizer_and_encodings(model_name, full_text)

    rows = []

    from transformers import AutoModelForCausalLM
    same_seeds()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE); model.eval()
    t0 = time.time()
    ppl = measure_ppl(model, encodings, f"{model_key}__BF16", DEVICE)
    elapsed = time.time() - t0
    print(f"[{fmt}][{model_key}] BF16 PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": model_key, "config": "BF16", "surface": "-", "mse_rel_g": 0.0, "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    del model
    cleanup_cuda()

    for config_name in config_names:
        same_seeds()
        per_layer = measure_axis_mse(model_name, arch, config_name, block_size, encodings)
        for layer_name, val in per_layer.items():
            per_layer_rows.append({"model": model_key, "config": config_name, "layer": layer_name, "mse_rel_g": val})
        linear_vals = {k: v for k, v in per_layer.items() if k not in ("Q", "K")}
        linear_mean = sum(linear_vals.values()) / len(linear_vals) if linear_vals else 0.0
        print(f"[{fmt}][{model_key}] {config_name} Linear MSE^rel(G) mean = {linear_mean:.6f}  "
              f"Q={per_layer['Q']:.6f} K={per_layer['K']:.6f}", flush=True)

        same_seeds()
        t0 = time.time()
        ppl = measure_ppl_all_quantized(model_name, arch, config_name, block_size, encodings)
        elapsed = time.time() - t0
        print(f"[{fmt}][{model_key}] {config_name} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)

        for surf, val in [("Linear", linear_mean), ("Q", per_layer["Q"]), ("K", per_layer["K"])]:
            rows.append({"model": model_key, "config": config_name, "surface": surf, "mse_rel_g": val,
                         "ppl": ppl if surf == "Linear" else "", "elapsed_sec": round(elapsed, 3) if surf == "Linear" else ""})

    return rows


def summarize(all_rows: list, config_names: list, base_config: str) -> list:
    """Recover_* per surface: relative MSE removed by relaxing each constraint, for Linear/Q/K."""
    by_model_surface = {}
    for row in all_rows:
        if row["surface"] == "-":
            continue
        key = (row["model"], row["surface"])
        by_model_surface.setdefault(key, {})[row["config"]] = row["mse_rel_g"]
    summary = []
    for (model_key, surface), vals in by_model_surface.items():
        if not all(c in vals for c in config_names):
            continue
        r = vals[base_config]
        summary.append({
            "model": model_key, "surface": surface, "R_BASE": r,
            "Recover_SF": r - vals["E2M1_optimal_none"],
            "Recover_Exp": r - vals[config_names[2]],
            "Recover_Man": r - vals[config_names[3]],
        })
    return summary


def main():
    fmt = sys.argv[1]
    model_keys = sys.argv[2:] if len(sys.argv) > 2 else ["llama_base"]
    cfg = FORMAT_CONFIG[fmt]
    config_names, base_config = cfg["config_names"], cfg["base_config"]
    all_rows = []
    per_layer_rows = []
    suffix = os.environ.get("RESULT_SUFFIX", "")
    if len(model_keys) == 1:  # keep the model in the file name (e.g. real_nvfp4_llama_base.csv)
        suffix = f"_{model_keys[0]}{suffix}"
    csv_path = os.path.join(RESULTS_DIR, f"real_{fmt}{suffix}.csv")
    layer_csv_path = os.path.join(RESULTS_DIR, f"real_per_layer_{fmt}{suffix}.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}{suffix}.csv")
    fieldnames = ["model", "config", "surface", "mse_rel_g", "ppl", "elapsed_sec"]
    layer_fieldnames = ["model", "config", "layer", "mse_rel_g"]
    summary_fieldnames = ["model", "surface", "R_BASE", "Recover_SF", "Recover_Exp", "Recover_Man"]
    for mk in model_keys:
        all_rows.extend(measure(fmt, mk, per_layer_rows))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_rows:
                w.writerow(row)
        with open(layer_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=layer_fieldnames)
            w.writeheader()
            for row in per_layer_rows:
                w.writerow(row)
        summary = summarize(all_rows, config_names, base_config)
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fieldnames)
            w.writeheader()
            for row in summary:
                w.writerow(row)
                print(f"[{fmt}][{row['model']}][{row['surface']}] Recover_SF={100*row['Recover_SF']:.4f}% "
                      f"Recover_Exp={100*row['Recover_Exp']:.4f}% Recover_Man={100*row['Recover_Man']:.4f}%", flush=True)
        print(f"Saved -> {csv_path}, {layer_csv_path}, {summary_csv_path}", flush=True)


if __name__ == "__main__":
    main()
