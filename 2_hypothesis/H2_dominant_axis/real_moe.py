"""H2 (dominant axis: scaling factor vs exponent vs mantissa) for the MoE models Qwen3-30B-A3B and
DeepSeek-MoE-16B-Base. Same experiment as real.py (imported helpers: rel_mse, summarize,
FORMAT_CONFIG): for each of the 4 configs, MSE^rel(G) per surface (Linear/Q/K, diagnostic pass
that returns the original tensor) plus end-to-end PPL with that config on Linear+Q+K
(V, P and weights stay BF16).

  qwen3_moe        : attention Linears are nn.Linear; expert gate_up/down are batched Parameters ->
                     probed / quantized inline via common/qwen3_moe_targets.patch_qwen3_moe_experts.
  deepseek_moe_16b : common/deepseek_moe_model.py (pure-PyTorch port), every Linear is an nn.Linear.

"Linear" MSE = mean over the four Linear categories (qkv_proj, o_proj, gateup_proj, down_proj) of
each category's mean, which equals the dense script's mean over per-(layer, category) entries.
MoE expert inputs are averaged over every expert call (Qwen: one entry per category over all layers;
DeepSeek: per (layer, category), all routed + shared experts of that layer).

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 real_moe.py <nvfp4|mxfp4> <qwen3_moe|deepseek_moe_16b>
"""
import csv
import os
import re
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from eval_utils import same_seeds, cleanup_cuda, load_wikitext2_text, load_tokenizer_and_encodings, measure_ppl
from formats import quantize_tensor_config
from quant_targets import should_skip_linear, replace_linear_modules
from real import rel_mse, summarize, FORMAT_CONFIG, PROBE_SEQ_LEN, RESULTS_DIR, identity

LINEAR_CATEGORY = {"q_proj": "qkv_proj", "o_proj": "o_proj", "gate_proj": "gateup_proj", "down_proj": "down_proj"}
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def build_model(model_key, q_fn, k_fn, gateup_fn=None, down_fn=None):
    """Returns (model, restore, encodings-model-name). Patches are applied BEFORE loading
    (accelerate captures forwards at load time). gateup_fn/down_fn only matter for qwen3_moe."""
    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME, patch_eager_qwen3_moe, patch_qwen3_moe_experts
        import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
        attn_cls, orig_attn = patch_eager_qwen3_moe(attn_mod, q_fn, k_fn, identity, identity)
        experts_cls, orig_experts = patch_qwen3_moe_experts(attn_mod, gateup_fn or identity, down_fn or identity)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                       attn_implementation="eager", device_map="auto")
        restore = lambda: (setattr(attn_cls, "forward", orig_attn), setattr(experts_cls, "forward", orig_experts))
        name = MODEL_NAME
    else:
        import deepseek_moe_model as dsm
        dsm.set_hooks(q=q_fn, k=k_fn)
        model = dsm.load_model()
        restore = dsm.set_hooks
        name = dsm.MODEL_NAME
    model.eval()
    return model, restore, name


def model_name_of(model_key):
    if model_key == "qwen3_moe":
        from qwen3_moe_targets import MODEL_NAME
    else:
        from deepseek_moe_model import MODEL_NAME
    return MODEL_NAME


@torch.no_grad()
def measure_axis_mse(model_key, config_name, block_size, encodings) -> dict:
    acc = {}  # key -> [sum, n]

    def record(key, x):
        x32 = x.detach().to(torch.float32)
        v = rel_mse(x32, quantize_tensor_config(x32, config_name, block_size), block_size)
        s = acc.setdefault(key, [0.0, 0])
        s[0] += v
        s[1] += 1

    def probe(key):
        def fn(x):
            record(key, x)
            return x
        return fn

    model, restore, _ = build_model(model_key, probe("Q"), probe("K"),
                                    gateup_fn=probe("experts.gateup_proj"), down_fn=probe("experts.down_proj"))
    handles = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear) or should_skip_linear(n):
            continue
        cat = LINEAR_CATEGORY.get(n.rsplit(".", 1)[-1])
        if cat is None:
            continue
        lm = LAYER_RE.search(n)
        key = f"layers.{lm.group(1)}.{cat}"
        handles.append(m.register_forward_pre_hook(lambda _m, inputs, key=key: record(key, inputs[0])))

    model(input_ids=encodings.input_ids[:, :PROBE_SEQ_LEN].to(model.device))
    restore()
    for h in handles:
        h.remove()
    del model
    cleanup_cuda()
    return {k: s[0] / s[1] for k, s in acc.items() if s[1] > 0}


def linear_mean_of(per_layer: dict) -> float:
    by_cat = {}
    for key, v in per_layer.items():
        if key in ("Q", "K"):
            continue
        by_cat.setdefault(key.rsplit(".", 1)[-1], []).append(v)
    means = [sum(v) / len(v) for v in by_cat.values()]
    return sum(means) / len(means) if means else 0.0


def measure_ppl_all_quantized(model_key, config_name, block_size, encodings) -> float:
    def qfn(x):
        return quantize_tensor_config(x.detach().to(torch.float32), config_name, block_size).to(x.dtype)

    model, restore, _ = build_model(model_key, qfn, qfn, gateup_fn=qfn, down_fn=qfn)
    replace_linear_modules(model, qfn)
    try:
        ppl = measure_ppl(model, encodings, f"{model_key}__{config_name}", model.device)
    finally:
        restore()
    del model
    cleanup_cuda()
    return ppl


def main():
    fmt, model_key = sys.argv[1], sys.argv[2]
    cfg = FORMAT_CONFIG[fmt]
    block_size, config_names, base_config = cfg["block_size"], cfg["config_names"], cfg["base_config"]
    same_seeds()
    _, encodings = load_tokenizer_and_encodings(model_name_of(model_key), load_wikitext2_text())

    csv_path = os.path.join(RESULTS_DIR, f"real_{fmt}_{model_key}.csv")
    layer_csv_path = os.path.join(RESULTS_DIR, f"real_per_layer_{fmt}_{model_key}.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, f"real_summary_{fmt}_{model_key}.csv")
    fieldnames = ["model", "config", "surface", "mse_rel_g", "ppl", "elapsed_sec"]
    layer_fieldnames = ["model", "config", "layer", "mse_rel_g"]
    summary_fieldnames = ["model", "surface", "R_BASE", "Recover_SF", "Recover_Exp", "Recover_Man"]
    rows, per_layer_rows = [], []

    def save():
        for path, fields, data in [(csv_path, fieldnames, rows), (layer_csv_path, layer_fieldnames, per_layer_rows),
                                   (summary_csv_path, summary_fieldnames, summarize(rows, config_names, base_config))]:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in data:
                    w.writerow(row)
        print(f"Saved -> {csv_path}", flush=True)

    # BF16 baseline PPL
    same_seeds()
    model, restore, _ = build_model(model_key, identity, identity)
    t0 = time.time()
    ppl = measure_ppl(model, encodings, f"{model_key}__BF16", model.device)
    elapsed = time.time() - t0
    print(f"[{fmt}][{model_key}] BF16 PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
    rows.append({"model": model_key, "config": "BF16", "surface": "-", "mse_rel_g": 0.0, "ppl": ppl, "elapsed_sec": round(elapsed, 3)})
    restore()
    del model
    cleanup_cuda()
    save()

    for config_name in config_names:
        same_seeds()
        per_layer = measure_axis_mse(model_key, config_name, block_size, encodings)
        for layer_name, val in per_layer.items():
            per_layer_rows.append({"model": model_key, "config": config_name, "layer": layer_name, "mse_rel_g": val})
        linear_mean = linear_mean_of(per_layer)
        print(f"[{fmt}][{model_key}] {config_name} Linear MSE^rel(G) mean = {linear_mean:.6f}  "
              f"Q={per_layer['Q']:.6f} K={per_layer['K']:.6f}", flush=True)

        same_seeds()
        t0 = time.time()
        ppl = measure_ppl_all_quantized(model_key, config_name, block_size, encodings)
        elapsed = time.time() - t0
        print(f"[{fmt}][{model_key}] {config_name} PPL = {ppl}  elapsed={elapsed:.1f}s", flush=True)
        for surf, val in [("Linear", linear_mean), ("Q", per_layer["Q"]), ("K", per_layer["K"])]:
            rows.append({"model": model_key, "config": config_name, "surface": surf, "mse_rel_g": val,
                         "ppl": ppl if surf == "Linear" else "", "elapsed_sec": round(elapsed, 3) if surf == "Linear" else ""})
        save()
    print("H2_MOE_DONE", flush=True)


if __name__ == "__main__":
    main()
