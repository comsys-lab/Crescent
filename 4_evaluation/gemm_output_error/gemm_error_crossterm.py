"""Czardas attention cross-term (dQ x dK) contribution to GEMM Output Relative MSE.

quant_targets.patch_eager_ropegqa_decomposed computes:
  attn_weights = Q_base@K_boost^T + Q_boost@K_base^T - Q_base@K_base^T
               = Q_base@K_base^T + Q_base@dK^T + dQ@K_base^T          (dQ@dK^T dropped)
where dQ = Q_boost - Q_base, dK = K_boost - K_base. Algebraically, the EXACT 4-term sum
(adding dQ@dK^T back in) equals (Q_base+dQ)@(K_base+dK)^T = Q_boost@K_boost^T exactly -- this
script verifies that identity numerically (the correctness check below) rather than assuming it,
then uses Q_boost@K_boost^T directly as "Czardas + cross-term" (no need to separately sum 4 terms
once the identity is confirmed).

Reuses each model's existing capture() function (gemm_error_llama/deepseek/qwen.py) verbatim --
one real forward pass per model to get real post-RoPE Q/K, same as the existing harness; no new
capture logic. GEMM-output level only -- does NOT measure downstream PPL/zero-shot impact.

Usage: python3 gemm_error_crossterm.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b>
"""
import csv
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
sys.path.insert(0, HERE)
from eval_utils import same_seeds, load_wikitext2_text, load_tokenizer_and_encodings  # noqa: E402
import gemm_error_common as G  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_llama(fmt):
    from quant_targets import MODEL_CONFIG
    import gemm_error_llama as M
    from transformers import AutoModelForCausalLM
    cfg = MODEL_CONFIG["llama_base"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(torch.device("cuda:0"))
    model.eval()
    n_layers = model.config.num_hidden_layers
    sample_layer_idxs = sorted({0, n_layers // 2, n_layers - 1})
    tok, encodings = load_tokenizer_and_encodings(cfg["model_name"], load_wikitext2_text())
    _, qk_by_layer, head_dim = M.capture(model, encodings, sample_layer_idxs)
    return [(li, qk_by_layer[li][0], qk_by_layer[li][1], head_dim) for li in sample_layer_idxs]


def run_deepseek(fmt):
    import deepseek_moe_model as dsm
    import gemm_error_deepseek as M
    os.environ.setdefault("DEEPSEEK_MAX_MEM_CPU", "150GiB")
    model = dsm.load_model()
    model.eval()
    n_layers = len(model.model.layers)
    first_k_dense = model.cfg["first_k_dense_replace"]
    moe_freq = model.cfg["moe_layer_freq"]
    moe_layers = [i for i in range(n_layers) if i >= first_k_dense and i % moe_freq == 0]
    sample_layer_idxs = sorted({0, moe_layers[len(moe_layers) // 3], moe_layers[-1]})
    tok, encodings = load_tokenizer_and_encodings(dsm.MODEL_NAME, load_wikitext2_text())
    _, qk_by_layer, head_dim, _ = M.capture(model, encodings, sample_layer_idxs)
    return [(li, qk_by_layer[li][0], qk_by_layer[li][1], head_dim) for li in sample_layer_idxs if li in qk_by_layer]


def run_qwen(fmt):
    from qwen3_moe_targets import MODEL_NAME
    import gemm_error_qwen as M
    from transformers import AutoModelForCausalLM
    os.environ.setdefault("QWEN_EVAL_MAX_MEM_PER_GPU", "19GiB")
    os.environ.setdefault("QWEN_EVAL_MAX_MEM_CPU", "150GiB")
    kw = {}
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        kw["max_memory"] = {i: os.environ["QWEN_EVAL_MAX_MEM_PER_GPU"] for i in range(n_gpu)}
        kw["max_memory"]["cpu"] = os.environ["QWEN_EVAL_MAX_MEM_CPU"]
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                   device_map="auto", attn_implementation="eager", **kw)
    model.eval()
    n_layers = model.config.num_hidden_layers
    sample_layer_idxs = sorted({0, n_layers // 2, n_layers - 1})
    tok, encodings = load_tokenizer_and_encodings(MODEL_NAME, load_wikitext2_text())
    _, qk_by_layer, _ = M.capture(model, encodings, sample_layer_idxs)
    return [(li, qk_by_layer[li][0], qk_by_layer[li][1], qk_by_layer[li][2]) for li in sample_layer_idxs]


RUNNERS = {"llama_base": run_llama, "deepseek_moe_16b": run_deepseek, "qwen3_moe": run_qwen}


def correctness_check(q_bf16, k_bf16, fmt):
    """4-term manual sum must equal Q_boost @ K_boost^T exactly (float precision)."""
    q_base = G.quantize_activation(q_bf16, fmt, "native", G.QK_DIV)
    q_boost = G.quantize_activation(q_bf16, fmt, "czardas", G.QK_DIV)
    k_base = G.quantize_activation(k_bf16, fmt, "native", G.QK_DIV)
    k_boost = G.quantize_activation(k_bf16, fmt, "czardas", G.QK_DIV)
    dQ = q_boost - q_base
    dK = k_boost - k_base
    four_term = (q_base @ k_base.transpose(-2, -1) + q_base @ dK.transpose(-2, -1)
                 + dQ @ k_base.transpose(-2, -1) + dQ @ dK.transpose(-2, -1))
    direct = q_boost @ k_boost.transpose(-2, -1)
    max_abs_diff = (four_term - direct).abs().max().item()
    return max_abs_diff, max_abs_diff < 1e-3


if __name__ == "__main__":
    fmt, model_key = sys.argv[1], sys.argv[2]
    same_seeds()
    print(f"[crossterm][{model_key}][{fmt}] capturing real Q/K (reusing existing capture())...", flush=True)
    layer_data = RUNNERS[model_key](fmt)

    li0, q0, k0, hd0 = layer_data[0]
    max_abs_diff, passed = correctness_check(q0, k0, fmt)
    status_path = os.path.join(RESULTS_DIR, f"correctness_crossterm_{fmt}_{model_key}.txt")
    line = f"[check 4term-sum-equals-Qboost@Kboost] layer={li0} max_abs_diff={max_abs_diff:.6e} {'PASS' if passed else 'FAIL'}"
    print("  " + line, flush=True)
    with open(status_path, "w") as f:
        f.write(line + f"\nALL_PASSED={passed}\n")
    if not passed:
        print(f"[crossterm][{model_key}][{fmt}] ABORTING: correctness check failed", flush=True)
        sys.exit(1)

    rows = []
    sse3 = sse4 = ssr_bf16 = n_total = 0.0
    for li, q_bf16, k_bf16, head_dim in layer_data:
        scale = 1.0 / math.sqrt(head_dim)
        qf, kf = q_bf16.to(torch.float32), k_bf16.to(torch.float32)
        y_bf16 = (qf @ kf.transpose(-2, -1)) * scale

        q_base = G.quantize_activation(q_bf16, fmt, "native", G.QK_DIV)
        q_boost = G.quantize_activation(q_bf16, fmt, "czardas", G.QK_DIV)
        k_base = G.quantize_activation(k_bf16, fmt, "native", G.QK_DIV)
        k_boost = G.quantize_activation(k_bf16, fmt, "czardas", G.QK_DIV)

        y_3term = (q_base @ k_boost.transpose(-2, -1) + q_boost @ k_base.transpose(-2, -1)
                   - q_base @ k_base.transpose(-2, -1)) * scale
        y_4term = (q_boost @ k_boost.transpose(-2, -1)) * scale  # verified identity above

        rel3, s3, ssr, n = G.relative_mse(y_3term, y_bf16)
        rel4, s4, ssr4, n4 = G.relative_mse(y_4term, y_bf16)
        sse3 += s3; sse4 += s4; ssr_bf16 += ssr; n_total += n

        cross_term_delta = rel3 - rel4  # absolute Relative-MSE drop from restoring dQ@dK^T
        pct_of_czardas_gap_closed = (cross_term_delta / rel3 * 100.0) if rel3 > 0 else float("nan")
        rows.append(dict(model=model_key, format=fmt, layer=f"model.layers.{li}.self_attn.qk",
                          relative_mse_czardas_3term=rel3, relative_mse_czardas_4term_exact=rel4,
                          crossterm_relative_mse_delta=cross_term_delta,
                          pct_of_czardas_gap_closed_by_crossterm=pct_of_czardas_gap_closed, n_elements=n))
        print(f"  layer {li}: 3term={rel3:.6e} 4term={rel4:.6e} delta={cross_term_delta:.6e} "
              f"({pct_of_czardas_gap_closed:.2f}% of 3term's own error closed)", flush=True)

    csv_path = os.path.join(RESULTS_DIR, f"gemm_error_crossterm_{fmt}_{model_key}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "layer", "relative_mse_czardas_3term",
                                           "relative_mse_czardas_4term_exact", "crossterm_relative_mse_delta",
                                           "pct_of_czardas_gap_closed_by_crossterm", "n_elements"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved per-layer -> {csv_path}", flush=True)

    agg_rel3 = sse3 / ssr_bf16
    agg_rel4 = sse4 / ssr_bf16
    agg_delta = agg_rel3 - agg_rel4
    agg_pct = (agg_delta / agg_rel3 * 100.0) if agg_rel3 > 0 else float("nan")
    agg_path = os.path.join(RESULTS_DIR, f"gemm_error_crossterm_aggregate_{fmt}_{model_key}.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "aggregate_relative_mse_czardas_3term",
                                           "aggregate_relative_mse_czardas_4term_exact",
                                           "aggregate_crossterm_relative_mse_delta",
                                           "pct_of_czardas_gap_closed_by_crossterm", "n_elements_total"])
        w.writeheader()
        w.writerow(dict(model=model_key, format=fmt, aggregate_relative_mse_czardas_3term=agg_rel3,
                         aggregate_relative_mse_czardas_4term_exact=agg_rel4,
                         aggregate_crossterm_relative_mse_delta=agg_delta,
                         pct_of_czardas_gap_closed_by_crossterm=agg_pct, n_elements_total=n_total))
    print(f"Saved aggregate -> {agg_path}", flush=True)
    print(f"[crossterm][{model_key}][{fmt}] AGGREGATE: 3term={agg_rel3:.6e} 4term={agg_rel4:.6e} "
          f"delta={agg_delta:.6e} ({agg_pct:.2f}% of Czardas's own error would be closed by the cross-term)", flush=True)
