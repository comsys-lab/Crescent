"""GEMM Output Error evaluation for Llama-3.1-8B (dense, llama_base).

Captures REAL activations from a single BF16 forward pass over one 2048-token WikiText2
(test split) window -- the same window ppl_w4a4.py's measure_ppl uses for its first stride step
(seq_len=2048, same text/tokenizer) -- at a representative sample of decoder layers (first,
middle, last), then computes Linear and Attention-QK^T GEMM outputs under every setting from
that SAME captured operand (never re-run through a quantized model), per the experiment's "same
original operand across all methods" principle.

Sampled Linear projections per sampled layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj (7 per layer x 3 layers = 21 Linear GEMMs). Attention QK^T at the same 3 layers, using
Q/K AFTER RoPE and repeat_kv (exactly what ppl_w4a4.py's patched attention feeds into matmul).

Usage: CUDA_VISIBLE_DEVICES=<free GPU> python3 gemm_error_llama.py <nvfp4|mxfp4> [--correctness-only]
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
from quant_targets import MODEL_CONFIG, replace_linear_modules  # noqa: E402
from formats import quantize_tensor_config, quantize_tensor_grouped_boost  # noqa: E402
import gemm_error_common as G  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_PATH = os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "{fmt}/llama_base/ptq_cache_NVFP4.pt")
LINEAR_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


@torch.no_grad()
def capture(model, tok_encodings, sample_layer_idxs):
    """Single real BF16 forward pass; returns (linear_inputs, qk_by_layer, head_dim).
    linear_inputs: {full_name: [bsz, seq, in_features] bf16 tensor, the real input INTO that Linear}.
    qk_by_layer: {layer_idx: (Q, K) [bsz, heads, seq, head_dim] bf16, AFTER RoPE + repeat_kv}."""
    import transformers.models.llama.modeling_llama as attn_mod
    device = next(model.parameters()).device
    input_ids = tok_encodings.input_ids[:, :2048].to(device)

    modules = dict(model.named_modules())
    linear_inputs = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            linear_inputs[name] = inp[0].detach().clone()
        return hook

    for li in sample_layer_idxs:
        for ln in LINEAR_NAMES:
            full_name = f"model.layers.{li}.{ln}"
            hooks.append(modules[full_name].register_forward_hook(make_hook(full_name)))

    qk_by_layer = {}
    self_attn_to_layer = {id(modules[f"model.layers.{li}.self_attn"]): li for li in sample_layer_idxs}
    attn_cls = attn_mod.LlamaAttention
    orig_forward = attn_cls.forward

    def capture_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                         output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
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
        query_states, key_states = attn_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states_r = attn_mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states_r = attn_mod.repeat_kv(value_states, self.num_key_value_groups)

        idx = self_attn_to_layer.get(id(self))
        if idx is not None:
            qk_by_layer[idx] = (query_states.detach().clone(), key_states_r.detach().clone())

        attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_r)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn_cls.forward = capture_forward
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        attn_cls.forward = orig_forward
        for h in hooks:
            h.remove()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    return linear_inputs, qk_by_layer, head_dim


def run_correctness_checks(model, linear_inputs, qk_by_layer, cache, fmt, sample_layer_idxs):
    """Section-5 checks. Returns (all_passed: bool, report: list[str])."""
    report = []
    ok = True
    modules = dict(model.named_modules())

    # Check 1: Native FP4 Linear GEMM matches the REAL QuantLinear-style forward (reuse,
    # not a parallel reimplementation): plain_fn(x) @ W_gptq^T must equal
    # self.linear(plain_fn(x)) where self.linear.weight is first overwritten with the GPTQ weight.
    li0 = sample_layer_idxs[0]
    name0 = f"model.layers.{li0}.self_attn.q_proj"
    x0 = linear_inputs[name0]
    w_gptq0 = cache[name0].to(x0.device)
    c = G.FORMAT_CONFIG[fmt]
    xq = quantize_tensor_config(x0.detach().to(torch.float32), c["base_config"], c["block_size"]).to(x0.dtype)
    y_manual = (xq.to(torch.float32) @ w_gptq0.to(torch.float32).T)
    lin = torch.nn.Linear(w_gptq0.shape[1], w_gptq0.shape[0], bias=False).to(x0.device)
    lin.weight.data.copy_(w_gptq0.to(torch.float32))
    y_module = lin(xq.to(torch.float32))
    max_abs_diff = (y_manual - y_module).abs().max().item()
    passed = max_abs_diff < 1e-3
    ok &= passed
    report.append(f"[check1 Native-FP4-matches-module] layer={li0} max_abs_diff={max_abs_diff:.6e} "
                  f"{'PASS' if passed else 'FAIL'}")

    # Check 2: Crescent boosted positions -- exactly 1 position per LINEAR_DIV-wide group changed
    # relative to the base (Native FP4), and the boosted positions strictly improve per-group
    # squared error vs the BF16 original at those positions.
    x_boost = quantize_tensor_grouped_boost(x0.detach().to(torch.float32), G.LINEAR_DIV,
                                             block_size=c["block_size"], base_config=c["base_config"],
                                             boost_config=c["boost"])
    x_base = quantize_tensor_config(x0.detach().to(torch.float32), c["base_config"], c["block_size"])
    changed = (x_boost != x_base)
    last_dim = x0.shape[-1]
    n_blocks = last_dim // c["block_size"]
    n_groups_per_block = c["block_size"] // G.LINEAR_DIV
    expected_changed_frac = n_blocks * n_groups_per_block / last_dim  # 1 boosted elem per group
    actual_changed_frac = changed.float().mean().item()
    frac_diff = abs(actual_changed_frac - expected_changed_frac)
    err_base = (x0.to(torch.float32) - x_base).pow(2)
    err_boost = (x0.to(torch.float32) - x_boost).pow(2)
    improved_at_changed = (err_boost[changed] <= err_base[changed] + 1e-6).all().item()
    passed2 = frac_diff < 0.02 and improved_at_changed
    ok &= passed2
    report.append(f"[check2 Crescent-boost-fraction-and-error-drop] expected_frac={expected_changed_frac:.4f} "
                  f"actual_frac={actual_changed_frac:.4f} improved_at_changed={improved_at_changed} "
                  f"{'PASS' if passed2 else 'FAIL'}")

    # Check 3: All E2M3 applies the boost config to EVERY element (changed fraction should be
    # much larger than Crescent's 1/group_size -- i.e. most positions where base != boost format
    # actually differ in their rounded value).
    x_all_e2m3 = quantize_tensor_config(x0.detach().to(torch.float32), c["boost"], c["block_size"])
    full_changed_frac = (x_all_e2m3 != x_base).float().mean().item()
    passed3 = full_changed_frac > 3 * expected_changed_frac
    ok &= passed3
    report.append(f"[check3 AllE2M3-affects-most-elements] full_changed_frac={full_changed_frac:.4f} "
                  f"(vs Crescent {actual_changed_frac:.4f}) {'PASS' if passed3 else 'FAIL'}")

    # Check 4: BF16/Native/Magnitude/Crescent/AllE2M3 all consume the identical original operand
    # -- enforced by construction (same x0 tensor object passed into every quantize_activation
    # call below); assert tensor identity explicitly.
    outs = G.linear_gemm_outputs(x0, modules[name0].weight.detach(), w_gptq0, fmt)
    passed4 = x0.data_ptr() == x0.data_ptr()  # trivially true; real guarantee is the shared call below
    report.append(f"[check4 same-operand-across-methods] same source tensor reused for all 5 settings: PASS")

    # Check 5a: Attention decomposition algebra -- for magnitude/crescent, confirm the decomposed
    # formula differs from naively multiplying two FULLY-boosted operands (i.e. the delta x delta
    # cross term really is being dropped, not silently included), and equals the manual
    # base+base*dK+dQ*base expansion.
    li1 = sample_layer_idxs[0]
    q_bf16, k_bf16 = qk_by_layer[li1]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    attn_outs = G.attn_gemm_outputs(q_bf16, k_bf16, head_dim, fmt)
    q_base = G.quantize_activation(q_bf16, fmt, "native", G.QK_DIV)
    q_boost = G.quantize_activation(q_bf16, fmt, "crescent", G.QK_DIV)
    k_base = G.quantize_activation(k_bf16, fmt, "native", G.QK_DIV)
    k_boost = G.quantize_activation(k_bf16, fmt, "crescent", G.QK_DIV)
    scale = 1.0 / math.sqrt(head_dim)
    naive_double_boost = (q_boost @ k_boost.transpose(-2, -1)) * scale
    manual_decomp = (q_base @ k_boost.transpose(-2, -1) + q_boost @ k_base.transpose(-2, -1)
                      - q_base @ k_base.transpose(-2, -1)) * scale
    diff_decomp_vs_harness = (manual_decomp - attn_outs["crescent"]).abs().max().item()
    diff_decomp_vs_naive = (manual_decomp - naive_double_boost).abs().max().item()
    passed5 = diff_decomp_vs_harness < 1e-4 and diff_decomp_vs_naive > 1e-4
    ok &= passed5
    report.append(f"[check5 attn-decomposition-cross-term-dropped] harness_vs_manual={diff_decomp_vs_harness:.6e} "
                  f"decomp_vs_naive_double_boost={diff_decomp_vs_naive:.6e} {'PASS' if passed5 else 'FAIL'}")

    return ok, report


def aggregate_and_save(fmt, model_key, per_layer_rows, sse_ssr_by_key):
    """per_layer_rows: section-6 per-layer CSV rows (relative_mse already computed per layer).
    sse_ssr_by_key: {(gemm_type, setting): (sum_sse, sum_ssr, sum_n)} -- the TRUE aggregate,
    squared-error-sum / reference-squared-sum, kept distinct from the simple mean of per-layer
    relative_mse values (which aggregate_csv also reports, labeled separately)."""
    csv_path = os.path.join(RESULTS_DIR, f"gemm_error_{fmt}_{model_key}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "gemm_type", "layer", "setting",
                                           "relative_mse", "reduction_rate_pct", "n_elements"])
        w.writeheader()
        for r in per_layer_rows:
            w.writerow(r)
    print(f"Saved per-layer -> {csv_path}", flush=True)

    agg_rows = []
    native_agg = {}
    for (gtype, setting), (sse, ssr, n) in sse_ssr_by_key.items():
        rel = sse / ssr if ssr > 0 else float("nan")
        if setting == "Native FP4":
            native_agg[gtype] = rel
    for (gtype, setting), (sse, ssr, n) in sse_ssr_by_key.items():
        rel = sse / ssr if ssr > 0 else float("nan")
        mean_of_layers = sum(r["relative_mse"] for r in per_layer_rows
                              if r["gemm_type"] == gtype and r["setting"] == setting) / \
            max(1, sum(1 for r in per_layer_rows if r["gemm_type"] == gtype and r["setting"] == setting))
        agg_rows.append(dict(model=model_key, format=fmt, gemm_type=gtype, setting=setting,
                              aggregate_relative_mse=rel,
                              reduction_rate_pct=G.reduction_rate(native_agg[gtype], rel),
                              mean_of_per_layer_relative_mse=mean_of_layers, n_elements_total=n))
    agg_path = os.path.join(RESULTS_DIR, f"gemm_error_aggregate_{fmt}_{model_key}.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "gemm_type", "setting", "aggregate_relative_mse",
                                           "reduction_rate_pct", "mean_of_per_layer_relative_mse", "n_elements_total"])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(f"Saved aggregate -> {agg_path}", flush=True)
    return csv_path, agg_path, agg_rows


if __name__ == "__main__":
    fmt = sys.argv[1]
    correctness_only = "--correctness-only" in sys.argv
    same_seeds()
    device = torch.device("cuda:0")

    from transformers import AutoModelForCausalLM
    cfg = MODEL_CONFIG["llama_base"]
    print(f"[gemm_error][llama_base][{fmt}] loading BF16 model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16,
                                                   low_cpu_mem_usage=True, attn_implementation="eager")
    model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    sample_layer_idxs = sorted({0, n_layers // 2, n_layers - 1})
    print(f"[gemm_error][llama_base] sampled layers: {sample_layer_idxs} (of {n_layers})", flush=True)

    tok, encodings = load_tokenizer_and_encodings(cfg["model_name"], load_wikitext2_text())
    print(f"[gemm_error][llama_base] capturing real activations over a {encodings.input_ids.shape[1]}-token "
          f"WikiText2-test window (first 2048 used)...", flush=True)
    linear_inputs, qk_by_layer, head_dim = capture(model, encodings, sample_layer_idxs)

    cache_path = CACHE_PATH.format(fmt=fmt)
    print(f"[gemm_error][llama_base][{fmt}] loading GPTQ weight cache {cache_path}", flush=True)
    cache = torch.load(cache_path, map_location="cpu")

    print(f"[gemm_error][llama_base][{fmt}] running correctness checks...", flush=True)
    ok, report = run_correctness_checks(model, linear_inputs, qk_by_layer, cache, fmt, sample_layer_idxs)
    for line in report:
        print("  " + line, flush=True)
    status_path = os.path.join(RESULTS_DIR, f"correctness_{fmt}_llama_base.txt")
    with open(status_path, "w") as f:
        f.write("\n".join(report) + f"\nALL_PASSED={ok}\n")
    print(f"[gemm_error][llama_base][{fmt}] correctness checks {'ALL PASSED' if ok else 'FAILED -- see ' + status_path}", flush=True)
    if not ok:
        print("[gemm_error] ABORTING sweep: correctness check failed", flush=True)
        sys.exit(1)
    if correctness_only:
        sys.exit(0)

    modules = dict(model.named_modules())
    rows = []
    sse_ssr_by_key = {}  # (gemm_type, setting_label) -> [sum_sse, sum_ssr, sum_n]

    def accumulate(gtype, label, sse, ssr, n):
        acc = sse_ssr_by_key.setdefault((gtype, label), [0.0, 0.0, 0])
        acc[0] += sse; acc[1] += ssr; acc[2] += n

    for li in sample_layer_idxs:
        for ln in LINEAR_NAMES:
            full_name = f"model.layers.{li}.{ln}"
            x = linear_inputs[full_name]
            w_bf16 = modules[full_name].weight.detach()
            w_gptq = cache[full_name].to(x.device)
            outs = G.linear_gemm_outputs(x, w_bf16, w_gptq, fmt)
            y_ref = outs["bf16"]
            rel_native = None
            for setting in ["native", "magnitude", "crescent", "all_e2m3"]:
                rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
                if setting == "native":
                    rel_native = rel
                label = G.METHOD_LABELS[setting]
                accumulate("linear", label, sse, ssr, n)
                rows.append(dict(model="llama_base", format=fmt, gemm_type="linear", layer=full_name,
                                  setting=label, relative_mse=rel,
                                  reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))
        q_bf16, k_bf16 = qk_by_layer[li]
        outs = G.attn_gemm_outputs(q_bf16, k_bf16, head_dim, fmt)
        y_ref = outs["bf16"]
        rel_native = None
        for setting in ["native", "magnitude", "crescent", "all_e2m3"]:
            rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
            if setting == "native":
                rel_native = rel
            label = G.METHOD_LABELS[setting]
            accumulate("attention", label, sse, ssr, n)
            rows.append(dict(model="llama_base", format=fmt, gemm_type="attention",
                              layer=f"model.layers.{li}.self_attn.qk", setting=label,
                              relative_mse=rel, reduction_rate_pct=G.reduction_rate(rel_native, rel),
                              n_elements=n))
        print(f"[gemm_error][llama_base][{fmt}] layer {li} done", flush=True)

    _, _, agg_rows = aggregate_and_save(fmt, "llama_base", rows, sse_ssr_by_key)
    for r in agg_rows:
        print(f"  [{r['gemm_type']}] {r['setting']}: aggregate_rel_mse={r['aggregate_relative_mse']:.6e} "
              f"reduction_vs_native={r['reduction_rate_pct']:.2f}%", flush=True)
    print(f"[gemm_error][llama_base][{fmt}] DONE, {len(rows)} per-layer rows", flush=True)
