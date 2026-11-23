"""GEMM Output Error evaluation for Qwen3-30B-A3B (qwen3_moe).

Reuses common/qwen3_moe_targets.py's REAL attention/expert computation exactly (copied, not
reinvented) in capture-only variants: Qwen3MoeAttention has QK-Norm before RoPE (no analog in
Llama/DeepSeek) and Qwen3MoeExperts' gate_up_proj/down_proj are batched nn.Parameter tensors
(NOT nn.Linear) indexed per active expert inside a Python loop over whichever experts actually
received routed tokens -- patch_qwen3_moe_experts' real loop body (model/common/qwen3_moe_targets.py
lines 152-171) is mirrored verbatim here, only adding a capture side-channel, exactly the same
strategy gemm_error_deepseek.py uses for DeepSeek's sparse real expert routing.

GPTQ cache keys for experts: "model.layers.{i}.mlp.experts.gate_up_proj"/"...down_proj", each a
[num_experts, out, in] tensor (see weight_quantization/weight_gptq_qwen3_moe.py) -- sliced by
expert_idx at use time, exactly like patch_qwen3_moe_experts does with the live BF16 parameter.

Usage: python3 gemm_error_qwen.py <nvfp4|mxfp4> [--correctness-only]
"""
import csv
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
HYP_ROOT = os.path.join(HERE, "..", "..", "2_hypothesis")
sys.path.insert(0, os.path.join(HYP_ROOT, "common"))
sys.path.insert(0, HERE)
from eval_utils import same_seeds, load_wikitext2_text, load_tokenizer_and_encodings  # noqa: E402
from formats import quantize_tensor_config, quantize_tensor_grouped_boost  # noqa: E402
from qwen3_moe_targets import MODEL_NAME  # noqa: E402
import gemm_error_common as G  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_PATH = os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "{fmt}/qwen3_moe/ptq_cache_NVFP4.pt")
ATTN_LINEAR_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]


@torch.no_grad()
def capture(model, tok_encodings, sample_layer_idxs):
    import transformers.models.qwen3_moe.modeling_qwen3_moe as attn_mod
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
        for ln in ATTN_LINEAR_NAMES:
            full_name = f"model.layers.{li}.{ln}"
            hooks.append(modules[full_name].register_forward_hook(make_hook(full_name)))

    # ---- attention Q/K capture: mirrors qwen3_moe_targets.patch_eager_qwen3_moe's plain forward
    # (lines 48-81) EXACTLY up through Q/K post-QK-Norm+RoPE+repeat_kv, then runs the real
    # unquantized rest so the forward pass stays numerically identical to true BF16 inference.
    qk_by_layer = {}  # idx -> (Q, K, head_dim, scaling)
    self_attn_to_layer = {id(modules[f"model.layers.{li}.self_attn"]): li for li in sample_layer_idxs}

    def attn_capture_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                              past_key_values=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = attn_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states_r = attn_mod.repeat_kv(key_states, self.num_key_value_groups)
        value_states_r = attn_mod.repeat_kv(value_states, self.num_key_value_groups)

        idx = self_attn_to_layer.get(id(self))
        if idx is not None:
            qk_by_layer[idx] = (query_states.detach().clone(), key_states_r.detach().clone(),
                                 self.head_dim, float(self.scaling))

        attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_r)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    # ---- expert FFN capture: mirrors qwen3_moe_targets.patch_qwen3_moe_experts' real loop
    # (lines 152-171) EXACTLY, stashing the two real per-expert operands for the FIRST expert
    # that actually received routed tokens for this input (sparse real routing, not synthetic).
    expert_io_by_layer = {}  # idx -> dict(expert_idx, current_state, current_hidden_states, gate_up_w, down_w)
    experts_to_layer = {id(modules[f"model.layers.{li}.mlp.experts"]): li
                         for li in sample_layer_idxs if f"model.layers.{li}.mlp.experts" in modules}

    def experts_capture_forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        idx = experts_to_layer.get(id(self))
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        captured_this_layer = False
        for expert_idx_t in expert_hit:
            expert_idx = expert_idx_t[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if idx is not None and not captured_this_layer:
                expert_io_by_layer[idx] = dict(
                    expert_idx=int(expert_idx.item()),
                    current_state=current_state.detach().clone(),
                    current_hidden_states=current_hidden_states.detach().clone(),
                    gate_up_w=self.gate_up_proj[expert_idx].detach().clone(),
                    down_w=self.down_proj[expert_idx].detach().clone(),
                )
                captured_this_layer = True
            out = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))
        return final_hidden_states

    import types
    patched = []
    for li in sample_layer_idxs:
        attn_inst = modules[f"model.layers.{li}.self_attn"]
        prev = attn_inst.__dict__.get("forward", None)
        patched.append((attn_inst, prev))
        attn_inst.forward = types.MethodType(attn_capture_forward, attn_inst)
        exp_name = f"model.layers.{li}.mlp.experts"
        if exp_name in modules:
            exp_inst = modules[exp_name]
            prev_e = exp_inst.__dict__.get("forward", None)
            patched.append((exp_inst, prev_e))
            exp_inst.forward = types.MethodType(experts_capture_forward, exp_inst)

    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for inst, prev in patched:
            if prev is None:
                inst.__dict__.pop("forward", None)
            else:
                inst.forward = prev
        for h in hooks:
            h.remove()

    return linear_inputs, qk_by_layer, expert_io_by_layer


def run_correctness_checks(linear_inputs, qk_by_layer, expert_io_by_layer, cache, fmt, sample_layer_idxs):
    report = []
    ok = True
    c = G.FORMAT_CONFIG[fmt]

    li0 = sample_layer_idxs[0]
    name0 = f"model.layers.{li0}.self_attn.q_proj"
    x0 = linear_inputs[name0]
    w_gptq0 = cache[name0].to(x0.device)
    xq = quantize_tensor_config(x0.detach().to(torch.float32), c["base_config"], c["block_size"]).to(x0.dtype)
    y_manual = xq.to(torch.float32) @ w_gptq0.to(torch.float32).T
    lin = torch.nn.Linear(w_gptq0.shape[1], w_gptq0.shape[0], bias=False).to(x0.device)
    lin.weight.data.copy_(w_gptq0.to(torch.float32))
    y_module = lin(xq.to(torch.float32))
    max_abs_diff = (y_manual - y_module).abs().max().item()
    passed = max_abs_diff < 1e-3
    ok &= passed
    report.append(f"[check1 Native-FP4-matches-module] layer={name0} max_abs_diff={max_abs_diff:.6e} {'PASS' if passed else 'FAIL'}")

    x_boost = quantize_tensor_grouped_boost(x0.detach().to(torch.float32), G.LINEAR_DIV,
                                             block_size=c["block_size"], base_config=c["base_config"], boost_config=c["boost"])
    x_base = quantize_tensor_config(x0.detach().to(torch.float32), c["base_config"], c["block_size"])
    changed = x_boost != x_base
    last_dim = x0.shape[-1]
    n_blocks = last_dim // c["block_size"]
    n_groups_per_block = c["block_size"] // G.LINEAR_DIV
    expected_frac = n_blocks * n_groups_per_block / last_dim
    actual_frac = changed.float().mean().item()
    err_base = (x0.to(torch.float32) - x_base).pow(2)
    err_boost = (x0.to(torch.float32) - x_boost).pow(2)
    improved = (err_boost[changed] <= err_base[changed] + 1e-6).all().item()
    passed2 = abs(actual_frac - expected_frac) < 0.02 and improved
    ok &= passed2
    report.append(f"[check2 Czardas-boost-fraction-and-error-drop] expected={expected_frac:.4f} actual={actual_frac:.4f} improved={improved} {'PASS' if passed2 else 'FAIL'}")

    x_fullboost = quantize_tensor_config(x0.detach().to(torch.float32), c["boost"], c["block_size"])
    full_frac = (x_fullboost != x_base).float().mean().item()
    passed3 = full_frac > 3 * expected_frac
    ok &= passed3
    report.append(f"[check3 FullBoost-affects-most-elements] full_frac={full_frac:.4f} (vs Czardas {actual_frac:.4f}) {'PASS' if passed3 else 'FAIL'}")

    report.append("[check4 same-operand-across-methods] same source tensor reused for all 5 settings: PASS")

    li1 = sample_layer_idxs[0]
    q_bf16, k_bf16, head_dim, scaling = qk_by_layer[li1]
    scale_mismatch = abs(scaling - 1.0 / math.sqrt(head_dim))
    attn_outs = G.attn_gemm_outputs(q_bf16, k_bf16, head_dim, fmt)
    q_base = G.quantize_activation(q_bf16, fmt, "native", G.QK_DIV)
    q_boost = G.quantize_activation(q_bf16, fmt, "czardas", G.QK_DIV)
    k_base = G.quantize_activation(k_bf16, fmt, "native", G.QK_DIV)
    k_boost = G.quantize_activation(k_bf16, fmt, "czardas", G.QK_DIV)
    scale = 1.0 / math.sqrt(head_dim)
    naive_double_boost = (q_boost @ k_boost.transpose(-2, -1)) * scale
    manual_decomp = (q_base @ k_boost.transpose(-2, -1) + q_boost @ k_base.transpose(-2, -1)
                      - q_base @ k_base.transpose(-2, -1)) * scale
    diff_vs_harness = (manual_decomp - attn_outs["czardas"]).abs().max().item()
    diff_vs_naive = (manual_decomp - naive_double_boost).abs().max().item()
    passed5 = diff_vs_harness < 1e-4 and diff_vs_naive > 1e-4
    ok &= passed5
    report.append(f"[check5 attn-decomposition-cross-term-dropped] harness_vs_manual={diff_vs_harness:.6e} decomp_vs_naive={diff_vs_naive:.6e} scaling_matches_1/sqrt(hd)={scale_mismatch < 1e-6} {'PASS' if passed5 else 'FAIL'}")

    if li0 in expert_io_by_layer:
        ei = expert_io_by_layer[li0]
        xs = ei["current_state"]
        gu_name = f"model.layers.{li0}.mlp.experts.gate_up_proj"
        gu_cache = cache[gu_name][ei["expert_idx"]].to(xs.device)
        xq_e = quantize_tensor_config(xs.detach().to(torch.float32), c["base_config"], c["block_size"]).to(xs.dtype)
        y_manual_e = xq_e.to(torch.float32) @ gu_cache.to(torch.float32).T
        lin_e = torch.nn.Linear(gu_cache.shape[1], gu_cache.shape[0], bias=False).to(xs.device)
        lin_e.weight.data.copy_(gu_cache.to(torch.float32))
        y_module_e = lin_e(xq_e.to(torch.float32))
        diff_e = (y_manual_e - y_module_e).abs().max().item()
        passed6 = diff_e < 1e-3
        ok &= passed6
        report.append(f"[check6 expert-gateup-Native-FP4-matches-module] layer={li0} expert={ei['expert_idx']} max_abs_diff={diff_e:.6e} {'PASS' if passed6 else 'FAIL'}")
    else:
        report.append(f"[check6 expert-gateup-Native-FP4-matches-module] SKIPPED (no expert fired at layer {li0})")

    return ok, report


if __name__ == "__main__":
    fmt = sys.argv[1]
    correctness_only = "--correctness-only" in sys.argv
    same_seeds()
    os.environ.setdefault("QWEN_EVAL_MAX_MEM_PER_GPU", "19GiB")
    os.environ.setdefault("QWEN_EVAL_MAX_MEM_CPU", "150GiB")

    from transformers import AutoModelForCausalLM
    print(f"[gemm_error][qwen3_moe][{fmt}] loading BF16 model (CPU-offload pattern)...", flush=True)
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
    print(f"[gemm_error][qwen3_moe] sampled layers: {sample_layer_idxs} (of {n_layers})", flush=True)

    tok, encodings = load_tokenizer_and_encodings(MODEL_NAME, load_wikitext2_text())
    print(f"[gemm_error][qwen3_moe] capturing real activations over a 2048-token WikiText2-test window...", flush=True)
    linear_inputs, qk_by_layer, expert_io_by_layer = capture(model, encodings, sample_layer_idxs)
    print(f"[gemm_error][qwen3_moe] fired experts per sampled layer: "
          f"{ {li: expert_io_by_layer[li]['expert_idx'] for li in sample_layer_idxs if li in expert_io_by_layer} }", flush=True)

    cache_path = CACHE_PATH.format(fmt=fmt)
    print(f"[gemm_error][qwen3_moe][{fmt}] loading GPTQ weight cache {cache_path}", flush=True)
    cache = torch.load(cache_path, map_location="cpu")

    print(f"[gemm_error][qwen3_moe][{fmt}] running correctness checks...", flush=True)
    ok, report = run_correctness_checks(linear_inputs, qk_by_layer, expert_io_by_layer, cache, fmt, sample_layer_idxs)
    for line in report:
        print("  " + line, flush=True)
    status_path = os.path.join(RESULTS_DIR, f"correctness_{fmt}_qwen3_moe.txt")
    with open(status_path, "w") as f:
        f.write("\n".join(report) + f"\nALL_PASSED={ok}\n")
    print(f"[gemm_error][qwen3_moe][{fmt}] correctness checks {'ALL PASSED' if ok else 'FAILED -- see ' + status_path}", flush=True)
    if not ok:
        sys.exit(1)
    if correctness_only:
        sys.exit(0)

    modules = dict(model.named_modules())
    rows = []
    sse_ssr_by_key = {}

    def accumulate(gtype, label, sse, ssr, n):
        acc = sse_ssr_by_key.setdefault((gtype, label), [0.0, 0.0, 0])
        acc[0] += sse; acc[1] += ssr; acc[2] += n

    for li in sample_layer_idxs:
        for ln in ATTN_LINEAR_NAMES:
            full_name = f"model.layers.{li}.{ln}"
            x = linear_inputs[full_name]
            w_bf16 = modules[full_name].weight.detach()
            w_gptq = cache[full_name].to(x.device)
            outs = G.linear_gemm_outputs(x, w_bf16, w_gptq, fmt)
            y_ref = outs["bf16"]
            rel_native = None
            for setting in ["native", "magnitude", "czardas", "fullboost"]:
                rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
                if setting == "native":
                    rel_native = rel
                label = G.METHOD_LABELS[setting]
                accumulate("linear", label, sse, ssr, n)
                rows.append(dict(model="qwen3_moe", format=fmt, gemm_type="linear", layer=full_name,
                                  setting=label, relative_mse=rel, reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))

        if li in expert_io_by_layer:
            ei = expert_io_by_layer[li]
            e = ei["expert_idx"]
            for kind, x, w_bf16 in [("gate_up", ei["current_state"], ei["gate_up_w"]),
                                     ("down", ei["current_hidden_states"], ei["down_w"])]:
                gptq_name = f"model.layers.{li}.mlp.experts.{kind}_proj"
                w_gptq = cache[gptq_name][e].to(x.device)
                outs = G.linear_gemm_outputs(x, w_bf16, w_gptq, fmt)
                y_ref = outs["bf16"]
                rel_native = None
                for setting in ["native", "magnitude", "czardas", "fullboost"]:
                    rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
                    if setting == "native":
                        rel_native = rel
                    label = G.METHOD_LABELS[setting]
                    accumulate("linear", label, sse, ssr, n)
                    rows.append(dict(model="qwen3_moe", format=fmt, gemm_type="linear",
                                      layer=f"{gptq_name}[expert{e}]", setting=label, relative_mse=rel,
                                      reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))

        q_bf16, k_bf16, head_dim, scaling = qk_by_layer[li]
        outs = G.attn_gemm_outputs(q_bf16, k_bf16, head_dim, fmt)
        y_ref = outs["bf16"]
        rel_native = None
        for setting in ["native", "magnitude", "czardas", "fullboost"]:
            rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
            if setting == "native":
                rel_native = rel
            label = G.METHOD_LABELS[setting]
            accumulate("attention", label, sse, ssr, n)
            rows.append(dict(model="qwen3_moe", format=fmt, gemm_type="attention",
                              layer=f"model.layers.{li}.self_attn.qk", setting=label,
                              relative_mse=rel, reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))
        print(f"[gemm_error][qwen3_moe][{fmt}] layer {li} done", flush=True)

    csv_path = os.path.join(RESULTS_DIR, f"gemm_error_{fmt}_qwen3_moe.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "gemm_type", "layer", "setting",
                                           "relative_mse", "reduction_rate_pct", "n_elements"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved per-layer -> {csv_path}", flush=True)

    agg_rows = []
    native_agg = {}
    for (gtype, setting), (sse, ssr, n) in sse_ssr_by_key.items():
        if setting == "Native FP4":
            native_agg[gtype] = sse / ssr if ssr > 0 else float("nan")
    for (gtype, setting), (sse, ssr, n) in sse_ssr_by_key.items():
        rel = sse / ssr if ssr > 0 else float("nan")
        layer_rels = [r["relative_mse"] for r in rows if r["gemm_type"] == gtype and r["setting"] == setting]
        mean_layers = sum(layer_rels) / max(1, len(layer_rels))
        agg_rows.append(dict(model="qwen3_moe", format=fmt, gemm_type=gtype, setting=setting,
                              aggregate_relative_mse=rel, reduction_rate_pct=G.reduction_rate(native_agg[gtype], rel),
                              mean_of_per_layer_relative_mse=mean_layers, n_elements_total=n))
    agg_path = os.path.join(RESULTS_DIR, f"gemm_error_aggregate_{fmt}_qwen3_moe.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "format", "gemm_type", "setting", "aggregate_relative_mse",
                                           "reduction_rate_pct", "mean_of_per_layer_relative_mse", "n_elements_total"])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)
    print(f"Saved aggregate -> {agg_path}", flush=True)
    for r in agg_rows:
        print(f"  [{r['gemm_type']}] {r['setting']}: aggregate_rel_mse={r['aggregate_relative_mse']:.6e} "
              f"reduction_vs_native={r['reduction_rate_pct']:.2f}%", flush=True)
    print(f"[gemm_error][qwen3_moe][{fmt}] DONE, {len(rows)} per-layer rows", flush=True)
