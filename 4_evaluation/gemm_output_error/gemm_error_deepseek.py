"""GEMM Output Error evaluation for DeepSeek-MoE-16B-base (deepseek_moe_16b).

Same single-real-forward-pass capture strategy as gemm_error_llama.py, adapted to
common/deepseek_moe_model.py's pure-PyTorch port: Attention is a plain nn.Module class (no
GQA/repeat_kv -- q/k share head count directly) with inline RoPE (non-interleaved rotate_half,
theta-based), and every Linear (attention projections, dense-layer MLP, every routed+shared MoE
expert) is a plain nn.Linear, keyed in the GPTQ cache by its exact named_modules() dotted name
(see weight_gptq_deepseek_moe.py's build_cache: target_names = every nn.Linear except lm_head).

Sampled decoder layers: layer 0 (dense MLP, first_k_dense_replace), one early-MoE and one
late-MoE layer. For MoE layers, ALL experts under that layer are hooked; whichever expert(s)
actually receive routed tokens for the sampled input fire naturally (sparse real routing, not
synthetic) -- the first Linear name that captures a non-empty input per (layer, projection kind)
is kept.

Usage: CUDA_VISIBLE_DEVICES=<free GPUs> python3 gemm_error_deepseek.py <nvfp4|mxfp4> [--correctness-only]
"""
import csv
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
sys.path.insert(0, HERE)
from eval_utils import same_seeds, load_wikitext2_text, load_tokenizer_and_encodings  # noqa: E402
from formats import quantize_tensor_config, quantize_tensor_grouped_boost  # noqa: E402
import deepseek_moe_model as dsm  # noqa: E402
import gemm_error_common as G  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CACHE_PATH = os.path.join(os.environ.get("PTQ_CACHE_ROOT", os.path.expanduser("~/ptq_cache")), "{fmt}/deepseek_moe_16b/ptq_cache_NVFP4.pt")


@torch.no_grad()
def capture(model, tok_encodings, sample_layer_idxs):
    device = next(model.parameters()).device
    input_ids = tok_encodings.input_ids[:, :2048].to(device)

    modules = dict(model.named_modules())
    linear_inputs = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if name not in linear_inputs:  # keep the FIRST real activation (e.g. first routed expert)
                linear_inputs[name] = inp[0].detach().clone()
        return hook

    hooked_names = []
    for li in sample_layer_idxs:
        prefix = f"model.layers.{li}."
        for name, m in modules.items():
            if name.startswith(prefix) and isinstance(m, nn.Linear):
                hooked_names.append(name)
                hooks.append(m.register_forward_hook(make_hook(name)))

    qk_by_layer = {}
    self_attn_to_layer = {}
    for li in sample_layer_idxs:
        self_attn_to_layer[id(modules[f"model.layers.{li}.self_attn"])] = li

    import types

    def capture_forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=x.device).float() / self.hd))
        fr = torch.outer(torch.arange(S, device=x.device, dtype=inv.dtype), inv)
        emb = torch.cat((fr, fr), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)

        def rotate_half(t):
            h = t.shape[-1] // 2
            return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        idx = self_attn_to_layer.get(id(self))
        if idx is not None:
            qk_by_layer[idx] = (q.detach().clone(), k.detach().clone())

        w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd)
        mask = torch.full((S, S), float("-inf"), device=x.device, dtype=w.dtype).triu(1)
        w = w + mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))

    # Patch at the INSTANCE level (not the class), and only for the sampled layers: accelerate's
    # device_map="auto" dispatch can attach its own instance-level forward override (device-align
    # hooks) to some layers, which would silently shadow a class-level monkeypatch for exactly
    # those instances -- observed in practice (layer 0's self_attn never fired the class patch).
    # Apply AFTER dispatch so this always wins, and save/restore whatever was in __dict__ before
    # (None = no instance override existed, just delete ours to fall back to the class method).
    patched_instances = []
    for li in sample_layer_idxs:
        attn_instance = modules[f"model.layers.{li}.self_attn"]
        prev = attn_instance.__dict__.get("forward", None)
        patched_instances.append((attn_instance, prev))
        attn_instance.forward = types.MethodType(capture_forward, attn_instance)
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for attn_instance, prev in patched_instances:
            if prev is None:
                attn_instance.__dict__.pop("forward", None)
            else:
                attn_instance.forward = prev
        for h in hooks:
            h.remove()

    head_dim = model.cfg["hidden_size"] // model.cfg["num_attention_heads"]
    return linear_inputs, qk_by_layer, head_dim, hooked_names


def pick_representative_linears(hooked_names, linear_inputs, sample_layer_idxs):
    """For each sampled layer, pick one representative attention projection set
    (q/k/v/o -- always fire, not sparse) and one representative MLP triple
    (gate/up/down -- for MoE layers, the first expert that actually received
    tokens; for the dense layer, the single shared gate/up/down)."""
    chosen = {}
    for li in sample_layer_idxs:
        prefix = f"model.layers.{li}."
        for kind in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
            name = prefix + kind
            if name in linear_inputs:
                chosen[name] = name
        mlp_candidates = sorted(n for n in hooked_names if n.startswith(prefix + "mlp.") and n in linear_inputs)
        # dense layer: mlp.gate_proj/up_proj/down_proj. MoE layer: mlp.experts.{e}.{gate,up,down}_proj
        # and mlp.shared_experts.{gate,up,down}_proj. Keep ONE firing expert's triple (lowest index)
        # plus the shared-experts triple (always fires, every token).
        picked_expert_prefix = None
        for n in mlp_candidates:
            if ".experts." in n:
                ep = n.rsplit(".", 1)[0]  # strip "gate_proj"/"up_proj"/"down_proj"
                if picked_expert_prefix is None:
                    picked_expert_prefix = ep
                if n.startswith(picked_expert_prefix):
                    chosen[n] = n
            else:
                chosen[n] = n  # dense mlp.* or mlp.shared_experts.*
    return chosen


def run_correctness_checks(model, chosen_names, linear_inputs, qk_by_layer, cache, fmt, sample_layer_idxs):
    report = []
    ok = True
    modules = dict(model.named_modules())
    c = G.FORMAT_CONFIG[fmt]

    name0 = next(n for n in chosen_names if n.endswith("q_proj"))
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
    q_bf16, k_bf16 = qk_by_layer[li1]
    head_dim = model.cfg["hidden_size"] // model.cfg["num_attention_heads"]
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
    report.append(f"[check5 attn-decomposition-cross-term-dropped] harness_vs_manual={diff_vs_harness:.6e} decomp_vs_naive={diff_vs_naive:.6e} {'PASS' if passed5 else 'FAIL'}")
    return ok, report


if __name__ == "__main__":
    fmt = sys.argv[1]
    correctness_only = "--correctness-only" in sys.argv
    same_seeds()
    os.environ.setdefault("DEEPSEEK_MAX_MEM_CPU", "150GiB")

    print(f"[gemm_error][deepseek_moe_16b][{fmt}] loading model (accelerate device_map=auto)...", flush=True)
    model = dsm.load_model()
    model.eval()

    n_layers = len(model.model.layers)
    first_k_dense = model.cfg["first_k_dense_replace"]
    moe_freq = model.cfg["moe_layer_freq"]
    moe_layers = [i for i in range(n_layers) if i >= first_k_dense and i % moe_freq == 0]
    sample_layer_idxs = sorted({0, moe_layers[len(moe_layers) // 3], moe_layers[-1]})
    print(f"[gemm_error][deepseek_moe_16b] sampled layers: {sample_layer_idxs} (of {n_layers}, dense<{first_k_dense})", flush=True)

    tok, encodings = load_tokenizer_and_encodings(dsm.MODEL_NAME, load_wikitext2_text())
    print(f"[gemm_error][deepseek_moe_16b] capturing real activations over a 2048-token WikiText2-test window...", flush=True)
    linear_inputs, qk_by_layer, head_dim, hooked_names = capture(model, encodings, sample_layer_idxs)
    chosen = pick_representative_linears(hooked_names, linear_inputs, sample_layer_idxs)
    print(f"[gemm_error][deepseek_moe_16b] {len(chosen)} representative Linear GEMMs selected "
          f"(includes 1 fired expert per MoE layer): {sorted(chosen)}", flush=True)

    cache_path = CACHE_PATH.format(fmt=fmt)
    print(f"[gemm_error][deepseek_moe_16b][{fmt}] loading GPTQ weight cache {cache_path}", flush=True)
    cache = torch.load(cache_path, map_location="cpu")

    print(f"[gemm_error][deepseek_moe_16b][{fmt}] running correctness checks...", flush=True)
    ok, report = run_correctness_checks(model, list(chosen), linear_inputs, qk_by_layer, cache, fmt, sample_layer_idxs)
    for line in report:
        print("  " + line, flush=True)
    status_path = os.path.join(RESULTS_DIR, f"correctness_{fmt}_deepseek_moe_16b.txt")
    with open(status_path, "w") as f:
        f.write("\n".join(report) + f"\nALL_PASSED={ok}\n")
    print(f"[gemm_error][deepseek_moe_16b][{fmt}] correctness checks {'ALL PASSED' if ok else 'FAILED -- see ' + status_path}", flush=True)
    if not ok:
        sys.exit(1)
    if correctness_only:
        sys.exit(0)

    rows = []
    sse_ssr_by_key = {}

    def accumulate(gtype, label, sse, ssr, n):
        acc = sse_ssr_by_key.setdefault((gtype, label), [0.0, 0.0, 0])
        acc[0] += sse; acc[1] += ssr; acc[2] += n

    modules = dict(model.named_modules())
    for name in sorted(chosen):
        x = linear_inputs[name]
        w_bf16 = modules[name].weight.detach()
        if name not in cache:
            print(f"  [skip] {name} not in GPTQ cache", flush=True)
            continue
        w_gptq = cache[name].to(x.device)
        outs = G.linear_gemm_outputs(x, w_bf16, w_gptq, fmt)
        y_ref = outs["bf16"]
        rel_native = None
        for setting in ["native", "magnitude", "czardas", "fullboost"]:
            rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
            if setting == "native":
                rel_native = rel
            label = G.METHOD_LABELS[setting]
            accumulate("linear", label, sse, ssr, n)
            rows.append(dict(model="deepseek_moe_16b", format=fmt, gemm_type="linear", layer=name,
                              setting=label, relative_mse=rel, reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))

    for li in sample_layer_idxs:
        if li not in qk_by_layer:
            continue
        q_bf16, k_bf16 = qk_by_layer[li]
        outs = G.attn_gemm_outputs(q_bf16, k_bf16, head_dim, fmt)
        y_ref = outs["bf16"]
        rel_native = None
        for setting in ["native", "magnitude", "czardas", "fullboost"]:
            rel, sse, ssr, n = G.relative_mse(outs[setting], y_ref)
            if setting == "native":
                rel_native = rel
            label = G.METHOD_LABELS[setting]
            accumulate("attention", label, sse, ssr, n)
            rows.append(dict(model="deepseek_moe_16b", format=fmt, gemm_type="attention",
                              layer=f"model.layers.{li}.self_attn.qk", setting=label,
                              relative_mse=rel, reduction_rate_pct=G.reduction_rate(rel_native, rel), n_elements=n))

    csv_path = os.path.join(RESULTS_DIR, f"gemm_error_{fmt}_deepseek_moe_16b.csv")
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
        agg_rows.append(dict(model="deepseek_moe_16b", format=fmt, gemm_type=gtype, setting=setting,
                              aggregate_relative_mse=rel, reduction_rate_pct=G.reduction_rate(native_agg[gtype], rel),
                              mean_of_per_layer_relative_mse=mean_layers, n_elements_total=n))
    agg_path = os.path.join(RESULTS_DIR, f"gemm_error_aggregate_{fmt}_deepseek_moe_16b.csv")
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
    print(f"[gemm_error][deepseek_moe_16b][{fmt}] DONE, {len(rows)} per-layer rows", flush=True)
