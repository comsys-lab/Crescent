#!/usr/bin/env python3
"""
Real prefill profiling for Qwen3-30B-A3B and DeepSeek-MoE-16B-base (MoE architectures).

Generalizes prefill_profile.py:
  - Classifies every nn.Linear by NAME-SUFFIX role (q_proj/k_proj/v_proj/o_proj/
    gate_proj/up_proj/down_proj) *and* its actual (in_features, out_features) shape
    read directly off the module -- so dense-MLP layers, routed-expert layers, and
    shared-expert layers (which can have different intermediate sizes, e.g. DeepSeek's
    layer-0 dense MLP vs its routed experts) land in separate (role, K, N) buckets
    automatically, instead of assuming a fixed shape map per model.
  - Also hooks the attention module itself (whatever class name contains "Attention")
    with the same CUDA-event pre/post pattern, to get a REAL measured time for the
    fused QK^T+softmax+PV kernel(s), analogous to the "attention" kernel-dump bucket
    prefill_profile.py already extracts via torch.profiler for the dense models.

Writes results/prefill_moe_detail.csv: one row per (model, M, role, K, N) bucket with
the real measured ms/forward and instance count, PLUS the real measured attention ms.
No largest-error-selection overhead is applied here -- that's a separate step once we
know which (K,N) shapes actually matter (dominate real time).
"""
import csv, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "prefill_moe_detail_16384.csv")

# HF's own "deepseek-moe-16b-base" remote code doesn't import on this transformers
# version -- reuse the user's own verified pure-PyTorch port instead of fighting that.
sys.path.insert(0, "/hyp_common")

MS = [16384]

MODELS = [
    dict(id="deepseek-ai/deepseek-moe-16b-base", loader="dsm"),
    dict(id="Qwen/Qwen3-30B-A3B", loader="hf"),
]

ROLE_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                  "gate_proj", "up_proj", "down_proj"]


def role_of(name):
    n = name.lower()
    for suf in ROLE_SUFFIXES:
        if n.endswith(suf):
            return "o_proj" if suf == "out_proj" else suf
    return None


def run_model(spec):
    model_id = spec["id"]
    print(f"\n{'='*92}\n{model_id}\n{'='*92}", flush=True)
    # Lower per-GPU weight cap than dsm.load_model()'s/HF's default -- leaves more
    # headroom for the O(seq_len^2) attention-score / logits activations at M=16384
    # (both OOM'd at the wider caps). Local override here, not touching the shared
    # hypothesis_v4/common file (other experiments there rely on its own default).
    ngpu = torch.cuda.device_count()
    # DeepSeek weights are ~32.75GB total, Qwen3 ~61GB total (4x24GB=96GB card budget) --
    # cap needs to comfortably clear the model's own weight footprint (or accelerate
    # spills to CPU/disk, which silently wrecks forward-pass timing) while still leaving
    # real per-GPU headroom for the O(seq_len^2) activation spike at M=16384.
    cap_gib = 10 if spec["loader"] == "dsm" else 20
    max_memory = {i: f"{cap_gib}GiB" for i in range(ngpu)}
    if spec["loader"] == "dsm":
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from huggingface_hub import snapshot_download
        import deepseek_moe_model as dsm
        d = snapshot_download(dsm.MODEL_NAME, allow_patterns=["*.safetensors", "*.json"])
        cfg = dsm.load_config(d)
        with init_empty_weights():
            model = dsm.DeepseekMoEForCausalLM(cfg)
        model = load_checkpoint_and_dispatch(model, d, device_map="auto", dtype=torch.bfloat16,
                                             max_memory=max_memory, no_split_module_classes=["DecoderLayer"])
        model = model.eval()
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory).eval()
    first_dev = next(model.parameters()).device
    print(f"  loaded, first param device={first_dev}", flush=True)

    # ---- hook every role-matching nn.Linear, bucketed by (role, K, N) ----
    linears = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            role = role_of(name)
            if role is None:
                continue
            mod._bucket = (role, mod.in_features, mod.out_features)
            mod._ev = []
            linears.append(mod)

    def pre_hook(mod, inp):
        s = torch.cuda.Event(True); s.record(); mod._pending = s

    def post_hook(mod, inp, out):
        e = torch.cuda.Event(True); e.record()
        mod._ev.append((mod._pending, e))

    hs = []
    for m in linears:
        hs.append(m.register_forward_pre_hook(pre_hook))
        hs.append(m.register_forward_hook(post_hook))

    # ---- hook attention modules (class name contains "Attention") ----
    attn_mods = []
    for name, mod in model.named_modules():
        if "Attention" in type(mod).__name__ and not isinstance(mod, torch.nn.Linear):
            mod._ev = []
            attn_mods.append(mod)
    for m in attn_mods:
        hs.append(m.register_forward_pre_hook(pre_hook))
        hs.append(m.register_forward_hook(post_hook))
    print(f"  hooked {len(linears)} Linear instances across "
          f"{len(set(m._bucket for m in linears))} (role,K,N) buckets, "
          f"{len(attn_mods)} attention modules", flush=True)

    rows = []
    for M in MS:
        bs, sl = 1, M
        ids = torch.randint(0, 30000, (bs, sl), device=first_dev)
        torch.cuda.synchronize()
        for d in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
        kw = dict(use_cache=False)
        try:
            import inspect
            sig = inspect.signature(model.forward)
            if "logits_to_keep" in sig.parameters:
                kw["logits_to_keep"] = 1
        except Exception:
            pass
        NREP = 2
        try:
            with torch.inference_mode():
                for _ in range(1):
                    model(ids, **kw)
                torch.cuda.synchronize()
                for m in linears: m._ev.clear()
                for m in attn_mods: m._ev.clear()
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                ev0.record()
                for _ in range(NREP):
                    model(ids, **kw)
                ev1.record()
                torch.cuda.synchronize()
                fwd_ms = ev0.elapsed_time(ev1) / NREP
        except torch.OutOfMemoryError:
            print(f"  M={M:>5}  OOM -> skipped", flush=True)
            for m in linears: m._ev.clear()
            for m in attn_mods: m._ev.clear()
            for d in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
            continue
        except Exception as ex:
            print(f"  M={M:>5}  ERROR: {ex} -> skipped", flush=True)
            for m in linears: m._ev.clear()
            for m in attn_mods: m._ev.clear()
            continue

        per_bucket, inst = {}, {}
        for m in linears:
            t = sum(a.elapsed_time(b) for a, b in m._ev) / NREP
            per_bucket[m._bucket] = per_bucket.get(m._bucket, 0.0) + t
            inst[m._bucket] = inst.get(m._bucket, 0) + 1
            m._ev.clear()
        attn_ms = 0.0
        for m in attn_mods:
            attn_ms += sum(a.elapsed_time(b) for a, b in m._ev) / NREP
            m._ev.clear()

        total_lin = sum(per_bucket.values())
        print(f"  M={M:>5}  fwd={fwd_ms:9.2f} ms   linear={total_lin:9.2f}   attn={attn_ms:8.2f}", flush=True)
        for (role, K, N), t in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
            n = inst[(role, K, N)]
            print(f"    {role:<10} K={K:<6} N={N:<6} n={n:<4} {t:>10.3f} ms/fwd  ({t/n:.5f} ms/call)", flush=True)
            rows.append(dict(model=model_id, M=M, role=role, K=K, N=N, n_instances=n,
                             measured_ms_per_fwd=round(t, 4), fwd_ms=round(fwd_ms, 3),
                             attn_ms_per_fwd=round(attn_ms, 4)))
        rows.append(dict(model=model_id, M=M, role="__attention__", K=0, N=0,
                         n_instances=len(attn_mods), measured_ms_per_fwd=round(attn_ms, 4),
                         fwd_ms=round(fwd_ms, 3), attn_ms_per_fwd=round(attn_ms, 4)))

    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    allrows = []
    for spec in MODELS:
        try:
            allrows += run_model(spec)
        except Exception:
            import traceback; traceback.print_exc()
        # write incrementally so partial progress survives if a later model fails
        if allrows:
            with open(OUT, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
                w.writeheader(); w.writerows(allrows)
            print(f"\n(incremental write) {OUT}", flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
