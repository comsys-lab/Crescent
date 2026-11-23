#!/usr/bin/env python3
"""
Per-GEMM-kernel rollup of the NVFP4 largest-error-selection overhead onto real prefill.

For Llama-3.1-8B-Instruct, OPT-6.7B, Mistral-7B-v0.3, INFERENCE PREFILL:

  1. Run one real prefill forward on this GPU. A CUDA-event hook on every
     nn.Linear measures the REAL time spent in each projection type
     (q_proj, k_proj, ..., down_proj / fc1 / fc2), summed over all 32 layers.
  2. Each projection maps to a shape in search_cost_sweep.csv. From the CSV we
     get, for that exact (M,K,N): the FP4 GEMM time, the activation-quant time,
     and the added largest-error-selection time.
  3. overhead ratio for the shape  =  search_ms / (Aquant_ms + gemm_ms)
        (search as a fraction of the fused quantize->GEMM it lives in)
  4. added time for that projection  =  measured_real_ms * overhead_ratio
        -> "apply the microbench overhead to each GEMM at its real measured cost"
  5. Sum added time over all projections; divide by the real forward time.

Also reported per shape: the microbench's own absolute search_ms * (#instances
* #layers), i.e. the overhead without rescaling to this GPU's GEMM time.

Weights are quantized offline (inference) so only the activation search costs
anything -- that is exactly what the CSV measures. use_cache=False and
logits_to_keep=1: we time prefill compute, not KV-cache writes or full logits.
"""
import csv, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # .../nvfp4_microbench/code
RESULTS = os.path.join(os.path.dirname(HERE), "results")   # .../nvfp4_microbench/results
os.makedirs(os.path.join(RESULTS, "kernels"), exist_ok=True)
CSV = os.path.join(RESULTS, "search_cost_sweep.csv")
OUT = os.path.join(RESULTS, "prefill_gemm_rollup.csv")

lut = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["gemm_ms"] == "NA":
            continue
        lut[(r["label"], int(r["M"]))] = dict(
            gemm=float(r["gemm_ms"]), aq=float(r["Aquant_ms"]), search=float(r["search_ms"])
        )

MS = [2048, 4096, 8192, 16384]

# projection-name substring -> (csv label, K, N)   [K=in_features, N=out_features]
LLAMA_MAP = {
    "q_proj":    ("qkv_o_proj",   4096, 4096),
    "k_proj":    ("kv_proj_GQA",  4096, 1024),
    "v_proj":    ("kv_proj_GQA",  4096, 1024),
    "o_proj":    ("qkv_o_proj",   4096, 4096),
    "gate_proj": ("gate_up_proj", 4096, 14336),
    "up_proj":   ("gate_up_proj", 4096, 14336),
    "down_proj": ("down_proj",    14336, 4096),
}
OPT_MAP = {
    "q_proj":   ("qkv_o_proj", 4096, 4096),
    "k_proj":   ("qkv_o_proj", 4096, 4096),
    "v_proj":   ("qkv_o_proj", 4096, 4096),
    "out_proj": ("qkv_o_proj", 4096, 4096),
    "fc1":      ("opt_fc1",    4096, 16384),
    "fc2":      ("opt_fc2",    16384, 4096),
}

MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": dict(map=LLAMA_MAP,
        seqs={2048: (1, 2048), 4096: (1, 4096), 8192: (1, 8192), 16384: (1, 16384)}),
    "mistralai/Mistral-7B-v0.3": dict(map=LLAMA_MAP,
        seqs={2048: (1, 2048), 4096: (1, 4096), 8192: (1, 8192), 16384: (1, 16384)}),
    "facebook/opt-6.7b": dict(map=OPT_MAP,
        seqs={2048: (1, 2048), 4096: (2, 2048), 8192: (4, 2048), 16384: (8, 2048)}),
}


GEMM_KEYS = ("gemm", "cutlass", "cublas", "sgemm", "hgemm", "ampere", "nvjet", "s16816",
             "h16816", "wgrad", "dgrad", "gemv", "splitk", "sm80", "sm90", "16x8", "8x8")
ATTN_KEYS = ("flash", "fmha", "attention", "mha", "sdpa", "softmax", "scaled_dot")
EW_KEYS = ("norm", "rope", "rotary", "silu", "gelu", "swiglu", "elementwise", "relu",
           "mul", "add", "cast", "copy", "layer_norm", "rms", "activation", "dropout",
           "index", "embedding", "reduce", "fill", "arange", "triu", "where", "bias")


def bucket(name):
    n = name.lower()
    if any(k in n for k in ATTN_KEYS):
        return "attention"
    if any(k in n for k in GEMM_KEYS):
        return "linear_gemm"
    if any(k in n for k in EW_KEYS):
        return "norm_rope_elementwise"
    return "other"


def classify(name, pmap):
    for key in pmap:
        if name.endswith(key) or name.endswith(key + ".weight"):
            return key
    return None


def run_model(model_id, cfg):
    from transformers import AutoModelForCausalLM
    print(f"\n{'='*92}\n{model_id}\n{'='*92}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()

    # attach CUDA-event timing hooks to every mapped nn.Linear
    linears = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            key = classify(name, cfg["map"])
            if key is None:
                continue
            mod._key = key
            mod._t_ms = 0.0
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

    rows = []
    detail = []
    for M in MS:
        bs, sl = cfg["seqs"][M]
        ids = torch.randint(0, 30000, (bs, sl), device="cuda")
        torch.cuda.empty_cache()
        kw = dict(use_cache=False, logits_to_keep=1)
        NREP = 3
        try:
            with torch.inference_mode():
                for _ in range(2):
                    model(ids, **kw)
                torch.cuda.synchronize()
                for m in linears:
                    m._ev.clear()
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                ev0.record()
                for _ in range(NREP):
                    model(ids, **kw)
                ev1.record()
                torch.cuda.synchronize()
                fwd_ms = ev0.elapsed_time(ev1) / NREP
        except torch.OutOfMemoryError as ex:
            print(f"  M={M:>5} (bs{bs}x{sl})  OOM -> skipped", flush=True)
            for m in linears:
                m._ev.clear()
            torch.cuda.empty_cache()
            continue

        # per-key measured Linear time (ms per forward, summed over all 32 layers)
        per_key, inst = {}, {}
        for m in linears:
            t = sum(a.elapsed_time(b) for a, b in m._ev) / NREP
            per_key[m._key] = per_key.get(m._key, 0.0) + t
            inst[m._key] = inst.get(m._key, 0) + 1
            m._ev.clear()

        # ---- full per-CUDA-kernel dump (every kernel in the prefill forward) ----
        from torch.profiler import profile, ProfilerActivity
        for m in linears:
            m._ev.clear()
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2):
                model(ids, **kw)
            torch.cuda.synchronize()
        for m in linears:
            m._ev.clear()
        kcsv = os.path.join(RESULTS, "kernels", f"{model_id.split('/')[-1]}_M{M}.csv")
        krows = []
        for e in prof.key_averages():
            us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if us <= 0:
                continue
            cnt = e.count
            krows.append((us / 1e3 / 2.0, cnt / 2.0, bucket(e.key), e.key))
        krows.sort(reverse=True)
        with open(kcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ms_per_forward", "launches_per_forward", "category", "kernel"])
            for ms, c, cat, name in krows:
                w.writerow([f"{ms:.5f}", f"{c:.1f}", cat, name])
        gpu_kernel_ms = sum(r[0] for r in krows)
        print(f"    kernel dump -> {os.path.basename(kcsv)} ({len(krows)} distinct kernels, "
              f"sum {gpu_kernel_ms:.1f} ms/fwd GPU-busy)", flush=True)

        total_lin = sum(per_key.values())
        nongemm = fwd_ms - total_lin

        added_scaled = 0.0
        added_direct = 0.0
        print(f"  M={M:>5} (bs{bs}x{sl})  fwd={fwd_ms:8.2f} ms   linear={total_lin:8.2f}   "
              f"non-GEMM={nongemm:7.2f}", flush=True)
        print(f"    {'proj':<11} {'n':>3} {'measured ms':>11} {'mb gemm ms':>11} "
              f"{'mb aq ms':>9} {'ovh%(s/(aq+g))':>14} {'added ms(scaled)':>16} {'added ms(direct)':>16}",
              flush=True)
        for key in cfg["map"]:
            if key not in per_key:
                continue
            label, K, N = cfg["map"][key]
            mb = lut[(label, M)]
            meas = per_key[key]
            n = inst[key]
            ratio = mb["search"] / (mb["aq"] + mb["gemm"])
            a_scaled = meas * ratio
            a_direct = mb["search"] * n
            added_scaled += a_scaled
            added_direct += a_direct
            print(f"    {key:<11} {n:>3} {meas:>11.3f} {mb['gemm']:>11.4f} {mb['aq']:>9.4f} "
                  f"{100*ratio:>13.3f}% {a_scaled:>16.4f} {a_direct:>16.4f}", flush=True)
            detail.append(dict(
                model=model_id, M=M, bs=bs, sl=sl, proj=key, csv_shape=label,
                gemm_M=bs * sl, K=K, N=N, n_layers=n,
                measured_ms_per_fwd=round(meas, 4),
                measured_ms_per_call=round(meas / n, 5),
                mb_gemm_ms=mb["gemm"], mb_aq_ms=mb["aq"], mb_search_ms=mb["search"],
                overhead_ratio_pct=round(100 * ratio, 4),
                added_ms_scaled=round(a_scaled, 4),
                added_ms_direct=round(a_direct, 4)))

        pct_scaled = 100 * added_scaled / fwd_ms
        pct_direct = 100 * added_direct / fwd_ms
        pct_lin_scaled = 100 * added_scaled / total_lin
        print(f"    => added(scaled)={added_scaled:.3f} ms  = {pct_scaled:.3f}% of forward "
              f"({pct_lin_scaled:.3f}% of linear-only) ;  added(direct)={added_direct:.3f} ms "
              f"= {pct_direct:.3f}% of forward", flush=True)

        rows.append(dict(model=model_id, M=M, bs=bs, sl=sl, fwd_ms=round(fwd_ms, 3),
                         gpu_busy_ms=round(gpu_kernel_ms, 3),
                         distinct_kernels=len(krows),
                         total_launches_per_fwd=round(sum(r[1] for r in krows), 1),
                         linear_ms=round(total_lin, 3), nongemm_ms=round(nongemm, 3),
                         added_scaled_ms=round(added_scaled, 4), added_direct_ms=round(added_direct, 4),
                         pct_of_forward_scaled=round(pct_scaled, 4),
                         pct_of_forward_direct=round(pct_direct, 4),
                         pct_of_linear_scaled=round(pct_lin_scaled, 4)))
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return rows, detail


def main():
    allrows, alldetail = [], []
    for mid, cfg in MODELS.items():
        try:
            r, d = run_model(mid, cfg)
            allrows += r
            alldetail += d
        except Exception as e:
            import traceback; traceback.print_exc()
    if allrows:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader(); w.writerows(allrows)
        print(f"\nwrote {OUT}")
    if alldetail:
        dpath = os.path.join(RESULTS, "prefill_gemm_detail.csv")
        with open(dpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(alldetail[0].keys()))
            w.writeheader(); w.writerows(alldetail)
        print(f"wrote {dpath}")


if __name__ == "__main__":
    main()
