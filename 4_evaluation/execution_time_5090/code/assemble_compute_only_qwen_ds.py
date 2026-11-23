#!/usr/bin/env python3
"""
Selection-search compute-only cost (no metadata memory write, no GEMM/TMA-load effect)
as a % of real total inference time, for Qwen3-30B-A3B and DeepSeek-MoE-16B -- the
Qwen/DeepSeek counterpart to the Llama number computed by hand in-conversation.

Reuses assemble_final_table.py's real call-counts, grouped-GEMM baseline (grp[]), and
denominator construction (qwen_row/ds_row's fwd_est) unchanged -- only the numerator
swaps delta()'s DRAM-store-based quantize-side term for a compute-only term read from
compute_only_qwen_ds.log (produced by ../../../../nvfp4_microbench/code/
sweep_compute_only_qwen_ds.sh, "selection on A adds (compute-only)" lines).
"""
import re, os
import assemble_final_table as T

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "..", "..", "..", "nvfp4_microbench", "code",
                    "compute_only_qwen_ds.log")

def parse_compute_only(path):
    co = {}
    for b in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = b.strip().splitlines()
        if not lines or lines[0] == "DONE":
            continue
        label = re.match(r"(\S+) m=", lines[0]).group(1)
        if len(lines) < 2:
            continue
        m = re.search(r"compute-only\)\s*([-\d.]+) ms\s*->.*A-quant \+ nvfp4 GEMM = ([\d.]+) ms\)", lines[1])
        if m:
            co[label] = (float(m.group(1)), float(m.group(2)))
    return co

CO = parse_compute_only(LOG)

def co_delta(label):
    return CO[label][0]

def co_pct(label):
    d, n = CO[label]
    return 100 * d / n

MS = T.MS

def qwen_compute_only(M):
    rest_added = sum(n * co_delta(f"{l}_M{M}") for n, l in T.items_qwen_rest())
    qkt_added = 1536 * 2 * co_delta(f"qwen_qkt_M{M}")   # 32 heads x 48 layers, doubled for Q+K
    moe_added = 48 * (T.grp[f"qwen_gateup_M{M}"] * (co_pct(f"qwen_gateup_M{M}") / 100) * 2
                     + T.grp[f"qwen_down_M{M}"] * (co_pct(f"qwen_down_M{M}") / 100))
    total = rest_added + qkt_added + moe_added
    _, fwd_est, _ = T.qwen_row(M)
    return total, fwd_est, 100 * total / fwd_est

def ds_compute_only(M):
    rest_added = sum(n * co_delta(f"{l}_M{M}") for n, l in T.items_ds_rest())
    qkt_added = 448 * 2 * co_delta(f"ds_qkt_M{M}")   # 16 heads x 28 layers, doubled for Q+K
    moe_added = 27 * (T.grp[f"ds_routed_gu_M{M}"] * (co_pct(f"ds_routed_gu_M{M}") / 100) * 2
                     + T.grp[f"ds_routed_down_M{M}"] * (co_pct(f"ds_routed_down_M{M}") / 100))
    total = rest_added + qkt_added + moe_added
    _, fwd_est, _ = T.ds_row(M)
    return total, fwd_est, 100 * total / fwd_est

if __name__ == "__main__":
    for M in MS:
        a, f, p = qwen_compute_only(M)
        print(f"Qwen     M={M:>6}  compute-only-added={a:7.3f}ms  denom={f:9.1f}ms  -> {p:.3f}%")
    for M in MS:
        a, f, p = ds_compute_only(M)
        print(f"DeepSeek M={M:>6}  compute-only-added={a:7.3f}ms  denom={f:9.1f}ms  -> {p:.3f}%")
