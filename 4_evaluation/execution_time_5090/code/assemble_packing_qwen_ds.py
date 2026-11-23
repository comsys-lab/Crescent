#!/usr/bin/env python3
"""
Real (not Llama-borrowed) packing-only GEMM-time cost for Qwen3-30B-A3B and
DeepSeek-MoE-16B, as a %p contribution to total inference time -- to check whether
reusing Llama's packing ratio (+2.87% Linear / +1.40% QK^T) for these two models was a
safe approximation, or whether their (different, MoE-heavy, many small-expert-shape)
kernels show a meaningfully different packing cost.

Reuses assemble_final_table.py's real call-counts, grouped-GEMM baseline (grp[]), and
qwen_row/ds_row's fwd_est denominator unchanged -- only the numerator is a packing-only
GEMM-time delta (pack3 2-atom/14-op build vs TMA-load-only build), read from
packing_qwen_ds.log (produced by ../../../../nvfp4_microbench/code/
sweep_packing_qwen_ds.sh).
"""
import re, os
import assemble_final_table as T

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "..", "..", "..", "nvfp4_microbench", "code",
                    "packing_qwen_ds.log")

def parse_packing(path):
    out = {}
    for b in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = b.strip().splitlines()
        if not lines or lines[0] == "DONE":
            continue
        label = re.match(r"(\S+) m=", lines[0]).group(1)
        tma = pack = None
        for l in lines[1:]:
            m = re.search(r"tma:\s*nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if m: tma = float(m.group(1))
            m = re.search(r"pack:\s*nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if m: pack = float(m.group(1))
        if tma is not None and pack is not None:
            out[label] = (pack - tma, tma)   # (delta_ms, tma_baseline_ms)
    return out

PK = parse_packing(LOG)

def pk_delta(label):
    return PK[label][0]

def pk_pct(label):
    d, tma = PK[label]
    return 100 * d / tma

MS = T.MS

def qwen_packing(M):
    rest_added = sum(n * pk_delta(f"{l}_M{M}") for n, l in T.items_qwen_rest())
    qkt_added = 1536 * pk_delta(f"qwen_qkt_M{M}")   # 32 heads x 48 layers, no doubling (GEMM-side)
    moe_added = 48 * (T.grp[f"qwen_gateup_M{M}"] * (pk_pct(f"qwen_gateup_M{M}") / 100) * 2
                     + T.grp[f"qwen_down_M{M}"] * (pk_pct(f"qwen_down_M{M}") / 100))
    total = rest_added + qkt_added + moe_added
    _, fwd_est, _ = T.qwen_row(M)
    return total, fwd_est, 100 * total / fwd_est

def ds_packing(M):
    rest_added = sum(n * pk_delta(f"{l}_M{M}") for n, l in T.items_ds_rest())
    qkt_added = 448 * pk_delta(f"ds_qkt_M{M}")   # 16 heads x 28 layers, no doubling
    moe_added = 27 * (T.grp[f"ds_routed_gu_M{M}"] * (pk_pct(f"ds_routed_gu_M{M}") / 100) * 2
                     + T.grp[f"ds_routed_down_M{M}"] * (pk_pct(f"ds_routed_down_M{M}") / 100))
    total = rest_added + qkt_added + moe_added
    _, fwd_est, _ = T.ds_row(M)
    return total, fwd_est, 100 * total / fwd_est

if __name__ == "__main__":
    for M in MS:
        a, f, p = qwen_packing(M)
        print(f"Qwen     M={M:>6}  packing-added={a:7.3f}ms  denom={f:9.1f}ms  -> {p:.3f}%")
    for M in MS:
        a, f, p = ds_packing(M)
        print(f"DeepSeek M={M:>6}  packing-added={a:7.3f}ms  denom={f:9.1f}ms  -> {p:.3f}%")
