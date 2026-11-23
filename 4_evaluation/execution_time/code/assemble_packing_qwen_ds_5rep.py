#!/usr/bin/env python3
"""
5x-repeated version of assemble_packing_qwen_ds.py -- single-shot GEMM deltas were
noise-dominated (one pair even showed packing "faster" than TMA-load-only, which is
physically impossible), so every shape is now averaged over 5 reps before taking the
pack-vs-tma delta. Reads packing_qwen_ds_5rep.log.
"""
import re, os
import assemble_final_table as T

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "..", "..", "..", "nvfp4_microbench", "code",
                    "packing_qwen_ds_5rep.log")

def parse_packing(path):
    tma_r, pack_r = {}, {}
    for b in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = b.strip().splitlines()
        if not lines or lines[0] == "DONE":
            continue
        label = re.match(r"(\S+) rep\d+ m=", lines[0]).group(1)
        for l in lines[1:]:
            m = re.search(r"tma:\s*nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if m: tma_r.setdefault(label, []).append(float(m.group(1)))
            m = re.search(r"pack:\s*nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if m: pack_r.setdefault(label, []).append(float(m.group(1)))
    avg = lambda d: {k: sum(v) / len(v) for k, v in d.items()}
    tma, pack = avg(tma_r), avg(pack_r)
    return {k: (pack[k] - tma[k], tma[k]) for k in tma}

PK = parse_packing(LOG)

def pk_delta(label):
    return PK[label][0]

def pk_pct(label):
    d, tma = PK[label]
    return 100 * d / tma

MS = T.MS

def qwen_packing(M):
    rest_added = sum(n * pk_delta(f"{l}_M{M}") for n, l in T.items_qwen_rest())
    qkt_added = 1536 * pk_delta(f"qwen_qkt_M{M}")
    moe_added = 48 * (T.grp[f"qwen_gateup_M{M}"] * (pk_pct(f"qwen_gateup_M{M}") / 100) * 2
                     + T.grp[f"qwen_down_M{M}"] * (pk_pct(f"qwen_down_M{M}") / 100))
    total = rest_added + qkt_added + moe_added
    _, fwd_est, _ = T.qwen_row(M)
    return total, fwd_est, 100 * total / fwd_est

def ds_packing(M):
    rest_added = sum(n * pk_delta(f"{l}_M{M}") for n, l in T.items_ds_rest())
    qkt_added = 448 * pk_delta(f"ds_qkt_M{M}")
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
