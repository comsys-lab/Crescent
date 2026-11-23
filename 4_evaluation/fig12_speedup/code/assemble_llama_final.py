#!/usr/bin/env python3
"""
Authoritative Llama-3.1-8B overhead calculation -- supersedes the Llama numbers in
assemble_final_table.py (that version used a baseline with an SMEM-padding confound,
see README.md "Stage-count confound"). Everything here is averaged over 5 repeated
measurements. Reads only from ../results/ and ../log/.

Four real, repeated measurements feed this:
  1. Linear: full model, NVFP4 true-baseline vs NVFP4+selection, both "full forward"
     and "GEMM-only" (profiler-isolated) deltas. -> log/linear_stage_corrected_5rep.log
  2. Real fused BF16 attention kernel (QK^T+softmax+PV, unmodified Llama). ->
     log/bf16_real_attention_5rep.log
  3. Standalone BF16 QK^T and PV GEMMs (cuBLAS), to split the fused kernel into
     QK^T / PV / "other" (softmax+masking). -> results/bf16_qkt_pv_repeat.txt
  4. Standalone NVFP4 QK^T (true-baseline vs selection) and PV (true-baseline only,
     PV never gets selection -- only Q and K do) GEMMs. -> results/qkt_pv_repeat.txt

Denominator construction (all real/repeated, no single-shot numbers):
  NVFP4_attention_baseline = QK^T(NVFP4,true-base) + PV(NVFP4,true-base) + other(BF16)
  full_denom = Linear_NVFP4_true_base_fwd - real_BF16_attn + NVFP4_attention_baseline
  (i.e.: take the real Linear-converted forward, swap out its still-BF16 attention
   portion for what attention would cost if QK^T and PV also became NVFP4 baseline)

Numerator = Linear_added (real, Linear-only full-forward delta) + QK^T_added (real
NVFP4 baseline-vs-selection GEMM+quantize delta, doubled for Q+K, x32 heads x32 layers).
PV never gets selection (only Q/K do, matching the group=8 QK^T convention), so it
contributes 0 to the numerator -- only to the denominator.
"""
import re, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results") + "/"
L = os.path.join(HERE, "..", "log") + "/"
NHEADLAYER = 32 * 32  # Llama-3.1-8B: 32 attention heads x 32 layers

def parse_linear_log():
    fwd_true, fwd_sel = {}, {}
    for line in open(L + "linear_stage_corrected_5rep.log"):
        m = re.match(r"rep\d+ M=\s*(\d+)\s+fwd: true=\s*([\d.]+) sel=\s*([\d.]+)", line)
        if m:
            M, t, s = int(m.group(1)), float(m.group(2)), float(m.group(3))
            fwd_true.setdefault(M, []).append(t)
            fwd_sel.setdefault(M, []).append(s)
    return fwd_true, fwd_sel

def parse_attn_log():
    attn = {}
    for line in open(L + "bf16_real_attention_5rep.log"):
        m = re.match(r"rep\d+ M=\s*(\d+)\s+real_fused_attn_bf16=\s*([\d.]+)ms", line)
        if m:
            attn.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return attn

def parse_bf16_qkt_pv():
    qkt, pv = {}, {}
    blocks = re.split(r"^### ", open(R + "bf16_qkt_pv_repeat.txt").read(), flags=re.M)[1:]
    for b in blocks:
        lines = b.strip().splitlines()
        if not lines or lines[0] == "DONE":
            continue
        h = re.match(r"rep(\d+) (\S+) m=(\d+)", lines[0])
        label, M = h.group(2), int(h.group(3))
        v = float(re.search(r"([\d.]+) ms", lines[1]).group(1))
        (qkt if "QKT" in label else pv).setdefault(M, []).append(v)
    return qkt, pv

def parse_nvfp4_qkt_pv():
    data = {}
    blocks = re.split(r"^### ", open(R + "qkt_pv_repeat.txt").read(), flags=re.M)[1:]
    for b in blocks:
        lines = b.strip().splitlines()
        if not lines or lines[0] == "DONE":
            continue
        h = re.match(r"rep(\d+) (\S+) m=(\d+)", lines[0])
        rep, label, M = int(h.group(1)), h.group(2), int(h.group(3))
        d = {}
        for line in lines[1:]:
            x = re.search(r"nvfp4 GEMM\s*:\s*([\d.]+) ms", line)
            if x: d['g'] = float(x.group(1))
            x = re.search(r"DRAM store\+reload.*?\)\s*([\d.]+) ms.*meta reload alone:\s*([\d.]+) ms", line)
            if x: d['dram'] = float(x.group(1)); d['rl'] = float(x.group(2))
        data[(rep, label, M)] = d
    return data

def avg(d):
    return {k: sum(v) / len(v) for k, v in d.items()}

MS = [2048, 4096, 8192]
fwd_true_r, fwd_sel_r = parse_linear_log()
fwd_true, fwd_sel = avg(fwd_true_r), avg(fwd_sel_r)
linear_added = {M: fwd_sel[M] - fwd_true[M] for M in MS}

attn_r = parse_attn_log()
real_attn_bf16 = avg(attn_r)

qkt_bf16_r, pv_bf16_r = parse_bf16_qkt_pv()
qkt_bf16 = {M: avg(qkt_bf16_r)[M] * NHEADLAYER for M in MS}
pv_bf16 = {M: avg(pv_bf16_r)[M] * NHEADLAYER for M in MS}
other_bf16 = {M: real_attn_bf16[M] - qkt_bf16[M] - pv_bf16[M] for M in MS}

nv = parse_nvfp4_qkt_pv()
qkt_added, qkt_base_nvfp4, pv_base_nvfp4 = {}, {}, {}
for M in MS:
    gemm_deltas, bases, pv_vals = [], [], []
    for rep in range(1, 6):
        b = nv[(rep, "QKT_true_base", M)]
        s = nv[(rep, "QKT_sel", M)]
        p = nv[(rep, "PV_true_base", M)]
        gemm_delta = s['g'] - b['g']
        dq_st = s['dram'] - s['rl']
        gemm_deltas.append((gemm_delta + 2 * dq_st) * NHEADLAYER)
        bases.append(b['g'] * NHEADLAYER)
        pv_vals.append(p['g'] * NHEADLAYER)
    qkt_added[M] = sum(gemm_deltas) / len(gemm_deltas)
    qkt_base_nvfp4[M] = sum(bases) / len(bases)
    pv_base_nvfp4[M] = sum(pv_vals) / len(pv_vals)

if __name__ == "__main__":
    print(f"{'M':>6} {'linear_added':>13} {'qkt_added':>10} {'total_added':>12} {'denom':>10} {'overhead%':>10}")
    for M in MS:
        nvfp4_attn_base = qkt_base_nvfp4[M] + pv_base_nvfp4[M] + other_bf16[M]
        denom = fwd_true[M] - real_attn_bf16[M] + nvfp4_attn_base
        total_added = linear_added[M] + qkt_added[M]
        pct = 100 * total_added / denom
        print(f"{M:>6} {linear_added[M]:>13.2f} {qkt_added[M]:>10.2f} {total_added:>12.2f} {denom:>10.2f} {pct:>9.2f}%")
