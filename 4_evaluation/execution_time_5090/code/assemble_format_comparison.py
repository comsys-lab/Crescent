#!/usr/bin/env python3
"""
Inference-time increase of Czardas and of the 6-bit (E2M3 activation) formats, each family against its own
4-bit baseline:
  NVFP family : NVFP4 -> Czardas (NVFP4-based), NVFP6
  MXFP family : MXFP4 -> Czardas (MXFP4-based), MXFP6
Weights are always E2M1. 6-bit formats: Linear activation E2M3 x weight E2M1, QK^T E2M3 x E2M3, PV stays 4-bit.

Czardas (NVFP4)  : assemble_final_table_gs8_16_pp_optden.py (Linear pre-packed 1-of-8, QK^T 1-of-16,
                   MoE grouped GEMM, rebuilt Qwen/DeepSeek denominators).
GEMM times, results/accel_sim/fc_dense.txt / fc_grouped.txt (measure_fmtcmp.sh, 3 interleaved reps):
  NVFP4 / MXFP4 / MXFP6 : pristine CUTLASS sm120 block-scaled GEMM (src/nvfp6_gemm_bench.cu), natural Stages
  NVFP6                 : sm_120 has no E4M3-scaled FP6 MMA -> MXFP6 GEMM + one extra TMA load per stage of
                          NVFP6's additional scale bytes (16- instead of 32-element blocks, A and B), same
                          pipeline / mbarrier (patched collective, NVFP4_META_GROUP_SIZE=32 dense /
                          16 one-sided grouped, M-contiguous, load only)
  Czardas (MXFP4)       : MXFP4 GEMM + correction data (same patches as the NVFP4 Czardas), vs an MXFP4
                          baseline with the same Stage count (extra carveout)
  grouped               : CUTLASS 79d grouped GEMM driver with the same element types, pingpong schedule
Quantize (results/accel_sim/fc_quant.txt, quant_fp6_bench): same kernel, E2M1 / E2M3 codes, NV (E4M3/16) or
  MX (UE8M0/32) scales, at every quantized activation (Linear inputs, Q/K/P/V per head, MoE full-batch inputs).
  Czardas (MXFP4) selection + correction-data store: the NVFP4 selection-quantizer measurement (same
  correction data; only the block scale differs).
Operand term (Czardas): Accel-Sim fraction x the family's 4-bit GEMM time.
MXFP4 denominator = NVFP4 denominator + (MXFP4 - NVFP4) GEMM and quantize time of every quantized GEMM.
Writes results/format_comparison.csv.
"""
import csv, importlib.util, os, re, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results"); A = os.path.join(R, "accel_sim")
spec = importlib.util.spec_from_file_location("od", os.path.join(HERE, "assemble_final_table_gs8_16_pp_optden.py"))
od = importlib.util.module_from_spec(spec); spec.loader.exec_module(od)
pp, ft, ll, sim = od.pp, od.ft, od.ll, od.sim

dense = {}
for l in open(os.path.join(A, "fc_dense.txt")):
    m = re.match(r"### rep\d+ (\S+) (\S+) gemm_ms=([\d.]+)", l)
    if m: dense.setdefault((m.group(1), m.group(2)), []).append(float(m.group(3)))
D = lambda label, b: st.mean(dense[(label, b)])
quant = {}
for l in open(os.path.join(A, "fc_quant.txt")):
    m = re.match(r"### rep\d+ (\S+) RESULT .*nv_e2m1_ms=([\d.]+) nv_e2m3_ms=([\d.]+) mx_e2m1_ms=([\d.]+) mx_e2m3_ms=([\d.]+)", l)
    if m:
        d = quant.setdefault(m.group(1), {})
        for k, v in zip(("nv4", "nv6", "mx4", "mx6"), m.groups()[1:]): d.setdefault(k, []).append(float(v))
Q = lambda label, k: st.mean(quant[label][k])
grp = {}
for blk in re.split(r"^### ", open(os.path.join(A, "fc_grouped.txt")).read(), flags=re.M)[1:]:
    lines = blk.strip().splitlines(); _, label, b = lines[0].split()
    t = [float(re.search(r"([\d.]+) ms", l).group(1)) for l in lines[1:] if "Avg runtime" in l]
    if len(t) == 2: grp.setdefault((label, b), []).append(t[1])
G = lambda label, b: st.mean(grp[(label, b)])


def cfg(name):
    if name == "Llama":
        return dict(lin=[(64, "llama_qoproj"), (64, "llama_kvproj"), (64, "llama_gateup"), (32, "llama_down")],
                    qn=1024, qk="llama_qkt", moe=None)
    if name == "Qwen":
        return dict(lin=list(ft.items_qwen_rest()), qn=1536, qk="qwen_qkt", moe=(48, "qwen_gateup", "qwen_down"))
    return dict(lin=list(ft.items_ds_rest()), qn=448, qk="ds_qkt", moe=(27, "ds_routed_gu", "ds_routed_down"))


def nv_denom(name, M):
    if name == "Llama":
        return ll.fwd_true[M] - ll.real_attn_bf16[M] + (ll.qkt_base_nvfp4[M] + ll.pv_base_nvfp4[M] + ll.other_bf16[M])
    return od.denom(name, M)[0]


def terms(name, M):
    c = cfg(name); out = {}
    qk, pv = f"{c['qk']}_M{M}", f"pv_M{M}"
    dq_ = lambda k1, k0, lab: Q(lab, k1) - Q(lab, k0)
    # per-forward added ms: (GEMM delta, quantize delta)
    def fam(lin_b, lin_0, qkt_b, qkt_0, grp_b, grp_0, qk1, qk0, pv_b=None, pv_0=None):
        s = 0.0
        for n, l in c["lin"]:
            s += n * (D(f"{l}_M{M}", lin_b) - D(f"{l}_M{M}", lin_0) + dq_(qk1, qk0, f"{l}_M{M}_a"))
        s += c["qn"] * (D(qk, qkt_b) - D(qk, qkt_0) + 2 * dq_(qk1, qk0, f"qk_M{M}"))
        if pv_b:
            s += c["qn"] * (D(pv, pv_b) - D(pv, pv_0) + dq_(qk1, qk0, f"p_M{M}") + dq_(qk1, qk0, f"v_M{M}"))
        if c["moe"]:
            nl, gu, dn = c["moe"]
            u, d = f"{gu}_M{M}", f"{dn}_M{M}"
            s += nl * (2 * (G(u, grp_b) - G(u, grp_0)) + G(d, grp_b) - G(d, grp_0)
                       + 2 * dq_(qk1, qk0, f"{gu}_q_M{M}") + dq_(qk1, qk0, f"{dn}_q_M{M}"))
        return s
    out["nvfp6"] = fam("fc_nv6lin_p0", "fc_nv4_p0", "fc_nv6qkt_p0", "fc_nv4_p0", "grp_nv6emu_p0", "grp_base_p0", "nv6", "nv4")
    out["mxfp6"] = fam("fc_mx6lin_p0", "fc_mx4_p0", "fc_mx6qkt_p0", "fc_mx4_p0", "grp_mx2_p0", "grp_mx1_p0", "mx6", "mx4")
    out["mx4_vs_nv4"] = fam("fc_mx4_p0", "fc_nv4_p0", "fc_mx4_p0", "fc_nv4_p0", "grp_mx1_p0", "grp_base_p0", "mx4", "nv4",
                            "fc_mx4_p0", "fc_nv4_p0")
    # Czardas (MXFP4): load (+ selection quantize from the NVFP4 measurement), packing, operand
    cur = pk = 0.0
    for n, l in c["lin"]:
        lab = f"{l}_M{M}"
        cur += n * (D(lab, "fc_czmxlin0_p16384") - D(lab, "fc_mx4_p16384")
                    + pp.mean(pp.pp, lab, "sel", "dram") - pp.mean(pp.pp, lab, "sel", "rl"))
        pk += n * (D(lab, "fc_czmxlin3_p16384") - D(lab, "fc_czmxlin0_p16384"))
    cur += c["qn"] * (D(qk, "fc_czmxqkt0_p16384") - D(qk, "fc_mx4_p16384")
                      + 2 * (pp.mean(pp.gs, qk, "sel", "dram") - pp.mean(pp.gs, qk, "sel", "rl")))
    pk += c["qn"] * (D(qk, "fc_czmxqkt3_p16384") - D(qk, "fc_czmxqkt0_p16384"))
    lin_g = sum(n * D(f"{l}_M{M}", "fc_mx4_p0") for n, l in c["lin"])
    if c["moe"]:
        nl, gu, dn = c["moe"]; u, d = f"{gu}_M{M}", f"{dn}_M{M}"
        cur += nl * (2 * (G(u, "grp_czmx_swc0_p8192") - G(u, "grp_mx1_p8192")) + G(d, "grp_czmx_swc0_p8192") - G(d, "grp_mx1_p8192")
                     + 2 * pp.dq_moe(f"{gu}_q_M{M}") + pp.dq_moe(f"{dn}_q_M{M}"))
        pk += nl * (2 * (G(u, "grp_czmx_swc3_p8192") - G(u, "grp_czmx_swc0_p8192")) + G(d, "grp_czmx_swc3_p8192") - G(d, "grp_czmx_swc0_p8192"))
        lin_g += nl * (2 * G(u, "grp_mx1_p0") + G(d, "grp_mx1_p0"))
    op = sim.OP_LIN[0] * lin_g + sim.OP_QKT[0] * c["qn"] * D(qk, "fc_mx4_p0")
    out["cz_mx"] = (cur, pk, op)
    return out


if __name__ == "__main__":
    rows = []
    for name in ("Llama", "Qwen", "DeepSeek"):
        for M in ft.MS:
            t = terms(name, M)
            dn = nv_denom(name, M); dm = dn + t["mx4_vs_nv4"]
            # Czardas (NVFP4): same as assemble_final_table_gs8_16_pp_optden.py
            cz_nv_c, cz_nv_p, cz_nv_o = od.pp.model(name, M)
            if name != "Llama":
                old = ft.qwen_row(M)[1] if name == "Qwen" else ft.ds_row(M)[1]
                f = old / dn
                cz_nv_c, cz_nv_p, cz_nv_o = cz_nv_c * f, cz_nv_p * f, [x * f for x in cz_nv_o]
            cz_nv = cz_nv_c + cz_nv_p + cz_nv_o[0]
            cm = t["cz_mx"]
            rows.append(dict(model=name, tokens=M, nvfp4_ms=dn, czardas_nv=cz_nv, nvfp6=100 * t["nvfp6"] / dn,
                             mxfp4_ms=dm, mxfp4_vs_nvfp4=100 * t["mx4_vs_nv4"] / dn,
                             czardas_mx=100 * sum(cm) / dm, cz_mx_load=100 * cm[0] / dm, cz_mx_pack=100 * cm[1] / dm,
                             cz_mx_op=100 * cm[2] / dm, mxfp6=100 * t["mxfp6"] / dm))
    print("NVFP family (vs NVFP4)                 MXFP family (vs MXFP4)")
    for r in rows:
        print(f"{r['model']:8s} {r['tokens']:5d}  Czardas {r['czardas_nv']:5.2f}%  NVFP6 {r['nvfp6']:6.2f}%   |  "
              f"Czardas {r['czardas_mx']:5.2f}% (load {r['cz_mx_load']:.2f} pack {r['cz_mx_pack']:.2f} op {r['cz_mx_op']:.2f})  "
              f"MXFP6 {r['mxfp6']:6.2f}%   [MXFP4 vs NVFP4 {r['mxfp4_vs_nvfp4']:+.2f}%]")
    with open(os.path.join(R, "format_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\nwrote results/format_comparison.csv")
