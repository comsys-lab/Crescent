#!/usr/bin/env python3
"""
Inference-time overhead of Czardas vs. plain NVFP4 with the NEW selection group sizes:
Linear 1-of-8 (activation only), QK^T 1-of-16 (Q and K). Same three terms as
assemble_final_table_accel_sim.py:

  total = current + packing + operand        (all as % of the same NVFP4 forward)

  current : selection quantize + metadata TMA load, per shape
            (GEMM_sel - GEMM_base) + (DRAM store+reload - meta reload)  [x2 for QK^T],
            the assemble_final_table.py definition, re-measured with the new group sizes.
  packing : GEMM_pack - GEMM_sel (TMA load + packing vs TMA load only), per shape.
  operand : Accel-Sim 13/14-operand OMMA cost as % of the GEMM (same as
            assemble_final_table_accel_sim.py; the operand count does not change with
            the group size).

All three models are assembled from the per-shape microbench (results/accel_sim/gs8_16_raw.txt,
5 interleaved reps, 10 for MoE routed-expert shapes), x calls per forward; MoE routed experts
apply the per-shape ratio to the real grouped-GEMM time (grouped_raw.txt), as in
assemble_final_table.py. Denominators: Llama = assemble_llama_final.py's Stage-corrected NVFP4
forward; Qwen/DeepSeek = assemble_final_table.py's. Builds (padded carveout for all, i.e. not
Stage-corrected): baseline = unpatched CUTLASS, NVFP4_META_ENABLE=0; selection/packing =
patched CUTLASS, NVFP4_META_SW_CORRECT=0/3; QK^T metadata stored transposed (M-contiguous),
since K/16 = 8-byte rows at K=128 violate TMA's 16-byte stride rule.
Note: the Llama Linear term here is microbench-assembled, not the end-to-end full-model delta
assemble_llama_final.py uses for the old group size.

Writes results/final_table_gs8_16.csv and results/final_table_gs8_16_breakdown.csv.
"""
import csv, importlib.util, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)   # module level only
    return m


ll = load("assemble_llama_final")
ft = load("assemble_final_table")
sim = load("assemble_final_table_accel_sim")   # operand term (OP_LIN, OP_QKT)

# per (label, build): field -> list over reps
raw = {}
for blk in re.split(r"^### ", open(os.path.join(R, "accel_sim", "gs8_16_raw.txt")).read(), flags=re.M)[1:]:
    lines = blk.strip().splitlines()
    h = re.match(r"rep\d+ (\S+) (\S+) n=", lines[0])
    d = raw.setdefault((h.group(1), h.group(2)), {})
    for l in lines[1:]:
        x = re.search(r"nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
        if x: d.setdefault("g", []).append(float(x.group(1)))
        x = re.search(r"DRAM store\+reload.*?\)\s*([\d.]+) ms.*meta reload alone:\s*([\d.]+) ms", l)
        if x:
            d.setdefault("dram", []).append(float(x.group(1)))
            d.setdefault("rl", []).append(float(x.group(2)))
        x = re.search(r"with store \(DRAM\):\s*([\d.]+)", l)
        if x: d.setdefault("aq", []).append(float(x.group(1)))
m_ = lambda label, build, f: st.mean(raw[(label, build)][f])


def cur(label, qkt=False):
    """Added ms per call: GEMM delta + quantize-side store/reload delta (x2 for Q and K)."""
    dq = m_(label, "sel", "dram") - m_(label, "sel", "rl")
    return (m_(label, "sel", "g") - m_(label, "baseline", "g")) + (2 * dq if qkt else dq)


pack = lambda label: m_(label, "pack", "g") - m_(label, "sel", "g")
nv = lambda label: m_(label, "baseline", "aq") + m_(label, "baseline", "g")
gb = lambda label: m_(label, "baseline", "g")


def model(name, M):
    if name == "Llama":
        lin = [(64, "llama_qoproj"), (64, "llama_kvproj"), (64, "llama_gateup"), (32, "llama_down")]
        qn, qk, moe = 1024, "llama_qkt", None
        denom = ll.fwd_true[M] - ll.real_attn_bf16[M] + (ll.qkt_base_nvfp4[M] + ll.pv_base_nvfp4[M] + ll.other_bf16[M])
    elif name == "Qwen":
        lin = list(ft.items_qwen_rest())
        qn, qk, moe = 1536, "qwen_qkt", (48, "qwen_gateup", "qwen_down")
        denom = ft.qwen_row(M)[1]
    else:
        lin = list(ft.items_ds_rest())
        qn, qk, moe = 448, "ds_qkt", (27, "ds_routed_gu", "ds_routed_down")
        denom = ft.ds_row(M)[1]
    c = sum(n * cur(f"{l}_M{M}") for n, l in lin) + qn * cur(f"{qk}_M{M}", True)
    p = sum(n * pack(f"{l}_M{M}") for n, l in lin) + qn * pack(f"{qk}_M{M}")
    lin_g = sum(n * gb(f"{l}_M{M}") for n, l in lin)
    if moe:
        nl, gu, dn = moe
        G = lambda l: ft.grp[f"{l}_M{M}"]
        c += nl * (2 * G(gu) * cur(f"{gu}_M{M}") / nv(f"{gu}_M{M}") + G(dn) * cur(f"{dn}_M{M}") / nv(f"{dn}_M{M}"))
        p += nl * (2 * G(gu) * pack(f"{gu}_M{M}") / gb(f"{gu}_M{M}") + G(dn) * pack(f"{dn}_M{M}") / gb(f"{dn}_M{M}"))
        lin_g += nl * (2 * G(gu) + G(dn))
    qkt_g = qn * gb(f"{qk}_M{M}")
    op = [100 * (sim.OP_LIN[i] * lin_g + sim.OP_QKT[i] * qkt_g) / denom for i in range(3)]
    return 100 * c / denom, 100 * p / denom, op


if __name__ == "__main__":
    rows, table = [], {}
    for name in ("Llama", "Qwen", "DeepSeek"):
        for M in ft.MS:
            c, p, op = model(name, M)
            table[(name, M)] = c + p + op[0]
            rows.append([name, M, c, p, op[0], op[1], op[2], c + p + op[0], c + p + op[1], c + p + op[2]])
            print(f"{name:8s} M={M:>5}  current {c:5.2f}  +packing {p:4.2f}  +operand {op[0]:4.2f} "
                  f"[{op[1]:.2f}, {op[2]:.2f}]  => {c + p + op[0]:5.2f}%")
    with open(os.path.join(R, "final_table_gs8_16.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_tokens", "llama_pct", "qwen_pct", "deepseek_pct"])
        for M in ft.MS:
            w.writerow([M] + [f"{table[(m, M)]:.2f}" for m in ("Llama", "Qwen", "DeepSeek")])
    with open(os.path.join(R, "final_table_gs8_16_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "total_tokens", "current_pct", "packing_pct", "operand_pct", "operand_ci_lo",
                    "operand_ci_hi", "total_pct", "total_ci_lo", "total_ci_hi"])
        for r in rows:
            w.writerow(r[:2] + [f"{x:.2f}" for x in r[2:]])
    print("\nwrote results/final_table_gs8_16.csv, results/final_table_gs8_16_breakdown.csv")
