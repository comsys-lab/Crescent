#!/usr/bin/env python3
"""
Inference-time overhead of Czardas vs. plain NVFP4, new group sizes (Linear 1-of-8, QK^T 1-of-16),
Linear correction data in the PRE-PACKED layout, MoE routed experts measured on the real grouped GEMM.

  total = current + packing + operand        (all as % of the same NVFP4 forward)

Linear correction data (NVFP4_META_PREPACK): the quantizer stores it already in the 13th-operand
format (one 32-bit word per row pair r/r+8, 64-K block and lane%4, one value per byte; same size
as the plain layout), so the GEMM reads its operand with one 32-bit LDS per M atom.

Dense Linear (results/accel_sim/pp_raw.txt, 3 interleaved reps; baseline = unpatched CUTLASS,
padded carveout; sel / pack = gs_lin8pp_swc0 / swc3):
  current : (GEMM_sel - GEMM_base) + (DRAM store+reload - meta reload) of the PREPACK quantizer
  packing : GEMM_pack - GEMM_sel
QK^T (results/accel_sim/gs8_16_raw.txt, unchanged: transposed layout, gs_qkt16_swc0/3), x2 quantize.
MoE routed experts, per layer (gate, up, down = 3 grouped-GEMM calls):
  GEMM terms : CUTLASS 79d grouped GEMM (pingpong, the schedule grouped_raw.txt uses) with the
               correction-data patch of sm120_blockscaled_mma_array_tma.hpp, real expert
               configuration (Qwen 128 experts x tokens/16, DeepSeek 64 experts), 5 interleaved reps
               (results/accel_sim/grouped_pp_raw.txt): current = pp_swc0 - base,
               packing = pp_swc3 - pp_swc0.
  quantize   : PREPACK quantizer at the tensors the MoE quantizes (results/accel_sim/
               moe_quant_pp_raw.txt, 3 reps): gate/up input = full token batch x 2048, down input =
               all experts' rows x K_down; counted once per GEMM call (gate, up, down) like the
               dense Linear terms. Qwen down at 8192 tokens (65536 rows) exceeds the quantize
               kernel's grid limit: 2 x the measured 32768-row value.
operand : Accel-Sim 13/14-operand OMMA cost as % of GEMM (assemble_final_table_accel_sim.py),
          MoE GEMM base = grouped_raw.txt as there.
Denominators as in assemble_final_table_gs8_16.py.
Writes results/final_table_gs8_16_pp.csv and results/final_table_gs8_16_pp_breakdown.csv.
"""
import csv, importlib.util, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ll = load("assemble_llama_final")
ft = load("assemble_final_table")
sim = load("assemble_final_table_accel_sim")


def parse_micro(path):
    """'### rep<r> <label> <build> ...' blocks of nvfp4_microbench output -> (label, build) -> field -> list."""
    out = {}
    for blk in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = blk.strip().splitlines()
        h = lines[0].split()
        d = out.setdefault((h[1], h[2]), {})
        for l in lines[1:]:
            x = re.search(r"nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if x: d.setdefault("g", []).append(float(x.group(1)))
            x = re.search(r"DRAM store\+reload.*?\)\s*([\d.]+) ms.*meta reload alone:\s*([\d.]+) ms", l)
            if x:
                d.setdefault("dram", []).append(float(x.group(1)))
                d.setdefault("rl", []).append(float(x.group(2)))
            x = re.search(r"with store \(DRAM\):\s*([\d.]+)", l)
            if x: d.setdefault("aq", []).append(float(x.group(1)))
    return out


A = os.path.join(R, "accel_sim")
gs = parse_micro(os.path.join(A, "gs8_16_raw.txt"))      # QK^T (and baseline 'aq' for all shapes)
pp = parse_micro(os.path.join(A, "pp_raw.txt"))           # dense Linear, PREPACK
mq = parse_micro(os.path.join(A, "moe_quant_pp_raw.txt")) # MoE quantize side, PREPACK

grp = {}
for blk in re.split(r"^### ", open(os.path.join(A, "grouped_pp_raw.txt")).read(), flags=re.M)[1:]:
    lines = blk.strip().splitlines()
    _, label, build = lines[0].split()
    t = [float(re.search(r"([\d.]+) ms", l).group(1)) for l in lines[1:] if "Avg runtime" in l]
    if len(t) == 2:
        grp.setdefault((label, build), []).append(t[1])   # pingpong
G = lambda label, build: st.mean(grp[(label, build)])

mean = lambda src, label, build, f: st.mean(src[(label, build)][f])


def cur_lin(label):
    dq = mean(pp, label, "sel", "dram") - mean(pp, label, "sel", "rl")
    return mean(pp, label, "sel", "g") - mean(pp, label, "baseline", "g") + dq


pack_lin = lambda label: mean(pp, label, "pack", "g") - mean(pp, label, "sel", "g")
gb_lin = lambda label: mean(pp, label, "baseline", "g")


def cur_qkt(label):
    dq = mean(gs, label, "sel", "dram") - mean(gs, label, "sel", "rl")
    return mean(gs, label, "sel", "g") - mean(gs, label, "baseline", "g") + 2 * dq


pack_qkt = lambda label: mean(gs, label, "pack", "g") - mean(gs, label, "sel", "g")
gb_qkt = lambda label: mean(gs, label, "baseline", "g")
def dq_moe(label):
    # qwen_down at 8192 tokens = 65536 rows, over the quantize kernel's 65535 grid-y limit:
    # 2 x the measured 32768-row (qwen_down_q_M4096, same K) value -- the kernel is per-row independent.
    if label == "qwen_down_q_M8192":
        return 2 * dq_moe("qwen_down_q_M4096")
    return mean(mq, label, "sel", "dram") - mean(mq, label, "sel", "rl")


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
    c = sum(n * cur_lin(f"{l}_M{M}") for n, l in lin) + qn * cur_qkt(f"{qk}_M{M}")
    p = sum(n * pack_lin(f"{l}_M{M}") for n, l in lin) + qn * pack_qkt(f"{qk}_M{M}")
    lin_g = sum(n * gb_lin(f"{l}_M{M}") for n, l in lin)
    if moe:
        nl, gu, dn = moe
        u, d = f"{gu}_M{M}", f"{dn}_M{M}"
        c += nl * (2 * (G(u, "grp_pp_swc0_p0") - G(u, "grp_base_p0") + dq_moe(f"{gu}_q_M{M}"))
                   + G(d, "grp_pp_swc0_p0") - G(d, "grp_base_p0") + dq_moe(f"{dn}_q_M{M}"))
        p += nl * (2 * (G(u, "grp_pp_swc3_p0") - G(u, "grp_pp_swc0_p0"))
                   + G(d, "grp_pp_swc3_p0") - G(d, "grp_pp_swc0_p0"))
        lin_g += nl * (2 * G(u, "grp_base_p0") + G(d, "grp_base_p0"))  # 5090: re-measured grouped baseline
    qkt_g = qn * gb_qkt(f"{qk}_M{M}")
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
    with open(os.path.join(R, "final_table_gs8_16_pp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_tokens", "llama_pct", "qwen_pct", "deepseek_pct"])
        for M in ft.MS:
            w.writerow([M] + [f"{table[(m, M)]:.2f}" for m in ("Llama", "Qwen", "DeepSeek")])
    with open(os.path.join(R, "final_table_gs8_16_pp_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "total_tokens", "current_pct", "packing_pct", "operand_pct", "operand_ci_lo",
                    "operand_ci_hi", "total_pct", "total_ci_lo", "total_ci_hi"])
        for r in rows:
            w.writerow(r[:2] + [f"{x:.2f}" for x in r[2:]])
    print("\nwrote results/final_table_gs8_16_pp.csv, results/final_table_gs8_16_pp_breakdown.csv")
