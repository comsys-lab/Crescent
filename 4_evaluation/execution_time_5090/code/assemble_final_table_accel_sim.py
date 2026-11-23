#!/usr/bin/env python3
"""
Final table WITH the two costs the real-hardware measurement could not include
(see ../../accel_sim/README.md): the GEMM really packing the selection metadata into
a register, and the hypothetical MMA instruction reading it as an extra operand.

  total = current + packing + operand        (all as % of the same NVFP4 forward)

  current : the existing scripts' selection-quantize + metaA/metaB TMA-load cost.
            Llama from assemble_llama_final.py (authoritative, Stage-corrected);
            Qwen/DeepSeek from assemble_final_table.py (not Stage-corrected, same
            caveat as there), except that the 12 MoE routed-expert shapes (qwen_gateup/
            down, ds_routed_gu/down, M=128..768), whose single-shot microbench deltas
            sit at the noise floor, use the 10-rep mean of the same two binaries
            (results/accel_sim/moe_expert_repeat.txt, measure_moe_repeat.sh).
  packing : real GPU, per shape: GEMM ms of the metadata-patched CUTLASS GEMM with
            TMA load + packing (NVFP4_META_SW_CORRECT=3) minus TMA load only (unset),
            same SMEM (81,920 B) / Stage count, 5 interleaved reps each
            (results/accel_sim/packing_raw.csv, results/accel_sim/measure_packing.sh).
            Added ms per call x calls per forward; MoE routed experts: the per-shape
            ratio applied to the real grouped-GEMM time (as for the TMA-load cost).
  operand : Accel-Sim 2.0 (Blackwell OMMA port, RTX PRO 4000 config), 12-register vs
            13-register (Linear) / 14-register (QK^T) OMMA on the same trace of the
            real mode-3 GEMM; per shape the mean of 20 launch-latency-perturbed pairs,
            shapes pooled by inverse-variance weighting (results/accel_sim/*.csv):
              Linear: Llama q_proj M=2048, Llama k/v_proj M=4096, DeepSeek routed
                      gate/up M=768
              QK^T  : Llama QK^T M=2048 / 4096 / 8192
            applied to every Linear / QK^T GEMM of the model.

GEMM ms per forward (the base the packing/operand % apply to):
  Llama   : Linear = real full-model GEMM-only true-baseline time
            (log/linear_stage_corrected_5rep.log, "gemm: true"), QK^T = true-baseline
            NVFP4 QK^T x 32 heads x 32 layers (assemble_llama_final.py).
  Qwen/DS : baseline NVFP4 GEMM ms per call (base_raw.txt) x calls per forward, MoE
            routed experts from the real grouped-GEMM time (grouped_raw.txt).
The 95% CI covers the operand term only.

Writes results/final_table_accel_sim.csv (same layout as final_table.csv) and
results/final_table_accel_sim_breakdown.csv. Does not modify any other file.
"""
import csv, importlib.util, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)   # module level only; their __main__ blocks are not run
    return m


ll = load("assemble_llama_final")
ft = load("assemble_final_table")

# MoE routed-expert shapes: replace the single-shot microbench values that
# assemble_final_table.py's delta()/nv()/pct() read with 10-rep means.
_rep = {}
for blk in re.split(r"^### ", open(os.path.join(R, "accel_sim", "moe_expert_repeat.txt")).read(), flags=re.M)[1:]:
    lines = blk.strip().splitlines()
    h = re.match(r"rep\d+ (\S+) (\S+) n=", lines[0])
    d = _rep.setdefault((h.group(1), h.group(2)), {})
    for l in lines[1:]:
        x = re.search(r"nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
        if x: d.setdefault("g", []).append(float(x.group(1)))
        x = re.search(r"DRAM store\+reload.*?\)\s*([\d.]+) ms.*meta reload alone:\s*([\d.]+) ms", l)
        if x:
            d.setdefault("dram", []).append(float(x.group(1)))
            d.setdefault("rl", []).append(float(x.group(2)))
        x = re.search(r"with store \(DRAM\):\s*([\d.]+)", l)
        if x: d.setdefault("aq", []).append(float(x.group(1)))
for (label, binary), d in _rep.items():
    mean = {k: st.mean(v) for k, v in d.items()}
    ft.runs[(label, binary)] = mean
    if binary == "nvfp4_microbench_baseline":
        ft.base[(label, None)] = mean


def op_stats(files):
    """Inverse-variance pooled mean (fraction) and 95% CI of per-shape paired pct means."""
    w = wm = 0.0
    for f in files:
        p = [float(r["pct"]) / 100 for r in csv.DictReader(open(os.path.join(R, "accel_sim", f)))]
        m, se = st.mean(p), st.stdev(p) / len(p) ** 0.5
        w += 1 / se ** 2
        wm += m / se ** 2
    m, se = wm / w, w ** -0.5
    return m, m - 1.96 * se, m + 1.96 * se


OP_LIN = op_stats(["logs_qproj_noise_20.csv", "logs_noise_kv4096.csv", "logs_noise_dsrgu768.csv"])
OP_QKT = op_stats(["logs_noise_qkt2048.csv", "logs_noise_qkt4096.csv", "logs_noise_qkt8192.csv"])
# packing: per-shape mean GEMM ms, TMA only (swc0) and TMA + packing (swc3)
_p = {}
for r in csv.DictReader(open(os.path.join(R, "accel_sim", "packing_raw.csv"))):
    _p.setdefault((r["label"], r["build"]), []).append(float(r["gemm_ms"]))
pk_add = lambda label: st.mean(_p[(label, "swc3")]) - st.mean(_p[(label, "swc0")])   # ms per call
pk_pct = lambda label: pk_add(label) / st.mean(_p[(label, "swc0")])

# Llama real full-model Linear GEMM-only time, true baseline, 5-rep mean
_g = {}
for line in open(os.path.join(ll.L, "linear_stage_corrected_5rep.log")):
    m = re.match(r"rep\d+ M=\s*(\d+)\s+fwd:.*gemm: true=\s*([\d.]+)", line)
    if m:
        _g.setdefault(int(m.group(1)), []).append(float(m.group(2)))
llama_lin_gemm = {M: sum(v) / len(v) for M, v in _g.items()}


def llama(M):
    denom = ll.fwd_true[M] - ll.real_attn_bf16[M] + (ll.qkt_base_nvfp4[M] + ll.pv_base_nvfp4[M] + ll.other_bf16[M])
    added = ll.linear_added[M] + ll.qkt_added[M]
    pack = (32 * (2 * pk_add(f"llama_qoproj_M{M}") + 2 * pk_add(f"llama_kvproj_M{M}")
                  + 2 * pk_add(f"llama_gateup_M{M}") + pk_add(f"llama_down_M{M}"))
            + 1024 * pk_add(f"llama_qkt_M{M}"))
    return added, denom, llama_lin_gemm[M], ll.qkt_base_nvfp4[M], pack


g = lambda label: ft.base[(label, None)]["g"]


def qwen(M):
    added, denom, _ = ft.qwen_row(M)
    lin = (sum(n * g(f"{l}_M{M}") for n, l in ft.items_qwen_rest())
           + 48 * (2 * ft.grp[f"qwen_gateup_M{M}"] + ft.grp[f"qwen_down_M{M}"]))
    pack = (sum(n * pk_add(f"{l}_M{M}") for n, l in ft.items_qwen_rest())
            + 48 * (2 * ft.grp[f"qwen_gateup_M{M}"] * pk_pct(f"qwen_gateup_M{M}")
                    + ft.grp[f"qwen_down_M{M}"] * pk_pct(f"qwen_down_M{M}"))
            + 1536 * pk_add(f"qwen_qkt_M{M}"))
    return added, denom, lin, 1536 * g(f"qwen_qkt_M{M}"), pack


def deepseek(M):
    added, denom, _ = ft.ds_row(M)
    lin = (sum(n * g(f"{l}_M{M}") for n, l in ft.items_ds_rest())
           + 27 * (2 * ft.grp[f"ds_routed_gu_M{M}"] + ft.grp[f"ds_routed_down_M{M}"]))
    pack = (sum(n * pk_add(f"{l}_M{M}") for n, l in ft.items_ds_rest())
            + 27 * (2 * ft.grp[f"ds_routed_gu_M{M}"] * pk_pct(f"ds_routed_gu_M{M}")
                    + ft.grp[f"ds_routed_down_M{M}"] * pk_pct(f"ds_routed_down_M{M}"))
            + 448 * pk_add(f"ds_qkt_M{M}"))
    return added, denom, lin, 448 * g(f"ds_qkt_M{M}"), pack


if __name__ == "__main__":
    print(f"operand (GEMM-level): Linear {100*OP_LIN[0]:+.2f}% [{100*OP_LIN[1]:+.2f}, {100*OP_LIN[2]:+.2f}]"
          f"  QK^T {100*OP_QKT[0]:+.2f}% [{100*OP_QKT[1]:+.2f}, {100*OP_QKT[2]:+.2f}]"
          "   packing: per shape (results/accel_sim/packing_raw.csv)")
    rows, table = [], {}
    for model, fn in (("Llama", llama), ("Qwen", qwen), ("DeepSeek", deepseek)):
        for M in ft.MS:
            added, denom, lin, qkt, pack_ms = fn(M)
            cur = 100 * added / denom
            pack = 100 * pack_ms / denom
            op = [100 * (OP_LIN[i] * lin + OP_QKT[i] * qkt) / denom for i in range(3)]
            total = cur + pack + op[0]
            table[(model, M)] = total
            rows.append([model, M, cur, pack, op[0], op[1], op[2], total, cur + pack + op[1], cur + pack + op[2]])
            print(f"{model:8s} M={M:>5}  current {cur:5.2f}  +packing {pack:4.2f}  +operand {op[0]:4.2f} "
                  f"[{op[1]:.2f}, {op[2]:.2f}]  => {total:5.2f}%")
    with open(os.path.join(R, "final_table_accel_sim.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_tokens", "llama_pct", "qwen_pct", "deepseek_pct"])
        for M in ft.MS:
            w.writerow([M] + [f"{table[(m, M)]:.2f}" for m in ("Llama", "Qwen", "DeepSeek")])
    with open(os.path.join(R, "final_table_accel_sim_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "total_tokens", "current_pct", "packing_pct", "operand_pct", "operand_ci_lo",
                    "operand_ci_hi", "total_pct", "total_ci_lo", "total_ci_hi"])
        for r in rows:
            w.writerow(r[:2] + [f"{x:.2f}" for x in r[2:]])
    print("\nwrote results/final_table_accel_sim.csv, results/final_table_accel_sim_breakdown.csv")
