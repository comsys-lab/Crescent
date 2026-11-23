#!/usr/bin/env python3
"""Inference-time overhead of Ours vs. plain NVFP4, per model and token count.

  current  : ../../4_evaluation/execution_time final table (selection quantize +
             metaA/metaB TMA load), reused through assemble_final_table.py's own
             functions (that script and its outputs are not modified).
  packing  : real-GPU mode-3 vs mode-0 GEMM delta from 4_evaluation/accel_sim/README.md
             (Linear M=2048,K=N=4096: +2.87% of the GEMM; QK^T M=N=4096,K=128: +1.40%),
             applied to every Linear / QK^T GEMM of the model.
  operand  : simulated 13th (Linear) / 13th+14th (QK^T) operand. Per shape, the mean
             of 20 launch-latency-perturbed 12op/widened pairs; shapes of one kind
             are pooled by inverse-variance weighting (no size trend was found):
             Linear: Llama q_proj M=2048, Llama k/v_proj M=4096, DeepSeek routed
                     gate/up M=768;
             QK^T  : Llama QK^T M=2048, 4096, 8192.
             Applied to every Linear / QK^T GEMM of the model.
GEMM time per forward uses the measured baseline NVFP4 GEMM ms per call (base_raw.txt)
and, for MoE routed experts, the measured grouped-GEMM time (grouped_raw.txt).
95% CI covers the operand terms only (normal approx. on the pooled mean).
Writes overhead_table.csv.
"""
import csv, importlib.util, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ET = os.path.join(HERE, "..", "..", "4_evaluation", "execution_time", "code")
spec = importlib.util.spec_from_file_location("aft", os.path.join(ET, "assemble_final_table.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)  # module level only; its __main__ block (which writes files) is not run


def op_stats(paths):
    """Inverse-variance pooled mean of per-shape paired-pct means."""
    w_sum = wm_sum = 0.0
    n = 0
    for path in paths:
        p = [float(r["pct"]) / 100 for r in csv.DictReader(open(os.path.join(HERE, path)))]
        m, se = st.mean(p), st.stdev(p) / len(p) ** 0.5
        w_sum += 1 / se ** 2
        wm_sum += m / se ** 2
        n += len(p)
    m, se = wm_sum / w_sum, w_sum ** -0.5
    return m, m - 1.96 * se, m + 1.96 * se, n


OP_LIN = op_stats(["logs_qproj_noise_20.csv", "logs_noise_kv4096.csv", "logs_noise_dsrgu768.csv"])
OP_QKT = op_stats(["logs_noise_qkt2048.csv", "logs_noise_qkt4096.csv", "logs_noise_qkt8192.csv"])
PACK_LIN = (0.1870 - 0.1821) / 0.1708
PACK_QKT = (0.0448 - 0.0442) / 0.0430
g = lambda label: a.base[(label, None)]["g"]


def gemm_ms(model, M):
    if model == "Llama":
        lin = 32 * (2 * g(f"llama_qoproj_M{M}") + 2 * g(f"llama_kvproj_M{M}")
                    + 2 * g(f"llama_gateup_M{M}") + g(f"llama_down_M{M}"))
        return lin, 1024 * g(f"llama_qkt_M{M}")
    if model == "Qwen":
        lin = (sum(n * g(f"{l}_M{M}") for n, l in a.items_qwen_rest())
               + 48 * (2 * a.grp[f"qwen_gateup_M{M}"] + a.grp[f"qwen_down_M{M}"]))
        return lin, 1536 * g(f"qwen_qkt_M{M}")
    lin = (sum(n * g(f"{l}_M{M}") for n, l in a.items_ds_rest())
           + 27 * (2 * a.grp[f"ds_routed_gu_M{M}"] + a.grp[f"ds_routed_down_M{M}"]))
    return lin, 448 * g(f"ds_qkt_M{M}")


rows = []
for model, rowf in (("Llama", a.llama_row), ("Qwen", a.qwen_row), ("DeepSeek", a.ds_row)):
    for M in a.MS:
        added, denom, cur = rowf(M)
        lin, qkt = gemm_ms(model, M)
        pack = 100 * (PACK_LIN * lin + PACK_QKT * qkt) / denom
        op = [100 * (OP_LIN[i] * lin + OP_QKT[i] * qkt) / denom for i in range(3)]
        rows.append([model, M, cur, pack, op[0], op[1], op[2],
                     cur + pack + op[0], cur + pack + op[1], cur + pack + op[2]])

hdr = ["model", "total_tokens", "current_pct", "packing_pct", "operand_pct", "operand_ci_lo",
       "operand_ci_hi", "total_pct", "total_ci_lo", "total_ci_hi"]
with open(os.path.join(HERE, "overhead_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    for r in rows:
        w.writerow(r[:2] + [f"{x:.2f}" for x in r[2:]])
print(f"operand (GEMM-level): Linear {100*OP_LIN[0]:+.2f}% [{100*OP_LIN[1]:+.2f}, {100*OP_LIN[2]:+.2f}] n={OP_LIN[3]}"
      f" | QK^T {100*OP_QKT[0]:+.2f}% [{100*OP_QKT[1]:+.2f}, {100*OP_QKT[2]:+.2f}] n={OP_QKT[3]}")
for r in rows:
    print(f"{r[0]:8s} {r[1]:5d}  current {r[2]:5.2f}  +pack {r[3]:4.2f}  +operand {r[4]:4.2f} "
          f"[{r[5]:.2f}, {r[6]:.2f}]  => total {r[7]:5.2f}% [{r[8]:.2f}, {r[9]:.2f}]")
