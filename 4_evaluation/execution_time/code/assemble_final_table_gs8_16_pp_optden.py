#!/usr/bin/env python3
"""
Same numerators as assemble_final_table_gs8_16_pp.py (Czardas: Linear 1-of-8 pre-packed, QK^T 1-of-16,
MoE on the real grouped GEMM), but the Qwen/DeepSeek NVFP4-forward denominator rebuilt from
components instead of the HF-eager MoE forward (whose per-expert Python loop is 61-70% of its GPU
kernel time, results/prefill_moe_detail.csv):

  denom = dense Linear  : sum n x (A-quant + NVFP4 GEMM), baseline (gs8_16_raw.txt)
        + MoE experts   : layers x (2 x grouped GEMM(gate/up) + grouped GEMM(down)), baseline
                          (grouped_pp_raw.txt) + their A-quant at the quantized tensors (moe_quant_pp_raw.txt)
        + attention     : heads x layers x (NVFP4 QK^T + NVFP4 PV + softmax/other) per head, the Llama
                          construction (assemble_llama_final.py); head_dim = 128 and M are the same
                          for all three models, so the per-head PV and softmax/other values are
                          Llama's measured ones; QK^T = gs8_16_raw.txt baseline
        + other non-GEMM: results/accel_sim/moe_other.txt (measure_moe_other.py): HF-eager
                          elementwise ops (norms, QK-norm, RoPE, residuals, activation), router,
                          top-k, sort-based permute / index_add unpermute (no per-expert loop),
                          final norm + last-token lm_head; per layer x layers (+ DeepSeek layer-0
                          dense activation; layer 0 has no routing)
Llama is unchanged (its denominator is its real full-model forward, no MoE).
Writes results/final_table_gs8_16_pp_optden.csv and ..._breakdown.csv.
"""
import csv, importlib.util, os, re, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
spec = importlib.util.spec_from_file_location("pp", os.path.join(HERE, "assemble_final_table_gs8_16_pp.py"))
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
ft, ll, sim = pp.ft, pp.ll, pp.sim

oth = {}
for l in open(os.path.join(R, "accel_sim", "moe_other.txt")):
    m = re.match(r"rep\d+ (\w+) M=(\d+) (.*)", l)
    if m:
        d = oth.setdefault((m.group(1), int(m.group(2))), {})
        for kv in m.group(3).split():
            k, v = kv.split("="); d.setdefault(k, []).append(float(v))
O = lambda name, M, k: st.mean(oth[(name, M)][k])
aq_moe = lambda label: (2 * aq_moe("qwen_down_q_M4096") if label == "qwen_down_q_M8192"
                        else pp.mean(pp.mq, label, "sel", "aq"))
nv_lin = lambda label: pp.mean(pp.gs, label, "baseline", "aq") + pp.mean(pp.gs, label, "baseline", "g")

MOD = {"Qwen": dict(key="qwen", lin=ft.items_qwen_rest, L=48, moeL=48, heads=32, qk="qwen_qkt", gu="qwen_gateup", dn="qwen_down"),
       "DeepSeek": dict(key="ds", lin=ft.items_ds_rest, L=28, moeL=27, heads=16, qk="ds_qkt", gu="ds_routed_gu", dn="ds_routed_down")}


def denom(name, M):
    c = MOD[name]; parts = {}
    parts["dense Linear"] = sum(n * nv_lin(f"{l}_M{M}") for n, l in c["lin"]())
    u, d = f"{c['gu']}_M{M}", f"{c['dn']}_M{M}"
    parts["MoE experts"] = c["moeL"] * (2 * pp.G(u, "grp_base_p0") + pp.G(d, "grp_base_p0")
                                        + 2 * aq_moe(f"{c['gu']}_q_M{M}") + aq_moe(f"{c['dn']}_q_M{M}"))
    hl = c["heads"] * c["L"]
    parts["attention"] = hl * (pp.mean(pp.gs, f"{c['qk']}_M{M}", "baseline", "g")
                               + ll.pv_base_nvfp4[M] / 1024 + ll.other_bf16[M] / 1024)
    parts["other non-GEMM"] = (c["L"] * O(c["key"], M, "layer_nonmoe") + c["moeL"] * O(c["key"], M, "layer_moe")
                               + O(c["key"], M, "fwd_once") + O(c["key"], M, "dense0_act"))
    return sum(parts.values()), parts


if __name__ == "__main__":
    rows, table = [], {}
    for name in ("Llama", "Qwen", "DeepSeek"):
        for M in ft.MS:
            c, p, op = pp.model(name, M)
            if name != "Llama":
                old = ft.qwen_row(M)[1] if name == "Qwen" else ft.ds_row(M)[1]
                new, parts = denom(name, M)
                c, p, op = c * old / new, p * old / new, [x * old / new for x in op]
                print(f"{name:8s} M={M:5d} denom {old:7.1f} -> {new:7.1f} ms  ("
                      + ", ".join(f"{k} {v:.1f}" for k, v in parts.items()) + ")")
            table[(name, M)] = c + p + op[0]
            rows.append([name, M, c, p, op[0], op[1], op[2], c + p + op[0], c + p + op[1], c + p + op[2]])
    print()
    for r in rows:
        print(f"{r[0]:8s} M={r[1]:>5}  current {r[2]:5.2f}  +packing {r[3]:4.2f}  +operand {r[4]:4.2f} [{r[5]:.2f}, {r[6]:.2f}]  => {r[7]:5.2f}%")
    with open(os.path.join(R, "final_table_gs8_16_pp_optden.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["total_tokens", "llama_pct", "qwen_pct", "deepseek_pct"])
        for M in ft.MS: w.writerow([M] + [f"{table[(m, M)]:.2f}" for m in ("Llama", "Qwen", "DeepSeek")])
    with open(os.path.join(R, "final_table_gs8_16_pp_optden_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "total_tokens", "current_pct", "packing_pct", "operand_pct", "operand_ci_lo",
                                       "operand_ci_hi", "total_pct", "total_ci_lo", "total_ci_hi"])
        for r in rows: w.writerow(r[:2] + [f"{x:.2f}" for x in r[2:]])
