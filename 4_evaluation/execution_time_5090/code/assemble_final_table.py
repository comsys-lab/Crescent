#!/usr/bin/env python3
"""
Reproduces the final "Inference Time Overhead" table (Linear real-measured + QK^T
assembled from microbench deltas; Qwen/DeepSeek MoE Linear assembled on top of REAL
grouped-GEMM baseline timing). Reads only from ../results/.

Methodology recap:
  - Baseline  = NVFP4 quantize (no selection) + plain NVFP4 GEMM (no metadata).
  - Our method = NVFP4 quantize+largest-error-selection + NVFP4 GEMM with metaA
    (and metaB for QK^T) TMA-loaded. Group size 4 for Linear (activation only),
    group size 8 for QK^T (both Q and K get selection).
  - overhead % = added_ms / NVFP4-forward-equivalent denominator, per (model, M).
  - Llama: Linear added_ms is a REAL measurement (baseline vs. selection NVFP4
    forward, both run end-to-end via nvfp4_lib.cu + e2e_llama.py). QK^T is assembled
    (per-head GEMM delta x heads x layers, doubled for Q+K selection cost).
  - Qwen/DeepSeek: MoE routed-expert Linear uses a REAL grouped-GEMM baseline
    (see run_grouped.sh / 79d_blackwell_geforce_nvfp4_grouped_gemm.cu), with the
    per-shape %-overhead ratio (measured via the single-GEMM microbench, since the
    L2-bandwidth-bound byte-ratio argument makes this ratio ~M/grouping-independent)
    applied to that real grouped baseline time. Non-MoE Linear (q/o/k/v, shared/dense
    experts) and QK^T are assembled the same way as Llama.
  - 16384 token row dropped: DeepSeek OOMs at this length on this 4x24GB GPU setup
    regardless of memory cap (non-flash attention + full-vocab logits), so no clean
    measurement exists for it.

Known limitations (see README.md):
  - QK^T is never directly measured end-to-end; it is always an assembled estimate.
  - The grouped-GEMM %-overhead-ratio transfer (from individual-GEMM microbench to
    grouped-GEMM baseline) is a physical-reasoning extrapolation, not a direct
    measurement of a metadata-enabled grouped GEMM kernel (that collective was not
    patched -- see README).
  - Small per-expert microbench deltas (M~128) are sometimes at the measurement noise
    floor (delta rounds to 0.00%).
"""
import re, csv, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results") + "/"

def parse_sweep(path):
    """Parses a '### label [binary] n=.. k=.. m=..' raw sweep dump into {(label,binary): {...}}."""
    out = {}
    for b in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = b.strip().splitlines()
        h = re.match(r"(\S+)(?: (\S+))? n=", lines[0])
        label, binary = h.group(1), h.group(2)
        d = {}
        for l in lines[1:]:
            x = re.search(r"nvfp4 GEMM\s*:\s*([\d.]+) ms", l)
            if x: d['g'] = float(x.group(1))
            x = re.search(r"DRAM store\+reload.*?\)\s*([\d.]+) ms.*meta reload alone:\s*([\d.]+) ms", l)
            if x: d['dram'] = float(x.group(1)); d['rl'] = float(x.group(2))
            x = re.search(r"with store \(DRAM\):\s*([\d.]+)", l)
            if x: d['aq'] = float(x.group(1))
        out[(label, binary)] = d
    return out

def parse_grouped(path):
    """Parses grouped_raw.txt ('### label' + 2x 'Avg runtime : X ms' lines, cooperative/pingpong)."""
    out = {}
    for b in re.split(r"^### ", open(path).read(), flags=re.M)[1:]:
        lines = b.strip().splitlines()
        label = lines[0]
        times = [float(re.search(r"([\d.]+) ms", l).group(1)) for l in lines[1:] if "Avg runtime" in l]
        out[label] = times[-1]  # pingpong schedule (consistently faster than cooperative here)
    return out

runs = {}
runs.update(parse_sweep(R + "shapes_raw.txt"))
runs.update(parse_sweep(R + "llama_raw.txt"))
base = parse_sweep(R + "base_raw.txt")  # binary=None; baseline A-quant ("aq") + GEMM ("g")
grp = parse_grouped(R + "grouped_raw.txt")

L, P = "nvfp4_microbench_linear", "nvfp4_microbench_qkt"

def delta(label, patched_bin, doubled=False):
    """Added ms per GEMM call: (patched_gemm_ms - baseline_gemm_ms) + quantize-side DRAM-store delta."""
    b = runs[(label, "nvfp4_microbench_baseline")]
    p = runs[(label, patched_bin)]
    dq = p['dram'] - p['rl']
    return (p['g'] - b['g']) + (2 * dq if doubled else dq)

def nv(label):
    """Baseline (no-selection) A-quant + NVFP4 GEMM ms per call -- the 'config 1' denominator unit."""
    b = base[(label, None)]
    return b['aq'] + b['g']

def pct(label):
    return 100 * delta(label, L) / nv(label)

def load_csv(path, key):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if key in r['model']:
            d[(int(r['M']), r['role'], int(r['K']), int(r['N']))] = r
    return d

MS = [2048, 4096, 8192]

# ---------------------------------------------------------------------------------------------
# LLAMA: real e2e NVFP4 forward (baseline vs. selection), measured via nvfp4_lib.cu + e2e_llama.py.
# ---------------------------------------------------------------------------------------------
# (base_fwd_ms, selection_fwd_ms) -- average of 2 runs each, GPU timer, static activation scale.
llama_e2e = {2048: (211.3, 221.2), 4096: (579.7, 598.1), 8192: (1716.7, 1750.4)}

def llama_row(M):
    base_fwd, sel_fwd = llama_e2e[M]
    linear_added = sel_fwd - base_fwd           # REAL measured delta (not assembled)
    qkt_added = 1024 * delta(f"llama_qkt_M{M}", P, True)   # 32 heads x 32 layers
    total_added = linear_added + qkt_added
    return total_added, base_fwd, 100 * total_added / base_fwd

# ---------------------------------------------------------------------------------------------
# QWEN / DEEPSEEK: non-MoE Linear + QK^T assembled; MoE routed-expert Linear assembled on top of
# a REAL grouped-GEMM baseline measurement.
# ---------------------------------------------------------------------------------------------
def items_qwen_rest(): return [(48, "qwen_qproj"), (48, "qwen_oproj"), (96, "qwen_kvproj")]
def items_ds_rest(): return [(54, "ds_shared_gu"), (27, "ds_shared_down"), (2, "ds_dense0_gu"),
                              (1, "ds_dense0_down"), (112, "ds_attn")]

q = load_csv(R + "prefill_moe_detail.csv", "Qwen")
d = load_csv(R + "prefill_moe_detail.csv", "deepseek")
qwen_nvfp4_fwd = {2048: 521.45, 4096: 984.5, 8192: 2544.0}   # real GPU-kernel-time NVFP4 forward
ds_nvfp4_fwd = {2048: 299.6, 4096: 777.55, 8192: 2544.75}

def qwen_row(M):
    nlayers = 48
    old_moe_ms = (float(q[(M, "gate_proj", 2048, 768)]["n_instances"]) * nv(f"qwen_gateup_M{M}")
                + float(q[(M, "up_proj", 2048, 768)]["n_instances"]) * nv(f"qwen_gateup_M{M}")
                + float(q[(M, "down_proj", 768, 2048)]["n_instances"]) * nv(f"qwen_down_M{M}"))
    new_moe_base = nlayers * (grp[f"qwen_gateup_M{M}"] * 2 + grp[f"qwen_down_M{M}"])
    new_moe_added = nlayers * (grp[f"qwen_gateup_M{M}"] * (pct(f"qwen_gateup_M{M}") / 100) * 2
                              + grp[f"qwen_down_M{M}"] * (pct(f"qwen_down_M{M}") / 100))
    rest_added = sum(n * delta(f"{l}_M{M}", L) for n, l in items_qwen_rest())
    qkt_added = 1536 * delta(f"qwen_qkt_M{M}", P, True)   # 32 heads x 48 layers
    fwd_est = qwen_nvfp4_fwd[M] - old_moe_ms + new_moe_base
    total_added = new_moe_added + rest_added + qkt_added
    return total_added, fwd_est, 100 * total_added / fwd_est

def ds_row(M):
    nlayers = 27
    old_moe_ms = (float(d[(M, "gate_proj", 2048, 1408)]["n_instances"]) * nv(f"ds_routed_gu_M{M}")
                + float(d[(M, "up_proj", 2048, 1408)]["n_instances"]) * nv(f"ds_routed_gu_M{M}")
                + float(d[(M, "down_proj", 1408, 2048)]["n_instances"]) * nv(f"ds_routed_down_M{M}"))
    new_moe_base = nlayers * (grp[f"ds_routed_gu_M{M}"] * 2 + grp[f"ds_routed_down_M{M}"])
    new_moe_added = nlayers * (grp[f"ds_routed_gu_M{M}"] * (pct(f"ds_routed_gu_M{M}") / 100) * 2
                              + grp[f"ds_routed_down_M{M}"] * (pct(f"ds_routed_down_M{M}") / 100))
    rest_added = sum(n * delta(f"{l}_M{M}", L) for n, l in items_ds_rest())
    qkt_added = 448 * delta(f"ds_qkt_M{M}", P, True)   # 16 heads x 28 layers
    fwd_est = ds_nvfp4_fwd[M] - old_moe_ms + new_moe_base
    total_added = new_moe_added + rest_added + qkt_added
    return total_added, fwd_est, 100 * total_added / fwd_est

if __name__ == "__main__":
    rows = {"Llama": {}, "Qwen": {}, "DeepSeek": {}}
    for M in MS:
        a, f, p_ = llama_row(M); rows["Llama"][M] = p_
        print(f"Llama    M={M:>6}  added={a:7.2f}ms  denom={f:9.1f}ms  -> {p_:.2f}%")
    for M in MS:
        a, f, p_ = qwen_row(M); rows["Qwen"][M] = p_
        print(f"Qwen     M={M:>6}  added={a:7.2f}ms  denom={f:9.1f}ms  -> {p_:.2f}%")
    for M in MS:
        a, f, p_ = ds_row(M); rows["DeepSeek"][M] = p_
        print(f"DeepSeek M={M:>6}  added={a:7.2f}ms  denom={f:9.1f}ms  -> {p_:.2f}%")

    out_csv = os.path.join(HERE, "..", "results", "final_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["total_tokens", "llama_pct", "qwen_pct", "deepseek_pct"])
        for M in MS:
            w.writerow([M, f"{rows['Llama'][M]:.2f}", f"{rows['Qwen'][M]:.2f}", f"{rows['DeepSeek'][M]:.2f}"])
    print(f"\nwrote {out_csv}")
