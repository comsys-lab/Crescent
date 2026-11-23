# Inference Time Overhead — largest-error selection metadata

Measures how much integrating quantization-error computation and grouped selection
into NVFP4 activation quantization increases real inference time, across Llama-3.1-8B,
Qwen3-30B-A3B (MoE), and DeepSeek-MoE-16B-base.

## Directory layout

- `code/` — all source: CUDA microbenchmarks, the CUTLASS collective patch, the
  end-to-end PyTorch harnesses, and the final assembly script.
- `results/` — see the split below; not everything here feeds the final table.
- `log/` — selected raw run logs backing specific numbers in the table (not every
  intermediate run was kept; see "What's not here" below).

### `results/` and `log/`: what feeds which script

| File | Used by | What it is |
|---|---|---|
| `shapes_raw.txt`, `llama_raw.txt`, `base_raw.txt` | `assemble_final_table.py` (Qwen/DeepSeek) | Raw `nvfp4_microbench` sweep dumps (padded-baseline vs. metadata-patched GEMM timing, per shape) |
| `grouped_raw.txt` | `assemble_final_table.py` (Qwen/DeepSeek) | Raw grouped-GEMM baseline timing (CUTLASS's unmodified grouped-GEMM example) |
| `prefill_moe_detail.csv` | `assemble_final_table.py` (Qwen/DeepSeek, for `n_instances` and real `fwd_ms`) | Real per-projection BF16 forward timing for Qwen/DeepSeek, from `prefill_profile_moe.py` |
| `qkt_pv_repeat.txt` | `assemble_llama_final.py` | NVFP4 true-baseline vs. selection QK^T, and true-baseline PV, standalone GEMM, 5 reps each |
| `bf16_qkt_pv_repeat.txt` | `assemble_llama_final.py` | Standalone BF16 cuBLAS QK^T and PV GEMM, 5 reps each (splits real fused attention into QK^T/PV/"other") |
| `log/linear_stage_corrected_5rep.log` | `assemble_llama_final.py` | Full-model Linear true-baseline vs. selection, full-forward and GEMM-only (profiler-isolated) deltas, 5 reps |
| `log/bf16_real_attention_5rep.log` | `assemble_llama_final.py` | Real fused BF16 attention kernel time (unmodified Llama), 5 reps |
| `prefill_gemm_detail.csv`, `prefill_gemm_rollup.csv` | neither (reference only) | Real per-projection BF16 forward timing **with shapes** (`K`, `N`, `gemm_M`) for Llama, from `prefill_profile.py` (that script also supports other dense models — Mistral-7B, opt-6.7b — but those rows were stripped here as unrelated to this study). Kept as reference for "what does a real BF16 forward actually spend time on, and at what matrix size." |
| `kernels/*.csv` | neither (reference only) | Real Nsight **per-kernel-name** dump during an actual Llama forward pass (`ms_per_forward`, `launches_per_forward`, `category`, raw kernel name). No shape column — cross-reference against `prefill_gemm_detail.csv`'s `K`/`N`/`gemm_M` by `category`/`proj` to connect a kernel to its matrix size. Qwen/DeepSeek have no equivalent dump. |
| `final_table.csv` | (output of `assemble_final_table.py`) | Qwen/DeepSeek numbers (+ stale Llama column — see "Stage-count confound") |
| `accel_sim/*.csv` | `assemble_final_table_accel_sim.py` | Accel-Sim cycles of 20 launch-latency-perturbed 12-register vs 13/14-register OMMA pairs per shape (q_proj M=2048, k/v_proj M=4096, DeepSeek routed gate/up M=768, QK^T M=2048/4096/8192) |
| `accel_sim/packing_raw.csv` (+ `measure_packing.sh`) | `assemble_final_table_accel_sim.py` | Real-GPU GEMM ms per shape, metadata TMA load only vs TMA load + packing, 5 interleaved reps (all 57 Linear/QK^T shapes of the three models) |
| `accel_sim/moe_expert_repeat.txt` (+ `measure_moe_repeat.sh`) | `assemble_final_table_accel_sim.py` | 10-rep re-measurement of the 12 MoE routed-expert shapes with the same baseline/selection binaries as `shapes_raw.txt`/`base_raw.txt` (replaces their noise-floor single-shot values in that script only) |
| `final_table_accel_sim.csv`, `final_table_accel_sim_breakdown.csv` | (output of `assemble_final_table_accel_sim.py`) | Final table + metadata packing + extra MMA operand cost, and its per-term breakdown |

## Final result

| Total Tokens | Llama | Qwen | DeepSeek |
|---|---|---|---|
| 2,048 | **5.98%** | 1.55% | 1.15% |
| 4,096 | **4.96%** | 2.26% | 0.91% |
| 8,192 | **3.74%** | 1.00% | 0.71% |

Llama: reproduce with `python3 code/assemble_llama_final.py` (reads `results/` +
`log/`). This is the authoritative Llama number — see "Stage-count confound" below for
why it differs from `assemble_final_table.py`'s Llama column (which is now stale).

Qwen/DeepSeek: reproduce with `python3 code/assemble_final_table.py` (reads only
`results/`, writes `results/final_table.csv`). These numbers were **not** re-measured
after the Stage-count confound was found — see the caveat in that section.

### Including metadata packing and the extra MMA operand

The table above is what real hardware can measure: the GEMM TMA-loads the metadata,
but no real MMA instruction consumes it. Adding the two remaining costs (see
`../accel_sim/README.md`) — the GEMM packing the metadata into a register (real GPU,
`NVFP4_META_SW_CORRECT=3`) and the hypothetical MMA reading it as a 13th (Linear) /
13th+14th (QK^T) operand (Accel-Sim 2.0, Blackwell OMMA port):

| Total Tokens | Llama | Qwen | DeepSeek |
|---|---|---|---|
| 2,048 | **7.19%** | 2.21% | 1.60% |
| 4,096 | **6.07%** | 2.19% | 1.07% |
| 8,192 | **4.53%** | 1.10% | 0.87% |

Breakdown (% of the same NVFP4 forward; CI = 95%, operand term only):

| Model | Tokens | Table above | + packing | + operand [CI] | Total |
|---|---|---|---|---|---|
| Llama | 2,048 | 5.98 | 1.06 | 0.14 [-0.07, 0.35] | 7.19 |
| Llama | 4,096 | 4.96 | 0.98 | 0.12 [-0.05, 0.30] | 6.07 |
| Llama | 8,192 | 3.74 | 0.66 | 0.13 [-0.04, 0.30] | 4.53 |
| Qwen | 2,048 | 2.12* | 0.03 | 0.06 [-0.02, 0.15] | 2.21 |
| Qwen | 4,096 | 1.71* | 0.43 | 0.05 [-0.01, 0.12] | 2.19 |
| Qwen | 8,192 | 0.99* | 0.06 | 0.06 [-0.00, 0.12] | 1.10 |
| DeepSeek | 2,048 | 1.41* | 0.16 | 0.04 [-0.02, 0.10] | 1.60 |
| DeepSeek | 4,096 | 0.91* | 0.13 | 0.03 [-0.01, 0.07] | 1.07 |
| DeepSeek | 8,192 | 0.73* | 0.12 | 0.02 [-0.01, 0.05] | 0.87 |

\* Differs from the table above (Qwen 1.55/2.26/1.00, DeepSeek 1.15/0.91/0.71): the 12
MoE routed-expert shapes (Qwen gate/up and down, DeepSeek routed gate/up and down,
M=128–768 per expert) are re-measured 10× with the same two binaries and averaged
(`results/accel_sim/moe_expert_repeat.txt`, `measure_moe_repeat.sh`). Their GEMMs take
~0.01 ms, so the single-shot values in `shapes_raw.txt`/`base_raw.txt` were noise-floor
readings — e.g. `qwen_down_M4096` +16.35% single-shot vs +0.60% ± 0.34 over 10 reps, which
had made Qwen's 4,096-token value jump. Large-M expert values reproduce (e.g.
`qwen_gateup_M8192` +12.57% vs +12.31% ± 0.27).

Reproduce with `python3 code/assemble_final_table_accel_sim.py` (writes
`results/final_table_accel_sim.csv` and `results/final_table_accel_sim_breakdown.csv`;
does not touch `final_table.csv`).

- **Packing**: real GPU, every shape: GEMM ms of the metadata-patched GEMM with TMA
  load + packing (`NVFP4_META_SW_CORRECT=3`) minus TMA load only, same SMEM (81,920 B,
  same Stage count), 5 interleaved reps per build, `--iters=50 --warmup=5`
  (`results/accel_sim/packing_raw.csv`, produced by `results/accel_sim/measure_packing.sh`
  on the RTX PRO 4000 GPUs). Added ms per call × calls per forward; MoE routed experts
  use the per-shape ratio on the real grouped-GEMM time. Large Linear GEMMs pay
  +2.0–3.7% of the GEMM; small MoE expert GEMMs (M=128–768) ≈ 0; QK^T is +1.8–3.1% at
  M=4096 but ≈ 0 at M=2048 and 8192 (consistent across the three models' identical
  QK^T shapes), which is why Qwen's packing term is non-monotonic in token count.
- **Operand**: simulated cost of the extra operand register(s), as a % of the GEMM:
  Linear +0.26% [-0.14, +0.65], QK^T +0.28% [+0.03, +0.53]. Each shape is 20
  12-register vs 13/14-register OMMA simulation pairs on the same trace of the real
  mode-3 GEMM, perturbed only by kernel launch latency (a single pair varies by
  ±1–2%); shapes are pooled by inverse-variance weighting, no size trend was found.
  Shapes: Linear = Llama q_proj M=2048, Llama k/v_proj M=4096, DeepSeek routed
  gate/up M=768; QK^T = Llama QK^T M=2048/4096/8192. Per-pair cycles are in
  `results/accel_sim/`. The simulated register file uses Ampere-class bank/port
  parameters (Blackwell's are not public).
- GEMM time per forward that these % apply to: Llama Linear = real full-model
  GEMM-only true-baseline time (`log/linear_stage_corrected_5rep.log`), Llama QK^T =
  true-baseline NVFP4 QK^T; Qwen/DeepSeek = `base_raw.txt` GEMM ms × calls per forward
  (routed experts from `grouped_raw.txt`). Qwen/DeepSeek carry the same
  not-Stage-corrected caveat as the table above.

16,384-token row is omitted: DeepSeek OOMs at this length on this 4x24GB-GPU setup
regardless of the per-GPU memory cap tried (its reference `Attention` implementation
isn't flash-attention, and it always computes full-vocabulary logits), so no clean
measurement exists there.

## Stage-count confound (found after the first results, Llama since corrected)

CUTLASS auto-picks how many pipeline Stages (K-tile buffers in SMEM) a GEMM uses from
however much SMEM is left after a caller-specified carveout (`StageCountAutoCarveout<N>`
in `core.inc`/`nvfp4_microbench.cu`). The metaA patch needs extra per-stage SMEM, so the
carveout was padded by `+25000` bytes (`NVFP4_SMEM_PAD`, default 25000) specifically to
free up room for it — tuned once, empirically, via `CUTLASS_DEBUG_TRACE_LEVEL=1`.

The problem: **the baseline (unpatched, no metaA) build includes this same padded
carveout**, even though it has no metaA to make room for. Measured via
`code/trace_test.py` (`CUTLASS_DEBUG_TRACE_LEVEL=1`, "Setting smem size to..." line):

| Build | SMEM | Stages |
|---|---|---|
| baseline, padded (`NVFP4_SMEM_PAD=25000`, unpatched CUTLASS) | 69,632 B | 3 |
| selection (`NVFP4_SMEM_PAD=25000`, patched CUTLASS) | 81,920 B | 3 |
| **"true" baseline** (`NVFP4_SMEM_PAD=0`, unpatched CUTLASS) | 88,064 B | **4** |

(81,920 − 69,632 = 12,288 = 3 × 4,096, exactly 3 stages' worth of metaA's own per-stage
footprint — confirming both padded builds land on Stage 3. 88,064 − 69,632 = 18,432,
matching one extra stage's worth of A+B+SFA+SFB — confirming the true baseline gets
Stage 4.)

So the original baseline wasn't measuring its own best case. The **direction of the
bias was not obvious a priori** — more pipeline stages isn't always faster (higher SMEM
use per CTA can lower occupancy enough to outweigh better latency hiding) — so this had
to be checked empirically, not assumed. Measured result: Stage-4 "true" baseline GEMMs
were actually **slightly slower** than the shared Stage-3 config for these problem
sizes, which **shrinks** the measured overhead relative to the original (biased)
baseline — the opposite of what "the baseline was handicapped" would naively suggest.

**Fix**: `NVFP4_SMEM_PAD` is now a build-time macro (default 25000, unchanged
selection/original behavior). Build the true baseline with `-DNVFP4_SMEM_PAD=0`
against an *unpatched* CUTLASS checkout (`git stash` the patch first).

**Llama was fully re-measured** with this fix, repeated 5x throughout:
- Linear: true-baseline vs. selection, full model, same process, 5 reps
  (`code/ab_truebase_vs_sel_repeat.py` → `log/linear_stage_corrected_5rep.log`).
- QK^T and PV: true-baseline vs. selection (PV has no selection — only Q/K do),
  standalone GEMM, 5 reps (`results/qkt_pv_repeat.txt`).
- Real fused BF16 attention kernel and standalone BF16 QK^T/PV GEMMs, 5 reps each, to
  split real attention into QK^T / PV / "other" (softmax+masking) and build an
  NVFP4-attention-baseline-equivalent denominator (see `code/assemble_llama_final.py`
  docstring for the exact construction).

**Qwen/DeepSeek were NOT re-measured.** Their grouped-GEMM and non-MoE-Linear `nv()`
baseline values (in `results/shapes_raw.txt`, `results/base_raw.txt`) all come from
`nvfp4_microbench.cu` built with the default padded carveout, so they likely have the
same confound. Given the Llama finding (true baseline ended up *slower*, shrinking
overhead), the Qwen/DeepSeek numbers' bias direction is **not assumed** either way —
re-measuring them the same way (`-DNVFP4_SMEM_PAD=0`, rebuild `nvfp4_microbench`, rerun
the shape sweep in `results/shapes_raw.txt`/`base_raw.txt`) is the natural next step if
those numbers need the same rigor.

## Methodology

**Baseline** = NVFP4 quantize (no selection) + plain NVFP4 GEMM (no metadata).
**Our method** = NVFP4 quantize + largest-error selection + NVFP4 GEMM with metaA
(and metaB for QK^T) TMA-loaded alongside A/B/scale-factors. Group size 4 for Linear
projections (activation only gets selection), group size 8 for QK^T (both Q and K get
selection, matching the `a[4]`/`b[2]` MMA fragment register counts).

Overhead % = added_ms / NVFP4-forward-equivalent denominator, per (model, token count).

- **Llama**: Linear added_ms is a **real, direct measurement** — `nvfp4_lib.cu`
  implements NVFP4 quantize+GEMM as a drop-in replacement for every `nn.Linear` in the
  model, built once against an unpatched CUTLASS with `-DNVFP4_SMEM_PAD=0` (the "true"
  baseline — see "Stage-count confound") and once against the metaA-patched CUTLASS
  (our method); both run the full model forward end-to-end, repeated 5x in the same
  process (`code/ab_truebase_vs_sel_repeat.py`), and the wall-clock difference is the
  real added cost. QK^T and PV are also real, repeated NVFP4 GEMM measurements (not
  full-model — no NVFP4 fused-attention kernel exists to run end-to-end); PV never gets
  selection metadata (only Q/K do), so it only affects the denominator, never the
  added-cost numerator. See `code/assemble_llama_final.py` for the exact construction
  and "Stage-count confound" above for why this supersedes the simpler description
  originally here (real e2e baseline vs. selection diff, QK^T assembled on top of a
  BF16-attention denominator) that `assemble_final_table.py` still implements.
- **Qwen/DeepSeek**: MoE routed-expert Linear layers are measured via a **real grouped
  GEMM** (CUTLASS's own `79d_blackwell_geforce_nvfp4_grouped_gemm` example, SM120,
  unmodified — see `code/run_grouped.sh`), which eliminates the per-expert kernel
  launch overhead that an individually-launched-expert baseline would have. Since
  CUTLASS's grouped-GEMM collective (`sm120_blockscaled_mma_array_tma.hpp`) was *not*
  patched with metaA support (a substantially bigger task than the single-GEMM patch —
  pointer-array/device-side TMA descriptor scheduling, not touched in this work), the
  metadata overhead for these shapes is applied as a **%-ratio extrapolation**: the
  per-shape %-overhead measured on the single-GEMM microbench (metadata-added ms /
  baseline ms, for the exact same (M,N,K) shape) is applied to the real grouped-GEMM
  baseline time for that shape. This is grounded in the single-GEMM finding that the
  GEMM is L2-bandwidth-bound and the overhead ratio is empirically ~M-independent for a
  given shape — but it is an extrapolation, not a direct measurement of a
  metadata-enabled grouped kernel. Non-MoE Linear (q/k/v/o-proj; DeepSeek's shared and
  dense-layer-0 experts) and QK^T are assembled the same way as Llama.

## Known limitations

- QK^T (and PV) overhead is **never measured end-to-end inside a real model forward**
  anywhere in this work — always a standalone-GEMM measurement (real, repeated for
  Llama; see "Stage-count confound"), scaled by head count x layer count. No NVFP4
  fused-attention kernel exists to run a real end-to-end attention measurement.
- The grouped-GEMM metadata overhead (Qwen/DeepSeek MoE) is an extrapolation (see
  above), not a direct measurement — the real grouped-GEMM collective was never
  patched with metaA support.
- Qwen/DeepSeek's baseline GEMM timings likely have the same Stage-count confound found
  and fixed for Llama (see "Stage-count confound") — not yet re-measured.
- Small per-expert microbench shapes (M≈128, the per-expert average at 2,048 tokens)
  sometimes have a metadata delta at the measurement noise floor (rounds to 0.00%);
  `code/assemble_final_table.py`'s `pct()` uses these as-is rather than smoothing them.
- Packing/layout experiments (nibble-packing, 16-bit word-packing, dense bit-packing,
  transposed TMA layout) were explored to see if the metadata byte footprint could be
  reduced further; none improved on the naive "1 byte/group, row-major, no packing"
  layout used throughout this table once both the quantize-kernel write side and the
  GEMM TMA-read side are accounted for together. That exploration's code/results were
  not carried into this directory (prototype-only, lived in a throwaway scratch dir).

## What's not here

- Most intermediate/exploratory run logs (docker pulls, pip installs, failed attempts,
  earlier superseded methodology versions) were not kept — this directory holds the
  final methodology and the logs/results that directly back the numbers above.
- The CUTLASS checkout itself (`sm120_blockscaled_mma_tma.hpp`, patched) is not copied
  here in full — `code/cutlass_sm120_blockscaled_mma_tma.patch` is the `git diff` of
  that file against upstream CUTLASS 4.2.1; apply it to a fresh checkout to rebuild.

## File manifest (`code/`)

| File | Purpose |
|---|---|
| `nvfp4_microbench.cu` | Standalone CUDA microbench: quantize(+selection) kernels, CUTLASS NVFP4 GEMM driver, all the packing-experiment macros (`NVFP4_NIBBLE_PACK`, `NVFP4_WORD_PACK8`, `NVFP4_DENSE_PACK8`, `NVFP4_META_TRANSPOSE` — see "Known limitations"). Build via `build.sh`. |
| `build.sh` | Builds `nvfp4_microbench` against a local (patchable) CUTLASS checkout. |
| `cutlass_sm120_blockscaled_mma_tma.patch` | `git diff` adding metaA/metaB TMA loading to CUTLASS's real SM120 blockscaled GEMM collective. |
| `core.inc` | First ~570 lines of `nvfp4_microbench.cu` (type/CUTLASS-GEMM setup), shared via `#include` with `nvfp4_lib.cu` so both stay in sync. Has the `NVFP4_SMEM_PAD` macro (default 25000) — see "Stage-count confound". |
| `nvfp4_lib.cu` | C API wrapping NVFP4 quantize(+selection)+GEMM as a drop-in `nn.Linear` replacement, for the e2e PyTorch harnesses (`#include "core.inc"`). Build with `-DNVFP4_META_ENABLE=0 -DNVFP4_SMEM_PAD=0` against *unpatched* CUTLASS for the true baseline, or `-DNVFP4_META_ENABLE=1` (default `NVFP4_SMEM_PAD=25000`) against the *patched* CUTLASS for selection. |
| `trace_test.py` | Minimal one-layer NVFP4 Linear smoke test; run with `CUTLASS_DEBUG_TRACE_LEVEL=1` to print the real SMEM size / Stage count CUTLASS picked (used to find and verify the Stage-count confound). |
| `e2e_llama.py` | Original (now superseded for Llama's final number, see "Stage-count confound") single-shot baseline-vs-selection e2e Llama timing. |
| `e2e_llama_breakdown.py` | Per-category (Linear/Attention/other) profiler breakdown of a real NVFP4 Llama forward — used to validate the "attention unchanged" assumption and to isolate the Linear-only GEMM time. |
| `ab_truebase_vs_sel_repeat.py` | **Authoritative Llama Linear measurement.** Loads true-baseline and selection libs in one process, swaps which is installed into the model, times both (full-forward event timer + profiler-isolated GEMM-only time), 5x repeated. Output → `log/linear_stage_corrected_5rep.log`. |
| `bf16_attn_repeat.py` | Real fused BF16 attention kernel time (unmodified Llama-3.1-8B-Instruct), profiler-isolated, 5x repeated. Output → `log/bf16_real_attention_5rep.log`. |
| `e2e_moe.py` | Same idea as `e2e_llama.py` for Qwen3-30B-A3B / DeepSeek-MoE-16B (per-expert individual GEMM calls — this is the "non-grouped" baseline referenced in the methodology section above, used to establish `old_moe_ms` and validate per-shape ratios, not the final MoE numbers). |
| `prefill_profile.py` / `prefill_profile_moe.py` | Real per-projection/per-attention BF16 forward-pass timing (CUDA-event hooks), source of `results/prefill_*_detail.csv`. |
| `nvfp4_mma_atom.cu` | Hand-written PTX kernel measuring the cost of packing Q+K metadata into one 32-bit register for a hypothetical future tensor-core instruction operand (not real hardware) — found negligible. |
| `run_grouped.sh` | Runs CUTLASS's unmodified `79d_blackwell_geforce_nvfp4_grouped_gemm` example at Qwen/DeepSeek's real MoE routed-expert shapes. |
| `assemble_final_table.py` | Qwen/DeepSeek numbers (Llama column is stale — see "Stage-count confound"). Reads `results/*.txt`, `results/*.csv`. |
| `assemble_llama_final.py` | **Authoritative Llama number.** Reads `results/qkt_pv_repeat.txt`, `results/bf16_qkt_pv_repeat.txt`, `log/linear_stage_corrected_5rep.log`, `log/bf16_real_attention_5rep.log`. |
| `assemble_final_table_accel_sim.py` | Final table **plus** metadata packing (real GPU) and the extra MMA operand (Accel-Sim) — see "Including metadata packing and the extra MMA operand". Reuses the two scripts above (Llama from `assemble_llama_final.py`, Qwen/DeepSeek from `assemble_final_table.py`) and reads `results/accel_sim/*.csv`. |

## Final results (2026-10-31): new group sizes, pre-packed correction data, grouped-GEMM MoE, format comparison

Group sizes: Linear 1-of-8, QK^T 1-of-16. All numbers are inference-time increase over the family's 4-bit
forward. Weights are always E2M1; 6-bit formats use E2M3 activations for Linear and QK^T, PV stays 4-bit.

### Format comparison (`results/format_comparison.csv`, token mean in `results/format_comparison_token_mean.csv`)

| Model | Tokens | Czardas (vs NVFP4) | NVFP6 | Czardas (vs MXFP4) | MXFP6 |
|---|---|---|---|---|---|
| Llama | 2048 | 3.43% | 34.76% | 3.39% | 32.07% |
| Llama | 4096 | 3.17% | 27.92% | 3.37% | 26.36% |
| Llama | 8192 | 2.22% | 22.00% | 1.92% | 20.31% |
| Qwen | 2048 | 2.72% | 6.44% | 1.22% | 5.52% |
| Qwen | 4096 | 2.59% | 10.51% | 2.93% | 10.03% |
| Qwen | 8192 | 1.90% | 8.10% | 2.34% | 7.01% |
| DeepSeek | 2048 | 2.85% | 14.30% | 3.10% | 14.69% |
| DeepSeek | 4096 | 3.67% | 23.07% | 4.64% | 21.66% |
| DeepSeek | 8192 | 2.83% | 16.15% | 3.39% | 14.46% |

### How each piece is obtained

- **Czardas (NVFP4)**, `code/assemble_final_table_gs8_16_pp_optden.py` (breakdown in
  `results/final_table_gs8_16_pp_optden_breakdown.csv`): current (selection quantize + correction-data store,
  + GEMM-side TMA load) + packing + operand (Accel-Sim).
  - Linear correction data in the **pre-packed** layout (`NVFP4_META_PREPACK`): stored by the quantizer in the
    13th-operand word format, the GEMM reads it with one 32-bit LDS per M atom. Dense shapes:
    `results/accel_sim/pp_raw.txt`.
  - QK^T: transposed layout, `results/accel_sim/gs8_16_raw.txt` (a row-pair layout did not reduce packing).
  - **MoE routed experts on the real grouped GEMM**: correction-data patch of CUTLASS
    `sm120_blockscaled_mma_array_tma.hpp` (per-group TMA descriptor, `experiments/blackwell_omma_port/patches/
    patch_grouped_meta.py`), real expert configurations, `results/accel_sim/grouped_pp_raw.txt`; quantize side
    at the tensors actually quantized (full token batch for gate/up, all experts' rows for down),
    `results/accel_sim/moe_quant_pp_raw.txt`.
  - **Qwen/DeepSeek denominator rebuilt**: the HF-eager MoE forward spends 61-70% of its GPU time outside
    GEMM/attention (per-expert Python loop). Replaced by components: dense NVFP4 Linear, grouped GEMM + quantize,
    attention = per-head NVFP4 QK^T + PV + softmax/other (Llama's construction; head_dim 128 for all models),
    and the remaining non-GEMM ops measured with a vectorized MoE dispatch
    (`results/accel_sim/measure_moe_other.py` -> `moe_other.txt`). Llama keeps its real full-model forward.
- **NVFP6 / MXFP6 / MXFP4 / Czardas (MXFP4)**, `code/assemble_format_comparison.py`, raw data
  `results/accel_sim/fc_dense.txt`, `fc_grouped.txt`, `fc_quant.txt` (3 interleaved reps):
  - NVFP6 GEMM: sm_120 has no E4M3-scaled FP6 MMA, so MXFP6 GEMM + one extra TMA load per stage of the
    additional scale bytes (16- instead of 32-element blocks, A and B). Emulation, not a native kernel.
  - Quantize: same kernel for E2M1 / E2M3 codes with NV (E4M3/16) or MX (UE8M0/32) scales, measured.
  - Czardas (MXFP4): MXFP4 GEMM + correction data, Stage-matched baseline; selection-quantize cost taken from
    the NVFP4 selection quantizer (same correction data).
  - MXFP4 denominator = NVFP4 denominator + (MXFP4 - NVFP4) time of every quantized GEMM and its quantize.

### Caveats

- Czardas (MXFP4) packing has only 3 reps and is noisy (Qwen 2048: -0.18%).
- 4096 tokens is the peak for several entries: at 8192 the QK^T GEMM time is dominated by writing the M x M
  output (format-insensitive), at 2048 the MoE per-expert M (128) hides format differences.
- Qwen is lowest because quantized GEMMs (Linear+MoE 10-25%) are the smallest share of its forward; PV, softmax
  and the unfused per-head QK-norm/RoPE dominate.
