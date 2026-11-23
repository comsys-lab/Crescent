# Source for Accel-Sim 2.0 — closing the last gap in the inference-time number

## The actual goal

Everything in this folder (`code/`, `results/`) measures **real inference-time overhead** (5.98%,
4.96%, 3.74% for Llama at 2048/4096/8192 tokens, etc.) for a version of the technique
where the GEMM TMA-loads the largest-error-selection metadata alongside A/B/scale-
factors, but **no real Tensor Core instruction actually consumes it** — no such
instruction exists on real hardware, so that part of the real measurement is a stand-in
("pay the cost of moving the metadata into place; pretend the MMA can use it").

The one piece still missing from the inference-time number: if a real instruction
*could* consume the metadata (reading one extra operand register, applying the E2M3
correction inside the MMA), would the GEMM's compute time change — and by how much?
That can only come from a simulator, since no real instruction with that operand exists
to measure. **The end goal of this accel-sim work is a new GEMM-timing number** (with
vs. without the hypothetical instruction actually consuming the metadata) that gets
fed back into this folder (`code/`, `results/`)'s existing calculation to produce an updated
inference-time-overhead percentage — not a standalone cycle count.

## Two files, two different jobs

- **`nvfp4_microbench.cu` + `cutlass_sm120_blockscaled_mma_tma.patch`** — the
  **simulation target**. This is the same real, full-scale, properly-pipelined CUTLASS
  SM120 GEMM used for every real-hardware number in this folder (`code/`, `results/`) (realistic
  tile sizes, Stages, warp specialization, real memory traffic). Trace *this* kernel,
  not the toy one below — a toy single-tile kernel won't give a realistic inference-time
  number. Build per `README.md` ("Build" section there applies
  unchanged); `build.sh` here is the same Docker-wrapped reference as before.
- **`nvfp4_mma_atom.cu`** — **not** what you trace/simulate. The real MMA opcode
  (`mma.sync.aligned.kind::mxf4nvf4.block_scale...`) is self-identifying in a trace —
  you don't need this file to *find* it. What this file actually provides is the
  **spec for what the hypothetical 13th operand register holds**, which doesn't exist
  anywhere else (CUTLASS's real trace obviously has no 13th operand to look at — you
  have to derive its intended bit layout from the quantize kernel's metadata format,
  and that derivation is what's worked out here):
  - `pack_qk_meta` — QK^T config (group=8, both Q and K get selection): 6 values (Q's
    4, matching `a[0..3]`; K's 2, matching `b[0..1]`), each 5 bits (3-bit index +
    2-bit E2M3 correction), packed into bits [0:30) of one `uint32_t`.
  - `pack_linear_meta` — Linear config (group=4, activation only — the weight operand
    never gets selection): 4 values (matching `a[0..3]`), each 4 bits (2-bit index +
    2-bit correction), packed into bits [0:16) of one `uint32_t`. Reads from a
    nibble-packed source (2 values/byte — group=4's real on-wire format, see
    this folder (`code/`, `results/`)'s `nvfp4_microbench.cu` `NVFP4_NIBBLE_PACK`), unlike
    `pack_qk_meta`'s byte-indexed (1 value/byte) source.

  Use whichever matches the GEMM shape you're simulating (QK^T shapes get a 13th
  operand per `pack_qk_meta`'s layout; Linear shapes get `pack_linear_meta`'s).

## What's already in the traced kernel, and what isn't

Updated after this session's work: **packing is now real too, not just TMA-load.**
Earlier versions of this README said packing had to be hand-inserted into the trace —
that's no longer true, and the remaining gap for the simulator is smaller than it was.

- **TMA-loading metaA/metaB from global memory into SMEM — already in the kernel.**
  The patched CUTLASS collective (`cutlass_sm120_blockscaled_mma_tma.patch`) really
  does this, compiled into the real GEMM binary, same as every other real-hardware
  number in this folder (`code/`, `results/`). Trace the binary and these load instructions show up
  on their own — nothing to add.
- **Packing SMEM metadata into a register, matching `pack_qk_meta`/`pack_linear_meta`'s
  exact bit layouts — now ALSO already in the kernel, real and measured.** Build with
  `-DNVFP4_META_SW_CORRECT=3` (the patch now includes this mode — see `pack_kblock` in
  `cutlass_sm120_blockscaled_mma_tma.patch`): the real GEMM reads metaA/metaB back out
  of SMEM per k_block and builds the exact packed register `pack_qk_meta`/
  `pack_linear_meta` compute in isolation, inside the real, full-pipeline mainloop. It's
  not fed to any MMA (no real instruction has a 13th operand) — it's XOR-accumulated
  into a per-thread register and flushed to a global sink once per CTA-tile, purely so
  the compiler can't prove the computation dead and elide it. **Measured real cost**
  (3 reps, GEMM-only time, correctness unaffected — same Frobenius-vs-BF16 PASS in
  every build at a given shape):

  | shape | build | GEMM time | vs no-metadata baseline |
  |---|---|---|---|
  | Linear M=2048,K=4096,N=4096 | no metadata (pristine CUTLASS) | 0.1708 ms | — |
  | Linear M=2048,K=4096,N=4096 | `SW_CORRECT` unset/0 (TMA-load only) | 0.1821 ms | +6.6% |
  | Linear M=2048,K=4096,N=4096 | `SW_CORRECT=3` (TMA-load + packing) | 0.1870 ms | **+9.5%** |
  | QK^T M=N=4096,K=128 | no metadata (pristine CUTLASS) | 0.0430 ms | — |
  | QK^T M=N=4096,K=128 | `SW_CORRECT` unset/0 (TMA-load only) | 0.0442 ms | +2.8% |
  | QK^T M=N=4096,K=128 | `SW_CORRECT=3` (TMA-load + packing) | 0.0448 ms | **+4.2%** |

  Getting a clean, trustworthy number took three false starts, worth knowing before
  trusting any *other* SW_CORRECT-mode timing:
  1. The first `atomicXor`-into-one-shared-global-slot implementation measured +51%
     (Linear) — all GPU-wide CTAs serializing on one address every k_block.
  2. Giving each CTA its own slot (still a per-k_block store, no atomic) brought that
     down to +14.7% — still dominated by *how often* the sink was touched, not the
     packing compute itself.
  3. Moving to a per-thread register accumulator flushed **once per CTA-tile** (not
     per k_block) dropped it to +1.3% (Linear, on top of TMA-load) — but this was
     *also* wrong, just in the other direction: `pack_kblock` only built ONE packed
     register per k_block, using the row(s) of a SINGLE M atom. Real SASS (confirmed
     by tracing the actual compiled GEMM) shows SFA already uses a *different*
     register per M atom within a warp (A data differs by M atom) — metadata must
     too, so a warp genuinely needs `MMA_M` distinct packed values per k_block, not
     one; QK^T additionally needs `MMA_N` distinct B-metadata values (kept as a
     *separate* 14th operand, not merged into A's 13th — merging would force
     `MMA_M*MMA_N` combinations per k_block instead of `MMA_M+MMA_N`, mirroring why
     real hardware already keeps SFA/SFB separate). Fixing `pack_kblock` to loop over
     all M atoms (and, for QK^T, all N atoms for B) roughly doubled Linear's
     packing-only delta (+1.3% → +2.9% on top of TMA-load) and gave the QK^T number
     above its own first-ever correct measurement.

  `-Xptxas -v` register/smem usage was identical with vs. without packing throughout
  all three stages (168 registers, 0 spills, same shapes) — occupancy was never the
  cause of any of the three measurements above; each was a real behavioral bug in how
  much packing work the code was doing per k_block, not a compiler/occupancy artifact.
  **So: don't trust a first measurement of a "just XOR it into a sink" dead-code guard
  without checking (a) it isn't dominated by how often/how the sink is written, and
  (b) it's actually doing the SAME amount of per-k_block work the real instruction's
  data flow would require — not just one representative instance of it.**
  Operand count for the hypothetical new instruction: Linear MMA stays at **13**
  (12 + 1 A-metadata operand); QK^T MMA grows to **14** (12 + 1 A-metadata + 1
  B-metadata operand, kept separate as explained above).
- **The one thing still NOT in the kernel, and still needs the simulator: the
  hypothetical MMA instruction actually reading the packed register(s) (13th operand
  for Linear; 13th + 14th for QK^T) and using them inside the tensor-core
  accumulation.** No real instruction has those operands, so this can only come from
  a simulator's cost model, not a real build.

## What to actually do

1. Build `nvfp4_microbench.cu` against the patched CUTLASS with
   `-DNVFP4_META_SW_CORRECT=3` (TMA-load + real packing both compiled in — see "What's
   already in the traced kernel, and what isn't" above). Build Linear shapes with
   default `NVFP4_META_GROUP_SIZE=4 NVFP4_META_ON_B=0`; build QK^T shapes with
   `NVFP4_META_GROUP_SIZE=8 NVFP4_META_ON_B=1`. This is the build to trace, *not* the
   plain selection build from `README.md` — packing needs to already
   be in the binary so the trace doesn't need any hand-inserted instructions for it,
   only the MMA operand widening in step 3.
2. Trace it (e.g. via NVBit) running a realistic shape — reuse the exact (M,N,K) shapes
   already measured in `results/` (4 distinct Linear shapes for
   Llama: q/o_proj K=N=4096, k/v_proj K=4096,N=1024, gate/up_proj K=4096,N=14336,
   down_proj K=14336,N=4096; plus QK^T: K=128, M=N=seq_len, one run per token count —
   2048/4096/8192) so each simulated delta is directly comparable to, and can replace,
   the real-hardware delta already computed for that exact shape. Run each shape
   separately; there's no single run that covers all of them.
3. In the trace, find the MMA instruction(s) — the opcode name alone identifies them,
   no decoder needed — and widen the operand count: **Linear shapes 12 → 13** (feed the
   packed A-metadata register `pack_kblock` built, one value per M atom); **QK^T shapes
   12 → 14** (feed packed A-metadata AND packed B-metadata as two separate operands,
   per M atom and per N atom respectively) — instead of discarding them into the sink.
   This is now the *only* thing missing from the real trace — packing itself no longer
   needs hand-inserted instructions, per step 1.
4. **Use a relative ratio, not an absolute cycle→time conversion.** Accel-Sim reports
   cycles, not wall-clock time, and only the kernel with the hypothetical instruction
   needs simulating at all (everything else already has a real ms number in
   this folder (`code/`, `results/`) — don't re-simulate the quantize kernel, attention, or other
   GEMM shapes, there's no reason to).
   - Run the simulation **twice** on the same shape, same simulated config: once with
     the stock, real 12-operand instruction, once with the widened variant (13 operands
     for Linear, 14 for QK^T).
   - Compute `pct = (cycles_widened - cycles_12op) / cycles_12op`. Don't convert either
     cycle count to absolute time — a relative ratio from two runs of the *same*
     simulated config is far more trustworthy than trusting the simulator's absolute
     cycle count against real hardware (systematic simulator bias — clock-frequency
     assumptions, microarchitectural modeling gaps — mostly cancels out in a ratio,
     the same way it doesn't in an absolute number).
   - Apply `pct` to the real, already-measured GEMM time for that exact shape
     (`results/`): `simulated_ms = real_ms * (1 + pct)`. This is the
     same pattern used for every other overhead number in this folder (`code/`, `results/`) — a
     measured ratio applied to a real absolute baseline, never an absolute number
     pulled from a different measurement source.
5. **Bring the resulting GEMM-timing delta back to this folder (`code/`, `results/`).** It plugs in
   exactly where the real `nv()`/`delta()` GEMM-only numbers currently come from
   (`results/shapes_raw.txt`, `llama_raw.txt`,
   `qkt_pv_repeat.txt` — see `assemble_final_table.py` / `assemble_llama_final.py` for
   where those values are consumed). Replacing or adding to those per-shape GEMM deltas
   with the simulated (metadata-consuming) numbers, then re-running those two scripts,
   is what turns this into an updated inference-time-overhead percentage.
