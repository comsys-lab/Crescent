# Figure 12: inference speedup

Speedup of Native FP4 and Crescent over All E2M3 (batch 1; average over 2,048 / 4,096 / 8,192 tokens),
for Llama-3.1-8B, Qwen3-30B-A3B and DeepSeek-MoE-16B under the NVFP4 and MXFP4 families.

## Pipeline

1. **Measure on an RTX 5090.** `runpod_5090/run_all_runpod.sh` runs every real-GPU measurement
   (CUTLASS GEMM microbenchmarks with and without Crescent's correction data, activation quantization,
   grouped MoE GEMMs, the remaining non-GEMM operators, and Llama full-model forward passes). See
   `runpod_5090/README.md`. The raw outputs of the run used in the paper are in `runpod_5090_results/`.
2. **Add the extra MMA operand cost (Accel-Sim).** No real MMA instruction reads the correction data,
   so its cost is simulated with Accel-Sim on a Blackwell OMMA port (`blackwell_omma_port/`,
   `README_accel_sim.md`). The simulated fractions are in `results/accel_sim/logs_*noise*.csv`.
3. **Assemble.**
   ```bash
   runpod_5090/assemble_5090.sh runpod_5090_results
   ```
   copies the measurements into `results/` and `log/`, then runs
   `code/assemble_format_comparison.py` (-> `results/format_comparison.csv`, per model and token count),
   writes the token mean (`results/format_comparison_token_mean.csv`), and runs `code/make_figure12.py`
   (-> `results/figure12_speedup.csv`, the plotted values).

CUTLASS with the correction-data patches: `blackwell_omma_port/setup_cutlass.sh`.

## Inputs of the assembly

| File | What it is |
|---|---|
| `results/accel_sim/gs8_16_raw.txt`, `pp_raw.txt`, `moe_quant_pp_raw.txt`, `grouped_pp_raw.txt` | GEMM and quantization timing, baseline vs Crescent (dense, MoE quantize, grouped) |
| `results/accel_sim/fc_dense.txt`, `fc_quant.txt`, `fc_grouped.txt` | NVFP4 / MXFP4 / 6-bit (All E2M3) GEMM and quantization timing |
| `results/accel_sim/moe_other.txt`, `results/prefill_moe_detail.csv` | Qwen / DeepSeek non-GEMM operators and per-projection forward time |
| `results/qkt_pv_repeat.txt`, `results/bf16_qkt_pv_repeat.txt`, `log/*_5rep.log` | Llama attention and full-model timing |
| `results/shapes_raw.txt`, `llama_raw.txt`, `base_raw.txt`, `grouped_raw.txt` | Earlier GEMM sweeps still read for Qwen / DeepSeek denominators |
| `results/accel_sim/packing_raw.csv`, `moe_expert_repeat.txt`, `logs_*noise*.csv` | Correction-data packing cost and Accel-Sim operand fractions |
