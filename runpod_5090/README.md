# RTX 5090 re-measurement (RunPod)

## Pod
- Image: `nvcr.io/nvidia/pytorch:25.12-py3` (the container all binaries were built and measured in).
- RTX 5090 (2 GPUs: shapes are split over both, shape i -> GPU i mod 2, as the original split over 4), host driver with CUDA 13.x support, volume >= 50 GB.
- No other process on the GPU (the script stops if `nvidia-smi` shows one).

## Run
```bash
tar xzf runpod_5090.tar.gz && cd runpod_5090
export HF_TOKEN=hf_...            # Llama-3.1-8B is gated
nohup setsid ./run_all_runpod.sh > run.log 2>&1 &
tail -f run.log
```
Phases: 0 sanity (every binary once on a small shape) -> 1 gs8_16_raw -> 2 pp / MoE quantize / grouped ->
3 format comparison (dense, quantize, grouped) -> 4 Llama QK^T/PV split -> 5 Qwen/DeepSeek non-GEMM ops ->
6 Llama full-model forward + BF16 attention. Llama weights download in the background during phases 1-5.
The script is resumable: rerun it after an interruption and finished phases are skipped.

## Bring back
Copy the whole `results/` directory, then locally:
```bash
runpod_5090/assemble_5090.sh <copied results dir>     # -> 4_evaluation/execution_time_5090/
```

## Differences from the RTX PRO 4000 runs
- Split over the pod's GPUs (was 4 GPUs); same binaries, command lines, rep counts, output formats.
- Llama QK^T/PV NVFP4 true baseline: `fc_nv4_p0` (pristine CUTLASS NVFP4 GEMM, natural Stage count) replaces
  the deleted `nvfp4_microbench_truebase` binary.
- Llama BF16 attention: measured on `meta-llama/Llama-3.1-8B` (was -Instruct; same architecture and kernels),
  so only one model is downloaded. Override with `LLAMA_MODEL=...`.
- Not re-measured: Accel-Sim operand fraction (reused, sm_120 SM model); old HF-eager Qwen/DeepSeek forward
  files (they cancel in the denominator rescaling).
