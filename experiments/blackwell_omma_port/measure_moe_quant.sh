#!/usr/bin/env bash
# MoE quantize-side correction-data cost (selection + correction-data DRAM store, minus the
# correction-data reload that the grouped GEMM's TMA load already covers), Linear 1-of-8,
# bin/gs_lin8_swc0, at the activation tensors the MoE actually quantizes:
#   gate/up input : the full token batch (tokens x 2048), quantized before expert dispatch
#   down input    : all experts' intermediate rows stacked = sum of per-expert M of the grouped GEMM
#                   (Qwen 128 x tokens/16, DeepSeek 64 x per-expert M) x K_down
# Waits for measure_grouped.sh (GPU otherwise busy). 5 reps. Output logs_moe_quant_raw.txt.
set -uo pipefail
cd "$(dirname "$0")"
while ! grep -q DONE logs_grouped_raw.txt 2>/dev/null; do sleep 30; done
OUT=logs_moe_quant_raw.txt; : > $OUT
run_gpu() { local g=$1 i=0
  while read -r LABEL M N K; do i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    for rep in 1 2 3 4 5; do
      { echo "### rep$rep $LABEL sel n=$N k=$K m=$M"
        CUDA_VISIBLE_DEVICES=$g ./bin/gs_lin8_swc0 --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$g
    done
  done < moe_quant_shapes.txt
}
for g in 0 1 2 3; do run_gpu $g & done; wait
cat $OUT.[0-3] > $OUT; echo DONE >> $OUT
