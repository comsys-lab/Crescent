#!/usr/bin/env bash
# Pre-packed correction-data layouts (packing-cost reduction), real GPU:
#   Linear GS=8 : NVFP4_META_PREPACK (stored in the 13th-operand word format, 1 LDS.32 per M atom)
#   QK^T GS=16  : NVFP4_META_PAIR16  (row-pair layout, 1 LDS.32 per atom)
# Phase 1 dense, all non-MoE-routed shapes of shapes57.txt, 3 interleaved reps:
#   baseline (bin/nvfp4_microbench_baseline_padded), sel (pp/p swc0), pack (pp/p swc3)
#   -> logs_pp_raw.txt  (same block format as logs_gs_raw.txt)
# Phase 2 MoE full-batch quantize side with the PREPACK store, bin/gs_lin8pp_swc0, 5 reps
#   -> logs_moe_quant_pp_raw.txt
# Phase 3 grouped GEMM, 5 reps, binaries grp_base / grp_pp_swc0 / grp_pp_swc3 (old swc3: logs_grouped_raw.txt)
#   -> logs_grouped_pp_raw.txt
# Waits for measure_grouped.sh and measure_moe_quant.sh to finish.
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); cd "$EXP"
while ! grep -q DONE logs_moe_quant_raw.txt 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') phase 1"
OUT=logs_pp_raw.txt; : > $OUT
p1() { local g=$1 i=0
  grep -vE '^(qwen_gateup|qwen_down|ds_routed_gu|ds_routed_down)_' shapes57.txt | while read -r LABEL CFG M N K; do
    i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    sel=gs_lin8pp; [ $CFG = qkt ] && sel=gs_qkt16p
    for rep in 1 2 3; do for b in baseline sel pack; do
      case $b in baseline) bin=nvfp4_microbench_baseline_padded ;; sel) bin=${sel}_swc0 ;; pack) bin=${sel}_swc3 ;; esac
      { echo "### rep$rep $LABEL $b n=$N k=$K m=$M"
        CUDA_VISIBLE_DEVICES=$g ./bin/$bin --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$g
    done; done
  done
}
for g in 0 1 2 3; do p1 $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT

echo "$(date '+%F %T') phase 2"
OUT=logs_moe_quant_pp_raw.txt; : > $OUT
p2() { local g=$1 i=0
  while read -r LABEL M N K; do i=$((i + 1)); [ $(( (i - 1) % 4 )) -eq $g ] || continue
    for rep in 1 2 3 4 5; do
      { echo "### rep$rep $LABEL selpp n=$N k=$K m=$M"
        CUDA_VISIBLE_DEVICES=$g ./bin/gs_lin8pp_swc0 --m=$M --n=$N --k=$K --no-check 2>&1 | grep -E "nvfp4 GEMM +:|DRAM store\+reload|with store \(DRAM\)"; } >> $OUT.$g
    done
  done < moe_quant_shapes.txt
}
for g in 0 1 2 3; do p2 $g & done; wait; cat $OUT.[0-3] > $OUT; echo DONE >> $OUT

echo "$(date '+%F %T') phase 3"
SHAPES=(
 "qwen_gateup_M2048 128 768 2048 128"  "qwen_down_M2048 128 2048 768 128"
 "qwen_gateup_M4096 256 768 2048 128"  "qwen_down_M4096 256 2048 768 128"
 "qwen_gateup_M8192 512 768 2048 128"  "qwen_down_M8192 512 2048 768 128"
 "ds_routed_gu_M2048 128 1408 2048 64" "ds_routed_down_M2048 128 2048 1408 64"
 "ds_routed_gu_M4096 384 1408 2048 64" "ds_routed_down_M4096 384 2048 1408 64"
 "ds_routed_gu_M8192 768 1408 2048 64" "ds_routed_down_M8192 768 2048 1408 64"
)
p3() { g=$1; out=logs_grouped_pp_raw.txt.$g; : > $out
  docker run --rm --gpus "\"device=$g\"" -v "$EXP/bin":/w -w /w --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c "
    B=(grp_base_p0 grp_pp_swc0_p0 grp_pp_swc3_p0)
    for rep in \$(seq 1 5); do
      $(for i in "${!SHAPES[@]}"; do [ $((i % 4)) = $g ] || continue; set -- ${SHAPES[$i]}
        echo "for j in 0 1 2; do b=\${B[\$(( (j + rep) % 3 ))]}; echo \"### rep\$rep $1 \$b\";
              ./\$b --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E 'Stages|Avg runtime|Kernel-only'; done;"
      done)
    done" >> $out 2>&1
}
for g in 0 1 2 3; do p3 $g & done; wait
cat logs_grouped_pp_raw.txt.[0-3] > logs_grouped_pp_raw.txt; echo DONE >> logs_grouped_pp_raw.txt
echo "$(date '+%F %T') done"
