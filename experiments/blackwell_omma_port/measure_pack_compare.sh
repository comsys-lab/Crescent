#!/usr/bin/env bash
# Packing vs pre-packed packing on representative kernels, 5 interleaved reps (order rotated):
#   GPU0 dense Linear  llama_gateup_M8192 : gs_lin8_swc0  gs_lin8_swc3  gs_lin8pp_swc0  gs_lin8pp_swc3
#   GPU1 dense QK^T    llama_qkt_M2048    : gs_qkt16_swc0 gs_qkt16_swc3 gs_qkt16p_swc0  gs_qkt16p_swc3
#   GPU2 grouped       qwen_gateup_M8192  : grp_swc0_p0   grp_swc3_p0   grp_pp_swc0_p0  grp_pp_swc3_p0
# -> logs_pack_compare.txt
set -uo pipefail
EXP=$(cd "$(dirname "$0")" && pwd); cd "$EXP"; OUT=logs_pack_compare.txt
dense() { local g=$1 label=$2 m=$3 n=$4 k=$5; shift 5; local B=("$@")
  for rep in 1 2 3 4 5; do for j in 0 1 2 3; do b=${B[$(( (j + rep) % 4 ))]}
    { echo "### rep$rep $label $b"; CUDA_VISIBLE_DEVICES=$g ./bin/$b --m=$m --n=$n --k=$k --no-check 2>&1 | grep -E "nvfp4 GEMM +:"; } >> $OUT.$g
  done; done; }
: > $OUT.0; : > $OUT.1; : > $OUT.2
dense 0 llama_gateup_M8192 8192 14336 4096 gs_lin8_swc0 gs_lin8_swc3 gs_lin8pp_swc0 gs_lin8pp_swc3 &
dense 1 llama_qkt_M2048 2048 2048 128 gs_qkt16_swc0 gs_qkt16_swc3 gs_qkt16p_swc0 gs_qkt16p_swc3 &
docker run --rm --gpus '"device=2"' -v "$EXP/bin":/w -w /w --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c '
  B=(grp_swc0_p0 grp_swc3_p0 grp_pp_swc0_p0 grp_pp_swc3_p0)
  for rep in 1 2 3 4 5; do for j in 0 1 2 3; do b=${B[$(( (j + rep) % 4 ))]}; echo "### rep$rep qwen_gateup_M8192 $b"
    ./$b --m=512 --n=768 --k=2048 --groups=128 --iterations=50 --no_verif 2>&1 | grep -E "Avg runtime|Kernel-only"; done; done' >> $OUT.2 2>&1 &
wait; cat $OUT.0 $OUT.1 $OUT.2 > $OUT; echo DONE >> $OUT
