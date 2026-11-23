#!/usr/bin/env bash
# Build ../4_evaluation/accel_sim/nvfp4_microbench.cu with the mode-3 (TMA-load + real packing)
# patched CUTLASS, in the same NGC container as accel_sim/build.sh (nvcc 13.1).
# Usage: build_mode3.sh linear|qkt [tie]
#   tie: -DNVFP4_META_TIE_OPERAND=1 (packing scheduled before the MMA that would read it)
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$EXP/../../4_evaluation/accel_sim" && pwd)
CFG=${1:-linear}
TIE=${2:-}
SWC=${SWC:-3}   # SWC=0: metadata TMA load only (no packing)
case $CFG in
  linear) DEFS="-DNVFP4_META_SW_CORRECT=$SWC -DNVFP4_META_GROUP_SIZE=4 -DNVFP4_META_ON_B=0" ;;
  qkt)    DEFS="-DNVFP4_META_SW_CORRECT=$SWC -DNVFP4_META_GROUP_SIZE=8 -DNVFP4_META_ON_B=1" ;;
esac
NAME=$CFG; [ "$SWC" = 3 ] || NAME=${CFG}_swc$SWC
if [ "$TIE" = tie ]; then DEFS="$DEFS -DNVFP4_META_TIE_OPERAND=1"; NAME=${CFG}_tie; fi
docker run --rm --gpus all -v "$SRC":/src:ro -v "$EXP/cutlass_mode3":/cutlass:ro -v "$EXP/bin":/out \
  -w /src --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c "
  set -euo pipefail; nvcc --version | tail -1
  nvcc -std=c++17 -gencode=arch=compute_120a,code=sm_120a -O3 -lineinfo --expt-relaxed-constexpr --expt-extended-lambda \
       $DEFS -I/cutlass/include -I/cutlass/tools/util/include -I/cutlass/examples/common \
       nvfp4_microbench.cu -o /out/nvfp4_microbench_mode3_$NAME -lcublas
  chown $(id -u):$(id -g) /out/nvfp4_microbench_mode3_$NAME"
cuobjdump -sass "$EXP/bin/nvfp4_microbench_mode3_$NAME" > "$EXP/sass/nvfp4_microbench_mode3_$NAME.sass"
