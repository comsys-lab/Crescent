#!/usr/bin/env bash
# New selection group sizes: Linear 1-of-8 (GS=8, metadata on A only), QK^T 1-of-16 (GS=16,
# Q and K, transposed M-contiguous metadata -- row-major K/16 = 8 B rows violate TMA's 16-B
# stride rule at K=128). Patched CUTLASS (cutlass_mode3), src/nvfp4_microbench_gs.cu,
# NGC 25.12 (nvcc 13.1). Mode 0 = TMA load only, mode 3 = TMA load + packing.
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docker run --rm --gpus all -v "$EXP/src":/src:ro -v "$EXP/cutlass_mode3":/cutlass:ro -v "$EXP/bin":/out -w /src \
  --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c '
  set -e
  b() { n=$1; shift
    nvcc -std=c++17 -gencode=arch=compute_120a,code=sm_120a -O3 --expt-relaxed-constexpr --expt-extended-lambda "$@" \
         -I/cutlass/include -I/cutlass/tools/util/include -I/cutlass/examples/common nvfp4_microbench_gs.cu -o /out/$n -lcublas &
  }
  b gs_lin8_swc0  -DNVFP4_META_GROUP_SIZE=8  -DNVFP4_META_ON_B=0 -DNVFP4_META_SW_CORRECT=0
  b gs_lin8_swc3  -DNVFP4_META_GROUP_SIZE=8  -DNVFP4_META_ON_B=0 -DNVFP4_META_SW_CORRECT=3
  b gs_qkt16_swc0 -DNVFP4_META_GROUP_SIZE=16 -DNVFP4_META_ON_B=1 -DNVFP4_META_TRANSPOSE=1 -DNVFP4_META_SW_CORRECT=0
  b gs_qkt16_swc3 -DNVFP4_META_GROUP_SIZE=16 -DNVFP4_META_ON_B=1 -DNVFP4_META_TRANSPOSE=1 -DNVFP4_META_SW_CORRECT=3
  wait
  chown '"$(id -u):$(id -g)"' /out/gs_*'
ls -la $EXP/bin/gs_*
