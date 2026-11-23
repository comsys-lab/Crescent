#!/usr/bin/env bash
# Compile the NVFP4 micro-benchmark inside the NGC container (nvcc 13.1).
# Uses the LOCAL CUTLASS checkout (not the container's bundled copy) because the
# metaA/metaB-in-the-real-GEMM experiment patches sm120_blockscaled_mma_tma.hpp
# directly -- needs a copy we can actually edit.
set -euo pipefail
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUTLASS=${CUTLASS:-$HERE/../../cutlass}

docker run --rm --gpus all -v "$HERE":/work -v "$CUTLASS":/cutlass -w /work --entrypoint bash "$IMAGE" -c '
  set -euo pipefail
  CUT=/cutlass
  nvcc -std=c++17 -gencode=arch=compute_120a,code=sm_120a -O3 --expt-relaxed-constexpr --expt-extended-lambda \
       -I"$CUT/include" -I"$CUT/tools/util/include" -I"$CUT/examples/common" \
       nvfp4_microbench.cu -o nvfp4_microbench -lcublas
  ls -la nvfp4_microbench
'
