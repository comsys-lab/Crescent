#!/usr/bin/env bash
# MXFP4 / MXFP6 / NVFP6-emulation / Czardas(MXFP4) dense GEMM benches (src/nvfp6_gemm_bench.cu).
# Usage: build_fmtcmp.sh PAD  -> bin/fc_<name>_p<PAD>
#   base builds against cutlass_clean; CZ builds (correction data / extra-scale TMA) against cutlass_mode3.
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); PAD=$1
docker run --rm -v "$EXP/src":/src:ro -v "$EXP/cutlass_clean":/cc:ro -v "$EXP/cutlass_mode3":/cm:ro -v "$EXP/bin":/out -w /src \
  --entrypoint bash nvcr.io/nvidia/pytorch:25.12-py3 -c '
F="-std=c++17 -gencode=arch=compute_120a,code=sm_120a -O3 --expt-relaxed-constexpr --expt-extended-lambda -DNVFP4_SMEM_PAD='$PAD'"
c() { r=$1; n=$2; shift 2; nvcc $F -I/$r/include -I/$r/tools/util/include -I/$r/examples/common "$@" nvfp6_gemm_bench.cu -o /out/fc_${n}_p'$PAD' 2>&1 | grep " error" | head -2; }
LIN="-DCZ=1 -DNVFP4_META_GROUP_SIZE=8 -DNVFP4_META_ON_B=0 -DNVFP4_META_PREPACK=1"
QKT="-DCZ=1 -DNVFP4_META_GROUP_SIZE=16 -DNVFP4_META_ON_B=1 -DNVFP4_META_TRANSPOSE=1"
EMU="-DCZ=1 -DNVFP4_META_GROUP_SIZE=32 -DNVFP4_META_ON_B=1 -DNVFP4_META_TRANSPOSE=1 -DNVFP4_META_SW_CORRECT=0"
c cc nv4    -DFMT=0 &
c cc mx4    -DFMT=1 &
c cc mx6lin -DFMT=2 &
c cc mx6qkt -DFMT=3 &
c cm nv6lin -DFMT=2 $EMU &
c cm nv6qkt -DFMT=3 $EMU &
c cm czmxlin0 -DFMT=1 $LIN -DNVFP4_META_SW_CORRECT=0 &
c cm czmxlin3 -DFMT=1 $LIN -DNVFP4_META_SW_CORRECT=3 &
c cm czmxqkt0 -DFMT=1 $QKT -DNVFP4_META_SW_CORRECT=0 &
c cm czmxqkt3 -DFMT=1 $QKT -DNVFP4_META_SW_CORRECT=3 &
wait; chown '"$(id -u):$(id -g)"' /out/fc_*'
