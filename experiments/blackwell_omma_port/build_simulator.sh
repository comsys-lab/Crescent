#!/bin/bash
# Build GPGPU-Sim + Accel-Sim trace-driven simulator with the documented CMake flow.
# The simulator is built against CUDA 12.8 headers (Accel-Sim README reference platform);
# GPGPU-Sim's libcuda (PTX mode, unused in trace mode) does not compile with CUDA 13 headers
# (cudaDeviceProp::clockRate removed). Kernels/tracer still use CUDA 13.2.
set -e
source "$(dirname "$0")/env.sh"
export CUDA_INSTALL_PATH=$HOME/cuda-12.8.1
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
cd "$ACCELSIM"
source ./gpu-simulator/setup_environment.sh
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build -DCMAKE_PREFIX_PATH="$EXP/tools/env"
cmake --build ./gpu-simulator/build -j"$(nproc)"
cmake --install ./gpu-simulator/build
ls -l gpu-simulator/bin/release/accel-sim.out
