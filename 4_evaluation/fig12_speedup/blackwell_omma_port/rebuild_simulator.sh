#!/bin/bash
# Incremental rebuild of accel-sim.out after source edits (see build_simulator.sh).
set -e
source "$(dirname "$0")/env.sh"
export CUDA_INSTALL_PATH=$HOME/cuda-12.8.1
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
cd "$ACCELSIM"
source ./gpu-simulator/setup_environment.sh >/dev/null
cmake --build ./gpu-simulator/build -j"$(nproc)"
cmake --install ./gpu-simulator/build >/dev/null
