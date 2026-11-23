# Source this before building/running. Paths relative to this experiment dir.
export EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ACCELSIM=$(cd "$EXP/../../accel-sim-framework" && pwd)
export CUDA_INSTALL_PATH=/usr/local/cuda-13.2
export PATH=$CUDA_INSTALL_PATH/bin:$EXP/tools/env/bin:$PATH
# user-space build deps (bison/flex/makedepend/zstd) from micromamba, no sudo on this server
export CPLUS_INCLUDE_PATH=$EXP/tools/env/include
export LIBRARY_PATH=$EXP/tools/env/lib
export LD_LIBRARY_PATH=$EXP/tools/env/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # GPUs 1-3 run other jobs on this server
