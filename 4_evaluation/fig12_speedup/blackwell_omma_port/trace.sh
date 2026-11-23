#!/bin/bash
# Trace a CUDA command with the Accel-Sim NVBit tracer.
# Usage: trace.sh <out_dir> <cmd> [args...]
# (Not using others/torch_hook/run.sh: it force-exports NVBIT_INSTRUMENTATION_ENABLED=0.)
set -e
source "$(dirname "$0")/env.sh"
OUT=$(realpath -m "$1"); shift
rm -rf "$OUT"; mkdir -p "$OUT"
export CUDA_INJECTION64_PATH=$ACCELSIM/util/tracer_nvbit/tracer_tool/tracer_tool.so
export USER_DEFINED_FOLDERS=1 TRACES_FOLDER=$OUT
export NVBIT_INSTRUMENTATION_ENABLED=1 TOOL_VERBOSE=${TOOL_VERBOSE:-1}
"$@" 2>&1 | tee "$OUT/tracer_stdout.txt"
unset CUDA_INJECTION64_PATH
$ACCELSIM/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing "$OUT/traces" 2>&1 | tee "$OUT/postproc_stdout.txt"
