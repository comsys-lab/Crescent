#!/bin/bash
# Run Accel-Sim on one trace; config CFG (default SM120_RTX_PRO_4000, e.g. CFG=SM100_B200_CLASS).
# Usage: run_sim.sh <trace_dir (contains traces/kernelslist.g)> <run_name> [extra accel-sim options...]
set -e
source "$(dirname "$0")/env.sh"
T=$(realpath "$1"); NAME=$2; shift 2
D=$EXP/sim_runs/$NAME; mkdir -p "$D"; cd "$D"
cp $ACCELSIM/gpu-simulator/gpgpu-sim/configs/tested-cfgs/${CFG:-SM120_RTX_PRO_4000}/*.config $ACCELSIM/gpu-simulator/gpgpu-sim/configs/tested-cfgs/${CFG:-SM120_RTX_PRO_4000}/*.icnt . 2>/dev/null || true
cp $ACCELSIM/gpu-simulator/configs/tested-cfgs/${CFG:-SM120_RTX_PRO_4000}/trace.config .
echo "$ACCELSIM/gpu-simulator/bin/release/accel-sim.out -trace $T/traces/kernelslist.g -config ./gpgpusim.config -config ./trace.config $*" > cmd.sh
$ACCELSIM/gpu-simulator/bin/release/accel-sim.out -trace $T/traces/kernelslist.g \
  -config ./gpgpusim.config -config ./trace.config "$@" > sim.log 2>&1
grep -E "^gpu_tot_sim_cycle|^gpu_tot_sim_insn|^gpu_tot_ipc" sim.log | tail -3
