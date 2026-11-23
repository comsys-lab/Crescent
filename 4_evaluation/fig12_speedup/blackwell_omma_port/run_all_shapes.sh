#!/usr/bin/env bash
# Full (unsampled) 12op vs widened simulation of every GEMM shape that feeds
# ../code, ../results (Llama-3.1-8B, Qwen3-30B-A3B, DeepSeek-MoE-16B;
# PV has no metadata). Shapes from ../results/{shapes_raw,llama_raw}.txt
# (where a label appears twice, the later entry is the one assemble_final_table.py uses).
#
# GPU part (trace, convert, post-process) runs one shape at a time, largest first;
# each shape's two simulations are then started in the background, at most
# MAX_SIMS accel-sim processes at once. Raw/intermediate .trace.xz files are
# deleted once the .tracez exist. Launch detached:
#   nohup setsid ./run_all_shapes.sh > logs_all_shapes.txt 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")"; source ./env.sh
MAX_SIMS=${MAX_SIMS:-24}
PP=$ACCELSIM/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
KID=9   # first GEMM launch of nvfp4_microbench with --warmup=0 --iters=1 --no-check

# label cfg M N K
SHAPES=$(python3 - <<'EOF'
L = []
for M in (2048, 4096, 8192):
    L += [(f"llama_qoproj_M{M}", "linear", M, 4096, 4096), (f"llama_kvproj_M{M}", "linear", M, 1024, 4096),
          (f"llama_gateup_M{M}", "linear", M, 14336, 4096), (f"llama_down_M{M}", "linear", M, 4096, 14336),
          (f"llama_qkt_M{M}", "qkt", M, M, 128),
          (f"qwen_qproj_M{M}", "linear", M, 4096, 2048), (f"qwen_kvproj_M{M}", "linear", M, 512, 2048),
          (f"qwen_oproj_M{M}", "linear", M, 2048, 4096), (f"qwen_qkt_M{M}", "qkt", M, M, 128),
          (f"ds_attn_M{M}", "linear", M, 2048, 2048), (f"ds_dense0_gu_M{M}", "linear", M, 10880, 2048),
          (f"ds_dense0_down_M{M}", "linear", M, 2048, 10880), (f"ds_shared_gu_M{M}", "linear", M, 2816, 2048),
          (f"ds_shared_down_M{M}", "linear", M, 2048, 2816), (f"ds_qkt_M{M}", "qkt", M, M, 128)]
for M, qm, dm in ((2048, 128, 128), (4096, 256, 384), (8192, 512, 768)):
    L += [(f"qwen_gateup_M{M}", "linear", qm, 768, 2048), (f"qwen_down_M{M}", "linear", qm, 2048, 768),
          (f"ds_routed_gu_M{M}", "linear", dm, 1408, 2048), (f"ds_routed_down_M{M}", "linear", dm, 2048, 1408)]
for s in sorted(L, key=lambda s: -s[2] * s[3] * s[4]):
    print(*s)
EOF
)

wait_for_slot() {
  while [ "$(pgrep -xc accel-sim.out)" -ge $((MAX_SIMS - 1)) ]; do sleep 30; done
}

while read -r LABEL CFG M N K; do
  # ONLY="label1 label2": restrict to these shapes
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $LABEL "* ]]; then continue; fi
  NAME=all_${LABEL}
  if grep -q "^RESULT $NAME " logs_all_results.txt 2>/dev/null; then continue; fi
  echo "$(date '+%F %T') == $LABEL ($CFG M=$M N=$N K=$K)"
  BIN=./bin/nvfp4_microbench_mode3_${CFG}_tie
  SASS=sass/nvfp4_microbench_mode3_${CFG}_tie.sass
  T=traces/$NAME
  if [ ! -f "${T}_widened/traces/kernel-$KID.tracez" ]; then
    DYNAMIC_KERNEL_RANGE="$KID-$KID@.*GemmUniversal.*" TOOL_VERBOSE=0 \
      ./trace.sh "$T" $BIN --m=$M --n=$N --k=$K --iters=1 --warmup=0 --no-check > /dev/null 2>&1
    if ! ls "$T"/traces/kernel-$KID-*.trace.xz > /dev/null 2>&1 || grep -aqE "cuda error|Assert" "$T/tracer_stdout.txt"; then
      echo "  TRACE FAILED for $LABEL"; continue
    fi
    python3 -I omma_operand_widen.py "$T/traces" "$T" $KID $SASS | sed 's/^/  /'
    PIDS=()
    for m in 12op widened; do $PP "${T}_$m/traces" > "${T}_$m/postproc.txt" 2>&1 & PIDS+=($!); done
    wait "${PIDS[@]}"   # only the post-processing, not earlier shapes' simulations
    # keep only the .tracez inputs of the simulator
    rm -f "${T:?}"/traces/*.trace.xz "${T:?}"/traces/*.tracez "${T:?}"_12op/traces/*.trace.xz "${T:?}"_widened/traces/*.trace.xz
  fi
  wait_for_slot
  (
    for m in 12op widened; do
      ./run_sim.sh "${T}_$m" ${NAME}_$m -gpgpu_kernel_launch_latency 0 > /dev/null 2>&1 &
    done
    wait
    a=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_12op/sim.log | awk '{print $3}')
    b=$(grep -m1 '^gpu_tot_sim_cycle' sim_runs/${NAME}_widened/sim.log | awk '{print $3}')
    if [ -n "$a" ] && [ -n "$b" ]; then
      python3 -c "print('RESULT $NAME label=$LABEL M=$M N=$N K=$K 12op=%d widened=%d pct=%+.4f%%' % ($a, $b, 100.0*($b-$a)/$a))" >> logs_all_results.txt
    else
      echo "FAILED $NAME" >> logs_all_results.txt
    fi
  ) &
  sleep 5
done < <(echo "$SHAPES")
wait
echo "$(date '+%F %T') all done"
