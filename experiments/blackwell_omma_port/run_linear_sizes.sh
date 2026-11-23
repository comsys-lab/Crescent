#!/usr/bin/env bash
# After the representative-shape batch (pid $1) finishes: trace + convert Llama k/v_proj M=4096,
# then 20 noise pairs each for it and for DeepSeek routed gate/up M=768 (traced by that batch).
set -uo pipefail
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
ONLY="llama_kvproj_M4096" ./run_all_shapes.sh
./run_noise_pairs.sh traces/all_llama_kvproj_M4096 kv4096 > logs_noise_kv4096.txt 2>&1
./run_noise_pairs.sh traces/all_ds_routed_gu_M8192 dsrgu768 > logs_noise_dsrgu768.txt 2>&1
echo "$(date '+%F %T') done"
