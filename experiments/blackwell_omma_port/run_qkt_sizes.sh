#!/usr/bin/env bash
# Trace + convert Llama QK^T M=2048 and M=4096 (run_all_shapes.sh), then 20 noise pairs each.
set -uo pipefail
cd "$(dirname "$0")"
ONLY="llama_qkt_M2048 llama_qkt_M4096" ./run_all_shapes.sh
for M in 2048 4096; do
  ./run_noise_pairs.sh traces/all_llama_qkt_M$M qkt$M > logs_noise_qkt$M.txt 2>&1
done
echo "$(date '+%F %T') done"
