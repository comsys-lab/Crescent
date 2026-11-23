#!/usr/bin/env bash
# Operand term on the B200-CLASS config (SM100_B200_CLASS): the same 6 trace pairs and 20 launch-latency
# perturbations as the SM120 study (logs_noise_*.csv). Output logs_noise_b200_<tag>.csv/.txt
set -uo pipefail
cd "$(dirname "$0")"
export CFG=SM100_B200_CLASS MAX_SIMS=${MAX_SIMS:-46}
./run_noise_pairs.sh traces/m3tie_linear_2048x4096x4096 b200_qproj > logs_noise_b200_qproj.txt 2>&1 &
./run_noise_pairs.sh traces/all_llama_kvproj_M4096   b200_kv4096  > logs_noise_b200_kv4096.txt 2>&1 &
./run_noise_pairs.sh traces/all_ds_routed_gu_M8192   b200_dsrgu768 > logs_noise_b200_dsrgu768.txt 2>&1 &
./run_noise_pairs.sh traces/all_llama_qkt_M2048      b200_qkt2048 > logs_noise_b200_qkt2048.txt 2>&1 &
./run_noise_pairs.sh traces/all_llama_qkt_M4096      b200_qkt4096 > logs_noise_b200_qkt4096.txt 2>&1 &
./run_noise_pairs.sh traces/all_llama_qkt_M8192      b200_qkt8192 > logs_noise_b200_qkt8192.txt 2>&1 &
wait; echo "$(date '+%F %T') done" > logs_b200_operand_done.txt
