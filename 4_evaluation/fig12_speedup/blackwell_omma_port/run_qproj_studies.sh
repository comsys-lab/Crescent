#!/usr/bin/env bash
# Two studies on the existing q_proj (Llama, M=2048 N=4096 K=4096) 12op/widened traces:
#  noise : kernel launch latency L in {0,1,2,3,5,8} cycles. Irrelevant to the kernel
#          itself; shifts its start phase against the DRAM/L2 clocks, so the spread
#          of the paired pct is the simulator's sensitivity floor.
#  rf    : register-file structure sensitivity (L=0). gpgpu_num_reg_banks counts all
#          4 sub-cores (8 = 2 banks/sub-core, the config default); port throughput
#          = reads per bank per cycle (default 2).
# Results: logs_qproj_studies.csv
set -uo pipefail
cd "$(dirname "$0")"
T=traces/m3tie_linear_2048x4096x4096
pair() {  # name, extra options...
  local n=$1; shift
  for m in 12op widened; do
    ./run_sim.sh ${T}_$m qp_${n}_$m -gpgpu_kernel_launch_latency ${L:-0} "$@" > /dev/null 2>&1 &
  done
}
for L in 0 1 2 3 5 8; do L=$L pair noise_L$L; done
pair rf_b4p2 -gpgpu_num_reg_banks 4 -gpgpu_reg_file_port_throughput 2
pair rf_b8p1 -gpgpu_num_reg_banks 8 -gpgpu_reg_file_port_throughput 1
pair rf_b4p1 -gpgpu_num_reg_banks 4 -gpgpu_reg_file_port_throughput 1
wait
echo "run,cycles_12op,cycles_widened,pct,bankwait_12op,bankwait_widened" > logs_qproj_studies.csv
for d in sim_runs/qp_*_12op; do
  n=${d#sim_runs/qp_}; n=${n%_12op}
  g() { grep -m1 "^$1" sim_runs/qp_${n}_$2/sim.log | awk '{print $3}'; }
  a=$(g gpu_tot_sim_cycle 12op); b=$(g gpu_tot_sim_cycle widened)
  ca=$(g gpu_reg_bank_conflict_stalls 12op); cb=$(g gpu_reg_bank_conflict_stalls widened)
  python3 -c "print('$n,$a,$b,%.4f,$ca,$cb' % (100.0*($b-$a)/$a))" >> logs_qproj_studies.csv
done
echo "$(date '+%F %T') done"
