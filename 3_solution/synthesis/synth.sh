#!/usr/bin/env bash
# Synthesize the baseline and Crescent DPUs (cu.v) with Yosys + Nangate45 (Section 3: area and delay).
#
# Usage: synth.sh     needs `yosys` on PATH, or YOSYS=<path>. The paper used Yosys 0.68:
#                     pip install yowasp-yosys==0.68.0.0.post1208 && YOSYS=yowasp-yosys ./synth.sh
# Output: results/<name>.log and results/<name>.json (mapped netlist) per synthesized module,
#         results/summary.csv   (whole DPUs, flattened: area in um^2, critical path in ps from sta.py),
#         results/breakdown.csv (each DPU block synthesized on its own, from breakdown.py).
set -euo pipefail
cd "$(dirname "$0")"
YOSYS=${YOSYS:-yosys}
mkdir -p results
[ -f nangate45_typ.lib ] || gunzip -k nangate45_typ.lib.gz

# synth <result name> <top module> [W parameter]
synth() {
  local chparam=""
  [ -n "${3:-}" ] && chparam="-chparam W $3"
  echo "[synth] $1"
  $YOSYS -q -l results/$1.log -p "
    read_verilog -sv cu.v;
    hierarchy -top $2 $chparam;
    proc; opt; flatten; opt;
    techmap; opt;
    abc -liberty nangate45_typ.lib;
    opt_clean;
    read_liberty -lib -ignore_miss_dir nangate45_typ.lib;
    stat -liberty nangate45_typ.lib;
    write_json results/$1.json
  " >/dev/null
}
area() { grep -oP "Chip area for (top )?module '\\\\[^']*': \K[0-9.]+" results/$1.log | tail -1; }

# Whole DPUs and the two datapaths that run in parallel.
TOPS="dpe_plain dpe_corrected main_path correction_unit"
for t in $TOPS; do synth $t $t; done
python3 sta.py $TOPS > results/sta.csv
{
  echo "module,area_um2,critical_path_ps,cells_on_path"
  while IFS=, read -r t ps n; do echo "$t,$(area $t),$ps,$n"; done < results/sta.csv
} > results/summary.csv
rm results/sta.csv
cat results/summary.csv; echo

# DPU blocks, for the area breakdown.
BLOCKS="products16_w20:products16:20 products16_w21:products16:21 tree16:tree16 tree18:tree18
        scale_acc_w20:scale_acc:20 scale_acc_w21:scale_acc:21 sel16:sel16 correction_arith:correction_arith"
{
  echo "block,area_um2"
  for b in $BLOCKS; do
    IFS=: read -r name top w <<< "$b"
    synth $name $top $w >&2
    echo "$name,$(area $name)"
  done
} > results/blocks.csv
python3 breakdown.py
