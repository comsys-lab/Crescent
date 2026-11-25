#!/usr/bin/env python3
"""Area breakdown of the Crescent DPU increase, from results/blocks.csv (run by synth.sh).

Every DPU block is synthesized on its own, so the blocks add up exactly:
  baseline DPU = products16(W=20) + tree16 + scale_acc(W=20)
  Crescent DPU = products16(W=21) + 4 x sel16 + correction_arith + tree18 + scale_acc(W=21)
The increase splits into
  multiplexers            4 x sel16 (1-of-16 selection of the extended element and its partner)
  correction multipliers  correction_arith (extra-mantissa value, E2M1 decode, two multiplies)
  adder tree              tree18 - tree16, plus the one-bit-wider product and scale stages
Output: results/breakdown.csv (um^2, and percentage points of the block-sum baseline DPU).
"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
b = {r["block"]: float(r["area_um2"]) for r in csv.DictReader(open(RES / "blocks.csv"))}
base = b["products16_w20"] + b["tree16"] + b["scale_acc_w20"]
parts = [
    ("multiplexers", 4 * b["sel16"]),
    ("correction_multipliers", b["correction_arith"]),
    ("adder_tree", b["tree18"] - b["tree16"] + b["products16_w21"] - b["products16_w20"]
                   + b["scale_acc_w21"] - b["scale_acc_w20"]),
]
parts.append(("total", sum(v for _, v in parts)))
with open(RES / "breakdown.csv", "w") as f:
    f.write("part,area_um2,pct_of_baseline\n")
    f.write(f"baseline_dpu,{base:.3f},100.00\n")
    for name, v in parts:
        f.write(f"{name},{v:.3f},{100 * v / base:.2f}\n")
print((RES / "breakdown.csv").read_text())
