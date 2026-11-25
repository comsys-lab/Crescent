# Section 3: Correction Unit area and critical path

`synthesis/` synthesizes the baseline DPU and the Crescent DPU (baseline + Correction Unit) with
Yosys 0.68 and the Nangate45 Open Cell Library. CPU only; it takes about a minute.

```bash
pip install yowasp-yosys==0.68.0.0.post1208               # the Yosys version used for the paper numbers
YOSYS=yowasp-yosys 3_solution/synthesis/synth.sh           # -> results/summary.csv, results/breakdown.csv
YOSYS=yowasp-yosys python3 3_solution/synthesis/verify.py  # RTL vs Python model, 1,000 random vectors
```

## Files

| File | What it is |
|---|---|
| `synthesis/cu.v` | RTL: `dpe_plain` (baseline DPU), `dpe_corrected` (Crescent DPU) and their blocks |
| `synthesis/synth.sh` | Maps the DPUs and each DPU block to Nangate45; writes the Yosys logs and the CSVs |
| `synthesis/sta.py` | Critical path delay of a mapped netlist (one representative liberty delay per cell) |
| `synthesis/breakdown.py` | Splits the DPU area increase into multiplexers, correction multipliers and adder tree |
| `synthesis/verify.py` | Simulates both DPUs with `yosys eval` and compares them with a Python model of the dot product |
| `synthesis/nangate45_typ.lib.gz` | Nangate45 typical-corner liberty file |
| `synthesis/results/` | Yosys logs, `summary.csv`, `blocks.csv`, `breakdown.csv` |

## Design

Both DPUs compute one 16-element E2M1 dot product per cycle: 16 products (`products16`), an adder tree,
then scale multiply and accumulate (`scale_acc`). The adder tree is a 3:2 carry-save tree with one final
carry-propagate adder.

The Correction Unit (`correction_unit`) has four 1-of-16 multiplexers (`sel16`), which pick each
operand's E2M3-extended element and the other operand's element at the same index, and the correction
arithmetic (`correction_arith`), which computes the value of the 2 extra mantissa bits and multiplies it
by the partner element. The two correction products enter the adder tree as two extra operands
(`tree18`, 16 + 2 operands, instead of `tree16`). A 3:2 tree needs 6 levels for both 16 and 18 operands,
and the correction products join after the first level, so the critical path does not grow.

## Numbers

Whole DPUs, flattened (`results/summary.csv`):

| Module | Area (um^2) | Critical path (ps) |
|---|---|---|
| `dpe_plain` | 8603.238 | 3084.11 |
| `dpe_corrected` | 9755.816 | 3023.71 |
| `main_path` (products + adder tree) | 4613.770 | 1565.30 |
| `correction_unit` | 797.734 | 872.08 |

- DPU area increase: (9755.816 - 8603.238) / 8603.238 = 13.4%.
- Critical path: unchanged (3023.71 ps vs 3084.11 ps). ABC's mapping is not timing-driven, so a
  difference of a few percent is mapping noise. The Correction Unit (872.08 ps) is shorter than the
  main datapath (1565.30 ps).
- Per-SM overhead on the RTX 5090: the DPU increase scaled from 45 nm to 4 nm by (4/45)^2, times 128
  DPUs per SM, over an SM area of 0.4 x 750 mm^2 / 170 SMs = 1.76 mm^2: 0.066%.

Breakdown (`results/breakdown.csv`). Each block is synthesized on its own so the parts add up exactly;
the block sums are larger than the flattened DPUs (no optimization across block boundaries), so the
parts are given as shares of the increase and then applied to the 13.4%:

| Part | Blocks | Area (um^2) | Share of increase | Of the 13.4% |
|---|---|---|---|---|
| Multiplexers | 4 x `sel16` | 417.088 | 32.4% | 4.3 pp |
| Correction multipliers | `correction_arith` | 439.432 | 34.1% | 4.6 pp |
| Adder tree | `tree18` - `tree16`, wider product and scale stages | 432.250 | 33.5% | 4.5 pp |

Area depends on the Yosys version and, by a few percent, on how the RTL is written (ABC is
sensitive to netlist structure): use Yosys 0.68 and this `cu.v` to reproduce these numbers.
