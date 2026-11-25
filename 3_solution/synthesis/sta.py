#!/usr/bin/env python3
"""Critical path delay of a Nangate45-mapped netlist (written by synth.sh as results/<top>.json).

Delay model: every cell gets one representative delay from nangate45_typ.lib, the largest
cell_rise / cell_fall value over its timing arcs at a fixed table corner (2nd input-slew row,
3rd output-load column). The critical path is the largest sum of cell delays from any primary
input to any primary output (combinational netlist, so a longest path over the DAG).

Usage: sta.py <top> [<top> ...]    (run from this directory, after synth.sh has mapped the tops)
Prints: <top>,<critical_path_ps>,<cells_on_path>
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "nangate45_typ.lib"


def _blocks(text, header):
    """Yield (name, body) for every `header(name) { ... }` block, matching braces."""
    for m in re.finditer(header + r"\s*\(([A-Za-z0-9_]+)\)\s*\{", text):
        depth, j = 0, m.end() - 1
        while True:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        yield m.group(1), text[m.end() - 1 : j + 1]


def cell_delays(lib=LIB):
    """cell type -> representative delay in ns."""
    text = lib.read_text()
    delays = {}
    for cname, body in _blocks(text, r"\bcell"):
        best = None
        for _, pbody in _blocks(body, r"\bpin"):
            if not re.search(r"direction\s*:\s*output", pbody):
                continue
            for tm in re.finditer(r"cell_(?:rise|fall)\([^)]*\)\s*\{(.*?)\n\s*\}", pbody, re.S):
                vals = re.search(r"values\s*\((.*?)\)\s*;", tm.group(1), re.S)
                rows = re.findall(r'"([^"]*)"', vals.group(1)) if vals else []
                if not rows:
                    continue
                cols = [float(x) for x in rows[min(1, len(rows) - 1)].split(",")]
                v = cols[min(2, len(cols) - 1)]
                best = v if best is None else max(best, v)
        if best is not None:
            delays[cname] = best
    return delays


def critical_path(netlist_json, top, delays):
    mod = json.loads(Path(netlist_json).read_text())["modules"][top]
    driver = {}  # net bit -> cell name
    for name, c in mod["cells"].items():
        for port, bits in c["connections"].items():
            if c["port_directions"][port] == "output":
                for b in bits:
                    driver[b] = name
    memo = {}

    def arrival(cell):  # (arrival time at the cell output, cells on the path)
        if cell in memo:
            return memo[cell]
        c = mod["cells"][cell]
        best = (0.0, 0)
        for port, bits in c["connections"].items():
            if c["port_directions"][port] != "input":
                continue
            for b in bits:
                if b in driver:
                    best = max(best, arrival(driver[b]))
        memo[cell] = (best[0] + delays[c["type"]], best[1] + 1)
        return memo[cell]

    sys.setrecursionlimit(100000)
    outs = [b for p in mod["ports"].values() if p["direction"] == "output" for b in p["bits"]]
    return max(arrival(driver[b]) for b in outs if b in driver)


if __name__ == "__main__":
    d = cell_delays()
    for top in sys.argv[1:]:
        ns, n = critical_path(HERE / "results" / f"{top}.json", top, d)
        print(f"{top},{ns * 1000:.2f},{n}")
