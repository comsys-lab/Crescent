#!/usr/bin/env python3
"""Check that the RTL in cu.v computes the intended dot products (random vectors vs a Python model).

Usage: verify.py [num_vectors]    (default 1000; needs `yosys` on PATH, or YOSYS=<path>)
Simulates dpe_plain and dpe_corrected with `yosys eval` and compares acc_out with the model below.
"""
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YOSYS = os.environ.get("YOSYS", "yosys")
E2M1_X8 = [0, 4, 8, 12, 16, 24, 32, 48]          # E2M1 magnitudes, 8x real value


def dec(code):
    v = E2M1_X8[code & 7]
    return -v if code & 8 else v


def delta(code, extra_m):                        # value of the 2 extra mantissa bits (8x)
    ee = (code >> 1) & 3
    v = extra_m << (0 if ee == 0 else ee - 1)
    return -v if code & 8 else v


def wrap(x, bits):
    x &= (1 << bits) - 1
    return x - (1 << bits) if x >> (bits - 1) else x


def model(a, w, s_a, s_b, c_in, corr=None):
    A = [(a >> 4 * i) & 15 for i in range(16)]
    W = [(w >> 4 * i) & 15 for i in range(16)]
    total = sum(dec(x) * dec(y) for x, y in zip(A, W))
    if corr:
        ia, ma, ib, mb = corr
        total += delta(A[ia], ma) * dec(W[ia]) + dec(A[ib]) * delta(W[ib], mb)
    return wrap(c_in + wrap(total * wrap(s_a * s_b, 32), 32), 32)


def simulate(top, vecs, corrected):
    cmds = [f"read_verilog -sv cu.v; hierarchy -top {top}; proc; flatten; opt_clean"]
    for a, w, s_a, s_b, c_in, corr in vecs:
        c = f"eval -set a_flat 64'd{a} -set w_flat 64'd{w} -set s_a {s_a} -set s_b {s_b} -set c_in {c_in}"
        if corrected:
            ia, ma, ib, mb = corr
            c += f" -set index_a {ia} -set mant_a {ma} -set index_b {ib} -set mant_b {mb}"
        cmds.append(c + " -show acc_out")
    script = HERE / "results" / f"verify_{top}.ys"
    script.parent.mkdir(exist_ok=True)
    script.write_text("\n".join(cmds) + "\n")
    out = subprocess.run([YOSYS, "-s", f"results/{script.name}"], cwd=HERE,
                         capture_output=True, text=True).stdout
    script.unlink()
    vals = re.findall(r"\\acc_out = (32'[01]+|-?\d+)\.", out)   # binary, or decimal when small
    return [wrap(int(v[3:], 2) if v.startswith("32'") else int(v), 32) for v in vals]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = random.Random(0)
    vecs = [(rng.getrandbits(64), rng.getrandbits(64), rng.randint(-2**15, 2**15 - 1),
             rng.randint(-2**15, 2**15 - 1), rng.randint(-2**31, 2**31 - 1),
             (rng.randrange(16), rng.randrange(4), rng.randrange(16), rng.randrange(4))) for _ in range(n)]
    ok = True
    for top, corrected in (("dpe_plain", False), ("dpe_corrected", True)):
        got = simulate(top, vecs, corrected)
        want = [model(a, w, sa, sb, c, corr if corrected else None) for a, w, sa, sb, c, corr in vecs]
        bad = sum(g != e for g, e in zip(got, want)) + abs(len(got) - n)
        print(f"{top}: {n} vectors, {bad} mismatches")
        ok &= bad == 0
    sys.exit(0 if ok else 1)
