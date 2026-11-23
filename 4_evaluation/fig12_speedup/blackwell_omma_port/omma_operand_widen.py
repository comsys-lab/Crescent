#!/usr/bin/env python3
"""Turn a trace of the NVFP4_META_TIE_OPERAND=1 GEMM build into the two traces
that are simulated (one pass, both outputs):

  <out>_12op     : the tie LOP3s removed -> the real 12-register OMMA.
  <out>_widened  : the tie LOP3s removed and, for every OMMA, the packed
                   metadata register(s) tied into that OMMA's accumulator
                   appended as extra sources (13th operand for Linear, 13th +
                   14th for QK^T).

The build ties each packed value to an OMMA accumulator with a value-preserving
    LOP3.LUT Rd, Ra, URzero, Rpacked, 0x78         (Rd = Ra ^ (URzero & Rpacked))
so ptxas schedules the packing before that OMMA. ptxas may chain ties through a
temporary (QK^T: acc -> tmp -> acc), so ties are followed as copies: Rd carries
Ra's root accumulator and the packed registers collected so far. An OMMA reading
C = Rd gets those packed registers (in tie order: A's, then B's) appended, and
C renamed to the root accumulator if it is a temporary. Ties are recognized by
static PC from the build's SASS (LOP3 0x78 with the zero uniform register).

If a packed register is overwritten between its tie and the OMMA (ptxas placed
the tie early and reused the register), the OMMA gets the alias R<n+256>: same
RF bank parity and no writer, so no false dependency on the unrelated writer.
The overwrite itself had to wait for the packing result (WAW), so the true
dependency was already satisfied.

Usage: omma_operand_widen.py <in_traces_dir> <out_prefix> <kernel_id> <sass_file>
Writes <out_prefix>_{12op,widened}/traces/kernel-<id>.trace.xz + kernelslist_edit;
run post-traces-processing on each afterwards.
"""
import collections, glob, os, re, subprocess, sys

in_dir, out_prefix, kid, sass_file = sys.argv[1:5]

# Static tie PCs of the GEMM kernel from cuobjdump SASS.
tie_re = re.compile(r"LOP3\.LUT (R\d+), (R\d+), (UR\d+), (R\d+), 0x78, !PT")
cands, in_gemm = [], False
for line in open(sass_file):
    if "Function :" in line:
        in_gemm = "GemmUniversal" in line
    m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
    if in_gemm and m:
        tm = tie_re.match(m.group(2).replace(".reuse", ""))
        if tm:
            cands.append((int(m.group(1), 16), tm.group(3)))
# ptxas may materialize the zero uniform register more than once (static-scheduler QK^T build:
# 64 ties on UR9 in one code copy, 64 on UR5 in the executed one), so accept every UR that
# carries a substantial number of ties.
cnt = collections.Counter(u for _, u in cands)
zero_urs = {u for u, c in cnt.items() if c >= 16}
zero_ur = ",".join(sorted(zero_urs))
tie_pcs = {f"{pc:04x}" for pc, u in cands if u in zero_urs}
assert tie_pcs, "no tie LOP3 found in SASS"

src = glob.glob(os.path.join(in_dir, f"kernel-{kid}-*.trace.xz"))
assert len(src) == 1, src
outs = {}
for mode in ("12op", "widened"):
    d = f"{out_prefix}_{mode}/traces"
    os.makedirs(d, exist_ok=True)
    # post-traces-processing input list (it writes kernelslist.g / .tracez itself)
    with open(os.path.join(d, "kernelslist_edit"), "w") as k:
        k.write(f"kernel-{kid}.trace.xz\n")
    fo = open(os.path.join(d, f"kernel-{kid}.trace.xz"), "wb")
    outs[mode] = subprocess.Popen(["xz", "-0", "-T4", "-c"], stdin=subprocess.PIPE,
                                  stdout=fo, text=True, bufsize=1 << 20)
w12, wwide = outs["12op"].stdin, outs["widened"].stdin
dec = subprocess.Popen(["xz", "-dc", "-T4", src[0]], stdout=subprocess.PIPE,
                       text=True, bufsize=1 << 20)

chain = collections.defaultdict(dict)  # warp -> {reg: (root acc reg, [packed regs])}
stats = collections.Counter()
for line in dec.stdout:
    f = line.split()
    if len(f) < 15 or not f[0].lstrip("-").isdigit():
        w12.write(line)
        wwide.write(line)
        continue
    warp = (f[0], f[1], f[2], f[3])
    nd = int(f[13])
    dsts = f[14:14 + nd]
    op_i = 14 + nd
    opcode = f[op_i]
    if f[11] in tie_pcs:
        ns = int(f[op_i + 1])
        srcs = f[op_i + 2:op_i + 2 + ns]
        assert opcode == "LOP3.LUT" and nd == 1 and ns == 3 and \
            srcs[1] in zero_urs and f[op_i + 2 + ns + 1] == "120", line
        ch = chain[warp]
        root, packed = ch.get(srcs[0], (srcs[0], []))
        ch[dsts[0]] = (root, packed + [srcs[2]])
        stats["ties_dropped"] += 1
        continue
    line12 = linew = line
    if opcode.startswith("OMMA"):
        ns = int(f[op_i + 1])
        stats["omma"] += 1
        root, extra = chain[warp].pop(f[op_i + 4], (f[op_i + 4], []))
        stats[f"omma_with_{len(extra)}_tied"] += 1
        if any(int(r[1:]) >= 256 for r in extra):
            stats["omma_with_aliased_packed"] += 1
        if root != f[op_i + 4]:
            stats["omma_C_renamed_to_root"] += 1
            f[op_i + 4] = root
            line12 = " ".join(f) + " \n"
        if extra:
            assert "NoVal" in f[op_i + 2 + ns:], "register values not supported"
            f[op_i + 1] = str(ns + len(extra))
            f[op_i + 2 + ns:op_i + 2 + ns] = extra
            linew = " ".join(f) + " \n"
        else:
            linew = line12
    ch = chain.get(warp)
    if ch:
        for d in dsts:
            ch.pop(d, None)
            if d.startswith("R"):
                for k, (root, packed) in ch.items():
                    if d in packed:
                        ch[k] = (root, [f"R{int(r[1:]) + 256}" if r == d else r
                                        for r in packed])
    w12.write(line12)
    wwide.write(linew)
stats["ties_unconsumed"] = sum(len(p) for ch in chain.values() for _, p in ch.values())
assert dec.wait() == 0
for p in outs.values():
    p.stdin.close()
    assert p.wait() == 0
print(f"zero={zero_ur} static tie PCs={len(tie_pcs)}", dict(stats))
