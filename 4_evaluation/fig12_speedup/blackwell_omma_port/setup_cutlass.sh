#!/usr/bin/env bash
# Recreates cutlass_clean/ and cutlass_mode3/ used by the build_*.sh scripts.
#   cutlass_clean : unmodified CUTLASS at commit 147295a3 (include, tools/util, examples/common, examples/79)
#   cutlass_mode3 : cutlass_clean + patches/cutlass/cutlass_mode3_full.patch (Crescent kernel changes)
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); cd "$EXP"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git clone --quiet https://github.com/NVIDIA/cutlass.git "$TMP/cutlass"
git -C "$TMP/cutlass" archive 147295a3 include tools/util examples/common examples/79_blackwell_geforce_gemm \
  | tar -x --one-top-level=cutlass_clean
cp -r cutlass_clean cutlass_mode3
(cd cutlass_mode3 && patch -p1 -s < "$EXP/patches/cutlass/cutlass_mode3_full.patch")
echo "cutlass_clean and cutlass_mode3 ready"
