#!/usr/bin/env bash
OUT=grouped_raw.txt; : > $OUT
r() { echo "### $1" >> $OUT; docker run --rm --gpus '"device=1"' -v "$PWD":/w -w /w --entrypoint ./grouped_gemm_example nvcr.io/nvidia/pytorch:25.12-py3 --m=$2 --n=$3 --k=$4 --groups=$5 --iterations=50 --no_verif 2>&1 | grep -E "Avg runtime" >> $OUT; }
# Qwen: 128 experts, per-expert M = M_seq/16
for p in "2048:128" "4096:256" "8192:512" "16384:1024"; do s=${p%%:*}; e=${p##*:}
  r qwen_gateup_M$s $e 768 2048 128
  r qwen_down_M$s $e 2048 768 128
done
# DeepSeek: 64 experts, per-expert M = M_seq*6/64 (rounded to 128 multiple)
for p in "2048:128" "4096:384" "8192:768"; do s=${p%%:*}; e=${p##*:}
  r ds_routed_gu_M$s $e 1408 2048 64
  r ds_routed_down_M$s $e 2048 1408 64
done
echo DONE >> $OUT
