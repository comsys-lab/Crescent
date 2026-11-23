"""Table 7, C4 columns: PPL of BF16, Native, Magnitude, All E2M3 and Crescent (settings defined in
common/ppl_w4a4.py), with GPTQ 4-bit weights for every quantized setting. 256 random 2048-token samples
from the first C4 validation shard, seed 0 (GPTQ protocol, common/c4_utils.py).

Usage: CUDA_VISIBLE_DEVICES=<GPUs> python3 ppl_c4.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b>
Output: results/ppl_c4_{fmt}_{model}.csv (resumes: settings already in the file are skipped)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import ppl_w4a4 as W  # noqa: E402
from c4_utils import c4_encodings, N_SAMPLES, SEQ_LEN  # noqa: E402


def main():
    fmt, model_key = sys.argv[1], sys.argv[2]
    W.same_seeds()
    tok, _ = W.load_tokenizer_and_encodings(W.model_name_of(model_key), "x")  # only the tokenizer is needed
    encodings = c4_encodings(tok)
    W.run_all(fmt, model_key, W.ARMS, encodings, os.path.join(HERE, "results", f"ppl_c4_{fmt}_{model_key}.csv"),
              seq_len=SEQ_LEN, stride=SEQ_LEN, max_steps=N_SAMPLES)


if __name__ == "__main__":
    main()
