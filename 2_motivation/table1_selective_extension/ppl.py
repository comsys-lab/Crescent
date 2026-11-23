"""Table 1: WikiText-2 PPL under selective mantissa extension, BF16 weights (activations only quantized).

Rows (one CSV per format x model):
  BF16           : no quantization
  Native         : every activation operand (Linear inputs, Q, K, V, P) in the original NVFP4 / MXFP4 format
  Selective_E2M3 : in each block, the top 12.5% elements by squared quantization error (NVFP4 k=2 of 16,
                   MXFP4 k=4 of 32) take their E2M3 value, the rest stay E2M1; Linear inputs, Q, K, V and P
QK^T and P@V use the decomposed form (selected x selected delta term dropped). P: causal positions only.
Weights stay BF16 (empty weight cache). Model loading, patching and PPL measurement: common/ppl_w4a4.py.

Usage: python3 ppl.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b>
Output: results/ppl_{fmt}_{model}.csv (resumes: rows already in the file are skipped)
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
import ppl_w4a4 as W  # noqa: E402
from formats import quantize_tensor_group_topk_boost  # noqa: E402

ARMS = ["bf16", "native", "selective"]
LABELS = {"bf16": "BF16", "native": "Native", "selective": "Selective_E2M3"}

orig_setup = W.arm_setup


def arm_setup(fmt, arm, model_key=None):
    if arm != "selective":
        return orig_setup(fmt, arm, model_key)
    c = W.FORMAT_CONFIG[fmt]
    bs, base, boost = c["block_size"], c["base_config"], c["boost"]
    k = bs // 8  # 12.5% of the block

    def topk(x):
        return quantize_tensor_group_topk_boost(x.detach().to(torch.float32), bs, k, block_size=bs,
                                                base_config=base, boost_config=boost, by="error").to(x.dtype)
    plain_fn, plain_p = W.plain_fns(fmt)
    base_fns = {"q": plain_fn, "k": plain_fn, "v": plain_fn, "p": plain_p}
    boost_fns = {"q": topk, "k": topk, "v": topk, "p": W.selective_p(topk, bs)}
    return topk, ("decomposed", base_fns, boost_fns)


W.arm_setup = arm_setup


def main():
    fmt, model_key = sys.argv[1:3]
    W.same_seeds()
    _, enc = W.load_tokenizer_and_encodings(W.model_name_of(model_key), W.load_wikitext2_text())
    W.run_all(fmt, model_key, ARMS, enc, os.path.join(HERE, "results", f"ppl_{fmt}_{model_key}.csv"),
              cache_for=lambda: {}, labels=LABELS)  # {}: no GPTQ weights -> BF16 weights


if __name__ == "__main__":
    main()
