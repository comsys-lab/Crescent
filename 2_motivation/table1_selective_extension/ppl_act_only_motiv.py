"""Activation-only PPL (WikiText2) for the Motivation proposal applied to EVERY activation operand:
weights BF16 (empty GPTQ cache); Linear/Q/K/V/P each: exact top-12.5% per scaling block by squared
quantization error -> E2M3, rest E2M1 (NVFP4 k=2 of 16, MXFP4 k=4 of 32). QK^T and P@V decomposed
(delta x delta dropped), as in H4/H5's QK^T. P: causal-full mask. V along head_dim.
Usage: python3 ppl_act_only_motiv.py <fmt> <model>"""
import csv, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import ppl_w4a4 as W
from quant_targets import _chunked_query
from formats import quantize_tensor_group_topk_boost

fmt, model_key = sys.argv[1:3]
c = W.FORMAT_CONFIG[fmt]
bs, base, boost = c["block_size"], c["base_config"], c["boost"]
K = bs // 8  # 12.5% of block


def topk(x):
    return quantize_tensor_group_topk_boost(x.detach().to(torch.float32), bs, K, block_size=bs,
                                            base_config=base, boost_config=boost, by="error").to(x.dtype)


def p_one(aw):
    k_len = aw.shape[3]
    def fn(chunk, row_offset):
        full = topk(chunk)
        rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
        cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
        return torch.where(cols <= rows, full, chunk)
    return _chunked_query(aw, bs, fn)


def p_boost(aw):
    return p_one(aw) if aw.shape[1] <= 4 else torch.cat([p_one(c_) for c_ in aw.split(4, dim=1)], dim=1)


orig_setup = W.arm_setup
ARM = "grouped"


def patched(fmt_, arm, model_key_=None):
    _, attn = orig_setup(fmt_, "grouped", model_key_)  # base_fns = plain q/k/v/p
    return topk, ("decomposed", attn[1], {"q": topk, "k": topk, "v": topk, "p": p_boost})


W.arm_setup = patched
W.LABELS[ARM] = "Motiv_Top12.5_E2M3_ActOnly"
W.ALL_ARMS = [ARM]
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"ppl_{fmt}_{model_key}_actonly_baseline.csv")
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"ppl_{fmt}_{model_key}_actonly_motiv.csv")
if not os.path.exists(dst):
    with open(src) as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "BF16"]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
        w.writeheader(); w.writerows(rows)
print(f"[{fmt}] motivation act-only: top-{K}/{bs} -> {boost} on Linear/Q/K/V/P, BF16 weights", flush=True)
W.same_seeds()
_, enc = W.load_tokenizer_and_encodings(W.model_name_of(model_key), W.load_wikitext2_text())
W.run_all(fmt, model_key, lambda kind: {}, enc, dst)
