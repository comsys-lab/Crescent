"""Czardas / Magnitude, final config: Linear = exact top-2 per 16-element group; Q/K/V/P = top-1 per 16
(decomposed QK^T and P@V, delta x delta dropped). Selection: squared error (czardas) or |a| (magnitude).
P: causal-full mask, same as plain_p / MX+. V along head_dim (V-axis issue deferred).
Usage: python3 ppl_czardas_top2.py <fmt> <model> <czardas|magnitude>"""
import csv, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppl_w4a4 as W
from quant_targets import _chunked_query
from formats import quantize_tensor_group_topk_boost

fmt, model_key, method = sys.argv[1:4]
c = W.FORMAT_CONFIG[fmt]
bs, base, boost = c["block_size"], c["base_config"], c["boost"]
by = "error" if method == "czardas" else "magnitude"
G = 16


def topk_fn(k):
    return lambda x: quantize_tensor_group_topk_boost(x.detach().to(torch.float32), G, k, block_size=bs,
                                                      base_config=base, boost_config=boost, by=by).to(x.dtype)


def p_boost_one(aw):
    k_len = aw.shape[3]
    g = topk_fn(1)

    def fn(chunk, row_offset):
        full = g(chunk)
        rows = (torch.arange(chunk.shape[2], device=chunk.device) + row_offset).view(1, 1, -1, 1)
        cols = torch.arange(k_len, device=chunk.device).view(1, 1, 1, k_len)
        return torch.where(cols <= rows, full, chunk)
    return _chunked_query(aw, bs, fn)


def p_boost(aw):
    if aw.shape[1] <= 4:
        return p_boost_one(aw)
    return torch.cat([p_boost_one(c_) for c_ in aw.split(4, dim=1)], dim=1)


orig_setup = W.arm_setup


def patched(fmt_, arm, model_key_=None):
    _, attn = orig_setup(fmt_, "grouped", model_key_)  # only for base_fns (plain q/k/v/p)
    a1 = topk_fn(1)
    boost_fns = {"q": a1, "k": a1, "v": a1, "p": p_boost}
    return topk_fn(2), ("decomposed", attn[1], boost_fns)


W.arm_setup = patched
ARM = "grouped"
W.LABELS[ARM] = f"W4A4_{'Czardas' if by == 'error' else 'Magnitude'}_top2"
W.ALL_ARMS = [ARM]
src = os.path.join(W.RESULTS_DIR, f"ppl_{fmt}_{model_key}.csv")
dst = os.path.join(W.RESULTS_DIR, f"ppl_{fmt}_{model_key}_{method}_top2.csv")
if not os.path.exists(dst):
    with open(src) as f:
        rows = [r for r in csv.DictReader(f) if r["setting"] == "BF16"]
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "model", "setting", "ppl", "increase", "elapsed_sec"])
        w.writeheader(); w.writerows(rows)
print(f"[{fmt}] {method}: Linear top-2/{G}, Q/K/V/P top-1/{G}, block={bs}", flush=True)
W.same_seeds()
_, enc = W.load_tokenizer_and_encodings(W.model_name_of(model_key), W.load_wikitext2_text())
W.run_all(fmt, model_key, W.make_cache_provider(fmt, model_key), enc, dst)
