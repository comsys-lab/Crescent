"""Shared model-loading, dataset, and PPL-measurement helpers."""

from __future__ import annotations

import csv
import gc
import math
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

PPL_SEQ_LEN = 2048
PPL_STRIDE = 512
PPL_MAX_STEPS = 64
PPL_DECIMAL_PLACES = 15


def same_seeds(seed: int = 1111) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


WIKITEXT2_REPO = "Salesforce/wikitext"  # "wikitext" itself is a legacy loading-script
# repo that newer `datasets` releases can no longer load (HfUriError on the
# .huggingface.yaml resolve); Salesforce/wikitext is the parquet mirror of the
# same data (test split row count matches: 4358) -- confirmed when bringing
# this project up on a fresh "super" server with transformers/datasets
# upgraded past the versions used on rtx4090.


def load_wikitext2_text() -> str:
    from datasets import load_dataset
    dataset = load_dataset(WIKITEXT2_REPO, "wikitext-2-raw-v1", split="test")
    return "\n\n".join(r["text"] for r in dataset if r.get("text") and r["text"].strip())


def load_wikitext2_calib_text() -> str:
    # Separate from load_wikitext2_text (the "test" split used for PPL eval):
    # calibration must not overlap with the eval text. Uses the "train" split.
    from datasets import load_dataset
    dataset = load_dataset(WIKITEXT2_REPO, "wikitext-2-raw-v1", split="train")
    rows = [r["text"] for r in dataset.select(range(2000)) if r.get("text") and r["text"].strip()]
    return "\n\n".join(rows)


def load_tokenizer_and_encodings(model_name: str, full_text: str, trust_remote_code: bool = True):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok, tok(full_text, return_tensors="pt")


def load_model(model_name: str, device: torch.device, trust_remote_code: bool = True) -> nn.Module:
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


@torch.no_grad()
def measure_ppl(
    model: nn.Module,
    encodings,
    desc: str,
    device: torch.device,
    seq_len: int = PPL_SEQ_LEN,
    stride: int = PPL_STRIDE,
    max_steps: int = PPL_MAX_STEPS,
) -> float:
    input_ids_cpu = encodings.input_ids
    total_len = input_ids_cpu.size(1)
    nll_sum, nll_tokens, steps, prev_end = 0.0, 0, 0, 0
    for begin in range(0, total_len, stride):
        end = min(begin + seq_len, total_len)
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        ids = input_ids_cpu[:, begin:end].to(device)
        labels = ids.clone(); labels[:, :-trg_len] = -100
        out = model(input_ids=ids, labels=labels, use_cache=False)
        loss = out.loss.float()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        n_tok = int((labels[:, 1:] != -100).sum().item())
        nll_sum += float(loss.item()) * n_tok; nll_tokens += n_tok
        steps += 1; prev_end = end
        if steps == 1 or steps % 8 == 0:
            print(f"[{desc}] step={steps} running_ppl={math.exp(nll_sum/nll_tokens):.6f}")
        del ids, labels, out, loss
        if steps >= max_steps or end >= total_len:
            break
    return float(math.exp(nll_sum / nll_tokens))
