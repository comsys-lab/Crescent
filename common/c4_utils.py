"""C4 PPL evaluation input, built with the exact GPTQ protocol (Frantar et al., datautils.py `get_c4`):
256 random documents (seed 0) from allenai/c4's English validation split, each trimmed to one random
2048-token window, concatenated into a single [1, 256*2048] tensor. Combined with eval_utils.measure_ppl
called at stride == seq_len (non-overlapping full-sequence loss per chunk, no window-boundary trg_len
trimming), this reproduces the standard GPTQ/AWQ/FlatQuant C4 PPL number bit-for-bit in methodology.

Usage: ids = build_c4_ppl_input_ids(tokenizer)   # [1, 256*2048]
       measure_ppl(model, SimpleNamespace(input_ids=ids), desc, device, stride=2048, max_steps=256)
"""
import random
from types import SimpleNamespace

import torch

C4_REPO = "allenai/c4"
N_SAMPLES = 256
SEQ_LEN = 2048
SEED = 0  # matches the original GPTQ protocol exactly


def build_c4_ppl_input_ids(tokenizer, n_samples: int = N_SAMPLES, seq_len: int = SEQ_LEN, seed: int = SEED):
    from datasets import load_dataset
    # data_files (not the "en" config default) -- the "en" config's legacy loading script pulls ALL 1024
    # train shards regardless of split="validation" (confirmed: a plain split="validation" load started
    # downloading multi-hundred-GB of train--of-01024 shards). One validation shard (~40k docs, 39MB) is
    # exactly the file the original GPTQ protocol (Frantar et al.) itself uses and is far more than enough
    # for n_samples=256.
    valdata = load_dataset(C4_REPO, data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    rng = random.Random(seed)
    chunks = []
    for _ in range(n_samples):
        while True:
            i = rng.randint(0, len(valdata) - 1)
            enc = tokenizer(valdata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seq_len:
                break
        start = rng.randint(0, enc.input_ids.shape[1] - seq_len - 1) if enc.input_ids.shape[1] > seq_len else 0
        chunks.append(enc.input_ids[:, start:start + seq_len])
    return torch.hstack(chunks)


def c4_encodings(tokenizer, n_samples: int = N_SAMPLES, seq_len: int = SEQ_LEN, seed: int = SEED):
    """Drop-in for eval_utils.load_tokenizer_and_encodings's second return value."""
    return SimpleNamespace(input_ids=build_c4_ppl_input_ids(tokenizer, n_samples, seq_len, seed))
