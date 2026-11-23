"""Table 6: zero-shot accuracy of BF16, Native, Magnitude, All E2M3 and Crescent (settings defined in
common/ppl_w4a4.py), with GPTQ 4-bit weights for every quantized setting. lm-eval's HFLM wraps the same
patched model that the PPL scripts build.

Tasks (0-shot): arc_easy, arc_challenge, hellaswag, piqa -> acc_norm; lambada_openai, winogrande -> acc.
Avg. = mean of the six.

Usage: CUDA_VISIBLE_DEVICES=<GPUs> python3 zeroshot.py <nvfp4|mxfp4> <llama_base|qwen3_moe|deepseek_moe_16b> [--limit N]
  --limit N : only N examples per task (smoke test; the reported results use the full sets)
Env: ZEROSHOT_BATCH_SIZE (default per model, see BATCH_SIZE)
Output: results/zeroshot_{fmt}_{model}.csv (resumes: settings already in the file are skipped)
"""
import argparse
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))
from eval_utils import same_seeds, cleanup_cuda  # noqa: E402
import ppl_w4a4 as W  # noqa: E402

torch.manual_seed(1111)

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "lambada_openai", "winogrande"]
METRIC = {"arc_easy": "acc_norm", "arc_challenge": "acc_norm", "hellaswag": "acc_norm", "piqa": "acc_norm",
          "lambada_openai": "acc", "winogrande": "acc"}
ARMS, LABELS = W.ARMS, W.LABELS
BATCH_SIZE = {"llama_base": 16, "qwen3_moe": 24, "deepseek_moe_16b": 32}
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FIELDS = ["format", "model", "setting"] + TASKS + ["avg"]


def run_arm(model_key, arm, fmt, cache, tok, limit, batch_size):
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    class RobustHFLM(HFLM):
        """DeepSeek's tokenizer's vocab has entries that merge a context's trailing single-letter word with
        the continuation's leading word into ONE token (e.g. context "...from a" + continuation " can" ->
        the single token "acan", id 81871, instead of separate 'a' and ' can' tokens) for a handful of common
        short-word pairs. The base _encode_pair diffs encode(context+continuation) against encode(context) to
        isolate the continuation; for exactly these merges the diff comes out EMPTY (confirmed on the real
        datasets: 9 of 65719 loglikelihood requests across the 6 tasks, all lambada_openai), which
        HFLM._loglikelihood_tokens then rejects with AssertionError. Falls back to encoding the continuation
        independently (add_special_tokens=False) only in that exact situation -- every other request (all but
        those 9) is unaffected and still uses the exact joint-diff tokenization lm_eval expects."""
        def _encode_pair(self, context, continuation):
            context_enc, continuation_enc = super()._encode_pair(context, continuation)
            if len(continuation_enc) == 0:
                continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
            return context_enc, continuation_enc

    same_seeds()
    model, restore = W.build_model(model_key, arm, fmt, cache)

    # Qwen3-30B-A3B and the DeepSeek-MoE-16B port are both sharded across several GPUs by
    # accelerate's device_map="auto" (they don't fit one 24GB card). `model.device` (what HFLM
    # uses to place ITS OWN comparison tensors -- labels, greedy-token ids) is the FIRST
    # parameter's device (embed_tokens), but the actual output (lm_head's logits, what those
    # tensors get compared against) lands on whichever GPU accelerate put the LAST layer on --
    # a different one, on a multi-GPU split. Confirmed as the exact cause of a real run's crash
    # ("Expected all tensors to be on the same device, but found at least two devices, cuda:2
    # and cuda:0"). accelerate's per-submodule dispatch hooks move inputs to wherever each
    # submodule needs them regardless of what device the caller put them on, so overriding this
    # class-level property to report the LAST parameter's device instead is safe for the model's
    # own computation -- it only changes where lm_eval's bookkeeping tensors are placed, to match
    # where the logits it compares them against actually are. Single-GPU llama_base: no-op
    # (embed_tokens and lm_head are already on the same device).
    model_cls = type(model)
    orig_device_prop = model_cls.device
    # CPU/disk-offloaded params (accelerate device_map="auto" overflow) show up as device "meta"
    # (a placeholder -- the real tensor lives on cpu/disk behind an AlignDevicesHook); "meta" is not
    # a valid torch.autocast device_type, so skip those and use the last REAL (non-meta) device.
    output_device = next(p.device for p in reversed(list(model.parameters())) if p.device.type != "meta")
    model_cls.device = property(lambda self, d=output_device: d)

    # add_bos_token=True (explicit, not the tokenizer's own default): DeepSeek's tokenizer defaults to
    # add_bos_token=False, and without a BOS anchoring position 0, lm_eval's _encode_pair (which tokenizes
    # context and context+continuation separately and diffs the token-id lists to isolate the continuation)
    # hit AssertionError: len(continuation_enc) == 0 on a real run -- a BPE boundary-merge artifact at the
    # very start of a string with no anchor token before it. Llama/Qwen's tokenizers already default to
    # add_bos_token=True, so this is a no-op for them.
    lm = RobustHFLM(pretrained=model, tokenizer=tok, batch_size=batch_size, add_bos_token=True)
    try:
        res = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=limit, confirm_run_unsafe_code=True)
    finally:
        model_cls.device = orig_device_prop
        restore()
    scores = {t: res["results"][t][f"{METRIC[t]},none"] for t in TASKS}
    del model, lm
    cleanup_cuda()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmt", choices=["nvfp4", "mxfp4"])
    ap.add_argument("model_key", choices=["llama_base", "qwen3_moe", "deepseek_moe_16b"])
    ap.add_argument("--limit", type=int, default=None,
                    help="sanity-check only -- final paper numbers must be run WITHOUT --limit (full validation sets)")
    args = ap.parse_args()
    fmt, model_key = args.fmt, args.model_key
    batch_size = int(os.environ.get("ZEROSHOT_BATCH_SIZE", BATCH_SIZE[model_key]))

    suffix = f"_limit{args.limit}" if args.limit else ""
    csv_path = os.path.join(RESULTS_DIR, f"zeroshot_{fmt}_{model_key}{suffix}.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    done = {r["setting"] for r in rows}

    cache = None

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(W.model_name_of(model_key), trust_remote_code=True)

    for arm in ARMS:
        label = LABELS[arm]
        if label in done:
            print(f"[zeroshot][{fmt}][{model_key}] {label} already measured -- skipping", flush=True)
            continue
        t0 = time.time()
        if arm != "bf16" and cache is None:
            cache = W.load_gptq_cache(fmt, model_key)
        scores = run_arm(model_key, arm, fmt, cache, tok, args.limit, batch_size)
        avg = sum(scores.values()) / len(scores)
        elapsed = time.time() - t0
        print(f"[zeroshot][{fmt}][{model_key}] {label}: " + "  ".join(f"{t}={100*v:.2f}" for t, v in scores.items())
              + f"  avg={100*avg:.2f}  elapsed={elapsed:.1f}s", flush=True)
        row = {"format": fmt, "model": model_key, "setting": label, "avg": avg, **scores}
        rows.append(row)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
