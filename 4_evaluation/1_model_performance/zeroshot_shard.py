"""Run ONE arm on a SUBSET of zeroshot.py's tasks, reusing zeroshot.run_arm unchanged, so a single arm's
6-task eval can be split across GPUs. Writes per-task scores to a shard CSV; merge with --merge.
Bypasses zeroshot.py's MX_ONLY_ARMS guard on purpose (used to run m2xfp at nvfp4 as our extension).

Run:   python3 zeroshot_shard.py <fmt> <model> <arm> <task1,task2,...>
Merge: python3 zeroshot_shard.py --merge <fmt> <model> <arm> <label>
"""
import csv
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zeroshot as Z  # noqa: E402

SHARD_DIR = os.path.join(Z.RESULTS_DIR, "shards")


def run(fmt, model_key, arm, tasks):
    os.makedirs(SHARD_DIR, exist_ok=True)
    Z.TASKS = tasks  # run_arm passes the module-level TASKS to lm_eval
    batch_size = int(os.environ.get("ZEROSHOT_BATCH_SIZE", Z.BATCH_SIZE[model_key]))
    cache = Z.W.make_cache_provider(fmt, model_key)(Z.W.cache_kind(arm))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Z.W.model_name_of(model_key), trust_remote_code=True)
    t0 = time.time()
    scores = Z.run_arm(model_key, arm, fmt, cache, tok, None, batch_size)
    print(f"[shard][{fmt}][{model_key}][{arm}] " + "  ".join(f"{t}={100*v:.2f}" for t, v in scores.items())
          + f"  elapsed={time.time() - t0:.1f}s", flush=True)
    out = os.path.join(SHARD_DIR, f"{fmt}_{model_key}_{arm}_{'+'.join(tasks)}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "score"])
        for t, v in scores.items():
            w.writerow([t, v])
    print(f"Saved -> {out}", flush=True)


def merge(fmt, model_key, arm, label):
    scores = {}
    for path in glob.glob(os.path.join(SHARD_DIR, f"{fmt}_{model_key}_{arm}_*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                scores[r["task"]] = float(r["score"])
    missing = [t for t in Z.TASKS if t not in scores]
    if missing:
        sys.exit(f"missing tasks for {fmt}/{model_key}/{arm}: {missing}")
    row = {"format": fmt, "model": model_key, "setting": label, **{t: scores[t] for t in Z.TASKS}}
    row["avg"] = sum(scores[t] for t in Z.TASKS) / len(Z.TASKS)
    csv_path = os.path.join(Z.RESULTS_DIR, f"zeroshot_{fmt}_{model_key}.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if r["setting"] != label]
    rows.append(row)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=Z.FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{label}: avg={100*row['avg']:.2f}  -> {csv_path}")


if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge(*sys.argv[2:6])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","))
