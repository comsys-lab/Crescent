"""Merge per-model real_{model}.csv files (written to avoid a shared-file
race when models run in parallel on different GPUs) into one real_summary.csv."""
import csv
import glob
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

rows = []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "real_*.csv"))):
    if os.path.basename(path) == "real_summary.csv":
        continue
    with open(path) as f:
        rows.extend(csv.DictReader(f))

out_path = os.path.join(RESULTS_DIR, "real_summary.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "target", "format", "match_rate", "blockmax_error_contribution", "largest_error_contribution", "n_blocks"])
    w.writeheader()
    for row in rows:
        w.writerow(row)
print(f"Merged {len(rows)} rows -> {out_path}")
