"""Builds the paper-figure plots and the combined aggregate table from every
gemm_error_aggregate_{fmt}_{model}.csv in results/. No emojis, no decorative styling beyond
what's needed to read the figure; method names as in the paper."""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
METHOD_ORDER = ["Native FP4", "Magnitude", "Crescent", "All E2M3"]
MODEL_ORDER = ["llama_base", "qwen3_moe", "deepseek_moe_16b"]
MODEL_LABEL = {"llama_base": "Llama-3.1-8B", "qwen3_moe": "Qwen3-30B-A3B", "deepseek_moe_16b": "DeepSeek-MoE-16B"}


def load_all_aggregates():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "gemm_error_aggregate_*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                r["aggregate_relative_mse"] = float(r["aggregate_relative_mse"])
                r["reduction_rate_pct"] = float(r["reduction_rate_pct"])
                rows.append(r)
    return rows


def write_combined_table(rows):
    path = os.path.join(RESULTS_DIR, "gemm_error_combined_table.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Format", "GEMM Type"] + METHOD_ORDER)
        seen = {}
        for r in rows:
            key = (r["model"], r["format"], r["gemm_type"])
            seen.setdefault(key, {})[r["setting"]] = r["aggregate_relative_mse"]
        for model in MODEL_ORDER:
            for fmt in ["nvfp4", "mxfp4"]:
                for gtype in ["linear", "attention"]:
                    key = (model, fmt, gtype)
                    if key not in seen:
                        continue
                    vals = [f"{seen[key].get(m, float('nan')):.4e}" for m in METHOD_ORDER]
                    w.writerow([model, fmt.upper(), gtype.capitalize()] + vals)
    print(f"Saved combined table -> {path}")
    return path


def plot_bars(rows):
    by_model_fmt = {}
    for r in rows:
        key = (r["model"], r["format"])
        by_model_fmt.setdefault(key, []).append(r)

    for (model, fmt), group in by_model_fmt.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        gtypes = ["linear", "attention"]
        x = range(len(METHOD_ORDER))
        width = 0.35
        for gi, gtype in enumerate(gtypes):
            vals = []
            for m in METHOD_ORDER:
                match = [r for r in group if r["gemm_type"] == gtype and r["setting"] == m]
                vals.append(match[0]["aggregate_relative_mse"] if match else float("nan"))
            offs = [xi + (gi - 0.5) * width for xi in x]
            ax.bar(offs, vals, width=width, label=gtype.capitalize())
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(METHOD_ORDER, rotation=20)
        ax.set_ylabel("GEMM Output Relative MSE (log scale)")
        ax.set_title(f"{MODEL_LABEL.get(model, model)} -- {fmt.upper()}")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"gemm_error_plot_{fmt}_{model}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot -> {out_path}")


if __name__ == "__main__":
    rows = load_all_aggregates()
    if not rows:
        print("No aggregate CSVs found yet in results/ -- run the per-model scripts first.")
    else:
        write_combined_table(rows)
        plot_bars(rows)
