# Crescent

Code and results for *Crescent: Accurate and Efficient 4-Bit LLM Inference Through Selective Mantissa
Extension* (ISCA 2027 submission). Crescent extends the mantissa of the activation elements with the
largest quantization error in each group from E2M1 to E2M3, under NVFP4 and MXFP4.

Settings in Section 4 (code name / CSV label): `bf16` / BF16, `native` / Native, `magnitude` / Magnitude,
`all_e2m3` / All_E2M3, `crescent` / Crescent. They are defined once, in `common/ppl_w4a4.py`.

## Layout

| Directory | Paper |
|---|---|
| `2_motivation/fig2_representation_constraints/` | Figure 2 |
| `2_motivation/fig3_error_concentration/` | Figure 3 |
| `2_motivation/fig4_magnitude_vs_error/` | Figure 4 |
| `2_motivation/fig5_mantissa_bitwidth/` | Figure 5 |
| `2_motivation/table1_selective_extension/` | Table 1 |
| `4_evaluation/table6_zeroshot/` | Table 6 |
| `4_evaluation/table7_perplexity/` | Table 7 |
| `4_evaluation/fig11_gemm_error/` | Figure 11 |
| `4_evaluation/fig12_speedup/` | Figure 12 |
| `weight_quantization/` | GPTQ weight caches used by Section 4 |
| `common/` | Shared quantization formats, model patches, data loading |

Every directory has a `results/` folder with the CSVs behind the paper's numbers. Each script states its
usage and output in its docstring. Scripts can be run from any working directory.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...   # meta-llama/Llama-3.1-8B is gated: request access on Hugging Face first
```

Models (downloaded from Hugging Face on first use):

| Key | Model | GPU memory (BF16) |
|---|---|---|
| `llama_base` | `meta-llama/Llama-3.1-8B` | 1 x 24 GB |
| `qwen3_moe` | `Qwen/Qwen3-30B-A3B` | ~61 GB, spread over the visible GPUs (`device_map="auto"`) |
| `deepseek_moe_16b` | `deepseek-ai/deepseek-moe-16b-base` | ~33 GB, spread over the visible GPUs |

Datasets: WikiText-2 (`wikitext-2-raw-v1`) and the first C4 validation shard, both through `datasets`.
Zero-shot tasks are run through `lm_eval`.

## Section 2: Motivation

No weight quantization (weights stay BF16). Formats: `nvfp4` or `mxfp4`.

Figures 2, 3 and 5 use the same three scripts per directory: `laplace.py` (Laplace model, CPU),
`dense.py` (Llama), `moe.py` (Qwen, DeepSeek).

```bash
cd 2_motivation/fig2_representation_constraints   # likewise fig3_error_concentration, fig5_mantissa_bitwidth
python3 laplace.py nvfp4
python3 dense.py nvfp4                     # llama_base
python3 moe.py nvfp4 qwen3_moe
python3 moe.py nvfp4 deepseek_moe_16b
```

Figure 4 computes both formats in one pass, so it takes no format argument:

```bash
cd 2_motivation/fig4_magnitude_vs_error
python3 laplace.py
python3 dense.py
python3 moe.py qwen3_moe
python3 moe.py deepseek_moe_16b
python3 merge_results.py                   # -> results/real_summary.csv
```

Table 1 (BF16, Native and Selective (E2M3) rows in one run):

```bash
python3 2_motivation/table1_selective_extension/ppl.py nvfp4 llama_base
```

## Section 4: Evaluation

### 1. GPTQ weight caches

Section 4 quantizes weights with GPTQ. Build the cache once per format and model; it is written to
`$PTQ_CACHE_ROOT/{fmt}/{model}/ptq_cache_NVFP4.pt` (default `PTQ_CACHE_ROOT=~/ptq_cache`; the file name
says NVFP4 for both formats).

```bash
                python3 weight_quantization/weight_gptq.py nvfp4 llama_base
GPTQ_CPU=0      python3 weight_quantization/weight_gptq_qwen3_moe.py nvfp4
                python3 weight_quantization/weight_gptq_deepseek_moe.py nvfp4
```

### 2. Table 7 (perplexity)

Each script runs all five settings and writes one CSV per format and model to
`4_evaluation/table7_perplexity/results/` (`ppl_wikitext2_{fmt}_{model}.csv`, `ppl_c4_{fmt}_{model}.csv`).

```bash
python3 4_evaluation/table7_perplexity/ppl_wikitext2.py nvfp4 llama_base
python3 4_evaluation/table7_perplexity/ppl_c4.py nvfp4 llama_base
```

### 3. Table 6 (zero-shot accuracy)

```bash
python3 4_evaluation/table6_zeroshot/zeroshot.py nvfp4 llama_base   # -> results/zeroshot_{fmt}_{model}.csv
```

All Section 4 scripts resume: settings already in the output CSV are skipped.

### 4. Figure 11 (GEMM output error)

```bash
python3 4_evaluation/fig11_gemm_error/gemm_error_llama.py nvfp4      # also gemm_error_qwen.py, gemm_error_deepseek.py
python3 4_evaluation/fig11_gemm_error/plot_gemm_error.py
```

### 5. Figure 12 (speedup)

Measured on an RTX 5090 (CUDA microbenchmarks with a patched CUTLASS), plus Accel-Sim for the cost of the
extra MMA operand. See `4_evaluation/fig12_speedup/README.md`.

```bash
4_evaluation/fig12_speedup/runpod_5090/assemble_5090.sh <measured results dir>   # -> results/figure12_speedup.csv
```
