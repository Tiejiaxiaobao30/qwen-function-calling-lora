# Qwen2.5-1.5B Function-Calling LoRA

Fine-tuning a small open decoder-only model (**Qwen2.5-1.5B-Instruct**) with **QLoRA**
to reliably generate structured function/tool calls, evaluated with an objective,
LLM-judge-free metric. Runs end to end on a single GPU and fits free-tier Colab (T4).

**Published adapter:** https://huggingface.co/Arunark/qwen2.5-1.5b-function-calling-lora

## Results

Held-out test set, greedy decoding, strict BFCL-style AST match — function name **and**
arguments compared as parsed JSON structures, order-insensitive. No LLM judge.

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| JSON validity | 74.5% | 99.4% | +24.8 |
| Function-name accuracy | 43.5% | 98.4% | +54.9 |
| **Exact-match (name + args)** | **31.1%** | **78.8%** | **+47.7** |

Fine-tuning essentially solved output format (JSON validity → 99%) and tool selection
(name accuracy → 98%). Of the residual ~21% exact-match misses, ~90% are argument-level
(wrong API values or keys); type-coercion artifacts (int vs. string) account for only
0.2%, so the strict metric is faithful.

## Method in brief

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ungated).
- **Data:** `Salesforce/xlam-function-calling-60k` — cleaned (drop rows whose target
  calls a tool not in the offered set), reformatted so tool schemas sit in the system
  message and the target is a raw JSON call list, length-filtered, split 5000/500/500.
- **Adapter:** QLoRA (4-bit NF4), LoRA r=16, α=32, dropout=0.05, on all attention +
  MLP projections. Completion-only loss (train on the JSON call, not the tool schemas).
- **Eval:** strict AST match decomposed into JSON validity → function-name accuracy →
  exact-match, computed with `json.loads` + structural comparison. Base and tuned models
  run through the identical prompt + greedy-decode path, so the adapter is the only
  variable.

## Quickstart

```bash
pip install -r requirements.txt          # on Colab, then RESTART the runtime
huggingface-cli login                     # paste a token
# Accept the dataset terms once, in a browser, on the same account:
#   https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

# Optional: persist artifacts to Google Drive on Colab
export OUTPUT_ROOT=/content/drive/MyDrive/llm-fc-finetune

python prepare_data.py     # load, clean, format, split  ->  outputs/data/
python train.py            # QLoRA fine-tune             ->  outputs/adapter-final/
python evaluate.py         # before/after table          ->  outputs/eval_results.json
python error_analysis.py   # categorize misses           ->  outputs/error_analysis.json

export HF_USERNAME=your-username
python publish.py          # build model card + push adapter to the Hub
```

## Repo layout

| File | Purpose |
|---|---|
| `config.py` | Every path and hyperparameter — the single source of truth. |
| `prepare_data.py` | Load, clean, reformat, length-filter, split the dataset. |
| `train.py` | Load base in 4-bit, attach LoRA, train, save the adapter. |
| `scoring.py` | JSON extraction, strict AST-match scoring, greedy generation. |
| `evaluate.py` | Base vs. tuned before/after table on the test set. |
| `error_analysis.py` | Categorize every exact-match miss; type-coercion check. |
| `publish.py` | Build the model card and push to the HuggingFace Hub. |

`outputs/` (data, checkpoints, adapter) is git-ignored — the xLAM splits are gated and
research-only and must not be redistributed; the pipeline regenerates them.

## Notes learned along the way

A few things that quietly break on the current `transformers` 5.x / Colab stack, now
handled in the code:

- `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not a flat token list —
  `len()` on it counts keys, not tokens. Extract `input_ids` explicitly (`prepare_data.py`,
  `scoring.py`).
- `save_total_limit` prunes checkpoints by recency, not by eval score, so it can delete the
  best one. Pair it with `load_best_model_at_end=True` (`train.py`).
- Colab doesn't render carriage-return (`\r`) progress — long loops look frozen. Use
  newline prints with `flush=True` (`scoring.py`).

## Reproducibility

Validated against the pinned versions in `requirements.txt`. Exact metrics vary by ~1%
run to run (GPU non-determinism, library patch versions); the methodology and the size of
the before/after gap reproduce. The published adapter was trained under the same pipeline.

## Licensing

Base model Apache-2.0. Training data (`xlam-function-calling-60k`) is research/non-commercial;
the adapter is released under **CC-BY-NC-4.0** to respect that. This is a
research/demonstration artifact.
