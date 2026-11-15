"""Stage 3 -- Before/after evaluation.

Loads the base model once, evaluates it on the held-out test set, then attaches the
trained adapter to the SAME base object and evaluates again -- so the adapter is the
only difference between the two numbers. Both use identical prompts and greedy
decoding. Prints the comparison table and writes eval_results.json.

Run:  python evaluate.py
"""
import json
import os

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from scoring import generate_calls, score


def build_bnb():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    return bnb, compute_dtype


def evaluate(outs, dataset):
    valid = names = exact = 0
    for text, row in zip(outs, dataset):
        target = json.loads(row["completion"][0]["content"])
        v, n, e = score(text, target)
        valid += v; names += n; exact += e
    k = len(dataset)
    return {"json_valid": valid / k, "name_acc": names / k, "exact_match": exact / k}


def main():
    test = load_from_disk(os.path.join(config.DATA_DIR, "test"))
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb, compute_dtype = build_bnb()

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, quantization_config=bnb, dtype=compute_dtype, device_map="auto")

    print("Base pass:")
    base_scores = evaluate(generate_calls(base, tok, test), test)

    print("Attaching adapter...")
    tuned = PeftModel.from_pretrained(base, config.ADAPTER_DIR)   # same base weights + adapter
    print("Tuned pass:")
    tuned_scores = evaluate(generate_calls(tuned, tok, test), test)

    print(f"\n{'metric':<16}{'base':>10}{'tuned':>10}{'delta':>10}")
    for m in ["json_valid", "name_acc", "exact_match"]:
        b, t = base_scores[m], tuned_scores[m]
        print(f"{m:<16}{b:>10.1%}{t:>10.1%}{t - b:>+10.1%}")

    results = {
        "base": base_scores, "tuned": tuned_scores,
        "model": config.BASE_MODEL, "adapter": config.ADAPTER_DIR,
        "test_n": len(test), "decoding": "greedy", "metric": "BFCL-style AST match",
    }
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved", config.RESULTS_PATH)


if __name__ == "__main__":
    main()
