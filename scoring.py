"""Evaluation primitives -- JSON extraction, strict AST-match scoring, and greedy
generation. Shared by evaluate.py and error_analysis.py. No LLM judge involved.

Run this file directly to self-test the scoring functions:  python scoring.py
"""
import json
import re

import torch


def extract_json(text):
    """Parse a model's raw output into a list of call dicts, or None if unparseable."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)   # fallback: grab the outermost [...] span
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def norm_call(c):
    """Canonicalize one call so argument order doesn't affect equality."""
    if not isinstance(c, dict):
        return None
    return (c.get("name"), json.dumps(c.get("arguments", {}), sort_keys=True))


def score(pred_text, target_list):
    """Return (json_valid, name_correct, exact_match) as 0/1 flags for one example.

    - json_valid:   output parsed into JSON at all
    - name_correct: predicted set of function names == target set
    - exact_match:  names AND all arguments match (structural, order-insensitive)
    """
    pred = extract_json(pred_text)
    if pred is None:
        return (0, 0, 0)
    pred_names = {c.get("name") for c in pred if isinstance(c, dict)}
    target_names = {c.get("name") for c in target_list}
    name_ok = int(pred_names == target_names)
    exact_ok = int({norm_call(c) for c in pred} == {norm_call(c) for c in target_list})
    return (1, name_ok, exact_ok)


@torch.no_grad()
def generate_calls(model, tok, dataset, max_new_tokens=200, log_every=25):
    """Greedy-decode one function-call completion per row. Deterministic and
    model-agnostic: base and tuned models run through this identical path, so the
    only variable between them is the adapter.
    """
    model.eval()
    outs, n = [], len(dataset)
    for i, row in enumerate(dataset):
        prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs.append(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)   # newline, not \r -- Colab hides carriage returns
    return outs


if __name__ == "__main__":
    assert score('[{"name":"f","arguments":{"a":1}}]', [{"name": "f", "arguments": {"a": 1}}]) == (1, 1, 1)
    assert score('[{"name":"f","arguments":{"a":2}}]', [{"name": "f", "arguments": {"a": 1}}]) == (1, 1, 0)
    assert score('not json', [{"name": "f", "arguments": {}}]) == (0, 0, 0)
    print("Scoring functions pass self-test.")
