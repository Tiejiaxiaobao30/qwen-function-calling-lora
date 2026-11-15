"""Stage 1 -- Data preparation.

Loads the xLAM function-calling dataset, drops rows whose target calls a tool that
isn't offered, reformats each row into a (prompt, completion) chat pair, filters out
any example that would exceed MAX_LEN tokens, then shuffles and splits into
train/val/test on disk.

Run:  python prepare_data.py
Prereq: accept the dataset terms on its HF page and `huggingface-cli login`.
"""
import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import config


def is_valid(row):
    """Drop rows that don't parse or whose answers reference a tool not in `tools`."""
    try:
        tools = json.loads(row["tools"])
        answers = json.loads(row["answers"])
    except Exception:
        return False
    names = {t.get("name") for t in tools}
    return len(answers) > 0 and all(a.get("name") in names for a in answers)


def to_chat(row):
    """Turn one raw row into a (prompt, completion) chat pair.

    The tool schemas go in the system message; the target is a raw JSON list of calls.
    Keeping prompt and completion separate lets the trainer compute loss on the
    completion only (see train.py, completion_only_loss=True).
    """
    tools = json.loads(row["tools"])
    answers = json.loads(row["answers"])
    return {
        "prompt": [
            {"role": "system", "content": config.SYS_PROMPT + json.dumps(tools, ensure_ascii=False)},
            {"role": "user", "content": row["query"]},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(answers, ensure_ascii=False)},
        ],
    }


def token_length(tok, messages):
    """Length of the fully-templated conversation in tokens.

    NOTE: in transformers 5.x, apply_chat_template(tokenize=True) returns a
    BatchEncoding (dict-like), NOT a flat list -- calling len() on it counts keys,
    not tokens. Extract input_ids explicitly.
    """
    out = tok.apply_chat_template(messages, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], list):   # guard against a batched/nested return
        ids = ids[0]
    return len(ids)


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    print(f"Loading {config.DATASET} ...")
    raw = load_dataset(config.DATASET, split="train")

    clean = raw.filter(is_valid)
    print(f"Cleaned: {len(raw)} -> {len(clean)} (dropped {len(raw) - len(clean)} invalid rows)")

    chat = clean.map(to_chat, remove_columns=clean.column_names)

    # Filter by token length BEFORE splitting so the splits come out to exact sizes.
    chat = chat.map(lambda ex: {"n_tokens": token_length(tok, ex["prompt"] + ex["completion"])})
    before = len(chat)
    chat = chat.filter(lambda ex: ex["n_tokens"] <= config.MAX_LEN)
    lens = np.array(chat["n_tokens"])
    print(f"Length filter (<= {config.MAX_LEN} tok): kept {len(chat)}/{before} | "
          f"p50={int(np.percentile(lens, 50))} "
          f"p95={int(np.percentile(lens, 95))} max={int(lens.max())}")

    chat = chat.remove_columns(["n_tokens"]).shuffle(seed=config.SEED)
    total = config.N_TRAIN + config.N_VAL + config.N_TEST
    assert len(chat) >= total, f"Only {len(chat)} valid rows; need {total} for the split."
    subset = chat.select(range(total))

    splits = {
        "train": subset.select(range(config.N_TRAIN)),
        "val":   subset.select(range(config.N_TRAIN, config.N_TRAIN + config.N_VAL)),
        "test":  subset.select(range(config.N_TRAIN + config.N_VAL, total)),
    }
    for name, d in splits.items():
        d.save_to_disk(os.path.join(config.DATA_DIR, name))
        print(f"  {name}: {len(d)}")

    # Record provenance so downstream scripts (and future sessions) know what built these.
    meta = {
        "base_model": config.BASE_MODEL, "dataset": config.DATASET,
        "max_len": config.MAX_LEN, "seed": config.SEED,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
    }
    with open(os.path.join(config.DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved splits + meta.json to", config.DATA_DIR)


if __name__ == "__main__":
    main()
