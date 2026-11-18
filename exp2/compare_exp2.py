"""Compare Exp0 and Exp2 on val by default; explicitly opt into frozen test."""
import argparse
import gc
import json
from pathlib import Path
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import set_seed
from scoring import generate_calls, score
from evaluate import evaluate
from exp2 import utils as cfg
from exp2.utils import key, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--adapter", help="Explicit final Exp2 adapter path")
    parser.add_argument("--models", choices=("both", "exp0", "exp2"), default="both")
    args = parser.parse_args()
    test = load_from_disk(str(Path(cfg.DATA) / args.split))
    if not len(test):
        raise ValueError(f"{args.split} set is empty")
    adapter = Path(args.adapter) if args.adapter else Path(cfg.ROOT) / "runs/main/adapter-final"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing Exp2 adapter: {adapter}")
    dest = Path(cfg.ROOT) / "evaluation" / args.split
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Evaluation destination is nonempty; use a fresh DPO_OUTPUT_ROOT")
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in (("exp0", "exp2") if args.models == "both" else (args.models,)):
        model, tok = load_model()
        set_seed(cfg.SEED)
        if name == "exp2":
            base = model.unload()
            model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
            del base
        outputs = generate_calls(model, tok, test, max_new_tokens=200)
        if len(outputs) != len(test):
            raise ValueError(f"{name}: expected {len(test)} predictions, got {len(outputs)}")
        rows = []
        for i, (row, text) in enumerate(zip(test, outputs)):
            gold = json.loads(row["completion"][0]["content"])
            rows.append(dict(id=i, prediction=text, gold=gold, scores=score(text, gold)))
        summary[name] = evaluate(outputs, test)
        (dest / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect(); torch.cuda.empty_cache()
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = dict(split=args.split, data_path=str((Path(cfg.DATA) / args.split).resolve()),
                    data_n=len(test), data_sha256=key(list(test)), decoding="greedy",
                    max_new_tokens=200, exp0_adapter=str(Path(cfg.ADAPTER).resolve()),
                    exp2_adapter=str(adapter.resolve()))
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
