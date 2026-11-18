"""Choose sampled-correct versus wrong responses; split by query group."""
import json
import random
from collections import Counter
from pathlib import Path
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from scoring import score
from exp2 import utils as cfg
from exp2.utils import calls, canonical, key


def group(prompt):
    return key(" ".join(prompt[-1]["content"].lower().split()))


def main():
    root = Path(cfg.ROOT)
    source = load_from_disk(str(Path(cfg.DATA) / "train"))
    blocked = {group(row["prompt"]) for split in ("val", "test")
               for row in load_from_disk(str(Path(cfg.DATA) / split))}
    tok = AutoTokenizer.from_pretrained(cfg.ADAPTER)
    result = {"train": [], "val": []}; counts = Counter(); seen = set()
    pairs = []
    for line in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line); prompt = row["prompt"]; gid = group(prompt)
        counts["sources"] += 1
        sid = row["source_id"]
        if type(sid) is not int or not 0 <= sid < len(source) or prompt != source[sid]["prompt"] or row["gold"] != source[sid]["completion"] or row["prompt_hash"] != key(prompt):
            raise ValueError(f"Candidate/source mismatch: {sid}")
        if gid in blocked or row["prompt_hash"] in seen:
            counts["duplicate_or_overlap"] += 1; continue
        seen.add(row["prompt_hash"])
        gold = calls(row["gold"][0]["content"])
        if gold is None:
            counts["invalid_gold"] += 1; continue
        correct, wrong = [], []
        for text in dict.fromkeys(row["candidates"]):
            counts["unique_candidates"] += 1
            parsed = calls(text)
            if parsed is None:  # Keep semantic negatives rather than JSON/truncation failures.
                counts["invalid_candidate"] += 1
                continue
            exact = score(text, gold)[2]
            if bool(exact) != (canonical(parsed) == canonical(gold)):
                counts["canonical_score_disagreement"] += 1
            (correct if exact else wrong).append(text)
            counts["correct_candidate" if exact else "wrong_candidate"] += 1
        if not correct or not wrong:
            counts["no_sampled_pair"] += 1; continue
        pair = dict(prompt=prompt, chosen=[dict(role="assistant", content=correct[0])],
                    rejected=[dict(role="assistant", content=wrong[0])])
        lengths = [len(tok.apply_chat_template(prompt + pair[k], tokenize=True,
                    return_dict=False)) for k in ("chosen", "rejected")]
        if max(lengths) > cfg.MAX_LENGTH:
            counts["too_long"] += 1; continue
        pairs.append((gid, sid, pair))
    groups = sorted({gid for gid, _, _ in pairs})
    if len(groups) < 2:
        raise ValueError("Need at least two valid query groups; generate more candidates")
    random.Random(cfg.SEED).shuffle(groups)
    val_groups = set(groups[:max(1, len(groups) // 10)])
    for gid, _, pair in pairs:
        split = "val" if gid in val_groups else "train"
        result[split].append(pair); counts[split] += 1
    counts["valid_query_groups"] = len(groups)
    counts["val_query_groups"] = len(val_groups)
    dest = root / "preferences"; dest.mkdir(exist_ok=True)
    if any((dest / split).exists() for split in result):
        raise FileExistsError("Preference splits already exist; keep them frozen or use a new output root")
    for split, rows in result.items():
        if not rows:
            raise ValueError(f"No {split} pairs; generate more source candidates first")
        Dataset.from_list(rows).save_to_disk(str(dest / split))
    (dest / "audit.json").write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")
    preview = [dict(source_id=sid, query_group=gid,
                    split="val" if gid in val_groups else "train", **pair)
               for gid, sid, pair in pairs[:10]]
    (dest / "preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    main()
