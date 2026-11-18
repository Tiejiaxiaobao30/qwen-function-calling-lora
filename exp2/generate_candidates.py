"""Sample Exp0 responses from frozen train prompts; append one source at a time."""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_from_disk
from transformers import set_seed
import config
from exp2 import utils as cfg
from exp2.utils import key, load_model


SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.9,
                num_return_sequences=4, max_new_tokens=200)


def generation_metadata(data):
    adapter = Path(cfg.ADAPTER)
    files = sorted(p for p in adapter.rglob("*") if p.is_file()
                   and (p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}))
    if not (adapter / "adapter_config.json").is_file() or not any(
        p.name in {"adapter_model.safetensors", "adapter_model.bin"} for p in files
    ):
        raise FileNotFoundError(f"Incomplete Exp0 adapter: {adapter}")
    hashes = {}
    for p in files:
        with p.open("rb") as stream:
            hashes[p.relative_to(adapter).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(version=1, base_model=config.BASE_MODEL, adapter=str(adapter.resolve()),
                adapter_sha256=hashes, data=str(Path(cfg.DATA).resolve()),
                train_sha256=key(list(data)), train_n=len(data), seed=cfg.SEED,
                sampling=SAMPLING)


def prepare_resume(path, metadata, data):
    manifest = path.with_suffix(".meta.json")
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Candidate configuration changed; use a new DPO_OUTPUT_ROOT")
    elif path.exists() and path.stat().st_size:
        raise ValueError("Existing candidates lack metadata; use a new DPO_OUTPUT_ROOT")
    else:
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    done = set()
    if path.exists():
        raw = path.read_bytes()
        # Discard only an unfinished last write after interruption.
        if raw and not raw.endswith(b"\n"):
            raw = raw[:raw.rfind(b"\n") + 1]
        for line in raw.splitlines():
            row = json.loads(line)
            i = row["source_id"]
            if type(i) is not int or i < 0 or i >= len(data) or i in done:
                raise ValueError("Invalid or duplicate source_id in candidates")
            source = data[i]
            if row["prompt_hash"] != key(source["prompt"]) or row["prompt"] != source["prompt"] or row["gold"] != source["completion"]:
                raise ValueError(f"Candidate/source mismatch: {i}")
            if len(row["candidates"]) != SAMPLING["num_return_sequences"] or not all(isinstance(t, str) for t in row["candidates"]):
                raise ValueError(f"Incomplete candidates: {i}")
            done.add(i)
        if raw != path.read_bytes():
            path.write_bytes(raw)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    root = Path(cfg.ROOT); root.mkdir(parents=True, exist_ok=True)
    path = root / "candidates.jsonl"
    data = load_from_disk(str(Path(cfg.DATA) / "train"))
    done = prepare_resume(path, generation_metadata(data), data)
    limit = min(args.limit or len(data), len(data))
    if all(i in done for i in range(limit)):
        print("Requested sources already complete")
        return
    model, tok = load_model(); model.eval()
    with path.open("a", encoding="utf-8") as out:
        for i in range(limit):
            if i in done:
                continue
            row = data[i]; set_seed(cfg.SEED + i)
            prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            import torch
            with torch.no_grad():
                generated = model.generate(**enc, **SAMPLING, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(generated[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            record = dict(source_id=i, prompt_hash=key(row["prompt"]), prompt=row["prompt"],
                          gold=row["completion"], candidates=texts)
            out.write(json.dumps(record, ensure_ascii=False) + "\n"); out.flush()
            print(f"Source {i + 1}/{len(data)} saved", flush=True)


if __name__ == "__main__":
    main()
