"""Batch candidate generation for Exp2 DPO."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import set_seed

from exp2 import utils as cfg
from exp2.utils import key, load_model
from exp2.generate_candidates import generation_metadata, prepare_resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    batch_size = args.batch_size
    if batch_size < 1:
        raise ValueError("batch-size must be positive")

    root = Path(cfg.ROOT)
    root.mkdir(parents=True, exist_ok=True)

    path = root / "candidates.jsonl"
    meta_path = root / "candidates.meta.json"
    batch_log = root / "candidate_batches.jsonl"

    # 保留原候选任务的元数据，不允许无元数据续做。
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing candidate metadata: {meta_path}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Invalid candidate metadata")

    meta_hash = hashlib.sha256(
        meta_path.read_bytes()
    ).hexdigest()

    # 完整核验当前数据、adapter/tokenizer、采样参数后才允许修复尾行。
    data = load_from_disk(str(Path(cfg.DATA) / "train"))
    if len(data) != 5000:
        raise ValueError("Expected 5000 Exp0 train samples")
    current = generation_metadata(data)
    if meta != current:
        raise ValueError("Candidate configuration changed; use a new DPO_OUTPUT_ROOT")
    done = prepare_resume(path, current, data)
    if not done:
        raise ValueError("No existing pilot candidates found")

    remaining = len(data) - len(done)

    print(f"Existing candidates: {len(done)}", flush=True)
    print(f"Remaining sources: {remaining}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)
    print(f"Candidates per source: 4", flush=True)

    if remaining == 0:
        print("All sources completed.")
        return

    # 加载 Exp0 Base + SFT Adapter。
    model, tok = load_model()
    model.eval()

    # Decoder-only 模型批量生成使用左侧 padding。
    tok.padding_side = "left"

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if tok.pad_token_id is None:
        raise ValueError("Tokenizer has no usable pad token")

    pad_id = tok.pad_token_id

    total_batches = sum(
        1 for start in range(0, len(data), batch_size)
        if any(
            i not in done
            for i in range(start, min(start + batch_size, len(data)))
        )
    )

    batch_no = 0
    start_time = time.time()

    # 按固定 source 索引窗口分组，保证断点续做顺序稳定。
    with path.open("a", encoding="utf-8") as out, \
         batch_log.open("a", encoding="utf-8") as log:

        for start in range(0, len(data), batch_size):

            indices = [
                i for i in range(start, min(start + batch_size, len(data)))
                if i not in done
            ]

            if not indices:
                continue

            batch_no += 1
            seed = cfg.SEED + start
            set_seed(seed)

            rows = [data[i] for i in indices]

            prompts = [
                tok.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for row in rows
            ]

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).to(model.device)

            input_width = enc.input_ids.shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=4,
                    max_new_tokens=200,
                    pad_token_id=pad_id
                )

            texts = tok.batch_decode(
                generated[:, input_width:],
                skip_special_tokens=True
            )

            if len(texts) != len(indices) * 4:
                raise ValueError(
                    f"Expected {len(indices)*4} outputs, got {len(texts)}"
                )

            # 每道题对应连续4条生成结果。
            for j, (sid, row) in enumerate(zip(indices, rows)):
                candidates = texts[j * 4:(j + 1) * 4]

                record = dict(
                    source_id=sid,
                    prompt_hash=key(row["prompt"]),
                    prompt=row["prompt"],
                    gold=row["completion"],
                    candidates=candidates
                )

                out.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                out.flush()
                done.add(sid)

            # 单独记录本批执行配置，不覆盖原始 pilot metadata。
            log.write(json.dumps({
                "source_ids": indices,
                "batch_size": len(indices),
                "requested_batch_size": batch_size,
                "seed": seed,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 4,
                "max_new_tokens": 200,
                "metadata_sha256": meta_hash
            }) + "\n")
            log.flush()

            print(
                f"Batch {batch_no}/{total_batches} | "
                f"Sources {indices[0]}-{indices[-1]} | "
                f"Completed {len(done)}/{len(data)} | "
                f"Elapsed {time.time()-start_time:.1f}s",
                flush=True
            )

            del enc, generated, texts

    print("Candidate generation completed.", flush=True)


if __name__ == "__main__":
    main()
