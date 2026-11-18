"""Exp1：用教师模型增强固定训练集，然后交给原 train.py 训练。"""
import argparse  # 从终端读取运行参数
from collections import Counter # 统计各种数据过滤数量
import json
import os # 读取环境变量
from pathlib import Path
import random
from urllib.request import Request, urlopen #发送 HTTP 请求，调用教师 API
from datasets import Dataset, concatenate_datasets, load_from_disk
# concatenate_datasets()：合并两个 Dataset
from transformers import AutoTokenizer
import config
from prepare_data import token_length
from exp1.augmentation_utils import (
    build_messages, from_chat, parse_json, query_key, to_chat, validate_candidate,
)

# 第一次真正调用教师模型 base_url:API 服务地址 timeout=60:请求超时时间，单位秒
def call_teacher(messages, model, base_url, api_key, temperature=0.7,
                 max_tokens=1024, timeout=60):
    """调用一次 OpenAI 兼容接口，返回教师生成的文本。"""
    payload = {
        "model": model,
        "messages": messages, # 教师模型收到的提示词
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False
    }
    # 创建 HTTP 请求
    request = Request(
        base_url.rstrip("/") + "/chat/completions", # 请求地址
        data=json.dumps(payload).encode("utf-8"),   # 请求数据
        headers={                                   # 请求头
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key},
    )
    # 实际调用 API
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8")) # 教师 API 返回的数据
    choice = result["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("教师输出被截断，请增大 --max-tokens")
    # 判断生成是否因为达到 Token 上限而被截断.如果是，就报错，不接受不完整的 JSON
    return choice["message"]["content"] # 提取教师模型真正生成的文本


def augment_seed(seed, teacher, mode="rewrite"):
    text = teacher(build_messages(seed, mode)) # 把种子样本转换为教师提示词，然后请求教师模型生成一条新数据
    row = validate_candidate(parse_json(text), seed["tools"], mode) # 将教师生成的文本解析成 Python 字典
    if mode == "rewrite" and row["answers"] != seed["answers"]: # 检查这条数据是否符合要求
        raise ValueError("改写模式不能改变原答案")
    return row

def source_info(splits):
    """记录固定划分身份；续跑和 Control 必须读取相同数据。"""
    return {name: {"rows": len(ds), "fingerprint": ds._fingerprint}
            for name, ds in splits.items()}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def save_event(progress, event):
    # 每次尝试独立保存，先写临时文件再改名；已保存的结果不覆盖。
    number = len(list(progress.glob("*.json")))
    temp = progress / "pending.tmp"
    with temp.open("w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temp.rename(progress / f"{number:06d}.json")


def save_split(ds, path):
    # 导出中断后，可以复用已完整保存且内容相同的 split。
    if path.exists():
        saved = load_from_disk(str(path))
        if len(saved) != len(ds) or any(a != b for a, b in zip(saved, ds)):
            raise ValueError(f"已有数据与待导出内容不同，拒绝覆盖：{path}")
        return
    temp = path.with_name(path.name + ".building")
    if temp.exists():
        raise ValueError(f"发现未完成的导出目录，请另存后移开再续跑：{temp}")
    ds.save_to_disk(str(temp))
    temp.rename(path)


def run(args, teacher):
    source, target = Path(args.source).resolve(), Path(args.output).resolve()
    control = Path(args.control_output).resolve()
    paths = (source, target, control)
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("源数据、Rewrite、Control 目录必须互相分开")
    if args.per_seed != 1 or args.limit < 1:
        raise ValueError("每个候选 seed 只尝试一次；limit 必须为正数")
    splits = {name: load_from_disk(str(source / name)) for name in ("train", "val", "test")}
    if args.limit > len(splits["train"]):
        raise ValueError("limit 不能超过训练集条数")
    indices = list(range(len(splits["train"])))
    random.Random(args.seed).shuffle(indices)
    manifest = {"source": str(source), "splits": source_info(splits),
                "seed": args.seed, "indices": indices[:args.limit]}
    if args.seed_indices or args.resume:
        path = Path(args.seed_indices) if args.seed_indices else target / "seed_indices.json"
        supplied = json.loads(path.read_text(encoding="utf-8"))
        indices = supplied["indices"]
        if (any(supplied.get(k) != manifest[k] for k in ("source", "splits", "seed"))
                or len(indices) != args.limit or any(type(i) is not int for i in indices)
                or len(set(indices)) != len(indices)
                or any(not 0 <= i < len(splits["train"]) for i in indices)):
            raise ValueError("候选清单与源数据、seed、limit 不一致或索引无效")
        manifest = supplied
    settings = {key: getattr(args, key) for key in
                ("model", "mode", "per_seed", "max_length", "temperature", "max_tokens")}
    settings["tokenizer"] = args.tokenizer
    if args.resume:
        saved = json.loads((target / "seed_indices.json").read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError("续跑来源、seed 或 limit 与原清单不一致")
        if json.loads((target / "settings.json").read_text(encoding="utf-8")) != settings:
            raise ValueError("续跑生成配置与原任务不一致")
    else:
        target.mkdir(parents=True, exist_ok=False)
        write_json(target / "seed_indices.json", manifest)
        write_json(target / "settings.json", settings)
        (target / "progress").mkdir()

    if control.exists() and not (target / "accepted_seed_indices.json").exists():
        raise ValueError("Control 输出已存在，请使用一个全新目录，避免混用旧实验")

    progress = target / "progress"
    events = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(progress.glob("*.json"))]
    rows = {event["index"]: event["row"] for event in events if event["status"] == "accepted"}
    if len(rows) != sum(event["status"] == "accepted" for event in events):
        raise ValueError("存在重复成功记录，请先检查旧进度")
    if any(event["index"] not in manifest["indices"] or event["status"] not in
           {"accepted", "invalid", "duplicate", "too_long", "request_error"} for event in events):
        raise ValueError("进度包含未知 seed 或状态")
    finished = {event["index"] for event in events if event["status"] != "request_error"}
    counts = Counter(event["status"] for event in events)
    counts["attempts"] = len(events)
    seen = {query_key(from_chat(row)["query"]) for ds in splits.values() for row in ds}
    seen.update(query_key(row["query"]) for row in rows.values())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    for index in manifest["indices"]:
        if index in finished:
            continue
        seed = from_chat(splits["train"][index])
        # 上限按每次启动计算；显式 --resume 表示继续尝试未完成的原 seed。
        # 当前规则：每个候选只尝试一次；不合格也算已处理，resume 不再请求它。
        event = {"index": index}
        try:
            row = augment_seed(seed, teacher, args.mode)
            key = query_key(row["query"])
            chat = to_chat(row)
            if key in seen:
                event["status"] = "duplicate"
            elif token_length(tokenizer, chat["prompt"] + chat["completion"]) > args.max_length:
                event["status"] = "too_long"
            else:
                event.update(status="accepted", row=row)
        except (ValueError, KeyError, TypeError, IndexError):
            event["status"] = "invalid"
        except Exception as exc:
            # HTTP/网络错误停止，不重试；不把可能含凭证的异常正文写入日志。
            event.update(status="request_error", error_type=type(exc).__name__,
                         http_status=getattr(exc, "code", None))
            save_event(progress, event)
            raise
        save_event(progress, event)
        counts["attempts"] += 1
        counts[event["status"]] += 1
        if event["status"] == "accepted":
            rows[index] = row
            seen.add(key)
        print(f"Processed seed {index}: {dict(counts)}")

    # 只有固定 seed 全部合格才导出，顺序与 Control 一致。
    # 当前按实际 accepted=N 导出；Control 只复制成功 seed 的原始样本。
    accepted = [index for index in manifest["indices"] if index in rows]
    raw_rows = [rows[index] for index in accepted]
    generated = (Dataset.from_list([to_chat(row) for row in raw_rows], features=splits["train"].features)
                 if raw_rows else splits["train"].select([], keep_in_memory=True))
    train = concatenate_datasets([splits["train"], generated])
    train = train.shuffle(seed=args.seed, keep_in_memory=True)
    save_split(train, target / "train")
    for name in ("val", "test"):
        save_split(splits[name], target / name)
    accepted_manifest = {"source": str(source), "splits": source_info(splits),
                         "seed": args.seed, "indices": accepted}
    if not (target / "accepted_seed_indices.json").exists():
        write_json(target / "accepted_seed_indices.json", accepted_manifest)
    elif json.loads((target / "accepted_seed_indices.json").read_text(encoding="utf-8")) != accepted_manifest:
        raise ValueError("已保存的成功清单与当前进度不一致")
    control.mkdir(parents=True, exist_ok=True)
    control_manifest = control / "seed_indices.json"
    if control_manifest.exists():
        if json.loads(control_manifest.read_text(encoding="utf-8")) != accepted_manifest:
            raise ValueError("已有 Control seed 清单不同，拒绝覆盖")
    else:
        write_json(control_manifest, accepted_manifest)
    extra = splits["train"].select(accepted, keep_in_memory=True)
    control_train = concatenate_datasets([splits["train"], extra]).shuffle(seed=args.seed, keep_in_memory=True)
    save_split(control_train, control / "train")
    for name in ("val", "test"):
        save_split(splits[name], control / name)
    if not (control / "meta.json").exists():
        write_json(control / "meta.json", {**accepted_manifest, "mode": "control", "complete": True,
                                           "train_rows": len(control_train), "accepted": len(accepted)})
    if not (target / "augmented.jsonl").exists():
        temp = target / "augmented.tmp"
        with temp.open("w", encoding="utf-8") as f:
            for row in raw_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        temp.rename(target / "augmented.jsonl")
    meta = {**manifest, **settings, "counts": dict(counts), "complete": True,
            "accepted_indices": accepted, "control_output": str(control),
            "policy": "one_attempt_skip_rejected", "semantic_validation": False}
    if not (target / "meta.json").exists():
        write_json(target / "meta.json", meta)
    print(f"Saved Rewrite/Control: accepted={len(accepted)}, train={len(train)}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")
    return meta

# 便于从终端灵活地控制实验参数
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.join(config.OUTPUT_ROOT, "data-redacted"))
    parser.add_argument("--output", default="exp1/local/new-rewrite/data")
    parser.add_argument("--control-output", default="exp1/local/new-control/data")
    parser.add_argument("--tokenizer", default=config.ADAPTER_DIR)
    parser.add_argument("--mode", choices=["rewrite", "hard", "parameters", "no-tool"], default="rewrite")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--per-seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-indices", help="Control 或其他同源任务保存的 seed_indices.json")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--max-length", type=int, default=min(config.MAX_LEN, config.TRAIN_MAX_LENGTH))
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not args.model:
        parser.error("请设置 TEACHER_MODEL 或传入 --model")
    return args


def main():
    args = parse_args() # 读取终端传入的参数
    # 读取教师 API 的地址和密钥
    base_url = os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("TEACHER_API_KEY", "")

    # 把之前配置好的模型名称、API 地址、密钥、温度等参数统一封装起来 即上面调用的调用教师模型teacher()
    def teacher(messages):
        return call_teacher(messages, args.model, base_url, api_key,
                            args.temperature, args.max_tokens, args.timeout)

    run(args, teacher)


if __name__ == "__main__":
    main()
