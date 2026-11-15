"""Stage 1 -- Data preparation.

Loads the xLAM function-calling dataset, drops rows whose target calls a tool that
isn't offered, reformats each row into a (prompt, completion) chat pair, filters out
any example that would exceed MAX_LEN tokens, then shuffles and splits into
train/val/test on disk.

Run:  python prepare_data.py
Prereq: accept the dataset terms on its HF page and `huggingface-cli login`.
"""
import json
import os   # 管理目录和文件路径

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

import config  # 导入刚刚读过的统一配置文件


def is_valid(row):
    """Drop rows that don't parse or whose answers reference a tool not in `tools`."""
    try:  # 尝试执行 try 中的代码，如果发生匹配的异常，就执行 except 中的代码
        tools = json.loads(row["tools"])  # json.loads() 的作用是将 JSON 字符串转换为 Python 对象
        answers = json.loads(row["answers"])
    except Exception:   # Exception 是一个比较宽泛的异常类型，能够捕获多种常见运行时错误
        return False
    names = {t.get("name") for t in tools}
    return len(answers) > 0 and all(a.get("name") in names for a in answers)
# all() 是 Python 内置函数。它要求所有条件都为 True，最终才返回 True。


def to_chat(row):  # 把原始数据转换成 SFT 格式
    """Turn one raw row into a (prompt, completion) chat pair.

    The tool schemas go in the system message; the target is a raw JSON list of calls.
    Keeping prompt and completion separate lets the trainer compute loss on the
    completion only (see train.py, completion_only_loss=True).
    """
    # prompt:模型输入的上下文   completion:希望模型学习生成的标准答案
    # system:系统指令 user：用户输入 assistant：模型回复
    tools = json.loads(row["tools"])
    answers = json.loads(row["answers"])
    return {
        "prompt": [
            {"role": "system", "content": config.SYS_PROMPT + json.dumps(tools, ensure_ascii=False)}, 
            # json.dumps（）把工具列表序列化为 JSON 字符串，并拼接到系统提示词后面。
            {"role": "user", "content": row["query"]},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(answers, ensure_ascii=False)},
        ],
    }

# 计算整条聊天样本在模型对应聊天模板下的 token 长度  messages是 to_chat() 转换之后的聊天消息
def token_length(tok, messages):  
    """Length of the fully-templated conversation in tokens.

    NOTE: in transformers 5.x, apply_chat_template(tokenize=True) returns a
    BatchEncoding (dict-like), NOT a flat list -- calling len() on it counts keys,
    not tokens. Extract input_ids explicitly.
    """
    out = tok.apply_chat_template(messages, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out  
    # hasattr(out, "keys") 检查 out 对象是否具有 keys 属性 如果有 就 if 前的 否则 out
    # 因为 apply_chat_template() 可能返回：token ID 列表 或者 包含 input_ids 等字段的字典式对象
    # 是后者就取出 out["input_ids"]，前者则直接使用 out
    if ids and isinstance(ids[0], list):   # guard against a batched/nested return
        ids = ids[0]   # 如果非空 或者是一个嵌套列表 就取出一层
    return len(ids)


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    print(f"Loading {config.DATASET} ...")
    raw = load_dataset(config.DATASET, split="train") # 读取该数据集上游提供的 train 分区

    # filter() 方法：遍历数据集，对每条样本调用 is_valid()。
    clean = raw.filter(is_valid)
    print(f"Cleaned: {len(raw)} -> {len(clean)} (dropped {len(raw) - len(clean)} invalid rows)")
    # map() 的作用是：对数据集里的每条数据执行某个转换函数
    chat = clean.map(to_chat, remove_columns=clean.column_names) # 移除原始数据集自带的列

    # Filter by token length BEFORE splitting so the splits come out to exact sizes.
    # 给上面的chat增加一列 “n_tokens" 内容就是将已有的两列内容拼接起来 方便计算token length
    chat = chat.map(lambda ex: {"n_tokens": token_length(tok, ex["prompt"] + ex["completion"])})
    before = len(chat)
    chat = chat.filter(lambda ex: ex["n_tokens"] <= config.MAX_LEN) # 2048
    lens = np.array(chat["n_tokens"]) # 转换为 NumPy 数组
    print(f"Length filter (<= {config.MAX_LEN} tok): kept {len(chat)}/{before} | "
          f"p50={int(np.percentile(lens, 50))} "
          f"p95={int(np.percentile(lens, 95))} max={int(lens.max())}")

    chat = chat.remove_columns(["n_tokens"]).shuffle(seed=config.SEED) # 删除前面为了计算长度而临时加入的列
    total = config.N_TRAIN + config.N_VAL + config.N_TEST
    assert len(chat) >= total, f"Only {len(chat)} valid rows; need {total} for the split."
    subset = chat.select(range(total))  # 从打乱后的数据集中取出前 6000 条

    splits = {
        "train": subset.select(range(config.N_TRAIN)), # range(0, 5000)
        "val":   subset.select(range(config.N_TRAIN, config.N_TRAIN + config.N_VAL)), # range(5000, 5500)
        "test":  subset.select(range(config.N_TRAIN + config.N_VAL, total)), # range(5500, 6000)
    }
    for name, d in splits.items():  # 遍历键值对
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
