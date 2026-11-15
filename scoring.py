"""Evaluation primitives -- JSON extraction, strict AST-match scoring, and greedy
generation. Shared by evaluate.py and error_analysis.py. No LLM judge involved.

Run this file directly to self-test the scoring functions:  python scoring.py
"""
import json
import re  # 正则表达式，提取 JSON
import torch

# text是模型生成的原始文本
def extract_json(text):
    """Parse a model's raw output into a list of call dicts, or None if unparseable."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else [obj] 
    # 如果解析结果已经是列表，直接返回 如果不是列表，就将它包装成列表再返回
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

# 评分函数 
def score(pred_text, target_list):
    """Return (json_valid, name_correct, exact_match) as 0/1 flags for one example.

    - json_valid:   output parsed into JSON at all
    - name_correct: predicted set of function names == target set
    - exact_match:  names AND all arguments match (structural, order-insensitive)
    """
    pred = extract_json(pred_text) # 把模型输出转换为 Python 列表
    if pred is None:
        return (0, 0, 0)
    pred_names = {c.get("name") for c in pred if isinstance(c, dict)} # 提取预测中每个调用的工具名称
    target_names = {c.get("name") for c in target_list} # 提取标准答案中的工具名称
    name_ok = int(pred_names == target_names)  # int(True) 就是 1
    exact_ok = int({norm_call(c) for c in pred} == {norm_call(c) for c in target_list})
    return (1, name_ok, exact_ok)


@torch.no_grad() # Python 装饰器 作用是让整个函数在执行过程中关闭梯度计算 这里是在做模型推理
def generate_calls(model, tok, dataset, max_new_tokens=200, log_every=25):
    # 默认最多生成 200 个新 token，每处理 25 条打印一次进度
    """Greedy-decode one function-call completion per row. Deterministic and
    model-agnostic: base and tuned models run through this identical path, so the
    only variable between them is the adapter.
    """
    model.eval()
    outs, n = [], len(dataset)
    for i, row in enumerate(dataset): # 模型推理时只使用 prompt
        prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        #  add_generation_prompt=True 在输入末尾添加 assistant 的生成起始标记
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        # 对上面的prompt 正式进行 tokenizer 编码 add_special_tokens=False 不要再次自动添加特殊 token
        # enc 是一个包含 input_ids、attention_mask 等信息的 BatchEncoding 对象
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        # 不进行随机采样，采用贪心解码 每一步都选择当前概率最高的 token
        outs.append(tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        # 只提取新生成的部分 把 token IDs 转换回文本 解码时跳过特殊 token
        if (i + 1) % log_every == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)   # newline, not \r -- Colab hides carriage returns
    return outs


if __name__ == "__main__":
    assert score('[{"name":"f","arguments":{"a":1}}]', [{"name": "f", "arguments": {"a": 1}}]) == (1, 1, 1)
    assert score('[{"name":"f","arguments":{"a":2}}]', [{"name": "f", "arguments": {"a": 1}}]) == (1, 1, 0)
    assert score('not json', [{"name": "f", "arguments": {}}]) == (0, 0, 0)
    print("Scoring functions pass self-test.")
