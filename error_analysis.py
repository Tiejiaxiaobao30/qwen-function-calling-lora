"""Stage 4 -- Error analysis.

Runs the tuned model over the test set and categorizes every exact-match miss
(wrong function, wrong arg keys, wrong arg values, etc.). Also reports a
type-insensitive score, so you can see how much of the "error" is just int-vs-string
formatting rather than genuine mistakes. Writes error_analysis.json.

Run:  python error_analysis.py
"""
import json
import os
from collections import Counter # 计数工具
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config
from scoring import extract_json, generate_calls, score

# 判断错误原因
def diagnose(pred_text, target):
    """Classify why a single prediction missed exact-match."""
    pred = extract_json(pred_text)
    if pred is None:
        return "invalid_json"
    pred_names = [c.get("name") for c in pred if isinstance(c, dict)]
    target_names = [c.get("name") for c in target]
    if set(pred_names) != set(target_names):
        return "wrong_call_count" if len(pred) != len(target) else "wrong_function_name"
    # Function names match -> the miss is in the arguments.
    for pc in pred:
        tc = next((t for t in target if t.get("name") == pc.get("name")), None)
        if tc is None:
            continue
        pa, ta = pc.get("arguments", {}), tc.get("arguments", {})
        if set(pa.keys()) != set(ta.keys()):
            return "wrong_arg_keys"
        if pa != ta:
            return "wrong_arg_values"
    return "other"

# 增加了一个宽松匹配指标
def norm_loose(c): 
    """Like norm_call, but coerces every argument value to str -- so 1 and "1" match.
    Used to measure how much error is pure type-coercion noise."""
    if not isinstance(c, dict):
        return None
    args = {k: str(v) for k, v in c.get("arguments", {}).items()} # 将参数值统一转换成字符串
    return (c.get("name"), json.dumps(args, sort_keys=True))


def main():
    test = load_from_disk(os.path.join(config.DATA_DIR, "test"))
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, 
        bnb_4bit_use_double_quant=True
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, 
        quantization_config=bnb, 
        dtype=compute_dtype, 
        device_map="auto"
    )
    tuned = PeftModel.from_pretrained(base, config.ADAPTER_DIR) # 只需要分析 Tuned 模型

    print("Generating tuned outputs over test set...")
    outs = generate_calls(tuned, tok, test) # 重新推理

    cats = Counter()
    strict = loose = 0 # 严格匹配成功和宽松匹配成功的数量
    examples = []      # 保存部分错误样例
    for text, row in zip(outs, test):
        target = json.loads(row["completion"][0]["content"]) # 获取标准答案
        _, _, exact = score(text, target) # 只需要第三个
        if exact:
            strict += 1
        else:
            cat = diagnose(text, target)
            cats[cat] += 1
            if len(examples) < 6: # 如果目前保存的错误样例少于 6 条，就继续保存
                examples.append({
                    "query": row["prompt"][1]["content"][:90],
                    "target": target,
                    "pred": text[:160],
                    "category": cat
                })
        pred = extract_json(text)
        if pred is not None and {norm_loose(c) for c in pred} == {norm_loose(c) for c in target}:
            loose += 1

    k = len(test)
    total_miss = sum(cats.values()) # 所有失败样本的总数量
    # 打印严格匹配率
    print(f"\nExact-match:{strict}/{k} = {strict / k:.1%}")
    # 打印宽松匹配率
    print(f"Type-insensitive match: {loose / k:.1%}  (coercion gap {loose / k - strict / k:+.1%})")
    # 打印错误总量
    print(f"\nMisses: {total_miss}/{k} ({total_miss / k:.1%})")
    # 将错误数量按照出现次数从高到低排列
    for cat, n in cats.most_common():
        print(f"  {cat:<20} {n:>3}  ({n / total_miss:.0%} of misses)")

    analysis = {
        "test_n": k,
        "strict_exact_match": strict / k, # 严格匹配率
        "lenient_exact_match": loose / k, # 宽松匹配率
        "type_coercion_gap": loose / k - strict / k, # 类型转换造成的匹配率差距
        "total_misses": total_miss, # 失败样本总数
        "breakdown": dict(cats), # 不同错误类型的数量
        "sample_misses": examples, # 最多六条错误样例
    }
    with open(config.ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nSaved", config.ANALYSIS_PATH)


if __name__ == "__main__":
    main()
