"""Exp1：转换样本、构造教师提示词、检查生成结果。"""
import json
import config
from prepare_data import to_chat as prepare_chat  # 重命名

# text是教师模型生成的数据 教师有时会把 JSON 放在 Markdown 代码块里
def parse_json(text):
    text = text.strip()
    if text.startswith("```"):  # 判断教师模型有没有使用 Markdown 代码块
        text = text.split("\n", 1)[1].rsplit("```", 1)[0] # 去掉 Markdown 代码块的开头和结尾，只保留中间的 JSON
    return json.loads(text) # 转化为python字典

# row含有prompt和completion
def from_chat(row):
    """把已划分好的训练样本还原为 query/tools/answers。"""
    return {
        "query": row["prompt"][1]["content"],
        "tools": json.loads(row["prompt"][0]["content"][len(config.SYS_PROMPT):]),
        "answers": json.loads(row["completion"][0]["content"]),
    }

def to_chat(row):
    # 复用原项目的格式转换，保证 train.py 可以直接读取。
    return prepare_chat({
        "query": row["query"],
        "tools": json.dumps(row["tools"], ensure_ascii=False),
        "answers": json.dumps(row["answers"], ensure_ascii=False),
    })

def query_key(query):
    return " ".join(query.split()).lower()

# candidata 是教师模型生成的新样本
def validate_candidate(candidate, tools, mode="rewrite"):
    """只检查基本结构、工具名和参数名，不做完整的类型系统校验。"""
    query, calls = candidate["query"], candidate["answers"]
    if not isinstance(query, str) or not query.strip() or not isinstance(calls, list):
    # query 是否非字符串 是否非空 calls是否非列表
        raise ValueError("query 应为非空字符串，answers 应为列表")
    if (mode == "no-tool") != (len(calls) == 0):
        raise ValueError("只有 no-tool 模式应返回空调用列表")

    functions = {tool["name"]: tool for tool in tools}
    for call in calls:
        name, arguments = call["name"], call["arguments"]
        if name not in functions or not isinstance(arguments, dict):
            raise ValueError("工具名不存在或 arguments 不是字典")
        parameters = functions[name]["parameters"]
        allowed = parameters.get("properties", parameters)
        if not set(arguments).issubset(allowed):
            raise ValueError("调用包含工具未定义的参数")
        # 教师生成的所有参数名，都必须出现在工具定义允许的参数名中
    return {"query": query.strip(), "tools": tools, "answers": calls}
    # 返回一条完整样本


def build_messages(seed, mode="rewrite"):
    instructions = {
        # 保持意思和答案不变，只改变用户问题的表达
        "rewrite": "Rewrite the query with different wording. Keep the meaning and answers exactly unchanged.",
        # 生成更复杂但明确的问题
        "hard": "Create a harder but unambiguous query using these tools. Include all necessary argument values in the query.",
        # 改变参数值，同时调整问题和答案
        "parameters": "Create a query with different explicit argument values and provide matching calls. Respect parameter types and constraints.",
        # 生成不需要任何可用工具的问题，答案为 []
        "no-tool": "Create a request unrelated to every available tool. Return answers as [].",
    }
    return [
        {"role": "system", "content": (
            "Generate one function-calling training example using the supplied tools. "
            "Do not change tools or invent arguments. "
            'Return only JSON: {"query": "...", "answers": [{"name": "...", "arguments": {}}]}. '
            + instructions[mode]
        )},
        {"role": "user", "content": json.dumps(seed, ensure_ascii=False)},
    ]
