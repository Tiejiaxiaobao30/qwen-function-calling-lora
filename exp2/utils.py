"""Shared data and model helpers; keep Exp0 SFT as the only starting point."""
import hashlib
import json
import os
import config

# 复用原配置；只有 Exp2 专用输出路径在这里新增。
ROOT = os.environ.get("DPO_OUTPUT_ROOT", "exp2/local/new-run")
DATA = os.environ.get("SFT_DATA_DIR", config.DATA_DIR)
ADAPTER = os.environ.get("EXP0_ADAPTER_DIR", config.ADAPTER_DIR)
SEED = config.SEED
MAX_LENGTH = config.MAX_LEN


def key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def calls(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(value, list):
        return None

    if not all(isinstance(c, dict) and set(c) == {"name", "arguments"}
               and isinstance(c["name"], str) and isinstance(c["arguments"], dict) for c in value):
        return None
    return value


def canonical(value):
    return sorted(json.dumps(c, sort_keys=True, ensure_ascii=False) for c in value)


def load_model(trainable=False):
    # 复用现有 SFT 底座加载，只新增 Exp0 adapter 的继续训练逻辑。
    from train import load_quantized_base
    from peft import PeftModel, prepare_model_for_kbit_training
    model, tok = load_quantized_base()
    if trainable:
        model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=trainable)
    model.config.use_cache = not trainable
    return model, tok
