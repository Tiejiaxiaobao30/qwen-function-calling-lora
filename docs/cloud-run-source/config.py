"""Central configuration. Change values here; every script imports from this module.

All paths live under OUTPUT_ROOT. On Colab, point OUTPUT_ROOT at Google Drive so
artifacts survive session timeouts, e.g.:

    export OUTPUT_ROOT=/content/drive/MyDrive/llm-fc-finetune
"""
import os

# --- Paths -------------------------------------------------------------------
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

DATA_DIR       = os.path.join(OUTPUT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
ADAPTER_DIR    = os.path.join(OUTPUT_ROOT, "adapter-final")
RESULTS_PATH   = os.path.join(OUTPUT_ROOT, "eval_results.json")
ANALYSIS_PATH  = os.path.join(OUTPUT_ROOT, "error_analysis.json")

# --- Model & data ------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"            # ungated, Apache-2.0
DATASET    = "Salesforce/xlam-function-calling-60k"  # GATED + research-only: accept terms on HF first

N_TRAIN, N_VAL, N_TEST = 5000, 500, 500
SEED    = 42
MAX_LEN = 2048   # hard data ceiling: no saved example exceeds this many tokens

# System prompt that frames the task. The tool schemas are appended to it as JSON.
# The adapter is trained on THIS exact format; inference must use it too.
SYS_PROMPT = (
    "You are a function-calling assistant. Given the available tools, respond with "
    'ONLY a JSON list of calls, each {"name": <tool>, "arguments": {<args>}}. '
    "If nothing applies, respond with []. Tools:\n"
)

# --- LoRA / training hyperparameters ----------------------------------------
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

TRAIN_MAX_LENGTH = 1024    # training ceiling; covers p95 (~850) with margin, < MAX_LEN for speed
EPOCHS           = 3
BATCH_SIZE       = 8
GRAD_ACCUM       = 2       # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
LEARNING_RATE    = 2e-4    # ~10x a full-FT rate; standard for LoRA
WARMUP_RATIO     = 0.03

# --- Publishing --------------------------------------------------------------
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
REPO_NAME   = "qwen2.5-1.5b-function-calling-lora"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
