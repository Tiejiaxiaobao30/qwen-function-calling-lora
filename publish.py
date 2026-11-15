"""Stage 5 -- Publish to the HuggingFace Hub.

Builds a model card from eval_results.json + error_analysis.json + config, writes it
into the adapter folder, and pushes the adapter (weights, config, tokenizer, card) to
a public repo. Set HF_USERNAME first, e.g.:

    export HF_USERNAME=your-username
    python publish.py

Note: the adapter is trained on a research-only dataset, so it's published under
CC-BY-NC-4.0 to respect those terms.
"""
import json
import os
from huggingface_hub import HfApi
import config

def pct(x):
    return f"{x * 100:.1f}%"


# Usage snippet with placeholders (avoids f-string brace escaping in the code block).
USAGE = '''```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

base  = AutoModelForCausalLM.from_pretrained("__BASE__", device_map="auto")
model = PeftModel.from_pretrained(base, "__REPO__")
tok   = AutoTokenizer.from_pretrained("__REPO__")

SYS = ("You are a function-calling assistant. Given the available tools, respond with "
       'ONLY a JSON list of calls, each {"name": <tool>, "arguments": {<args>}}. '
       "If nothing applies, respond with []. Tools:\\n")

tools = [{"name": "get_weather", "parameters": {"city": {"type": "string"}}}]
messages = [{"role": "system", "content": SYS + json.dumps(tools)},
            {"role": "user",   "content": "What is the weather in Paris?"}]

prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```'''


def build_card(results, analysis, repo_id):
    b, t = results["base"], results["tuned"]
    frontmatter = (
        "---\n"
        "license: cc-by-nc-4.0\n"
        f"base_model: {config.BASE_MODEL}\n"
        "library_name: peft\n"
        "pipeline_tag: text-generation\n"
        "tags: [function-calling, tool-use, lora, qlora, peft]\n"
        f"datasets: [{config.DATASET}]\n"
        "language: [en]\n"
        "---"
    )
    table = (
        "| Metric | Base | Fine-tuned | \u0394 |\n|---|---|---|---|\n"
        f"| JSON validity | {pct(b['json_valid'])} | {pct(t['json_valid'])} | {pct(t['json_valid'] - b['json_valid'])} |\n"
        f"| Function-name accuracy | {pct(b['name_acc'])} | {pct(t['name_acc'])} | {pct(t['name_acc'] - b['name_acc'])} |\n"
        f"| **Exact-match (AST)** | **{pct(b['exact_match'])}** | **{pct(t['exact_match'])}** | **{pct(t['exact_match'] - b['exact_match'])}** |"
    )
    total = analysis["total_misses"]
    breakdown = "\n".join(
        f"- `{k}`: {v} ({v / total * 100:.0f}% of misses)"
        for k, v in sorted(analysis["breakdown"].items(), key=lambda kv: -kv[1])
    )
    usage = USAGE.replace("__BASE__", config.BASE_MODEL).replace("__REPO__", repo_id)

    return "\n\n".join([
        frontmatter,
        f"# {config.REPO_NAME}\n\nA LoRA adapter fine-tuning **{config.BASE_MODEL}** for structured "
        "function-calling / tool-call generation. Given available tools and a natural-language "
        "query, it emits a JSON list of function calls.",
        f"## Results\n\nHeld-out test set ({results['test_n']} examples), greedy decoding, "
        "strict BFCL-style AST match (function name + arguments compared as parsed structures; "
        f"no LLM judge).\n\n{table}",
        "## Intended prompt format\n\nThe adapter was trained on one specific format -- tool schemas "
        "in the system message, reply is a raw JSON list. Use this exact shape:\n\n" + usage,
        "## Training\n\n"
        f"- **Method:** QLoRA (4-bit NF4), LoRA r={config.LORA_R}, \u03b1={config.LORA_ALPHA}, "
        f"dropout={config.LORA_DROPOUT}\n"
        f"- **Target modules:** {', '.join(config.LORA_TARGETS)}\n"
        f"- **Data:** {config.N_TRAIN} examples from `{config.DATASET}`, {config.EPOCHS} epochs, "
        f"effective batch {config.BATCH_SIZE * config.GRAD_ACCUM}, lr {config.LEARNING_RATE}, "
        f"max_length {config.TRAIN_MAX_LENGTH}\n"
        "- **Hardware:** single GPU, ~35 min (fits free-tier Colab T4 via QLoRA)",
        "## Error analysis\n\n"
        f"Of {total}/{results['test_n']} exact-match misses "
        f"({pct(total / results['test_n'])}), the breakdown by failure type:\n\n" + breakdown + "\n\n"
        "Residual errors are almost entirely argument-level (wrong API values or keys), not format "
        "or tool-selection. Type-insensitive scoring changes the result by only "
        f"{pct(analysis['type_coercion_gap'])}, so the strict metric is faithful.",
        "## Limitations\n\n"
        "- Fine-tuned on a research-only dataset; intended for research/demonstration.\n"
        "- Small (1.5B) model; not a substitute for larger tool-calling models in production.",
        f"## Attribution\n\nBase model `{config.BASE_MODEL}` (Apache-2.0). Training data "
        f"`{config.DATASET}` (research/non-commercial). Adapter released under CC-BY-NC-4.0 "
        "to respect the dataset's terms.",
    ])


def main():
    if config.HF_USERNAME == "your-username":
        raise SystemExit("Set HF_USERNAME first, e.g.  export HF_USERNAME=your-username")

    results = json.load(open(config.RESULTS_PATH))
    analysis = json.load(open(config.ANALYSIS_PATH))
    repo_id = config.REPO_ID

    card = build_card(results, analysis, repo_id)
    with open(os.path.join(config.ADAPTER_DIR, "README.md"), "w") as f:
        f.write(card)
    print("Model card written. Repo:", repo_id)

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    # upload_folder is simplest; if a flaky connection resets it, switch to per-file
    # api.upload_file(...) calls, which are smaller and more resilient.
    api.upload_folder(folder_path=config.ADAPTER_DIR, repo_id=repo_id, repo_type="model")
    print("Published:", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
