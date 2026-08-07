"""Stage 2 -- QLoRA fine-tuning.

Loads the base model in 4-bit, attaches a LoRA adapter, and trains with a
completion-only objective. Checkpoints to disk every 100 steps and auto-resumes,
so a Colab session timeout doesn't cost the run. Precision (bf16/fp16) is detected
from the GPU so the same code runs on an A100 or a free-tier T4.

Run:  python train.py
"""
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config


def load_quantized_base():
    """Load the base model in 4-bit NF4 with a runtime-appropriate compute dtype."""
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # QLoRA-paper quantization; better than plain int4
        bnb_4bit_compute_dtype=compute_dtype,  # matmuls run in bf16/fp16, not 4-bit
        bnb_4bit_use_double_quant=True,        # quantize the quant constants too (small win)
    )
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, quantization_config=bnb,
        dtype=compute_dtype, device_map="auto",
    )
    model.config.use_cache = False   # required alongside gradient checkpointing
    return model, tok


def main():
    train_ds = load_from_disk(os.path.join(config.DATA_DIR, "train"))
    val_ds = load_from_disk(os.path.join(config.DATA_DIR, "val"))

    model, tok = load_quantized_base()
    print(f"Base loaded | footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    model = prepare_model_for_kbit_training(model)   # fp32 norms, enable input grads
    lora = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=config.LORA_TARGETS,
    )
    # Wrap here. Do NOT also pass peft_config to SFTTrainer -- double-wrapping a
    # PeftModel re-runs prepare_model_for_kbit_training and silently freezes the adapter.
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()   # expect ~1% trainable

    bf16 = torch.cuda.is_bf16_supported()
    cfg = SFTConfig(
        output_dir=config.CHECKPOINT_DIR,
        max_length=config.TRAIN_MAX_LENGTH,
        completion_only_loss=True,          # loss on the assistant JSON only, not the tool schemas
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        num_train_epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100,
        save_total_limit=3,
        # save_total_limit prunes by recency, not by score -- without the next three
        # lines the best-eval checkpoint can be deleted mid-run. Keep the best one.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,   # trl 1.x name for the tokenizer
    )

    has_ckpt = os.path.isdir(config.CHECKPOINT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(config.CHECKPOINT_DIR))
    print("Resuming from checkpoint." if has_ckpt else "Starting fresh.")
    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_model(config.ADAPTER_DIR)   # saves the best-eval adapter (load_best_model_at_end)
    tok.save_pretrained(config.ADAPTER_DIR)
    print("Saved final adapter to", config.ADAPTER_DIR)


if __name__ == "__main__":
    main()
