"""Stage 2 -- QLoRA fine-tuning.
# 以 4-bit 量化方式加载底座模型 只对 completion 部分计算训练损失 每 100 步保存一次 checkpoint
Loads the base model in 4-bit, attaches a LoRA adapter, and trains with a
completion-only objective. Checkpoints to disk every 100 steps and auto-resumes,
so a Colab session timeout doesn't cost the run. Precision (bf16/fp16) is detected
from the GPU so the same code runs on an A100 or a free-tier T4.
# 检测到已有 checkpoint 时，自动尝试恢复训练 根据 GPU 能力选择计算精度
Run:  python train.py
"""
import os
import torch
from datasets import load_from_disk #从硬盘加载之前处理好的数据集
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# prepare_model_for_kbit_training 为量化模型进行参数高效训练做准备
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# 配置模型的低比特量化方式
from trl import SFTConfig, SFTTrainer
import config

# 加载 Qwen 底座 + 4-bit NF4
def load_quantized_base():
    """Load the base model in 4-bit NF4 with a runtime-appropriate compute dtype."""
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用 4-bit 量化加载模型
        bnb_4bit_quant_type="nf4",            # QLoRA-paper quantization; better than plain int4
        bnb_4bit_compute_dtype=compute_dtype,  # matmuls run in bf16/fp16, not 4-bit
        bnb_4bit_use_double_quant=True,        # quantize the quant constants too (small win)
    )
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 对于某些没有单独定义 pad token 的模型，使用 EOS 作为 PAD 是一种常见处理方式
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, quantization_config=bnb,
        dtype=compute_dtype, device_map="auto", # 指定模型非量化部分等使用的计算数据类型 由 Transformers 自动安排模型放置在哪些设备上
    )
    model.config.use_cache = False   
    # required alongside gradient checkpointing 
    # 训练时关闭 KV Cache，配合梯度检查点节省显存  推理时通常开启 KV Cache，加快生成
    return model, tok

# 训练和验证交替进行，而不是同时进行
def main():
    train_ds = load_from_disk(os.path.join(config.DATA_DIR, "train"))
    val_ds = load_from_disk(os.path.join(config.DATA_DIR, "val"))

    model, tok = load_quantized_base()
    print(f"Base loaded | footprint: {model.get_memory_footprint() / 1e9:.2f} GB") # 打印模型显存占用
    # 到这里我们已经将模型加载成 4-bit 量化模型 下面为量化训练做准备
    model = prepare_model_for_kbit_training(model)   # fp32 norms, enable input grads
    lora = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=config.LORA_TARGETS,
    ) # bias="none"表示不额外训练原始线性层的 bias 参数
    # Wrap here. Do NOT also pass peft_config to SFTTrainer -- double-wrapping a
    # PeftModel re-runs prepare_model_for_kbit_training and silently freezes the adapter.
    model = get_peft_model(model, lora) # 挂载lora
    model.print_trainable_parameters()   # expect ~1% trainable 打印可训练参数

    bf16 = torch.cuda.is_bf16_supported() # 先检查 GPU 是否支持 BF16
    cfg = SFTConfig(
        output_dir=config.CHECKPOINT_DIR,   # 指定训练 checkpoint 的保存位置
        max_length=config.TRAIN_MAX_LENGTH, # 最大训练序列长度 1024
        completion_only_loss=True,          # loss on the assistant JSON only, not the tool schemas
        per_device_train_batch_size=config.BATCH_SIZE, # batch_size 这里是8
        gradient_accumulation_steps=config.GRAD_ACCUM, # 梯度累计 所以 相当于8*2= 16条样本更新一次
        num_train_epochs=config.EPOCHS, # epoch轮数
        learning_rate=config.LEARNING_RATE, # lr
        lr_scheduler_type="cosine", # 学习率调度策略 
        warmup_ratio=config.WARMUP_RATIO, # 学习率预热比例 0.03 预热阶段约占总训练步数的 3
        bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, # 开启梯度检查点
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20, # 每 20 个训练 step 记录一次训练日志
        eval_strategy="steps", eval_steps=100,  # 每 100 步评估一次
        save_strategy="steps", save_steps=100,  # 按照训练步数保存 checkpoint
        save_total_limit=3, # 最多保留三个 checkpoint
        # save_total_limit prunes by recency, not by score -- without the next three
        # lines the best-eval checkpoint can be deleted mid-run. Keep the best one.
        load_best_model_at_end=True, # 加载验证效果最好的 checkpoint
        metric_for_best_model="eval_loss", # 指定使用eval_loss指标判断最佳模型
        greater_is_better=False, # 这个指标越小越好
        report_to="none", # 关闭外部实验跟踪平台的自动日志上报
    )
    # 正式构建训练器
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,   # trl 1.x name for the tokenizer
    ) # 使用 processing_class 传入 tokenizer

    # 实现自动恢复训练
    has_ckpt = os.path.isdir(config.CHECKPOINT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(config.CHECKPOINT_DIR))
    print("Resuming from checkpoint." if has_ckpt else "Starting fresh.")

    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_model(config.ADAPTER_DIR)   # saves the best-eval adapter (load_best_model_at_end)
    tok.save_pretrained(config.ADAPTER_DIR)
    print("Saved final adapter to", config.ADAPTER_DIR)
# 以后每次做新的消融实验，最好独立保存模型、checkpoint、配置和评测报告。这样才能保证实验之间真正可比较


if __name__ == "__main__":
    main()
