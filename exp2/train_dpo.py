"""Continue Exp0 LoRA with a separate frozen SFT reference on one 4090."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from datasets import load_from_disk
from transformers import TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer
from exp2 import utils as cfg
from exp2.utils import key, load_model


class SaveFinalStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = control.should_save = True
        return control


class StopSanity(TrainerCallback):
    """Stop after checkpoint 5 without changing the ten-step LR schedule."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 5:
            control.should_evaluate = control.should_save = True
            control.should_training_stop = True
        return control


def step_plan(n, batch_size, accumulation, sanity):
    total = 10 if sanity else 3 * math.ceil(math.ceil(n / batch_size) / accumulation)
    # Tiny runs evaluate every step, so even a single update can be selected.
    return total, 5 if sanity else (1 if total <= 25 else 25)


def prepare_run(run, manifest, resume):
    """Reject changed data/configuration and incomplete or foreign checkpoints."""
    record = run / "run.meta.json"
    if record.exists():
        if json.loads(record.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Training data/config changed; start a new run, never resume")
    elif resume:
        raise ValueError("Resume requires the original run.meta.json")
    if resume:
        checkpoint = Path(resume).resolve()
        if checkpoint.parent != (run / "checkpoints").resolve():
            raise ValueError("Checkpoint must belong to this sanity/main run")
        required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "adapter_config.json"]
        if any(not (checkpoint / name).is_file() for name in required) or not any(
            (checkpoint / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            raise ValueError("Incomplete checkpoint; adapter-final is not resumable")
    elif (run / "adapter-final").exists() or any((run / "checkpoints").glob("checkpoint-*")):
        raise FileExistsError("Run already has outputs; use --resume or a new run name")
    run.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def check_initialization(policy, reference):
    import torch
    reference.requires_grad_(False)
    reference.eval()
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("Policy must train only the existing LoRA adapter")
    ref_params = dict(reference.named_parameters())
    for name, param in trainable.items():
        other = ref_params.get(name)
        if other is None or other.requires_grad or param is other or (
            param.device == other.device and param.data_ptr() == other.data_ptr()
        ) or not torch.equal(param.detach().cpu(), other.detach().cpu()):
            raise ValueError(f"Policy/reference adapter mismatch or shared storage: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--run-name", help="Optional new run directory for OOM retries")
    parser.add_argument("--stop-after-step-5", action="store_true", help="Sanity recovery rehearsal only")
    args = parser.parse_args(); set_seed(cfg.SEED)
    if args.stop_after_step_5 and (not args.sanity or args.resume):
        parser.error("--stop-after-step-5 requires --sanity and no --resume")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}):
        parser.error("--run-name must be one directory name")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This entry point supports one GPU; effective batch is 16 pairs")
    data = Path(cfg.ROOT) / "preferences"
    train = load_from_disk(str(data / "train"))
    val = load_from_disk(str(data / "val"))
    if not len(train) or not len(val):
        raise ValueError("Both preference train and val must be nonempty")
    data_hashes = dict(train_sha256=key(list(train)), val_sha256=key(list(val)))
    if args.sanity:
        train = train.select(range(min(32, len(train))))
        val = val.select(range(min(8, len(val))))
    run = Path(cfg.ROOT) / "runs" / (args.run_name or ("sanity" if args.sanity else "main"))
    batch_size = args.batch_size
    accumulation = 16 // batch_size
    total_steps, interval = step_plan(len(train), batch_size, accumulation, args.sanity)
    import config
    adapter = Path(cfg.ADAPTER)
    hashes = {p.relative_to(adapter).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(adapter.rglob("*")) if p.is_file()}
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError("Incomplete Exp0 adapter")
    manifest = dict(version=1, sanity=args.sanity, **data_hashes,
                    adapter_sha256=hashes, base_model=config.BASE_MODEL,
                    seed=cfg.SEED, max_length=cfg.MAX_LENGTH, epochs=3, max_steps=10 if args.sanity else -1,
                    batch_size=batch_size, accumulation=accumulation, interval=interval,
                    learning_rate=5e-6, beta=0.1)
    prepare_run(run, manifest, args.resume)
    policy, tok = load_model(trainable=True)
    reference, _ = load_model()
    check_initialization(policy, reference)
    policy.print_trainable_parameters()
    print(f"train={len(train)}, val={len(val)}, effective_batch={batch_size * accumulation}, steps={total_steps}")
    training = DPOConfig(
        output_dir=str(run / "checkpoints"), seed=cfg.SEED, data_seed=cfg.SEED,
        beta=0.1, loss_type=["sigmoid"], learning_rate=5e-6,
        num_train_epochs=3, max_steps=10 if args.sanity else -1,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        max_length=cfg.MAX_LENGTH, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        logging_steps=1 if args.sanity else 5, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", push_to_hub=False)
    trainer = DPOTrainer(model=policy, ref_model=reference, args=training,
        train_dataset=train, eval_dataset=val, processing_class=tok,
        callbacks=[SaveFinalStep()] + ([StopSanity()] if args.stop_after_step_5 else []))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(run / "adapter-final"))
    tok.save_pretrained(str(run / "adapter-final"))


if __name__ == "__main__":
    main()
