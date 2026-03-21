import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def apply_chat_template(tokenizer: AutoTokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode_example(tokenizer: AutoTokenizer, row: dict, max_length: int) -> dict[str, list[int]]:
    messages = row.get("messages", [])
    if len(messages) < 2:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_text = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if not full_ids:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    input_ids = full_ids[:max_length]
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, labels, masks = [], [], []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        masks.append(row["attention_mask"] + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def resolve_model_path(model_id: str, cache_dir: str | None = None, retries: int = 5) -> str:
    if Path(model_id).exists():
        return model_id

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading model snapshot ({attempt}/{retries}): {model_id}")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
            )
            print(f"Model snapshot ready at: {local_path}")
            return local_path
        except Exception as exc:
            last_exc = exc
            wait_s = min(10, attempt * 2)
            print(f"Snapshot download failed: {exc}")
            if attempt < retries:
                print(f"Retrying in {wait_s}s...")
                time.sleep(wait_s)

    raise RuntimeError(f"Failed to download model snapshot for {model_id}: {last_exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen/Qwen3-8B")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", default="data/sft_train.jsonl")
    parser.add_argument("--output", default="adapters/qwen3-8b-ghost-lora")
    parser.add_argument("--cache-dir", default="", help="Optional Hugging Face cache directory")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    dataset = Dataset.from_list(rows)
    model_path = resolve_model_path(args.model, cache_dir=args.cache_dir or None, retries=5)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_config)

    tokenized = dataset.map(lambda row: encode_example(tokenizer, row, args.max_length), remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 0)
    if len(tokenized) == 0:
        raise ValueError("No trainable rows left after tokenization.")

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )

    model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter: {args.output}")


if __name__ == "__main__":
    main()
