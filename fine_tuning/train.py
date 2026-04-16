from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common.liar import get_split, load_liar_dataset, to_liar_example
from common.prompts import build_training_text

LOGGER = logging.getLogger("fine_tuning.train")

CONFIDENCE_PRIORS: dict[str, float] = {
    "false": 0.92,
    "mixed": 0.65,
    "true": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning on the LIAR dataset.")
    parser.add_argument("--dataset-name", default="ucsbnlp/liar")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output-dir", default="fine_tuned/mistral-liar-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--wandb-project", default="real-time-factcheck-stream")
    parser.add_argument("--log-steps", type=int, default=10)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _format_row(example: dict[str, Any]) -> dict[str, str]:
    record = to_liar_example(example)
    return {
        "text": build_training_text(
            claim=record.claim,
            label=record.label,
            reasoning=record.reasoning,
            confidence=CONFIDENCE_PRIORS[record.label],
            speaker=record.speaker,
            speaker_title=record.speaker_title,
            party_affiliation=record.party_affiliation,
            context=record.context,
        )
    }


def _maybe_limit(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def _get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    args = parse_args()
    configure_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_liar_dataset(args.dataset_name)
    train_split = _maybe_limit(get_split(dataset, "train"), args.max_train_samples)
    eval_split = _maybe_limit(get_split(dataset, "validation", fallback="test"), args.max_eval_samples)

    train_dataset = train_split.map(_format_row, remove_columns=train_split.column_names)
    eval_dataset = eval_split.map(_format_row, remove_columns=eval_split.column_names)

    train_labels = Counter(to_liar_example(example).label for example in train_split)
    LOGGER.info("Training label distribution: %s", dict(train_labels))

    compute_dtype = _get_compute_dtype()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    wandb_enabled = bool(os.getenv("WANDB_API_KEY"))
    if wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.log_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to=["wandb"] if wandb_enabled else "none",
        bf16=compute_dtype is torch.bfloat16,
        fp16=compute_dtype is torch.float16,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        packing=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "target_modules": ["q_proj", "v_proj"],
        "quantization": "4-bit NF4",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved LoRA adapter and tokenizer to %s", output_dir)


if __name__ == "__main__":
    main()
