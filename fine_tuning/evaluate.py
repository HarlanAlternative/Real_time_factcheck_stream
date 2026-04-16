from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.liar import get_split, load_liar_dataset, to_liar_example
from common.parsing import FACTCHECK_LABELS, parse_model_output
from common.prompts import build_inference_prompt

LOGGER = logging.getLogger("fine_tuning.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned LIAR classifier.")
    parser.add_argument("--dataset-name", default="ucsbnlp/liar")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--adapter-path", default="fine_tuned/mistral-liar-lora")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--report-path", default="reports/benchmark.md")
    parser.add_argument("--json-path", default="reports/classification_report.json")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def render_markdown_report(
    accuracy: float,
    report: dict[str, dict[str, float] | float],
    invalid_predictions: int,
    sample_count: int,
) -> str:
    rows = []
    for label in FACTCHECK_LABELS:
        label_metrics = report.get(label, {})
        if not isinstance(label_metrics, dict):
            continue
        rows.append(
            f"| {label} | {label_metrics.get('precision', 0.0):.3f} | "
            f"{label_metrics.get('recall', 0.0):.3f} | "
            f"{label_metrics.get('f1-score', 0.0):.3f} | "
            f"{int(label_metrics.get('support', 0))} |"
        )

    macro_avg = report.get("macro avg", {})
    if not isinstance(macro_avg, dict):
        macro_avg = {}

    return "\n".join(
        [
            "# Benchmark Report",
            "",
            f"- Samples evaluated: {sample_count}",
            f"- Accuracy: {accuracy:.4f}",
            f"- Macro F1: {macro_avg.get('f1-score', 0.0):.4f}",
            f"- Invalid / fallback parses: {invalid_predictions}",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    dataset = load_liar_dataset(args.dataset_name)
    test_split = get_split(dataset, "test", fallback="validation")
    if args.max_samples is not None:
        test_split = test_split.select(range(min(args.max_samples, len(test_split))))

    tokenizer_source = args.adapter_path if Path(args.adapter_path).exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_get_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    prompts: list[str] = []
    labels: list[str] = []
    for row in test_split:
        example = to_liar_example(row)
        prompts.append(build_inference_prompt(example.claim))
        labels.append(example.label)

    predictions: list[str] = []
    invalid_predictions = 0

    for prompt_batch in batched(prompts, args.batch_size):
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        outputs = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        for output in outputs:
            try:
                prediction = parse_model_output(output)
                predictions.append(prediction.label)
            except ValueError:
                invalid_predictions += 1
                predictions.append("false")

    accuracy = accuracy_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=list(FACTCHECK_LABELS),
        output_dict=True,
        zero_division=0,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown_report(
            accuracy=accuracy,
            report=report,
            invalid_predictions=invalid_predictions,
            sample_count=len(labels),
        ),
        encoding="utf-8",
    )

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Accuracy: %.4f", accuracy)
    LOGGER.info("Macro F1: %.4f", report["macro avg"]["f1-score"])
    LOGGER.info("Saved markdown report to %s", report_path)


if __name__ == "__main__":
    main()
