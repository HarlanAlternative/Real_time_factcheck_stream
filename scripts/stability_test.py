"""
Stability test: run 100 claims from the LIAR test split through the fine-tuned
LoRA model and report parse success rate, latency, and label distribution.

Runs directly against the transformers stack (no vLLM/Docker required).
Usage:
    python scripts/stability_test.py
"""
from __future__ import annotations

import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from common.liar import get_split, load_liar_dataset, to_liar_example
from common.parsing import parse_model_output
from common.prompts import build_inference_prompt

N_CLAIMS = 100
ADAPTER_PATH = "fine_tuned/mistral-liar-lora"
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
MAX_NEW_TOKENS = 200


def load_model():
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def infer(model, tokenizer, claim: str) -> tuple[str, float]:
    prompt = build_inference_prompt(claim)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return text, latency


def main():
    print(f"Loading model from {ADAPTER_PATH} ...")
    model, tokenizer = load_model()
    print("Model loaded. Loading LIAR test split ...")

    dataset = load_liar_dataset("ucsbnlp/liar")
    test_split = get_split(dataset, "test", fallback="validation")
    claims = [to_liar_example(test_split[i]) for i in range(N_CLAIMS)]
    print(f"Running {N_CLAIMS} claims ...\n")

    successes = 0
    fallbacks = 0
    errors = 0
    latencies: list[float] = []
    label_counts: Counter = Counter()
    error_details: list[str] = []

    for i, example in enumerate(claims):
        try:
            raw, latency = infer(model, tokenizer, example.claim)
            latencies.append(latency)
            try:
                prediction = parse_model_output(raw)
                successes += 1
                label_counts[prediction.label] += 1
                # detect fallback (regex path) by checking if raw had valid JSON
                import json
                from common.parsing import extract_first_json_object
                try:
                    json.loads(extract_first_json_object(raw))
                except (ValueError, json.JSONDecodeError):
                    fallbacks += 1
            except ValueError as e:
                errors += 1
                error_details.append(f"claim {i}: parse failed — {e}")
        except Exception as e:
            errors += 1
            latencies.append(0.0)
            error_details.append(f"claim {i}: inference failed — {e}")

        if (i + 1) % 10 == 0:
            done = i + 1
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            print(f"  [{done:3d}/{N_CLAIMS}] ok={successes} err={errors} avg_lat={avg_lat:.2f}s")

    total = N_CLAIMS
    parse_rate = successes / total * 100
    json_rate = (successes - fallbacks) / total * 100
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print("\n" + "=" * 60)
    print("STABILITY TEST RESULTS")
    print("=" * 60)
    print(f"Claims tested:          {total}")
    print(f"Parse successes:        {successes}  ({parse_rate:.1f}%)")
    print(f"  — clean JSON:         {successes - fallbacks}  ({json_rate:.1f}%)")
    print(f"  — regex fallback:     {fallbacks}")
    print(f"Parse failures:         {errors}")
    print(f"Avg latency/claim:      {avg_lat:.2f}s")
    print(f"P95 latency/claim:      {p95_lat:.2f}s")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {label:<14} {count:3d}  {bar}")
    if error_details:
        print(f"\nErrors ({len(error_details)}):")
        for e in error_details[:5]:
            print(f"  {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
