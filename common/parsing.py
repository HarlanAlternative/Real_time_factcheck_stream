from __future__ import annotations

import json
import re
from typing import Any

from common.schemas import FactCheckPrediction

FACTCHECK_LABELS: tuple[str, ...] = (
    "true",
    "false",
    "half-true",
    "mostly-true",
    "barely-true",
    "pants-fire",
)

LABEL_ALIASES: dict[str, str] = {
    "pants on fire": "pants-fire",
    "pants_fire": "pants-fire",
    "half true": "half-true",
    "mostly true": "mostly-true",
    "barely true": "barely-true",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = LABEL_ALIASES.get(normalized, normalized.replace(" ", "-"))
    if normalized not in FACTCHECK_LABELS:
        raise ValueError(f"Unsupported fact-check label: {label}")
    return normalized


def extract_first_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Model output did not contain a JSON object.")

    depth = 0
    for index in range(start, len(cleaned)):
        character = cleaned[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]

    raise ValueError("Model output contained an incomplete JSON object.")


def _fallback_payload(text: str) -> dict[str, Any]:
    label_match = re.search(
        r"\b(pants on fire|pants[- ]fire|mostly[- ]true|barely[- ]true|half[- ]true|false|true)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not label_match:
        raise ValueError("Model output did not contain a recognizable label.")

    confidence_match = re.search(
        r"confidence(?:\s+score)?\s*(?:is|=|:)?\s*(0(?:\.\d+)?|1(?:\.0+)?)\b",
        text,
        flags=re.IGNORECASE,
    )
    return {
        "label": label_match.group(1),
        "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
        "reasoning": text.strip(),
    }


def parse_model_output(text: str) -> FactCheckPrediction:
    try:
        payload = json.loads(extract_first_json_object(text))
    except (json.JSONDecodeError, ValueError):
        payload = _fallback_payload(text)

    payload["label"] = normalize_label(str(payload["label"]))
    payload["confidence"] = max(0.0, min(1.0, float(payload["confidence"])))
    payload["reasoning"] = str(payload["reasoning"]).strip()
    return FactCheckPrediction.model_validate(payload)
