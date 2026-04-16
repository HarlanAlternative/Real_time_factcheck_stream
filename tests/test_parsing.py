from __future__ import annotations

from common.parsing import normalize_label, parse_model_output


def test_normalize_label_accepts_human_variant() -> None:
    assert normalize_label("Pants on Fire") == "pants-fire"


def test_parse_model_output_extracts_json_from_code_fence() -> None:
    output = """```json
    {"label": "mostly true", "confidence": 0.74, "reasoning": "The claim is directionally correct."}
    ```"""

    prediction = parse_model_output(output)

    assert prediction.label == "mostly-true"
    assert prediction.confidence == 0.74


def test_parse_model_output_falls_back_to_label_regex() -> None:
    output = "Likely false with confidence 0.62 because the evidence conflicts with the claim."

    prediction = parse_model_output(output)

    assert prediction.label == "false"
    assert prediction.confidence == 0.62

