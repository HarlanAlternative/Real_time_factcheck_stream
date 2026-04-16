from __future__ import annotations

import json


PROMPT_TEMPLATE = """### Claim:
{claim}

### Task:
Assess the credibility of the above claim. Respond in JSON format:
{{"label": "true|false|half-true|mostly-true|barely-true|pants-fire", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

### Response:
"""


def build_inference_prompt(claim: str) -> str:
    return PROMPT_TEMPLATE.format(claim=claim.strip())


def build_training_text(
    claim: str,
    label: str,
    reasoning: str,
    confidence: float,
) -> str:
    response = json.dumps(
        {
            "label": label,
            "confidence": round(confidence, 2),
            "reasoning": reasoning.strip(),
        },
        ensure_ascii=True,
    )
    return f"{build_inference_prompt(claim)}{response}"

