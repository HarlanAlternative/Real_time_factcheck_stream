from __future__ import annotations

import json


PROMPT_TEMPLATE = """### Claim:
{claim}
### Speaker:
{speaker}, {speaker_title}, {party_affiliation}
### Context:
{context}
### Task:
Assess the credibility of the above claim. Respond in JSON format:
{{"label": "true|mixed|false", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
### Response:
"""


def build_inference_prompt(
    claim: str,
    speaker: str = "Unknown",
    speaker_title: str = "Unknown",
    party_affiliation: str = "Unknown",
    context: str = "Unknown",
) -> str:
    return PROMPT_TEMPLATE.format(
        claim=claim.strip(),
        speaker=speaker or "Unknown",
        speaker_title=speaker_title or "Unknown",
        party_affiliation=party_affiliation or "Unknown",
        context=context or "Unknown",
    )


def build_training_text(
    claim: str,
    label: str,
    reasoning: str,
    confidence: float,
    speaker: str = "Unknown",
    speaker_title: str = "Unknown",
    party_affiliation: str = "Unknown",
    context: str = "Unknown",
) -> str:
    response = json.dumps(
        {
            "label": label,
            "confidence": round(confidence, 2),
            "reasoning": reasoning.strip(),
        },
        ensure_ascii=True,
    )
    return f"{build_inference_prompt(claim, speaker, speaker_title, party_affiliation, context)}{response}"

