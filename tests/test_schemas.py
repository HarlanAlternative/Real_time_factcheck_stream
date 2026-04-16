from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from common.schemas import ClaimMessage, FactCheckPrediction


def test_claim_message_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        ClaimMessage(id=uuid4(), text="A test claim", timestamp=datetime.now())


def test_factcheck_prediction_rejects_invalid_label() -> None:
    with pytest.raises(ValidationError):
        FactCheckPrediction(label="unsupported", confidence=0.5, reasoning="Nope")


def test_claim_message_accepts_valid_payload() -> None:
    message = ClaimMessage(
        id=uuid4(),
        text="The inflation rate fell in the last quarter.",
        timestamp=datetime.now(tz=timezone.utc),
    )

    assert message.text.startswith("The inflation")
