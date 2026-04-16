from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

FactCheckLabel = Literal[
    "true",
    "mixed",
    "false",
]


class ClaimMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=3)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class FactCheckPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: FactCheckLabel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=3)


class FactCheckResult(ClaimMessage, FactCheckPrediction):
    model_config = ConfigDict(extra="forbid")

    processed_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    model_name: str = Field(min_length=1)
    raw_response: str | None = None

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("processed_at must be timezone-aware")
        return value
