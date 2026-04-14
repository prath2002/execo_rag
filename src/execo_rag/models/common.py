"""Shared model conventions and reusable schema helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecoBaseModel(BaseModel):
    """Base model with strict defaults for internal and external schemas."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload suitable for structured logging."""

        return self.model_dump(mode="json", exclude_none=True)


class TimestampedModel(ExecoBaseModel):
    """Base model carrying created and updated timestamps."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
