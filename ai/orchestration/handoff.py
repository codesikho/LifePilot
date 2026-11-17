
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HandoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination_agent: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    user_request: str = Field(min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("destination_agent")
    @classmethod
    def normalize_agent_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("destination_agent cannot be blank")
        return normalized
