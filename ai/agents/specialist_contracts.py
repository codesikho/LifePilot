
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai.orchestration.handoff import HandoffRequest


class AgentStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class SpecialistRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    user_request: str = Field(min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class SpecialistFinding(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    description: str
    estimated_cost: float | None = Field(default=None, ge=0)
    currency: str = "INR"
    source: str = "unknown"
    is_live: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class SpecialistResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    task_id: str
    status: AgentStatus
    summary: str
    findings: list[SpecialistFinding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    total_estimated_cost: float | None = Field(default=None, ge=0)
    currency: str = "INR"
    handoff: HandoffRequest | None = None
