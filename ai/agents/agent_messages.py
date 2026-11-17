"""Typed messages exchanged between LifePilot agents."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentMessageType(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"


class AgentMessage(BaseModel):
    """A single request or response in an internal A2A conversation."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    in_reply_to: str | None = None
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    message_type: AgentMessageType
    task_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
