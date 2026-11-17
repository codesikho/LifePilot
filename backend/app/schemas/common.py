"""
Shared schemas: money, prices, sources and agent context.

LifePilot module: common
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pydantic import BaseModel, Field, ConfigDict


class CurrencyCode(StrEnum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class Money(BaseModel):
    amount: Decimal = Field(ge=0)
    currency: CurrencyCode


class PriceBreakdown(BaseModel):
    base_amount: Money
    taxes: Money | None = None
    fees: Money | None = None
    total_amount: Money

    # True when the total is not yet fully verified.
    is_estimated: bool = False


class SourceMode(StrEnum):
    REAL = "real"
    MOCK = "mock"


class SourceInfo(BaseModel):
    provider: str
    mode: SourceMode
    retrieved_at: datetime
    source_url: str | None = None
    external_reference: str | None = None


class Preference(BaseModel):
    category: str
    value: str


class Constraint(BaseModel):
    name: str
    value: str
    mandatory: bool = True


class AgentExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str
    task_id: str
    budget_limit: Money

    preferences: list[Preference] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
