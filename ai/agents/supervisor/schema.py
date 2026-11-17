from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpecialistTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: str
    task_id: str
    user_request: str
    constraints: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


class SupervisorPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    tasks: list[SpecialistTask] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    requires_budget_check: bool = False