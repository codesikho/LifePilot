
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.agent_registry import (
    AgentRegistry,
    AgentRegistryError,
)
from ai.agents.specialist_contracts import (
    AgentStatus,
    SpecialistRequest,
    SpecialistResult,
)


class HandoffRecord(BaseModel):
    source_agent: str
    destination_agent: str
    reason: str
    task_id: str


class SwarmExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    results: list[SpecialistResult] = Field(default_factory=list)
    handoff_history: list[HandoffRecord] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(default_factory=list)


class SwarmRouter:
    def __init__(
        self,
        registry: AgentRegistry,
        allowed_handoffs: dict[str, tuple[str, ...]],
        max_handoffs: int = 5,
    ) -> None:
        if max_handoffs < 0:
            raise ValueError("max_handoffs cannot be negative")

        self.registry = registry
        self.max_handoffs = max_handoffs

        # Normalize the agent names once during initialization.
        self.allowed_handoffs = {
            source.strip().lower(): tuple(
                destination.strip().lower()
                for destination in destinations
            )
            for source, destinations in allowed_handoffs.items()
        }

    def run(
        self,
        initial_agent: str,
        initial_request: SpecialistRequest,
    ) -> SwarmExecutionResult:

        current_agent = initial_agent.strip().lower()
        current_request = initial_request

        results: list[SpecialistResult] = []
        history: list[HandoffRecord] = []
        visited_agents = {current_agent}

        while True:
            # 1. Execute the current specialist.
            result = self.registry.invoke(
                current_agent,
                current_request,
            )

            results.append(result)

            # 2. A request for user input always stops this path,
            # even when the specialist did not request a handoff.
            if result.status == AgentStatus.NEEDS_INPUT:
                return SwarmExecutionResult(
                    status="needs_input",
                    results=results,
                    handoff_history=history,
                    warnings=[
                        "Agent requested more information. "
                        "Handoff was not executed."
                    ] if result.handoff is not None else [],
                )

            # 3. No handoff means this execution path is complete.
            if result.handoff is None:
                return SwarmExecutionResult(
                    status="completed",
                    results=results,
                    handoff_history=history,
                )

            handoff = result.handoff
            destination = handoff.destination_agent

            # 4. Validate task identity.
            if handoff.task_id != result.task_id:
                return self._stop(
                    "handoff_rejected",
                    results,
                    history,
                    "Handoff task_id does not match the "
                    "specialist result.",
                )

            # 5. Check whether this handoff is permitted.
            permitted = self.allowed_handoffs.get(
                current_agent,
                (),
            )

            if destination not in permitted:
                return self._stop(
                    "handoff_rejected",
                    results,
                    history,
                    f"{current_agent} is not allowed to hand "
                    f"off to {destination}.",
                )

            # 6. Check that the destination is registered
            # and enabled.
            try:
                self.registry.get(destination)
            except AgentRegistryError:
                return self._stop(
                    "handoff_rejected",
                    results,
                    history,
                    f"Destination agent '{destination}' "
                    "is unavailable.",
                )

            # 7. Stop repeated-agent loops.
            if destination in visited_agents:
                return self._stop(
                    "handoff_rejected",
                    results,
                    history,
                    f"Repeated handoff to '{destination}' "
                    "was blocked.",
                )

            # 8. Enforce the handoff limit.
            if len(history) >= self.max_handoffs:
                return self._stop(
                    "max_handoffs_reached",
                    results,
                    history,
                    "Maximum number of handoffs reached.",
                )

            # 9. Record the transfer.
            history.append(
                HandoffRecord(
                    source_agent=current_agent,
                    destination_agent=destination,
                    reason=handoff.reason,
                    task_id=handoff.task_id,
                )
            )

            # 10. Build the next specialist's request.
            current_request = SpecialistRequest(
                task_id=handoff.task_id,
                user_request=handoff.user_request,
                constraints={
                    **current_request.constraints,
                    **handoff.constraints,
                },
                context={
                    **current_request.context,
                    **handoff.context,
                },
            )

            visited_agents.add(destination)
            current_agent = destination

    @staticmethod
    def _stop(
        status: str,
        results: list[SpecialistResult],
        history: list[HandoffRecord],
        warning: str,
    ) -> SwarmExecutionResult:
        return SwarmExecutionResult(
            status=status,
            results=results,
            handoff_history=history,
            warnings=[warning],
        )
