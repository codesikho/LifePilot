from __future__ import annotations

from typing import Protocol

from ai.agents.agent_registry import AgentRegistry
from ai.agents.specialist_contracts import (
    AgentStatus,
    SpecialistRequest,
    SpecialistResult,
)
from ai.orchestration.swarm_router import SwarmRouter


class AgentDispatcher(Protocol):
    """Contract for executing one specialist task."""

    def execute(
        self,
        agent_name: str,
        request: SpecialistRequest,
    ) -> SpecialistResult: ...


class RegistryDispatcher:
    """Execute exactly one specialist through the registry."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        agent_name: str,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        return self._registry.invoke(agent_name, request)


class SwarmDispatcher:
    """Execute a specialist and follow validated handoffs."""

    def __init__(self, swarm_router: SwarmRouter) -> None:
        self._swarm_router = swarm_router

    def execute(
        self,
        agent_name: str,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        execution = self._swarm_router.run(
            initial_agent=agent_name,
            initial_request=request,
        )

        if not execution.results:
            raise RuntimeError(
                f"Swarm execution produced no result for task {request.task_id}"
            )

        final_result = execution.results[-1]
        warnings = list(dict.fromkeys([
            warning
            for item in execution.results
            for warning in item.warnings
        ] + execution.warnings))

        if execution.status == "needs_input":
            status = AgentStatus.NEEDS_INPUT
        elif execution.status == "completed":
            status = final_result.status
        else:
            status = AgentStatus.PARTIAL

        # Preserve the whole handoff chain in one result because Supervisor
        # expects one result per planned task. Do not silently discard the
        # initial specialist's findings when a later specialist takes over.
        combined_summary = "\n\n".join(
            f"[{item.agent_name}] {item.summary}"
            for item in execution.results
        )
        combined_findings = [
            finding
            for item in execution.results
            for finding in item.findings
        ]

        return SpecialistResult(
            agent_name=agent_name,
            task_id=request.task_id,
            status=status,
            summary=combined_summary,
            findings=combined_findings,
            warnings=warnings,
            missing_information=[
                missing
                for item in execution.results
                for missing in item.missing_information
            ],
            total_estimated_cost=final_result.total_estimated_cost,
            currency=final_result.currency,
        )
