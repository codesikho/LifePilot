from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.agent_registry import AgentRegistry
from ai.orchestration.agent_dispatcher import AgentDispatcher, RegistryDispatcher
from ai.agents.specialist_contracts import (
    AgentStatus,
    SpecialistRequest,
    SpecialistResult,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.agents.supervisor.planner import OllamaSupervisorPlanner
from ai.agents.supervisor.schema import (
    SpecialistTask,
    SupervisorPlan,
)


class SupervisorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: SupervisorPlan
    specialist_results: list[SpecialistResult] = Field(
        default_factory=list
    )
    status: str
    warnings: list[str] = Field(default_factory=list)


class Supervisor:
    def __init__(
        self,
        registry: AgentRegistry,
        planner: OllamaSupervisorPlanner,
        dispatcher: AgentDispatcher | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner
        self.dispatcher = dispatcher or RegistryDispatcher(registry)

    def run(self, user_request: str) -> SupervisorResult:
        if not user_request or not user_request.strip():
            raise ValueError("User request cannot be empty")

        available_agents = self.registry.list_agents()

        # Step 1: Ask Ollama to create a structured plan.
        plan = self.planner.create_plan(
            user_request=user_request,
            available_agents=available_agents,
        )

        # Step 2: If information is missing, do not execute.
        if plan.missing_information:
            return SupervisorResult(
                plan=plan,
                status="needs_input",
                warnings=[
                    "Specialist execution was paused because "
                    "the request is missing information."
                ],
            )

        results: list[SpecialistResult] = []
        warnings: list[str] = []

        # Step 3: Run research specialists first.
        research_tasks = [
            task
            for task in plan.tasks
            if task.agent_name.lower() != "budget"
        ]

        budget_tasks = [
            task
            for task in plan.tasks
            if task.agent_name.lower() == "budget"
        ]

        for task in research_tasks:
            result = self._execute_task(task)
            results.append(result)

        # Step 4: Run Budget only after collecting structured
        # cost estimates from the research specialists.
        if budget_tasks:
            cost_items = self._collect_cost_items(results)

            if not cost_items:
                warnings.append(
                    "Budget calculation was not run because "
                    "the research specialists did not return "
                    "structured cost estimates."
                )
            else:
                for task in budget_tasks:
                    budget_task = SpecialistTask(
                        agent_name="budget",
                        task_id=task.task_id,
                        user_request=task.user_request,
                        constraints=task.constraints,
                        context={
                            **task.context,
                            "cost_items": cost_items,
                        },
                    )

                    result = self._execute_task(budget_task)
                    results.append(result)

        # Step 5: Report execution status.
        failed_results = [
            result
            for result in results
            if result.status in {
                AgentStatus.FAILED,
                AgentStatus.UNAVAILABLE,
            }
        ]

        status = (
            "partial"
            if failed_results or warnings
            else "completed"
        )

        return SupervisorResult(
            plan=plan,
            specialist_results=results,
            status=status,
            warnings=warnings,
        )

    def _execute_task(self, task: SpecialistTask) -> SpecialistResult:
        request = SpecialistRequest(
            task_id=task.task_id,
            user_request=task.user_request,
            constraints=task.constraints,
            context=task.context,
        )

        return self.dispatcher.execute(
            agent_name=task.agent_name,
            request=request,
        )

    @staticmethod
    def _collect_cost_items(
        results: list[SpecialistResult],
    ) -> list[dict]:
        cost_items = []

        for result in results:
            for finding in result.findings:
                if finding.estimated_cost is None:
                    continue

                cost_items.append(
                    {
                        "name": finding.title,
                        "amount": finding.estimated_cost,
                        "currency": finding.currency,
                    }
                )

        return cost_items