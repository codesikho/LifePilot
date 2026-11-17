
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ai.agents.specialist_contracts import (
    SpecialistRequest,
    SpecialistResult,
)
def create_default_agent_registry(
    model_name: str = "qwen3:8b",
) -> AgentRegistry:
    from ai.agents.specialists import (
        BudgetSpecialist,
        TravelSpecialist,
    )

    registry = AgentRegistry()

    registry.register(
        AgentDefinition(
            name="travel",
            description=(
                "Search and analyse travel options, including hotels."
            ),
            capabilities=(
                "hotel_search",
                "travel_research",
            ),
            agent=TravelSpecialist(model_name=model_name),
        )
    )

    registry.register(
        AgentDefinition(
            name="budget",
            description=(
                "Calculate combined costs and check budget constraints."
            ),
            capabilities=(
                "cost_calculation",
                "budget_validation",
            ),
            agent=BudgetSpecialist(),
        )
    )

    return registry



class SpecialistAgent(Protocol):
    """Interface every specialist agent must implement."""

    def invoke(
        self,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        ...


class AgentRegistryError(Exception):
    pass


class DuplicateAgentError(AgentRegistryError):
    pass


class AgentNotFoundError(AgentRegistryError):
    pass


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    capabilities: tuple[str, ...]
    agent: SpecialistAgent
    enabled: bool = True


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        name = definition.name.strip().lower()

        if not name:
            raise ValueError("Agent name cannot be empty")

        if name in self._agents:
            raise DuplicateAgentError(
                f"Agent '{name}' is already registered"
            )

        normalized = AgentDefinition(
            name=name,
            description=definition.description.strip(),
            capabilities=tuple(
                capability.strip().lower()
                for capability in definition.capabilities
                if capability.strip()
            ),
            agent=definition.agent,
            enabled=definition.enabled,
        )

        self._agents[name] = normalized

    def get(self, name: str) -> AgentDefinition:
        normalized_name = name.strip().lower()

        definition = self._agents.get(normalized_name)

        if definition is None:
            raise AgentNotFoundError(
                f"Agent '{normalized_name}' is not registered"
            )

        if not definition.enabled:
            raise AgentRegistryError(
                f"Agent '{normalized_name}' is disabled"
            )

        return definition

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "capabilities": list(definition.capabilities),
                "enabled": definition.enabled,
            }
            for definition in self._agents.values()
        ]

    def invoke(
        self,
        name: str,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        definition = self.get(name)

        result = definition.agent.invoke(request)

        # Verify that the specialist returned the correct task result.
        if result.task_id != request.task_id:
            raise AgentRegistryError(
                f"Agent '{name}' returned a mismatched task_id"
            )

        if result.agent_name != definition.name:
            raise AgentRegistryError(
                f"Agent '{name}' returned an unexpected agent_name"
            )

        return result
