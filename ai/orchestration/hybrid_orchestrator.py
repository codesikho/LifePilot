from ai.agents.agent_registry import AgentRegistry
from ai.agents.supervisor.planner import OllamaSupervisorPlanner
from ai.agents.supervisor.supervisor import Supervisor, SupervisorResult
from ai.orchestration.agent_dispatcher import SwarmDispatcher
from ai.orchestration.swarm_router import SwarmRouter


class HybridOrchestrator:
    """
    Combines central Supervisor planning with dynamic Swarm handoffs.

    Supervisor:
        Plans the work and assigns the initial specialist.

    Swarm Router:
        Executes that specialist and validates any handoff requests.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        planner: OllamaSupervisorPlanner,
        allowed_handoffs: dict[str, list[str]],
        max_handoffs: int = 5,
    ) -> None:
        self.registry = registry

        self.swarm_router = SwarmRouter(
            registry=registry,
            allowed_handoffs=allowed_handoffs,
            max_handoffs=max_handoffs,
        )

        dispatcher = SwarmDispatcher(
            swarm_router=self.swarm_router,
        )

        self.supervisor = Supervisor(
            registry=registry,
            planner=planner,
            dispatcher=dispatcher,
        )

    def run(self, user_request: str) -> SupervisorResult:
        """
        Plan and execute a user request using Supervisor + Swarm.
        """
        return self.supervisor.run(user_request)