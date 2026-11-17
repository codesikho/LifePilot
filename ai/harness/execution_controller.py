from dataclasses import dataclass
from typing import Any, Protocol

from ai.state.agent_context import AgentContext

from ai.harness.step_limiter import StepLimiter
from ai.harness.loop_guard import LoopGuard
from ai.harness.execution_tracker import ExecutionTracker


# ---------------------------------------------
# 1. Model decision
# ---------------------------------------------

@dataclass(frozen=True)
class AgentDecision:

    kind: str

    content: str = ""

    tool_name: str | None = None

    tool_input: dict[str, Any] | None = None


# ---------------------------------------------
# 2. Model interface
# ---------------------------------------------

class AgentModel(Protocol):

    def decide(
        self,
        context: AgentContext,
    ) -> AgentDecision:
        ...


# ---------------------------------------------
# 3. Tool interface
# ---------------------------------------------

class AgentTool(Protocol):

    name: str

    requires_approval: bool

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> str:
        ...


# ---------------------------------------------
# 4. Result
# ---------------------------------------------

@dataclass
class AgentResult:

    answer: str

    steps: int

    observations: list[str]

    events: list


# ---------------------------------------------
# 5. Application exception
# ---------------------------------------------

class DecisionRejected(RuntimeError):
    pass


# ---------------------------------------------
# 6. Execution controller
# ---------------------------------------------

class ExecutionController:

    def __init__(
        self,
        model: AgentModel,
        tools: list[AgentTool],
        *,
        max_steps: int = 8,
        max_identical_tool_calls: int = 2,
    ):

        self.model = model

        # Application owns the available tools.
        self.tools = {
            tool.name: tool
            for tool in tools
        }

        # Safety components are scoped to this controller.
        self.step_limiter = StepLimiter(max_steps)

        self.loop_guard = LoopGuard(
            max_identical_tool_calls
        )

        self.tracker = ExecutionTracker()

    def run(
        self,
        user_input: str,
        *,
        approved_actions: set[str] | None = None,
    ) -> AgentResult:

        approved_actions = approved_actions or set()

        context = AgentContext(
            user_input=user_input
        )

        self.tracker.record(
            "started",
            "Agent execution started",
            0,
        )

        try:

            while True:

                # ---------------------------------
                # A. Check maximum steps
                # ---------------------------------

                self.step_limiter.ensure_capacity(
                    context.step_count
                )

                # ---------------------------------
                # B. Ask model for next decision
                # ---------------------------------

                decision = self.model.decide(context)

                context.step_count += 1

                self.tracker.record(
                    "decision",
                    f"Model decision: {decision.kind}",
                    context.step_count,
                )

                # ---------------------------------
                # C. Handle final answer
                # ---------------------------------

                if decision.kind == "final":

                    self.tracker.record(
                        "completed",
                        "Agent returned final answer",
                        context.step_count,
                    )

                    return AgentResult(
                        answer=decision.content,
                        steps=context.step_count,
                        observations=context.observations.copy(),
                        events=self.tracker.events.copy(),
                    )

                # ---------------------------------
                # D. Validate decision
                # ---------------------------------

                if decision.kind != "tool":

                    raise DecisionRejected(
                        f"Unsupported decision: {decision.kind}"
                    )

                if not decision.tool_name:

                    raise DecisionRejected(
                        "Tool name is required"
                    )

                arguments = decision.tool_input or {}

                # ---------------------------------
                # E. Validate registered tool
                # ---------------------------------

                tool = self.tools.get(
                    decision.tool_name
                )

                if tool is None:

                    raise DecisionRejected(
                        f"Tool not registered: "
                        f"{decision.tool_name}"
                    )

                # ---------------------------------
                # F. Check approval
                # ---------------------------------

                if tool.requires_approval:

                    if decision.tool_name not in approved_actions:

                        raise DecisionRejected(
                            f"{decision.tool_name} "
                            f"requires explicit approval"
                        )

                # ---------------------------------
                # G. Check repeated calls
                # ---------------------------------

                self.loop_guard.check_and_register(
                    decision.tool_name,
                    arguments,
                )

                self.tracker.record(
                    "tool_started",
                    f"Executing {decision.tool_name}",
                    context.step_count,
                )

                # ---------------------------------
                # H. Execute tool
                # ---------------------------------

                result = tool.execute(arguments)

                # ---------------------------------
                # I. Save observation
                # ---------------------------------

                observation = (
                    f"{decision.tool_name}: {result}"
                )

                context.observations.append(
                    observation
                )

                self.tracker.record(
                    "observation",
                    observation,
                    context.step_count,
                )

        except Exception as exc:

            self.tracker.record(
                "failed",
                f"{type(exc).__name__}: {exc}",
                context.step_count,
            )

            raise