from ai.harness.execution_controller import (
    ExecutionController,
    AgentDecision,
    AgentResult,
)


class AgentRunner:

    def __init__(
        self,
        model,
        tools,
        *,
        max_steps=8,
        max_identical_tool_calls=2,
    ):

        self.model = model
        self.tools = tools

        self.max_steps = max_steps

        self.max_identical_tool_calls = (
            max_identical_tool_calls
        )

    def run(
        self,
        user_input: str,
        *,
        approved_actions: set[str] | None = None,
    ) -> AgentResult:

        # Fresh controller for every execution.
        controller = ExecutionController(
            model=self.model,
            tools=self.tools,
            max_steps=self.max_steps,
            max_identical_tool_calls=(
                self.max_identical_tool_calls
            ),
        )

        return controller.run(
            user_input,
            approved_actions=approved_actions,
        )


__all__ = [
    "AgentRunner",
    "AgentDecision",
    "AgentResult",
]