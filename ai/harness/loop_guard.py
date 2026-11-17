from ai.harness.repeated_call_detector import (
    RepeatedToolCallDetector,
)


class RepeatedToolCallError(RuntimeError):
    pass


class LoopGuard:

    def __init__(self, max_identical_calls: int = 2):

        self.detector = RepeatedToolCallDetector(
            max_identical_calls
        )

    def check_and_register(
        self,
        tool_name: str,
        arguments: dict,
    ):

        count = self.detector.register(
            tool_name,
            arguments,
        )

        if count > self.detector.max_identical_calls:
            raise RepeatedToolCallError(
                f"Repeated tool call blocked: "
                f"{tool_name} called {count} times"
            )