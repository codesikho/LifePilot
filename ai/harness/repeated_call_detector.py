import json
from collections import Counter


class RepeatedToolCallDetector:

    def __init__(self, max_identical_calls: int = 2):

        if max_identical_calls < 1:
            raise ValueError(
                "max_identical_calls must be at least 1"
            )

        self.max_identical_calls = max_identical_calls
        self.calls = Counter()

    def register(
        self,
        tool_name: str,
        arguments: dict,
    ) -> int:

        # Same tool + same arguments = identical call
        key = (
            tool_name,
            json.dumps(
                arguments,
                sort_keys=True,
                default=str,
            ),
        )

        self.calls[key] += 1

        return self.calls[key]