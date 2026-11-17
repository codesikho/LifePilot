
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AgentLoopEvaluation:
    completed: bool
    total_steps: int
    tool_calls: int
    repeated_tool_calls: int
    failures: int
    duration_seconds: float
    passed: bool
    warnings: list[str]


class AgentLoopEvaluator:

    def evaluate(
        self,
        events: list[Any],
        max_steps: int = 10,
        max_tool_calls: int = 5,
    ) -> AgentLoopEvaluation:

        warnings = []

        if not events:
            return AgentLoopEvaluation(
                completed=False,
                total_steps=0,
                tool_calls=0,
                repeated_tool_calls=0,
                failures=0,
                duration_seconds=0,
                passed=False,
                warnings=["No execution events available"],
            )

        started_events = [
            event for event in events
            if event.event_type == "agent_started"
        ]

        completed = any(
            event.event_type == "agent_completed"
            for event in events
        )

        total_steps = max(
            (event.step for event in events),
            default=0,
        )

        tool_calls = sum(
            event.event_type == "tool_started"
            for event in events
        )

        failures = sum(
            event.event_type == "failed"
            for event in events
        )

        repeated_tool_calls = sum(
            event.event_type == "repeated_tool_call"
            for event in events
        )

        duration_seconds = 0.0

        if started_events:
            start_time = started_events[0].timestamp
            end_time = max(
                event.timestamp for event in events
            )

            duration_seconds = (
                end_time - start_time
            ).total_seconds()

        if not completed:
            warnings.append("Agent did not complete successfully")

        if total_steps > max_steps:
            warnings.append("Maximum step limit exceeded")

        if tool_calls > max_tool_calls:
            warnings.append("Too many tool calls")

        if repeated_tool_calls > 0:
            warnings.append("Repeated tool calls detected")

        if failures > 0:
            warnings.append("Execution failures detected")

        passed = (
            completed
            and total_steps <= max_steps
            and tool_calls <= max_tool_calls
            and repeated_tool_calls == 0
            and failures == 0
        )

        return AgentLoopEvaluation(
            completed=completed,
            total_steps=total_steps,
            tool_calls=tool_calls,
            repeated_tool_calls=repeated_tool_calls,
            failures=failures,
            duration_seconds=duration_seconds,
            passed=passed,
            warnings=warnings,
        )
