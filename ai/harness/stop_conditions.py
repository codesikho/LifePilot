from dataclasses import dataclass
from enum import Enum

class StopReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: StopReason | None = None

class StopConditionEvaluator:
    def evaluate(self, *, cancelled=False, final_answer=False, max_steps_reached=False):
        if cancelled:
            return StopDecision(True, StopReason.CANCELLED)
        if final_answer:
            return StopDecision(True, StopReason.FINAL_ANSWER)
        if max_steps_reached:
            return StopDecision(True, StopReason.MAX_STEPS)
        return StopDecision(False)
