from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class ExecutionEvent:
    event_type: str
    message: str
    step: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ExecutionTracker:
    def __init__(self):
        self.events = []

    def record(self, event_type, message, step):
        self.events.append(ExecutionEvent(event_type, message, step))
