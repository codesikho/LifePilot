from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:

    user_input: str

    observations: list[str] = field(
        default_factory=list
    )

    step_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )