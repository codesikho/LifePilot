from typing import TypedDict, Annotated
from operator import add


class LifePilotState(TypedDict):
    user_request: str

    budget_limit: float

    hotel_findings: list[dict]

    activity_findings: list[dict]

    agent_messages: Annotated[list[dict], add]

    final_result: str