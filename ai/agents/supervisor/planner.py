from __future__ import annotations

import json

from ai.agents.supervisor.schema import SupervisorPlan


SUPERVISOR_PROMPT = """
You are the planning component of LifePilot.

Your job is to break the user's request into tasks for
the available specialist agents.

Rules:

1. Select only agents listed as available.
2. Never invent an agent.
3. Create separate tasks for separate responsibilities.
4. Do not ask the Budget Agent to calculate before cost
   estimates are available.
5. If important information is missing, add it to
   missing_information.
6. Do not invent dates, budgets, prices or guest counts.
7. Keep each task focused on one specialist's responsibility.
8. Do not perform purchases, bookings or payments.
9. If the request needs a specialist that is not available,
   explain this in missing_information.
10. Return only the requested structured plan.

The Travel Agent searches hotels using the available hotel tool.
The Shopping Agent researches products using its product tools.
The Budget Agent calculates totals from supplied cost_items.
"""


class OllamaSupervisorPlanner:
    def __init__(
        self,
        model_name: str = "qwen3:8b",
    ) -> None:
        # Import lazily so the orchestration layer and its tests can run
        # with fake planners without requiring an Ollama installation.
        from langchain_ollama import ChatOllama

        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
        ).with_structured_output(
            SupervisorPlan,
            method="json_schema",
        )

    def create_plan(
        self,
        user_request: str,
        available_agents: list[dict],
    ) -> SupervisorPlan:
        agent_description = json.dumps(
            available_agents,
            ensure_ascii=False,
        )

        response = self.llm.invoke(
            [
                (
                    "system",
                    SUPERVISOR_PROMPT,
                ),
                (
                    "human",
                    f"""
User request:
{user_request}

Available agents:
{agent_description}

Create the execution plan.
""",
                ),
            ]
        )

        return SupervisorPlan.model_validate(response)