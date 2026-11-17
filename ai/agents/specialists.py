
from __future__ import annotations

from ai.agents.specialist_contracts import (
    AgentStatus,
    SpecialistFinding,
    SpecialistRequest,
    SpecialistResult,
)
from ai.graphs.execution_strategy_comparison import (
    build_react_graph,
)


class TravelSpecialist:
    def __init__(
        self,
        model_name: str = "qwen3:8b",
        max_steps: int = 6,
    ) -> None:
        # Build once and reuse for multiple requests.
        self._graph = build_react_graph(
            model_name=model_name,
            max_steps=max_steps,
        )

    def invoke(
        self,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        from langchain_core.messages import HumanMessage

        state = {
            "messages": [
                HumanMessage(content=request.user_request)
            ],
            "steps_used": 0,
            "final_status": "running",
        }

        result = self._graph.invoke(
            state,
            config={"recursion_limit": 30},
        )

        final_text = ""

        for message in reversed(result["messages"]):
            if message.type == "ai" and message.content:
                final_text = str(message.content)
                break

        if not final_text:
            return SpecialistResult(
                agent_name="travel",
                task_id=request.task_id,
                status=AgentStatus.PARTIAL,
                summary="The travel agent did not produce a final answer.",
                warnings=["No final AI response was available."],
            )

        return SpecialistResult(
            agent_name="travel",
            task_id=request.task_id,
            status=AgentStatus.SUCCESS,
            summary=final_text,
            warnings=[
                "Hotel results currently come from mock data, "
                "not live provider inventory."
            ],
        )


class BudgetSpecialist:
    """
    Deterministic budget calculator.

    Expected context:
    {
        "cost_items": [
            {"name": "Laptop", "amount": 65000, "currency": "INR"},
            {"name": "Hotel", "amount": 8000, "currency": "INR"}
        ],
        "budget_limit": 150000
    }
    """

    def invoke(
        self,
        request: SpecialistRequest,
    ) -> SpecialistResult:
        context = request.context

        cost_items = context.get("cost_items", [])
        budget_limit = context.get("budget_limit")

        if not isinstance(cost_items, list):
            return self._needs_input(
                request,
                "cost_items must be a list.",
            )

        if budget_limit is None:
            return self._needs_input(
                request,
                "A budget_limit is required.",
            )

        try:
            budget_limit = float(budget_limit)

            if budget_limit < 0:
                raise ValueError("Budget cannot be negative")

            total = 0.0
            findings = []

            for item in cost_items:
                amount = float(item["amount"])
                currency = item.get("currency", "INR")

                if amount < 0:
                    raise ValueError(
                        "Cost amounts cannot be negative"
                    )

                if currency != "INR":
                    raise ValueError(
                        "All cost items must use INR for this calculation"
                    )

                total += amount

                findings.append(
                    SpecialistFinding(
                        title=str(item["name"]),
                        description=f"Estimated cost: ₹{amount:,.2f}",
                        estimated_cost=amount,
                        currency=currency,
                        source="provided_context",
                        is_live=False,
                    )
                )

        except (KeyError, TypeError, ValueError) as exc:
            return self._needs_input(
                request,
                f"Could not calculate the budget: {exc}",
            )

        remaining = budget_limit - total
        within_budget = remaining >= 0

        return SpecialistResult(
            agent_name="budget",
            task_id=request.task_id,
            status=AgentStatus.SUCCESS,
            summary=(
                f"Total estimated cost is ₹{total:,.2f}. "
                f"Budget is ₹{budget_limit:,.2f}. "
                f"{'Within budget' if within_budget else 'Over budget'} "
                f"by ₹{abs(remaining):,.2f}."
            ),
            findings=findings,
            total_estimated_cost=total,
            warnings=(
                []
                if within_budget
                else ["The combined estimated cost exceeds the budget."]
            ),
        )

    @staticmethod
    def _needs_input(
        request: SpecialistRequest,
        reason: str,
    ) -> SpecialistResult:
        return SpecialistResult(
            agent_name="budget",
            task_id=request.task_id,
            status=AgentStatus.NEEDS_INPUT,
            summary=reason,
            missing_information=[reason],
        )
