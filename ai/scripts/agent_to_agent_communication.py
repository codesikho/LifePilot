"""
LifePilot - Module 4: Five Concepts Integration Test

Concepts:
1. Shared State
2. Structured Messages
3. LangGraph Command
4. Parent Graph with child workflows
5. Send + parallel execution

All prices are mock estimates. No LLM or external tools are used.
"""

from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send


# ============================================================
# 1. STRUCTURED MESSAGE
# ============================================================

class AgentMessage(BaseModel):
    """A consistent message format used by agents."""

    model_config = ConfigDict(extra="forbid")

    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    message_type: str = "result"
    task: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 2. STATE SCHEMAS
# ============================================================

class SpecialistState(TypedDict, total=False):
    """
    State used inside an individual child workflow.
    It contains the inputs that specialist needs and its outputs.
    """

    user_request: str
    budget_limit: float
    trip_cost: float
    laptop_cost: float

    agent_results: Annotated[list[dict], add]
    agent_messages: Annotated[list[dict], add]
    execution_trace: Annotated[list[str], add]


class LifePilotState(TypedDict, total=False):
    """
    State owned by the parent workflow.
    Reducers allow parallel agents to append results/messages/traces.
    """

    user_request: str
    budget_limit: float
    trip_cost: float
    laptop_cost: float

    agent_results: Annotated[list[dict], add]
    agent_messages: Annotated[list[dict], add]
    execution_trace: Annotated[list[str], add]

    final_summary: str


# ============================================================
# 3. FIVE TEST INPUTS
# ============================================================

SAMPLE_INPUTS = [
    {
        "user_request": "Plan a family trip and find a laptop",
        "budget_limit": 200000,
        "trip_cost": 60000,
        "laptop_cost": 90000,
    },
    {
        "user_request": "Plan a short trip and find a work laptop",
        "budget_limit": 120000,
        "trip_cost": 45000,
        "laptop_cost": 65000,
    },
    {
        "user_request": "Find a premium family holiday and a laptop",
        "budget_limit": 150000,
        "trip_cost": 95000,
        "laptop_cost": 70000,
    },
    {
        "user_request": "Plan a budget trip and find a basic laptop",
        "budget_limit": 100000,
        "trip_cost": 35000,
        "laptop_cost": 45000,
    },
    {
        "user_request": "Plan a large family trip and find a gaming laptop",
        "budget_limit": 250000,
        "trip_cost": 130000,
        "laptop_cost": 145000,
    },
]


# ============================================================
# 4. TRAVEL SPECIALIST NODE
# ============================================================

def travel_node(state: SpecialistState):
    cost = float(state["trip_cost"])

    result = {
        "agent": "travel",
        "cost": cost,
        "currency": "INR",
    }

    message = AgentMessage(
        sender="travel_agent",
        recipient="lifepilot_parent",
        task="travel_estimate_ready",
        payload={"trip_cost": cost},
    )

    print(f"[Travel Agent] Trip estimate: ₹{cost:,.0f}")

    return {
        "agent_results": [result],
        "agent_messages": [message.model_dump()],
        "execution_trace": [f"Travel Agent estimated ₹{cost:,.0f}"],
    }


# ============================================================
# 5. SHOPPING SPECIALIST NODE
# ============================================================

def shopping_node(state: SpecialistState):
    cost = float(state["laptop_cost"])

    result = {
        "agent": "shopping",
        "cost": cost,
        "currency": "INR",
    }

    message = AgentMessage(
        sender="shopping_agent",
        recipient="lifepilot_parent",
        task="laptop_estimate_ready",
        payload={"laptop_cost": cost},
    )

    print(f"[Shopping Agent] Laptop estimate: ₹{cost:,.0f}")

    return {
        "agent_results": [result],
        "agent_messages": [message.model_dump()],
        "execution_trace": [f"Shopping Agent estimated ₹{cost:,.0f}"],
    }


# ============================================================
# 6. CHILD GRAPH: TRAVEL
# ============================================================

def build_travel_child_graph():
    builder = StateGraph(SpecialistState)

    builder.add_node("travel_specialist", travel_node)
    builder.add_edge(START, "travel_specialist")
    builder.add_edge("travel_specialist", END)

    return builder.compile()


# ============================================================
# 7. CHILD GRAPH: SHOPPING
# ============================================================

def build_shopping_child_graph():
    builder = StateGraph(SpecialistState)

    builder.add_node("shopping_specialist", shopping_node)
    builder.add_edge(START, "shopping_specialist")
    builder.add_edge("shopping_specialist", END)

    return builder.compile()


# ============================================================
# 8. PARENT WRAPPERS FOR CHILD GRAPHS
# ============================================================
# Important:
# A child graph may return its full state. These wrappers deliberately
# pass only the output fields back to the parent graph.
# This prevents parallel child workflows from writing the same
# single-value parent keys such as user_request and budget_limit.

OUTPUT_KEYS = ("agent_results", "agent_messages", "execution_trace")


def run_travel_workflow(parent_state: LifePilotState):
    child_input = {
        "user_request": parent_state["user_request"],
        "budget_limit": parent_state["budget_limit"],
        "trip_cost": parent_state["trip_cost"],
        "agent_results": [],
        "agent_messages": [],
        "execution_trace": [],
    }

    child_result = build_travel_child_graph().invoke(child_input)

    return {
        key: child_result.get(key, [])
        for key in OUTPUT_KEYS
    }


def run_shopping_workflow(parent_state: LifePilotState):
    child_input = {
        "user_request": parent_state["user_request"],
        "budget_limit": parent_state["budget_limit"],
        "laptop_cost": parent_state["laptop_cost"],
        "agent_results": [],
        "agent_messages": [],
        "execution_trace": [],
    }

    child_result = build_shopping_child_graph().invoke(child_input)

    return {
        key: child_result.get(key, [])
        for key in OUTPUT_KEYS
    }


# ============================================================
# 9. SUPERVISOR USING LANGGRAPH COMMAND
# ============================================================

def supervisor_node(state: LifePilotState) -> Command:
    print("[Supervisor] Received user request")

    return Command(
        update={
            "execution_trace": [
                "Supervisor selected Travel and Shopping workflows"
            ]
        },
        goto="dispatch",
    )


# ============================================================
# 10. DISPATCH NODE
# ============================================================

def dispatch_node(state: LifePilotState):
    print("[Dispatcher] Preparing parallel tasks")

    return {
        "execution_trace": [
            "Dispatcher preparing two independent tasks"
        ]
    }


# ============================================================
# 11. SEND: PARALLEL EXECUTION
# ============================================================

def dispatch_tasks(state: LifePilotState):
    common_input = {
        "user_request": state["user_request"],
        "budget_limit": state["budget_limit"],
        "agent_results": [],
        "agent_messages": [],
        "execution_trace": [],
    }

    print("[Send] Dispatching Travel and Shopping")

    return [
        Send(
            "travel_workflow",
            {
                **common_input,
                "trip_cost": state["trip_cost"],
            },
        ),
        Send(
            "shopping_workflow",
            {
                **common_input,
                "laptop_cost": state["laptop_cost"],
            },
        ),
    ]


# ============================================================
# 12. BUDGET AGENT
# ============================================================

def budget_node(state: LifePilotState):
    results = state.get("agent_results", [])

    # Defensive check: we expect one result from each specialist.
    received_agents = {item["agent"] for item in results}
    expected_agents = {"travel", "shopping"}

    if received_agents != expected_agents:
        raise ValueError(
            "Budget Agent expected Travel and Shopping results. "
            f"Received: {sorted(received_agents)}"
        )

    total_cost = sum(float(item["cost"]) for item in results)
    budget = float(state["budget_limit"])
    remaining = budget - total_cost

    status = "WITHIN BUDGET" if remaining >= 0 else "OVER BUDGET"

    message = AgentMessage(
        sender="budget_agent",
        recipient="lifepilot_parent",
        task="combined_budget_checked",
        payload={
            "total_cost": total_cost,
            "budget_limit": budget,
            "remaining": remaining,
        },
    )

    print("[Budget Agent] Received both agent results")
    print(f"[Budget Agent] Combined cost: ₹{total_cost:,.0f}")

    return {
        "agent_messages": [message.model_dump()],
        "final_summary": (
            f"{status}: "
            f"Total = ₹{total_cost:,.0f}; "
            f"Budget = ₹{budget:,.0f}; "
            f"{'Remaining' if remaining >= 0 else 'Excess'} "
            f"= ₹{abs(remaining):,.0f}"
        ),
        "execution_trace": ["Budget Agent checked combined results"],
    }


# ============================================================
# 13. PARENT GRAPH
# ============================================================

def build_parent_graph():
    # The parent graph MUST use LifePilotState, not SpecialistState.
    builder = StateGraph(LifePilotState)

    # Register wrappers, not raw child graphs.
    # Wrappers filter child outputs before updating parent state.
    builder.add_node("travel_workflow", run_travel_workflow)
    builder.add_node("shopping_workflow", run_shopping_workflow)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("budget_agent", budget_node)

    builder.add_edge(START, "supervisor")

    # Supervisor Command routes to dispatch.
    # Dispatch then uses Send to fan out to both specialists.
    builder.add_conditional_edges("dispatch", dispatch_tasks)

    # This join waits for both parallel branches before Budget Agent.
    builder.add_edge(
        ["travel_workflow", "shopping_workflow"],
        "budget_agent",
    )

    builder.add_edge("budget_agent", END)

    return builder.compile()


# ============================================================
# 14. RUN FIVE INPUTS
# ============================================================

def main():
    graph = build_parent_graph()

    print("\n" + "=" * 70)
    print("LIFEPILOT - MODULE 4 INTEGRATION TEST")
    print("=" * 70)

    for index, sample in enumerate(SAMPLE_INPUTS, start=1):
        print("\n" + "#" * 70)
        print(f"TEST INPUT {index}")
        print("#" * 70)

        print("Request:", sample["user_request"])
        print(f"Budget: ₹{sample['budget_limit']:,.0f}")

        initial_state: LifePilotState = {
            **sample,
            "agent_results": [],
            "agent_messages": [],
            "execution_trace": [],
        }

        result = graph.invoke(initial_state)

        print("\n--- EXECUTION TRACE ---")
        for event in result.get("execution_trace", []):
            print("-", event)

        print("\n--- AGENT RESULTS ---")
        for item in result.get("agent_results", []):
            print(item)

        print("\n--- STRUCTURED MESSAGES ---")
        for message in result.get("agent_messages", []):
            print(
                f"{message['sender']} -> {message['recipient']} | "
                f"Task: {message['task']} | "
                f"Payload: {message['payload']}"
            )

        print("\n--- FINAL RESULT ---")
        print(result["final_summary"])

    print("\n" + "=" * 70)
    print("ALL FIVE TEST INPUTS COMPLETED")
    print("=" * 70)

    print(
        """
CONCEPTS DEMONSTRATED:

1. Shared State
   Parent state collects results, messages and execution traces.

2. Structured Messages
   AgentMessage defines a consistent message format.

3. Command
   Supervisor routes execution to the dispatcher.

4. Parent Graph
   Parent workflow coordinates the Travel and Shopping workflows.

5. Send + Parallel Execution
   Travel and Shopping are dispatched independently.
   Budget runs after both workflows complete.

Note: Costs are hardcoded mock estimates. This script does not use an LLM.
"""
    )


if __name__ == "__main__":
    main()
