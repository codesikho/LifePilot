
from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from ai.tools.langgraph_tool_node import search_hotels_tool


SYSTEM_PROMPT = """
You are LifePilot, a helpful hotel-search assistant.

Your job is to understand the user's hotel-search request and
use the available hotel search tool when sufficient information
is available.

IMPORTANT RULES:

1. The hotel dataset is MOCK data, not live provider data.
2. Never claim that a hotel is live, verified, or bookable.
3. Never invent hotel prices, ratings, availability, or amenities.
4. Never invent missing travel dates, guest counts, or budget.
5. If required search information is missing, ask the user
   one concise clarification question.
6. Use the search_hotels tool for hotel searches.
7. Explain that results are based on mock data.
8. You can search hotels, but you cannot book or pay for them.
9. After receiving tool results, inspect them before answering.
10. If no hotels match, say so clearly.

Required search information:
- City
- Check-in date
- Check-out date
- Number of adults
- Maximum price per night

Keep your language simple and your response practical.
"""


class HotelAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps_used: int
    final_status: str


def route_after_agent(
    state: HotelAgentState,
    max_steps: int,
) -> Literal["tools", "limit_reached", "end"]:
    """
    Decide what happens after the LLM responds.

    - Tool calls present: execute tools.
    - No tool calls: finish the conversation.
    - Step limit reached: stop before executing another tool.
    """

    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
        return "end"

    if state.get("steps_used", 0) >= max_steps:
        return "limit_reached"

    return "tools"


def build_react_graph(
    model_name: str,
    max_steps: int = 6,
):
    """
    Build and compile the LifePilot ReAct graph.

    max_steps limits the number of LLM reasoning rounds.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )

    # Make the tool schema available to the LLM.
    llm_with_tools = llm.bind_tools(
        [search_hotels_tool]
    )

    # ToolNode executes the tool calls requested by the LLM.
    tool_node = ToolNode(
        tools=[search_hotels_tool]
    )

    def agent_node(state: HotelAgentState):
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]
        )

        return {
            "messages": [response],
            "steps_used": state.get("steps_used", 0) + 1,
        }

    def limit_reached_node(state: HotelAgentState):
        from langchain_core.messages import AIMessage

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I stopped because the search reached its "
                        "configured execution limit. The results "
                        "may be incomplete. Please narrow or simplify "
                        "your request."
                    )
                )
            ],
            "final_status": "limit_reached",
        }

    def route(state: HotelAgentState):
        return route_after_agent(
            state=state,
            max_steps=max_steps,
        )

    graph = StateGraph(HotelAgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("limit_reached", limit_reached_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route,
        {
            "tools": "tools",
            "limit_reached": "limit_reached",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("limit_reached", END)

    return graph.compile()


def run_hotel_react_agent(
    query: str,
    model_name: str | None = None,
    max_steps: int = 6,
) -> HotelAgentState:
    """
    Run a hotel-search request through the ReAct graph.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    selected_model = (
        model_name
        or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    )

    graph = build_react_graph(
        model_name=selected_model,
        max_steps=max_steps,
    )

    initial_state: HotelAgentState = {
        "messages": [HumanMessage(content=query)],
        "steps_used": 0,
        "final_status": "running",
    }

    return graph.invoke(
        initial_state,
        config={
            "recursion_limit": 30,
        },
    )
