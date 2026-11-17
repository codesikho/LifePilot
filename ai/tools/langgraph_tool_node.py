
from datetime import date

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from ai.tools.schema.hotel import HotelSearchInput
from ai.tools.implementation.hotel_search import (
    HotelSearchService,
)


# Create one reusable search service.
# The dataset is loaded and validated when the service starts.
hotel_search_service = HotelSearchService()


@tool(
    "search_hotels",
    args_schema=HotelSearchInput,
)
def search_hotels_tool(
    city: str,
    check_in: date,
    check_out: date,
    adults: int,
    max_price_per_night: int,
    min_stars: int | None = None,
    required_amenities: list[str] | None = None,
    limit: int = 10,
) -> dict:
    """
    Search hotels in an Indian city using the configured
    hotel dataset.

    Use this tool when a user wants to find or compare hotels.

    The current dataset contains MOCK records only.
    It does not represent real hotel inventory or live availability.

    This tool only searches. It never books a hotel.
    """

    criteria = HotelSearchInput(
        city=city,
        check_in=check_in,
        check_out=check_out,
        adults=adults,
        max_price_per_night=max_price_per_night,
        min_stars=min_stars,
        required_amenities=required_amenities or [],
        limit=limit,
    )

    result = hotel_search_service.search(criteria)

    return result.model_dump(mode="json")


def create_tool_node() -> ToolNode:
    """
    Create the LangGraph node responsible for executing
    hotel-search tool calls.
    """

    return ToolNode(
        tools=[search_hotels_tool],
    )
