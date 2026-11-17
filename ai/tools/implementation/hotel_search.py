
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from ai.tools.schema.hotel import HotelSearchInput



# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "hotels_india.json"
)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class HotelDatasetError(RuntimeError):
    """Raised when the hotel dataset cannot be loaded or validated."""


# ---------------------------------------------------------
# Dataset models
# ---------------------------------------------------------

class HotelRecord(BaseModel):
    """Validated representation of one hotel record."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    city: str = Field(min_length=2)
    locality: str
    address: str

    stars: int = Field(ge=1, le=5)
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)

    price_per_night: int = Field(gt=0)
    max_guests: int = Field(ge=1)

    room_types: list[str]
    amenities: list[str]

    check_in: str
    check_out: str
    cancellation_policy: str

    available_from: date
    available_to: date

    availability_status: Literal["AVAILABLE", "UNAVAILABLE", "sample_available", "sample_unavailable"]

    @model_validator(mode="after")
    def validate_availability_dates(self):
        if self.available_to < self.available_from:
            raise ValueError(
                f"Invalid availability range for hotel {self.id}"
            )
        return self

from datetime import date
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AvailabilityPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date = Field(alias="from")
    end: date = Field(alias="to")

    @model_validator(mode="after")
    def validate_period(self):
        if self.end < self.start:
            raise ValueError(
                "Dataset availability end date must be "
                "on or after its start date"
            )
        return self

class DatasetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    version: str
    source: str
    is_live: bool
    currency: str
    disclaimer: str
    availability_period: AvailabilityPeriod

class HotelDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: DatasetMetadata
    hotels: list[HotelRecord]

    @model_validator(mode="after")
    def validate_unique_hotel_ids(self):
        ids = [hotel.id for hotel in self.hotels]

        if len(ids) != len(set(ids)):
            raise ValueError("Hotel dataset contains duplicate IDs.")

        return self


# ---------------------------------------------------------
# Search response models
# ---------------------------------------------------------

class HotelSearchResult(BaseModel):
    hotel_id: str
    name: str
    city: str
    locality: str
    address: str

    stars: int
    rating: float
    review_count: int

    price_per_night: int
    total_price: int
    number_of_nights: int
    currency: Literal["INR"] = "INR"

    max_guests: int
    room_types: list[str]
    amenities: list[str]

    check_in_time: str
    check_out_time: str
    cancellation_policy: str

    availability_status: Literal["AVAILABLE"]


class HotelSearchResponse(BaseModel):
    source: Literal["MOCK"] = "MOCK"
    is_live: Literal[False] = False
    searched_at: datetime
    search_criteria: dict
    total_matches: int
    returned_count: int
    hotels: list[HotelSearchResult]
    disclaimer: str


# ---------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------

@lru_cache(maxsize=4)
def load_hotel_dataset(dataset_path: str) -> HotelDataset:
    """
    Load and validate the hotel dataset.

    The cached dataset avoids reading and parsing the JSON
    file for every individual search.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise HotelDatasetError(
            f"Hotel dataset not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)

        return HotelDataset.model_validate(raw_data)

    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HotelDatasetError(
            f"Unable to load valid hotel dataset: {path}"
        ) from exc


# ---------------------------------------------------------
# Hotel search service
# ---------------------------------------------------------

class HotelSearchService:
    """
    Searches the configured hotel dataset.

    This is a development data provider. It does not check
    real hotel inventory or live room availability.
    """

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
    ) -> None:
        self.dataset_path = str(dataset_path.resolve())
        self.dataset = load_hotel_dataset(self.dataset_path)

    def search(
        self,
        criteria: HotelSearchInput,
    ) -> HotelSearchResponse:

        matching_hotels: list[HotelRecord] = []

        for hotel in self.dataset.hotels:

            if not self._matches_city(hotel, criteria):
                continue

            if not self._matches_dates(hotel, criteria):
                continue

            if not self._matches_guests(hotel, criteria):
                continue

            if not self._matches_budget(hotel, criteria):
                continue

            if not self._matches_stars(hotel, criteria):
                continue

            if not self._matches_amenities(hotel, criteria):
                continue

            matching_hotels.append(hotel)

        # Best rated first; cheaper hotel wins rating ties.
        matching_hotels.sort(
            key=lambda hotel: (
                -hotel.rating,
                hotel.price_per_night,
                hotel.id,
            )
        )

        total_matches = len(matching_hotels)

        selected_hotels = matching_hotels[:criteria.limit]

        number_of_nights = (
            criteria.check_out - criteria.check_in
        ).days

        results = [
            HotelSearchResult(
                hotel_id=hotel.id,
                name=hotel.name,
                city=hotel.city,
                locality=hotel.locality,
                address=hotel.address,
                stars=hotel.stars,
                rating=hotel.rating,
                review_count=hotel.review_count,
                price_per_night=hotel.price_per_night,
                total_price=(
                    hotel.price_per_night * number_of_nights
                ),
                number_of_nights=number_of_nights,
                max_guests=hotel.max_guests,
                room_types=hotel.room_types,
                amenities=hotel.amenities,
                check_in_time=hotel.check_in,
                check_out_time=hotel.check_out,
                cancellation_policy=hotel.cancellation_policy,
                availability_status="AVAILABLE",
            )
            for hotel in selected_hotels
        ]

        return HotelSearchResponse(
            searched_at=datetime.now(timezone.utc),
            search_criteria=criteria.model_dump(mode="json"),
            total_matches=total_matches,
            returned_count=len(results),
            hotels=results,
            disclaimer=self.dataset.metadata.disclaimer,
        )

    @staticmethod
    def _matches_city(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        return hotel.city.casefold() == criteria.city.casefold()

    @staticmethod
    def _matches_dates(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        """
        The entire stay must fit within the configured
        mock availability window.

        Checkout is treated as the departure date, not
        as another overnight stay.
        """

        return (
            hotel.availability_status.casefold() in {"available", "sample_available"}
            and criteria.check_in >= hotel.available_from
            and criteria.check_out <= hotel.available_to
        )

    @staticmethod
    def _matches_guests(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        return hotel.max_guests >= criteria.adults

    @staticmethod
    def _matches_budget(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        return (
            hotel.price_per_night
            <= criteria.max_price_per_night
        )

    @staticmethod
    def _matches_stars(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        if criteria.min_stars is None:
            return True

        return hotel.stars >= criteria.min_stars

    @staticmethod
    def _matches_amenities(
        hotel: HotelRecord,
        criteria: HotelSearchInput,
    ) -> bool:
        if not criteria.required_amenities:
            return True

        available_amenities = {
            amenity.casefold()
            for amenity in hotel.amenities
        }

        requested_amenities = {
            amenity.casefold()
            for amenity in criteria.required_amenities
        }

        return requested_amenities.issubset(
            available_amenities
        )


# ---------------------------------------------------------
# Public function
# ---------------------------------------------------------

def search_hotels(
    criteria: HotelSearchInput,
) -> HotelSearchResponse:
    """
    Search hotels using validated input.

    This function will later be wrapped as a LangChain
    tool for the ReAct agent.
    """

    service = HotelSearchService()
    return service.search(criteria)
