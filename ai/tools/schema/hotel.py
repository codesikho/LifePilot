
from datetime import date
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class HotelSearchInput(BaseModel):
    """
    Validated input contract for the LifePilot hotel-search tool.

    All monetary amounts are in INR.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    city: Annotated[
        str,
        Field(
            min_length=2,
            max_length=100,
            description="Indian destination city, e.g. Jaipur, Mumbai or Noida.",
        ),
    ]

    check_in: date = Field(
        description="Hotel check-in date in YYYY-MM-DD format."
    )

    check_out: date = Field(
        description="Hotel check-out date in YYYY-MM-DD format."
    )

    adults: Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=20,
            description="Number of adult guests.",
        ),
    ]

    max_price_per_night: Annotated[
        int,
        Field(
            strict=True,
            gt=0,
            le=1_000_000,
            description="Maximum hotel price per night in INR.",
        ),
    ]

    min_stars: Annotated[
        int | None,
        Field(
            default=None,
            strict=True,
            ge=1,
            le=5,
            description="Minimum hotel star category, from 1 to 5.",
        ),
    ]

    required_amenities: Annotated[
        list[str],
        Field(
            default_factory=list,
            max_length=15,
            description="Amenities that the hotel must provide.",
        ),
    ]

    limit: Annotated[
        int,
        Field(
            default=10,
            strict=True,
            ge=1,
            le=20,
            description="Maximum number of matching hotels to return.",
        ),
    ]

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        value = " ".join(value.split())

        if not value:
            raise ValueError("City cannot be empty.")

        return value

    @field_validator("required_amenities")
    @classmethod
    def normalize_amenities(cls, values: list[str]) -> list[str]:
        normalized = []

        for value in values:
            value = value.strip().lower()

            if not value:
                raise ValueError("Amenity cannot be empty.")

            if value not in normalized:
                normalized.append(value)

        return normalized

    @model_validator(mode="after")
    def validate_stay_dates(self):
        if self.check_out <= self.check_in:
            raise ValueError(
                "Check-out date must be after check-in date."
            )

        return self
