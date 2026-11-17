class SearchHotels:
    name = "search_hotels"
    requires_approval = False

    def execute(self, arguments: dict) -> str:

        city = arguments["city"]

        return (
            f"Hotel A in {city}: ₹4,500/night\n"
            f"Hotel B in {city}: ₹4,000/night"
        )


class BookHotel:

    name = "book_hotel"
    requires_approval = True

    def execute(self, arguments: dict) -> str:

        hotel_id = arguments["hotel_id"]

        # Demonstration only. No real booking occurs.
        return f"Demo booking request executed for {hotel_id}"