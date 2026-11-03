"""Small builders shared by the test-suite."""

from __future__ import annotations

from uuid import UUID, uuid4

from pulso_matching.models import Location, Order, Place, Pulser, Vehicle

# Zócalo, Mexico City.
CDMX = (19.4326, -99.1332)


def make_order(lat: float = CDMX[0], lng: float = CDMX[1], order_id: UUID | None = None) -> Order:
    return Order(id=order_id or uuid4(), pickup=Place(lat=lat, lng=lng, address="pickup"))


def make_pulser(
    pulser_id: str,
    lat: float = CDMX[0],
    lng: float = CDMX[1],
    *,
    rating: float = 4.5,
    active_tasks: int = 0,
    max_tasks: int = 3,
    vehicle: Vehicle = Vehicle.MOTO,
    online: bool = True,
) -> Pulser:
    return Pulser(
        id=pulser_id,
        name=f"Pulser {pulser_id}",
        location=Location(lat=lat, lng=lng),
        rating=rating,
        active_tasks=active_tasks,
        max_tasks=max_tasks,
        vehicle=vehicle,
        online=online,
    )
