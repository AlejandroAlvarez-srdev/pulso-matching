"""Pair scoring: how good is pulser *p* for order *o*?

Every component is normalised to ``[0, 1]`` where 0 is best, then combined with the configured
weights into a **cost** in ``[0, 1]``. The public **score** is ``1 - cost`` (higher is better).
Assignment algorithms minimise cost; the API reports score.
"""

from __future__ import annotations

from dataclasses import dataclass

from pulso_matching.config import Settings
from pulso_matching.geo import haversine_km
from pulso_matching.models import Order, Pulser, Vehicle


@dataclass(frozen=True, slots=True)
class PairScore:
    cost: float
    distance_km: float
    eta_minutes: float

    @property
    def score(self) -> float:
        return 1.0 - self.cost


class Scorer:
    """Stateless scorer bound to a :class:`Settings` snapshot. Each call is O(1)."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        total = (
            settings.weight_distance
            + settings.weight_eta
            + settings.weight_rating
            + settings.weight_load
        )
        self._w_distance = settings.weight_distance / total
        self._w_eta = settings.weight_eta / total
        self._w_rating = settings.weight_rating / total
        self._w_load = settings.weight_load / total
        self._speed = {
            Vehicle.BIKE: settings.speed_bike_kmh,
            Vehicle.MOTO: settings.speed_moto_kmh,
            Vehicle.CAR: settings.speed_car_kmh,
        }

    @property
    def max_radius_km(self) -> float:
        return self._s.max_radius_km

    def eta_minutes(self, distance_km: float, vehicle: Vehicle) -> float:
        """Straight-line distance → road distance (detour factor) → minutes, plus overhead."""
        road_km = distance_km * self._s.road_factor
        return self._s.pickup_overhead_minutes + road_km / self._speed[vehicle] * 60.0

    def is_eligible(self, pulser: Pulser) -> bool:
        return pulser.online and pulser.has_capacity

    def score(self, order: Order, pulser: Pulser) -> PairScore | None:
        """Return the pair score, or ``None`` if the pulser cannot take the order."""
        if not self.is_eligible(pulser):
            return None
        distance = haversine_km(
            pulser.location.lat, pulser.location.lng, order.pickup.lat, order.pickup.lng
        )
        if distance > self._s.max_radius_km:
            return None
        eta = self.eta_minutes(distance, pulser.vehicle)
        cost = (
            self._w_distance * (distance / self._s.max_radius_km)
            + self._w_eta * min(eta / self._s.max_eta_minutes, 1.0)
            + self._w_rating * (1.0 - pulser.rating / 5.0)
            + self._w_load * (pulser.active_tasks / pulser.max_tasks)
        )
        return PairScore(cost=cost, distance_km=distance, eta_minutes=eta)
