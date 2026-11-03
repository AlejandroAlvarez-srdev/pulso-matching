"""Pair scoring: how good is pulser *p* for order *o*?

Every component is normalised to ``[0, 1]`` where 0 is best, then combined with the configured
weights into a **cost** in ``[0, 1]``. The public **score** is ``1 - cost`` (higher is better).
Assignment algorithms minimise cost; the API reports score.

The cost splits into a part that depends only on the pair (distance, rating, travel time) and a
part that depends on the batch slot (load, stacking delay). :class:`PairTerms` caches the first
part, so the expensive bit — haversine — is computed once per pair and reused for every slot,
every component matrix and any fallback re-solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True, slots=True)
class PairTerms:
    """Slot-independent part of a pair's cost, computed once per (order, pulser)."""

    distance_km: float
    base_eta_minutes: float
    static_cost: float  # weighted distance + rating terms


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

    def batch_slots(self, pulser: Pulser) -> int:
        """How many orders this pulser may receive in one batch (0 if ineligible)."""
        if not self.is_eligible(pulser):
            return 0
        free = pulser.max_tasks - pulser.active_tasks
        return min(free, self._s.max_orders_per_pulser_per_batch)

    def terms(self, pulser: Pulser, distance_km: float) -> PairTerms:
        return PairTerms(
            distance_km=distance_km,
            base_eta_minutes=self.eta_minutes(distance_km, pulser.vehicle),
            static_cost=self._w_distance * (distance_km / self._s.max_radius_km)
            + self._w_rating * (1.0 - pulser.rating / 5.0),
        )

    def slot_score(self, terms: PairTerms, pulser: Pulser, slot: int) -> PairScore | None:
        """Full score of a cached pair when it is the pulser's ``slot``-th order in the batch.

        ``slot`` raises the load term to ``(activeTasks + slot) / maxTasks`` and delays the ETA
        by ``slot · stacked_order_delay_minutes`` (they finish the earlier ones first).
        """
        if not pulser.online or pulser.active_tasks + slot >= pulser.max_tasks:
            return None
        eta = terms.base_eta_minutes + slot * self._s.stacked_order_delay_minutes
        cost = (
            terms.static_cost
            + self._w_eta * min(eta / self._s.max_eta_minutes, 1.0)
            + self._w_load * ((pulser.active_tasks + slot) / pulser.max_tasks)
        )
        return PairScore(cost=cost, distance_km=terms.distance_km, eta_minutes=eta)

    def slot_costs(
        self,
        static_cost: NDArray[np.float64],
        base_eta: NDArray[np.float64],
        active_tasks: NDArray[np.float64],
        max_tasks: NDArray[np.float64],
        slot: int,
    ) -> NDArray[np.float64]:
        """Vectorised :meth:`slot_score` cost for many cached pairs at the same slot."""
        eta = base_eta + slot * self._s.stacked_order_delay_minutes
        result: NDArray[np.float64] = (
            static_cost
            + self._w_eta * np.minimum(eta / self._s.max_eta_minutes, 1.0)
            + self._w_load * ((active_tasks + slot) / max_tasks)
        )
        return result

    def score(self, order: Order, pulser: Pulser, slot: int = 0) -> PairScore | None:
        """Score one pair from scratch, or ``None`` if the pulser cannot take the order."""
        if not pulser.online or pulser.active_tasks + slot >= pulser.max_tasks:
            return None
        distance = haversine_km(
            pulser.location.lat, pulser.location.lng, order.pickup.lat, order.pickup.lng
        )
        if distance > self._s.max_radius_km:
            return None
        return self.slot_score(self.terms(pulser, distance), pulser, slot)
