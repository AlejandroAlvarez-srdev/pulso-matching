"""Pydantic v2 domain and API models.

Field names follow the shared Pulso contract (camelCase on the wire, snake_case in Python).
Unknown fields are ignored, so a full ``Order`` payload from the orders service can be posted
as-is: matching only needs the id and the pickup point.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)


class Vehicle(StrEnum):
    BIKE = "BIKE"
    MOTO = "MOTO"
    CAR = "CAR"


class Location(CamelModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Place(Location):
    address: str = ""


class Pulser(CamelModel):
    id: str = Field(min_length=1)
    name: str
    location: Location
    rating: float = Field(ge=0, le=5)
    active_tasks: int = Field(ge=0)
    max_tasks: int = Field(ge=1)
    vehicle: Vehicle
    online: bool

    @property
    def has_capacity(self) -> bool:
        return self.active_tasks < self.max_tasks


class Order(CamelModel):
    """The subset of the Pulso ``Order`` that matching needs."""

    id: UUID
    pickup: Place


class Strategy(StrEnum):
    HUNGARIAN = "hungarian"
    GREEDY = "greedy"


class Candidate(CamelModel):
    pulser_id: str
    score: float
    eta_minutes: float
    distance_km: float


class Assignment(CamelModel):
    order_id: UUID
    pulser_id: str
    score: float
    eta_minutes: float
    assigned_at: datetime


class MatchRequest(CamelModel):
    order: Order
    pulsers: list[Pulser]
    limit: int = Field(default=5, ge=1, le=100)


class MatchResponse(CamelModel):
    order_id: UUID
    candidates: list[Candidate]


class BatchMatchRequest(CamelModel):
    orders: list[Order] = Field(min_length=1)
    pulsers: list[Pulser]
    strategy: Strategy = Strategy.HUNGARIAN

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        if len({o.id for o in self.orders}) != len(self.orders):
            raise ValueError("order ids must be unique")
        if len({p.id for p in self.pulsers}) != len(self.pulsers):
            raise ValueError("pulser ids must be unique")
        return self


class BatchMatchResponse(CamelModel):
    strategy: Strategy
    assignments: list[Assignment]
    unassigned_order_ids: list[UUID]
    total_score: float
    elapsed_ms: float
