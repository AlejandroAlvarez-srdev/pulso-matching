"""Runtime configuration, overridable through ``PULSO_*`` environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PULSO_", frozen=True)

    # Scoring weights. They are normalised to sum to 1, so only their ratios matter.
    weight_distance: float = Field(default=0.2, ge=0)
    weight_eta: float = Field(default=0.4, ge=0)
    weight_rating: float = Field(default=0.2, ge=0)
    weight_load: float = Field(default=0.2, ge=0)

    # Hard limits: a pulser farther than this from the pickup is never a candidate.
    max_radius_km: float = Field(default=8.0, gt=0)
    # ETA at which the ETA component saturates to its worst value.
    max_eta_minutes: float = Field(default=45.0, gt=0)

    # Average urban speeds (km/h) and a detour factor from straight line to road distance.
    speed_bike_kmh: float = Field(default=14.0, gt=0)
    speed_moto_kmh: float = Field(default=28.0, gt=0)
    speed_car_kmh: float = Field(default=22.0, gt=0)
    road_factor: float = Field(default=1.3, ge=1)
    pickup_overhead_minutes: float = Field(default=3.0, ge=0)

    # Capacity-aware batching: a pulser with spare capacity may take up to this many orders per
    # batch; each stacked order waits for the previous ones.
    max_orders_per_pulser_per_batch: int = Field(default=3, ge=1)
    stacked_order_delay_minutes: float = Field(default=12.0, ge=0)

    # Wall-clock budget for Hungarian per batch; past it the batch degrades to greedy.
    hungarian_budget_ms: float = Field(default=300.0, gt=0)
    # Hard limit for a whole request; past it the client gets a 504 problem+json.
    request_timeout_ms: float = Field(default=2000.0, gt=0)

    # Admission control. The request timeout only starts once the body has been parsed and
    # validated on the event loop, so these limits are what bound that work (and the solver's).
    # Bodies are counted as they stream in, with or without a Content-Length header.
    max_body_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    max_batch_orders: int = Field(default=2000, ge=1)
    max_pulsers: int = Field(default=5000, ge=1)

    log_level: str = "INFO"
    log_json: bool = True

    @model_validator(mode="after")
    def _weights_not_all_zero(self) -> Self:
        if self.weight_distance + self.weight_eta + self.weight_rating + self.weight_load <= 0:
            raise ValueError("at least one scoring weight must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
