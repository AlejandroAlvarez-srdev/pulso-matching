"""Geodesic helpers."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088  # IUGG mean Earth radius


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometres.

    Spherical model: error vs. the ellipsoid is below 0.5 %, irrelevant for city-scale
    dispatch. O(1).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    # min() guards against a > 1 from floating point error on antipodal points.
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))
