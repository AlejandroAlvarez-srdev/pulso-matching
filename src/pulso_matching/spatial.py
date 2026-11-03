"""Uniform lat/lng grid index used to pre-filter candidate pulsers.

Why
---
Scoring every order against every pulser is Θ(n · m). Dispatch only cares about pulsers within
``max_radius_km`` of the pickup, so we bucket pulsers into fixed-size cells (a geohash-like
bucketing without the base-32 encoding) and, per query, scan only the cells that intersect the
search circle's bounding box.

Complexity
----------
* Build: **O(m)** for ``m`` points (one dict insertion each).
* Query: **O(min(c, o) + k)** where ``c`` is the number of cells touched (≈ 9 when the cell size
  equals the radius), ``o`` the number of occupied cells, and ``k`` the number of points in the
  scanned cells; each of the ``k`` is checked exactly with haversine. When the search box covers
  more cells than are occupied, the index walks the occupied cells instead, so a query is never
  worse than a linear scan.
* A batch of ``n`` orders therefore costs **O(m + n · k)** instead of O(n · m). With roughly
  uniform density, ``k ≈ m · (area of 9 cells / city area)``, which is a small constant fraction.

Trade-offs
----------
* Cell size is fixed in degrees. Picking it equal to the search radius keeps queries at ~3x3 cells;
  smaller cells touch more (often empty) buckets, bigger cells return more false positives.
* The bounding box is a square around a circle, so ~1 - π/9 ≈ 65 % of scanned points can be false
  positives. They are discarded by the exact haversine check, so results are exact either way.
* Longitude cells shrink towards the poles; the query widens its longitude span by ``1/cos(lat)``
  and falls back to scanning a whole latitude band near the poles. Columns wrap around the
  antimeridian.
* Dense hotspots degrade towards the linear scan. A quadtree / H3 would adapt to density at the
  cost of complexity that city-scale dispatch does not need.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from pulso_matching.geo import EARTH_RADIUS_KM, haversine_km

KM_PER_DEGREE_LAT = math.pi * EARTH_RADIUS_KM / 180.0


class GridIndex[T]:
    """Spatial hash of items keyed by ``(lat, lng)``; queries return exact radius matches."""

    def __init__(self, cell_km: float) -> None:
        if cell_km <= 0:
            raise ValueError("cell_km must be positive")
        cell_deg = min(cell_km / KM_PER_DEGREE_LAT, 180.0)
        # Snap cell sizes so they tile the globe exactly; columns then wrap cleanly at ±180°.
        self._n_rows = math.ceil(180.0 / cell_deg)
        self._n_cols = math.ceil(360.0 / cell_deg)
        self._cell_lat = 180.0 / self._n_rows
        self._cell_lng = 360.0 / self._n_cols
        self._cells: defaultdict[tuple[int, int], list[tuple[T, float, float]]] = defaultdict(list)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _row(self, lat: float) -> int:
        return min(int((lat + 90.0) // self._cell_lat), self._n_rows - 1)

    def _col(self, lng: float) -> int:
        return int((lng + 180.0) // self._cell_lng) % self._n_cols

    def insert(self, item: T, lat: float, lng: float) -> None:
        """O(1) amortised."""
        self._cells[(self._row(lat), self._col(lng))].append((item, lat, lng))
        self._size += 1

    def query(self, lat: float, lng: float, radius_km: float) -> list[tuple[T, float]]:
        """Items within ``radius_km`` of ``(lat, lng)`` as ``(item, distance_km)`` pairs."""
        dlat = radius_km / KM_PER_DEGREE_LAT
        row_lo = self._row(max(-90.0, lat - dlat))
        row_hi = self._row(min(90.0, lat + dlat))

        # Widest longitude span happens at the box edge closest to a pole.
        extreme_lat = min(90.0, abs(lat) + dlat)
        cos_lat = math.cos(math.radians(extreme_lat))
        dlng = dlat / cos_lat if cos_lat > 1e-9 else 360.0
        if dlng >= 180.0:
            cols: set[int] | None = None  # every column
        else:
            start = math.floor((lng - dlng + 180.0) / self._cell_lng)
            stop = math.floor((lng + dlng + 180.0) / self._cell_lng)
            cols = (
                None
                if stop - start + 1 >= self._n_cols
                else {c % self._n_cols for c in range(start, stop + 1)}
            )

        n_cols = self._n_cols if cols is None else len(cols)
        if (row_hi - row_lo + 1) * n_cols <= len(self._cells):
            rows = range(row_lo, row_hi + 1)
            keys: Iterable[tuple[int, int]] = (
                (r, c) for r in rows for c in (range(self._n_cols) if cols is None else cols)
            )
        else:
            # The box covers more cells than are occupied: walk the occupied ones instead, so a
            # huge radius never costs more than a linear scan.
            keys = (
                (r, c)
                for r, c in self._cells
                if row_lo <= r <= row_hi and (cols is None or c in cols)
            )

        hits: list[tuple[T, float]] = []
        for key in keys:
            for item, ilat, ilng in self._cells.get(key, ()):
                d = haversine_km(lat, lng, ilat, ilng)
                if d <= radius_km:
                    hits.append((item, d))
        return hits
