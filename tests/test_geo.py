import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pulso_matching.geo import EARTH_RADIUS_KM, haversine_km

lat = st.floats(min_value=-90, max_value=90, allow_nan=False)
lng = st.floats(min_value=-180, max_value=180, allow_nan=False)


def test_zero_distance() -> None:
    assert haversine_km(19.4326, -99.1332, 19.4326, -99.1332) == 0.0


@pytest.mark.parametrize(
    ("a", "b", "expected_km", "tolerance_km"),
    [
        # Mexico City (Zócalo) → Guadalajara (Catedral): ~461 km great-circle.
        ((19.4326, -99.1332), (20.6767, -103.3475), 461.0, 3.0),
        # Bogotá → Medellín: ~240 km.
        ((4.7110, -74.0721), (6.2442, -75.5812), 240.0, 3.0),
        # One degree of latitude along a meridian.
        ((0.0, 0.0), (1.0, 0.0), 2 * math.pi * EARTH_RADIUS_KM / 360, 1e-6),
    ],
)
def test_known_distances(
    a: tuple[float, float], b: tuple[float, float], expected_km: float, tolerance_km: float
) -> None:
    assert haversine_km(*a, *b) == pytest.approx(expected_km, abs=tolerance_km)


def test_antipodal_points_are_half_circumference() -> None:
    assert haversine_km(0, 0, 0, 180) == pytest.approx(math.pi * EARTH_RADIUS_KM)


@given(lat, lng, lat, lng)
def test_symmetric_and_bounded(lat1: float, lng1: float, lat2: float, lng2: float) -> None:
    d = haversine_km(lat1, lng1, lat2, lng2)
    assert d == pytest.approx(haversine_km(lat2, lng2, lat1, lng1), abs=1e-9)
    assert 0.0 <= d <= math.pi * EARTH_RADIUS_KM + 1e-9


@given(lat, lng, lat, lng, lat, lng)
def test_triangle_inequality(
    lat1: float, lng1: float, lat2: float, lng2: float, lat3: float, lng3: float
) -> None:
    ab = haversine_km(lat1, lng1, lat2, lng2)
    bc = haversine_km(lat2, lng2, lat3, lng3)
    ac = haversine_km(lat1, lng1, lat3, lng3)
    assert ac <= ab + bc + 1e-6
