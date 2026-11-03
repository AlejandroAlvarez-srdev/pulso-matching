import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulso_matching.geo import haversine_km
from pulso_matching.spatial import GridIndex

lat = st.floats(min_value=-89.9, max_value=89.9, allow_nan=False)
lng = st.floats(min_value=-180, max_value=179.999, allow_nan=False)
points = st.lists(st.tuples(lat, lng), max_size=60)


def brute(pts: list[tuple[float, float]], qlat: float, qlng: float, r: float) -> set[int]:
    return {i for i, (a, b) in enumerate(pts) if haversine_km(qlat, qlng, a, b) <= r}


def build(pts: list[tuple[float, float]], cell_km: float) -> GridIndex[int]:
    index: GridIndex[int] = GridIndex(cell_km)
    for i, (a, b) in enumerate(pts):
        index.insert(i, a, b)
    return index


@settings(max_examples=200, deadline=None)
@given(
    points,
    lat,
    lng,
    st.floats(min_value=0.1, max_value=3000),
    st.floats(min_value=0.5, max_value=2000),
)
def test_query_equals_linear_scan(
    pts: list[tuple[float, float]], qlat: float, qlng: float, radius: float, cell_km: float
) -> None:
    index = build(pts, cell_km)
    assert len(index) == len(pts)
    assert {i for i, _ in index.query(qlat, qlng, radius)} == brute(pts, qlat, qlng, radius)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=19.2, max_value=19.6), st.floats(min_value=-99.3, max_value=-98.9)
        ),
        max_size=80,
    ),
    st.floats(min_value=19.2, max_value=19.6),
    st.floats(min_value=-99.3, max_value=-98.9),
)
def test_city_scale_queries_match_linear_scan(
    pts: list[tuple[float, float]], qlat: float, qlng: float
) -> None:
    index = build(pts, cell_km=8)
    assert {i for i, _ in index.query(qlat, qlng, 8)} == brute(pts, qlat, qlng, 8)


def test_wraps_around_antimeridian() -> None:
    index = build([(-17.8, 179.99), (-17.8, -179.99)], cell_km=5)
    assert {i for i, _ in index.query(-17.8, 179.995, 5)} == {0, 1}


def test_returns_distances() -> None:
    index = build([(0.0, 0.0)], cell_km=1)
    [(item, distance)] = index.query(0.0, 0.01, 5)
    assert item == 0
    assert distance == pytest.approx(haversine_km(0, 0, 0, 0.01))


def test_rejects_non_positive_cell() -> None:
    with pytest.raises(ValueError, match="cell_km"):
        GridIndex[int](0)
