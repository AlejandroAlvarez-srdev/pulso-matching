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


# --- Edge cases: antimeridian, poles, exact boundary values -------------------------------------

edge_lat = st.one_of(
    st.sampled_from([-90.0, -89.999, 0.0, 89.999, 90.0]),
    st.floats(min_value=85, max_value=90),
    st.floats(min_value=-90, max_value=-85),
    st.floats(min_value=-90, max_value=90),
)
antimeridian_lng = st.one_of(
    st.sampled_from([-180.0, -179.9999, 179.9999, 180.0]),
    st.floats(min_value=179, max_value=180),
    st.floats(min_value=-180, max_value=-179),
)
edge_lng = st.one_of(antimeridian_lng, st.floats(min_value=-180, max_value=180))


@settings(max_examples=400, deadline=None)
@given(
    st.lists(st.tuples(st.floats(-60, 60), antimeridian_lng), max_size=40),
    st.floats(-60, 60),
    antimeridian_lng,
    st.floats(min_value=0, max_value=400),
    st.floats(min_value=0.5, max_value=300),
)
def test_antimeridian_queries_match_linear_scan(
    pts: list[tuple[float, float]], qlat: float, qlng: float, radius: float, cell_km: float
) -> None:
    index = build(pts, cell_km)
    assert {i for i, _ in index.query(qlat, qlng, radius)} == brute(pts, qlat, qlng, radius)


@settings(max_examples=400, deadline=None)
@given(
    st.lists(st.tuples(edge_lat, edge_lng), max_size=40),
    edge_lat,
    edge_lng,
    st.floats(min_value=0, max_value=1500),
    st.floats(min_value=0.5, max_value=500),
)
def test_polar_and_boundary_queries_match_linear_scan(
    pts: list[tuple[float, float]], qlat: float, qlng: float, radius: float, cell_km: float
) -> None:
    index = build(pts, cell_km)
    assert {i for i, _ in index.query(qlat, qlng, radius)} == brute(pts, qlat, qlng, radius)


@pytest.mark.parametrize("lng", [-180.0, 180.0])
def test_plus_and_minus_180_are_the_same_meridian(lng: float) -> None:
    index = build([(10.0, 180.0), (10.0, -180.0)], cell_km=1)
    assert {i for i, _ in index.query(10.0, lng, 0.001)} == {0, 1}


@pytest.mark.parametrize("pole", [90.0, -90.0])
def test_every_longitude_meets_at_the_pole(pole: float) -> None:
    pts = [(pole, float(lng)) for lng in range(-180, 180, 30)]
    index = build(pts, cell_km=2)
    assert len(index.query(pole, 77.0, 0.001)) == len(pts)


def test_zero_radius_returns_only_coincident_points() -> None:
    index = build([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0001)], cell_km=1)
    assert {i for i, _ in index.query(1.0, 1.0, 0.0)} == {0, 1}


def test_cell_larger_than_the_globe_degenerates_to_one_bucket() -> None:
    pts = [(-45.0, -170.0), (45.0, 170.0), (0.0, 0.0)]
    index = build(pts, cell_km=50_000)
    assert {i for i, _ in index.query(0.0, 0.0, 20_100)} == {0, 1, 2}
