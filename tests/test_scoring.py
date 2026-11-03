import pytest

from pulso_matching.config import Settings
from pulso_matching.models import Vehicle
from pulso_matching.scoring import Scorer
from tests.factories import CDMX, make_order, make_pulser

# ~0.009 degrees of latitude ≈ 1 km.
ONE_KM_LAT = 1 / 111.195


@pytest.fixture
def scorer() -> Scorer:
    return Scorer(Settings())


def test_eta_depends_on_vehicle(scorer: Scorer) -> None:
    bike = scorer.eta_minutes(5.0, Vehicle.BIKE)
    car = scorer.eta_minutes(5.0, Vehicle.CAR)
    moto = scorer.eta_minutes(5.0, Vehicle.MOTO)
    assert moto < car < bike


def test_eta_formula() -> None:
    s = Settings(road_factor=1.0, pickup_overhead_minutes=0, speed_moto_kmh=30)
    assert Scorer(s).eta_minutes(10.0, Vehicle.MOTO) == pytest.approx(20.0)


def test_offline_or_full_pulsers_are_ineligible(scorer: Scorer) -> None:
    order = make_order()
    assert scorer.score(order, make_pulser("off", online=False)) is None
    assert scorer.score(order, make_pulser("full", active_tasks=3, max_tasks=3)) is None


def test_outside_radius_is_ineligible(scorer: Scorer) -> None:
    far = make_pulser("far", lat=CDMX[0] + 10 * ONE_KM_LAT)
    assert scorer.score(make_order(), far) is None


def test_closer_better_rated_less_loaded_wins(scorer: Scorer) -> None:
    order = make_order()
    near = scorer.score(order, make_pulser("near", lat=CDMX[0] + 1 * ONE_KM_LAT))
    far = scorer.score(order, make_pulser("far", lat=CDMX[0] + 4 * ONE_KM_LAT))
    good = scorer.score(order, make_pulser("good", rating=5.0))
    bad = scorer.score(order, make_pulser("bad", rating=3.0))
    idle = scorer.score(order, make_pulser("idle", active_tasks=0))
    busy = scorer.score(order, make_pulser("busy", active_tasks=2))
    assert near and far and good and bad and idle and busy
    assert near.score > far.score
    assert good.score > bad.score
    assert idle.score > busy.score


def test_score_is_in_unit_interval(scorer: Scorer) -> None:
    best = scorer.score(make_order(), make_pulser("best", rating=5.0))
    worst = scorer.score(
        make_order(),
        make_pulser(
            "worst",
            lat=CDMX[0] + 7.9 * ONE_KM_LAT,
            rating=0.0,
            active_tasks=4,
            max_tasks=5,
            vehicle=Vehicle.BIKE,
        ),
    )
    assert best and worst
    assert 0.0 <= worst.score < best.score <= 1.0
    assert best.score == pytest.approx(1.0, abs=0.1)


def test_weights_are_normalised() -> None:
    order = make_order()
    pulser = make_pulser("p", rating=2.5)
    a = Scorer(Settings(weight_distance=0, weight_eta=0, weight_rating=1, weight_load=0))
    b = Scorer(Settings(weight_distance=0, weight_eta=0, weight_rating=7, weight_load=0))
    sa, sb = a.score(order, pulser), b.score(order, pulser)
    assert sa and sb
    assert sa.score == pytest.approx(sb.score) == pytest.approx(0.5)


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(ValueError, match="weight"):
        Settings(weight_distance=0, weight_eta=0, weight_rating=0, weight_load=0)
