import random

import pytest

from pulso_matching.config import Settings
from pulso_matching.models import Strategy
from pulso_matching.scoring import Scorer
from pulso_matching.service import MatchingService
from tests.factories import CDMX, make_order, make_pulser

KM = 1 / 111.195  # degrees of latitude per km


@pytest.fixture
def service() -> MatchingService:
    return MatchingService(Scorer(Settings()))


def test_rank_orders_best_first_and_respects_limit(service: MatchingService) -> None:
    order = make_order()
    pulsers = [make_pulser(f"p{k}", lat=CDMX[0] + k * KM) for k in range(6)]
    pulsers.append(make_pulser("offline", online=False))
    ranked = service.rank(order, pulsers, limit=3)
    assert [c.pulser_id for c in ranked] == ["p0", "p1", "p2"]
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_hungarian_saves_order_that_greedy_strands(service: MatchingService) -> None:
    # B only reaches P1; A is closest to P1 but can also use P2.
    b = make_order(lat=CDMX[0])
    a = make_order(lat=CDMX[0] + 3.1 * KM)
    p1 = make_pulser("p1", lat=CDMX[0] + 3 * KM)
    p2 = make_pulser("p2", lat=CDMX[0] + 9 * KM)
    greedy = service.assign_batch([a, b], [p1, p2], Strategy.GREEDY)
    optimal = service.assign_batch([a, b], [p1, p2], Strategy.HUNGARIAN)
    assert greedy.unassigned_order_ids == [b.id]
    assert optimal.unassigned_order_ids == []
    assert {(x.order_id, x.pulser_id) for x in optimal.assignments} == {(a.id, "p2"), (b.id, "p1")}


def test_independent_cities_are_solved_separately(service: MatchingService) -> None:
    bogota = (4.711, -74.0721)
    orders = [make_order(), make_order(*bogota)]
    pulsers = [make_pulser("cdmx"), make_pulser("bog", *bogota)]
    result = service.assign_batch(orders, pulsers, Strategy.HUNGARIAN)
    assert [a.pulser_id for a in result.assignments] == ["cdmx", "bog"]
    assert result.total_score == pytest.approx(sum(a.score for a in result.assignments))


def test_order_without_nearby_pulser_is_unassigned(service: MatchingService) -> None:
    lonely = make_order(lat=CDMX[0] + 50 * KM)
    result = service.assign_batch([lonely], [make_pulser("p")], Strategy.HUNGARIAN)
    assert result.assignments == []
    assert result.unassigned_order_ids == [lonely.id]


@pytest.mark.parametrize("strategy", list(Strategy))
def test_batch_invariants_on_random_city(service: MatchingService, strategy: Strategy) -> None:
    rng = random.Random(7)

    def jitter() -> tuple[float, float]:
        return CDMX[0] + rng.uniform(-0.15, 0.15), CDMX[1] + rng.uniform(-0.15, 0.15)

    orders = [make_order(*jitter()) for _ in range(40)]
    pulsers = [
        make_pulser(f"p{k}", *jitter(), active_tasks=rng.randint(0, 3), max_tasks=3)
        for k in range(60)
    ]
    result = service.assign_batch(orders, pulsers, strategy)
    by_id = {p.id: p for p in pulsers}
    used = [a.pulser_id for a in result.assignments]
    assert len(used) == len(set(used))
    assert all(by_id[pid].has_capacity for pid in used)
    assert len(result.assignments) + len(result.unassigned_order_ids) == len(orders)


def test_grid_prefilter_finds_exactly_the_feasible_pairs(service: MatchingService) -> None:
    rng = random.Random(11)
    orders = [
        make_order(CDMX[0] + rng.uniform(-0.3, 0.3), CDMX[1] + rng.uniform(-0.3, 0.3))
        for _ in range(30)
    ]
    pulsers = [
        make_pulser(f"p{k}", CDMX[0] + rng.uniform(-0.3, 0.3), CDMX[1] + rng.uniform(-0.3, 0.3))
        for k in range(80)
    ]
    scorer = Scorer(Settings())
    expected = {
        (i, j)
        for i, o in enumerate(orders)
        for j, p in enumerate(pulsers)
        if scorer.score(o, p) is not None
    }
    assert set(service.candidate_edges(orders, pulsers)) == expected
