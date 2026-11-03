import itertools
import random
import time
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulso_matching.assignment import BudgetExceededError, FloatMatrix, Pairs
from pulso_matching.config import Settings
from pulso_matching.models import Order, Pulser, Strategy
from pulso_matching.scoring import Scorer
from pulso_matching.service import MatchingService
from pulso_matching.timeouts import DeadlineExceededError
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
    p1 = make_pulser("p1", lat=CDMX[0] + 3 * KM, max_tasks=1)
    p2 = make_pulser("p2", lat=CDMX[0] + 9 * KM, max_tasks=1)
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
    taken = Counter(a.pulser_id for a in result.assignments)
    for pid, count in taken.items():
        p = by_id[pid]
        assert count <= min(
            p.max_tasks - p.active_tasks, Settings().max_orders_per_pulser_per_batch
        )
    assert len({a.order_id for a in result.assignments}) == len(result.assignments)
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


def test_pulser_with_spare_capacity_takes_several_orders(service: MatchingService) -> None:
    orders = [make_order(lat=CDMX[0] + k * 0.2 * KM) for k in range(4)]
    busy_hub = make_pulser("hub", active_tasks=1, max_tasks=3)
    result = service.assign_batch(orders, [busy_hub], Strategy.HUNGARIAN)
    # Two free slots: two orders, the second one waits for the first.
    assert [a.pulser_id for a in result.assignments] == ["hub", "hub"]
    assert len(result.unassigned_order_ids) == 2
    etas = sorted(a.eta_minutes for a in result.assignments)
    assert etas[1] - etas[0] >= Settings().stacked_order_delay_minutes - 1


def test_batch_cap_limits_orders_per_pulser() -> None:
    capped = MatchingService(Scorer(Settings(max_orders_per_pulser_per_batch=1)))
    orders = [make_order(), make_order()]
    result = capped.assign_batch(orders, [make_pulser("p", max_tasks=5)], Strategy.HUNGARIAN)
    assert len(result.assignments) == 1


def test_stacked_slot_costs_more_than_first(service: MatchingService) -> None:
    scorer = Scorer(Settings())
    pulser = make_pulser("p", max_tasks=3)
    first, second = scorer.score(make_order(), pulser, 0), scorer.score(make_order(), pulser, 1)
    assert first and second
    assert second.cost > first.cost
    assert second.eta_minutes == pytest.approx(first.eta_minutes + 12.0)
    assert scorer.score(make_order(), pulser, 3) is None


def _brute_force_capacity(
    scorer: Scorer, orders: list[Order], pulsers: list[Pulser]
) -> tuple[int, float]:
    """(max matched, min total cost) over every capacity-respecting order→pulser mapping."""
    best: tuple[int, float] | None = None
    for choice in itertools.product([None, *range(len(pulsers))], repeat=len(orders)):
        groups: dict[int, list[int]] = {}
        for i, j in enumerate(choice):
            if j is not None:
                groups.setdefault(j, []).append(i)
        total = 0.0
        ok = True
        for j, members in groups.items():
            if len(members) > scorer.batch_slots(pulsers[j]):
                ok = False
                break
            # Best way to sequence this pulser's orders into slots 0..k-1.
            costs = []
            for perm in itertools.permutations(members):
                scores = [scorer.score(orders[i], pulsers[j], k) for k, i in enumerate(perm)]
                if all(sc is not None for sc in scores):
                    costs.append(sum(sc.cost for sc in scores if sc is not None))
            if not costs:
                ok = False
                break
            total += min(costs)
        if not ok:
            continue
        matched = len(orders) - choice.count(None)
        if best is None or (matched, -total) > (best[0], -best[1]):
            best = (matched, total)
    assert best is not None
    return best


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.tuples(st.floats(-3, 3), st.floats(-3, 3)), min_size=1, max_size=4),
    st.lists(
        st.tuples(
            st.floats(-3, 3),
            st.floats(-3, 3),
            st.integers(0, 2),
            st.integers(1, 3),
            st.floats(0, 5),
        ),
        min_size=1,
        max_size=3,
    ),
)
def test_capacity_aware_hungarian_matches_brute_force(
    order_offsets: list[tuple[float, float]],
    pulser_specs: list[tuple[float, float, int, int, float]],
) -> None:
    scorer = Scorer(Settings())
    service = MatchingService(scorer)
    orders = [make_order(CDMX[0] + dy * KM, CDMX[1] + dx * KM) for dy, dx in order_offsets]
    pulsers = [
        make_pulser(
            f"p{k}",
            CDMX[0] + dy * KM,
            CDMX[1] + dx * KM,
            active_tasks=min(active, max_tasks - 1),
            max_tasks=max_tasks,
            rating=rating,
        )
        for k, (dy, dx, active, max_tasks, rating) in enumerate(pulser_specs)
    ]
    result = service.assign_batch(orders, pulsers, Strategy.HUNGARIAN)
    matched, cost = _brute_force_capacity(scorer, orders, pulsers)
    assert len(result.assignments) == matched
    assert len(result.assignments) - result.total_score == pytest.approx(cost, abs=1e-4)


def _random_city(seed: int, n: int, m: int) -> tuple[list[Order], list[Pulser]]:
    rng = random.Random(seed)

    def jitter() -> tuple[float, float]:
        return CDMX[0] + rng.uniform(-0.1, 0.1), CDMX[1] + rng.uniform(-0.1, 0.1)

    return [make_order(*jitter()) for _ in range(n)], [
        make_pulser(f"p{k}", *jitter(), max_tasks=1) for k in range(m)
    ]


def test_exhausted_budget_degrades_to_greedy_on_the_same_matrix() -> None:
    orders, pulsers = _random_city(3, 30, 40)
    scorer = Scorer(Settings())
    starved = MatchingService(scorer, hungarian_budget_ms=1e-9)
    result = starved.assign_batch(orders, pulsers, Strategy.HUNGARIAN)
    greedy = MatchingService(scorer).assign_batch(orders, pulsers, Strategy.GREEDY)
    assert result.degraded
    assert result.degraded_components >= 1
    pairs = {(a.order_id, a.pulser_id) for a in result.assignments}
    assert pairs == {(a.order_id, a.pulser_id) for a in greedy.assignments}


def test_passed_request_deadline_stops_the_batch() -> None:
    orders, pulsers = _random_city(4, 10, 10)
    service = MatchingService(Scorer(Settings()))
    for strategy in Strategy:
        with pytest.raises(DeadlineExceededError):
            service.assign_batch(orders, pulsers, strategy, deadline=0.0)


def test_request_deadline_during_hungarian_stops_instead_of_degrading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders, pulsers = _random_city(4, 10, 10)
    service = MatchingService(Scorer(Settings()))

    def slow_hungarian(matrix: FloatMatrix, deadline: float | None = None) -> Pairs:
        time.sleep(0.2)  # the request deadline passes while "solving"
        raise BudgetExceededError

    def greedy_must_not_run(matrix: FloatMatrix) -> Pairs:
        raise AssertionError("greedy fallback ran after the request deadline")

    monkeypatch.setattr("pulso_matching.service.hungarian_assignment_np", slow_hungarian)
    monkeypatch.setattr("pulso_matching.service.greedy_assignment", greedy_must_not_run)
    with pytest.raises(DeadlineExceededError):
        service.assign_batch(
            orders, pulsers, Strategy.HUNGARIAN, deadline=time.perf_counter() + 0.1
        )


def test_generous_budget_keeps_the_optimum() -> None:
    orders, pulsers = _random_city(5, 30, 40)
    scorer = Scorer(Settings())
    budgeted = MatchingService(scorer, hungarian_budget_ms=60_000)
    result = budgeted.assign_batch(orders, pulsers, Strategy.HUNGARIAN)
    unbounded = MatchingService(scorer).assign_batch(orders, pulsers, Strategy.HUNGARIAN)
    assert not result.degraded
    assert result.total_score == pytest.approx(unbounded.total_score)


def test_greedy_strategy_is_never_reported_as_degraded() -> None:
    orders, pulsers = _random_city(6, 10, 10)
    starved = MatchingService(Scorer(Settings()), hungarian_budget_ms=1e-9)
    assert not starved.assign_batch(orders, pulsers, Strategy.GREEDY).degraded
