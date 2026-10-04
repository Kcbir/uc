import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import experiments as E  # noqa: E402
from src.dp import decision_advantages, single_crossing_violations, solve  # noqa: E402
from src.fleet import brute_force_fleet, joint_index, solve_fleet  # noqa: E402
from src.instances import random_chain, random_unit  # noqa: E402
from src.model import build_counter  # noqa: E402
from src.price_chain import deterministic_chain  # noqa: E402


@pytest.mark.parametrize("seed", range(30))
def test_theorem_b1(seed):
    rng = np.random.default_rng(10_000 + seed)
    u = random_unit(rng, UT=1, DT=1, const_S=True)
    ch = random_chain(rng)
    s = solve(u, ch, gamma=1.0 if seed % 2 else 0.9)
    r = E.b1_checks(s, 1e-9 * s.scale)
    assert r["ident"] and r["mono"] and r["thresh"] and r["hyst"] and r["be"]


@pytest.mark.parametrize("seed", range(20))
def test_stationary_time_monotone(seed):
    rng = np.random.default_rng(20_000 + seed)
    u = random_unit(rng, UT=1, DT=1, const_S=True)
    s = solve(u, random_chain(rng, flat=True))
    D = E.b1_checks(s, 1e-9 * s.scale)["D"]
    assert np.all(np.diff(D, axis=0) <= 1e-9 * s.scale)


@pytest.mark.parametrize("seed", range(30))
def test_iid_threshold_and_sawtooth(seed):
    rng = np.random.default_rng(30_000 + seed)
    u = random_unit(rng)
    s = solve(u, random_chain(rng, rho=0.0))
    a = E.analyse_general(s)
    assert a["sc"] == 0
    assert a["Gsaw"] == 0


@pytest.mark.parametrize("seed", range(30))
def test_sawtooth_general(seed):
    rng = np.random.default_rng(40_000 + seed)
    s = solve(random_unit(rng), random_chain(rng))
    assert E.analyse_general(s)["Gsaw"] == 0


def test_counterexamples():
    for name, u, ch, t, c, kind in E.counterexample_instances():
        s = solve(u, ch)
        G = s.advantage()[t, :, c]
        assert G[0] > 0 > G[1], name
        Gs, Gk = decision_advantages(s)
        tol = 1e-9 * s.scale
        total = sum(single_crossing_violations(Gs[:, :, j], tol) for j in range(Gs.shape[2]))
        total += single_crossing_violations(Gk, tol)
        assert total > 0


@pytest.mark.parametrize("seed", range(50))
def test_counter_is_minimal(seed):
    rng = np.random.default_rng(50_000 + seed)
    u = random_unit(rng, max_ut=10, max_dt=10)
    from dataclasses import replace
    u = replace(u, S_hot=max(u.S_hot, 1.0), S_warm=max(u.S_warm, 1.0), S_cold=max(u.S_cold, 1.0))
    assert E.moore_classes(build_counter(u)) == u.UT + max(u.DT, u.T_cold_eff)


@pytest.mark.parametrize("seed", range(3))
def test_fleet_bruteforce(seed):
    rng = np.random.default_rng(60_000 + seed)
    us = [random_unit(rng, max_ut=3, max_dt=3) for _ in range(2)]
    dem = rng.uniform(0, sum(u.Pmax for u in us), 5)
    init = [(int(rng.integers(0, 2)), int(rng.integers(1, 4))) for _ in us]
    out = solve_fleet(us, deterministic_chain(dem))
    assert abs(out["V0"][0, joint_index(us, init)] - brute_force_fleet(us, dem, init)) < 1e-6 * (1 + abs(out["V0"]).max())
