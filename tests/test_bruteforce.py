import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dp import brute_force_deterministic, full_history_value, solve  # noqa: E402
from src.instances import random_chain, random_det_path, random_initial, random_unit  # noqa: E402
from src.mip import schedule_value, solve_deterministic_mip  # noqa: E402
from src.price_chain import is_stochastically_monotone, rouwenhorst, tauchen  # noqa: E402


def close(a, b, rel=1e-7):
    return abs(a - b) <= rel * (1 + abs(a) + abs(b))


@pytest.mark.parametrize("seed", range(40))
def test_dp_matches_enumeration(seed):
    rng = np.random.default_rng(1000 + seed)
    unit = random_unit(rng, max_ut=5, max_dt=5)
    chain = random_det_path(rng, 10)
    u0, tau0 = random_initial(rng, unit)
    sol = solve(unit, chain)
    v_bf, _ = brute_force_deterministic(unit, chain.mu, u0, tau0)
    assert close(sol.V[0, 0, sol.cnt.index(u0, tau0)], v_bf)


@pytest.mark.parametrize("seed", range(6))
def test_augmented_state_equals_full_history(seed):
    rng = np.random.default_rng(2000 + seed)
    unit = random_unit(rng, max_ut=3, max_dt=3)
    chain = random_chain(rng, T=6, K=3)
    u0, tau0 = random_initial(rng, unit)
    k0 = int(rng.integers(0, 3))
    sol = solve(unit, chain)
    assert close(sol.V[0, k0, sol.cnt.index(u0, tau0)], full_history_value(unit, chain, k0, u0, tau0))


@pytest.mark.parametrize("seed", range(15))
def test_dp_matches_mip(seed):
    rng = np.random.default_rng(3000 + seed)
    unit = random_unit(rng)
    chain = random_det_path(rng, 24)
    u0, tau0 = random_initial(rng, unit)
    sol = solve(unit, chain)
    v_mip, sched = solve_deterministic_mip(unit, chain.mu, u0, tau0)
    assert close(sol.V[0, 0, sol.cnt.index(u0, tau0)], v_mip, rel=1e-6)
    assert close(schedule_value(unit, chain.mu, sched, u0, tau0), v_mip, rel=1e-6)


@pytest.mark.parametrize("K", [2, 5, 11, 21])
@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.98])
def test_chains_stochastically_monotone(K, rho):
    for x, P in (rouwenhorst(K, rho, 5.0), tauchen(K, rho, 5.0)):
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.all(np.diff(x) > 0)
        assert is_stochastically_monotone(P)
