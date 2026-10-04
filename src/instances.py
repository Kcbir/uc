"""Random instance generators for verification sweeps."""
from __future__ import annotations

import numpy as np

from .model import Unit
from .price_chain import daily_profile, deterministic_chain, make_chain


def random_unit(rng: np.random.Generator, UT=None, DT=None, const_S=False, max_ut=8, max_dt=8,
                price_level=30.0) -> Unit:
    Pmax = float(rng.uniform(20, 400))
    Pmin = float(Pmax * rng.uniform(0.0, 0.8))
    c = float(price_level * rng.uniform(0.5, 1.3))
    F = float(rng.uniform(0, 0.2) * c * Pmax)
    UT = int(rng.integers(1, max_ut + 1)) if UT is None else UT
    DT = int(rng.integers(1, max_dt + 1)) if DT is None else DT
    hot = float(rng.uniform(0.0, 20.0) * c * Pmax / 10)
    if const_S:
        warm = cold = hot
        T_warm = T_cold = 1
    else:
        warm = hot + float(rng.uniform(0, 1.0) * hot)
        cold = warm + float(rng.uniform(0, 1.0) * hot)
        T_warm = int(rng.integers(1, 8))
        T_cold = T_warm + int(rng.integers(0, 10))
    return Unit("rand", round(Pmin, 3), round(Pmax, 3), round(c, 3), round(F, 3), UT, DT,
                round(hot, 3), round(warm, 3), round(cold, 3), T_warm, T_cold)


def random_chain(rng: np.random.Generator, T=None, K=None, rho=None, flat=False, method=None,
                 price_level=30.0):
    T = int(rng.choice([12, 24, 36, 48])) if T is None else T
    K = int(rng.integers(5, 22)) if K is None else K
    rho = float(rng.uniform(0.0, 0.98)) if rho is None else rho
    sigma = float(rng.uniform(0.05, 0.6) * price_level)
    method = str(rng.choice(["rouwenhorst", "tauchen"])) if method is None else method
    if flat:
        mu = np.full(T, price_level)
    else:
        mu = daily_profile(T, price_level * rng.uniform(0.7, 1.1), price_level * rng.uniform(0.0, 1.0),
                           peak_shift=float(rng.uniform(-2, 2)))
        mu = mu + rng.normal(0, 0.05 * price_level, T)
    return make_chain(mu, K=K, rho=rho, sigma_stat=sigma, method=method)


def random_det_path(rng: np.random.Generator, T: int, price_level=30.0):
    mu = daily_profile(T, price_level * rng.uniform(0.6, 1.1), price_level * rng.uniform(0.2, 1.2),
                       peak_shift=float(rng.uniform(-3, 3)))
    return deterministic_chain(mu + rng.normal(0, 0.25 * price_level, T))


def random_initial(rng: np.random.Generator, unit: Unit):
    """Random (u0, tau0) with tau0 possibly beyond the counter cap (tests the cap)."""
    u0 = int(rng.integers(0, 2))
    tau0 = int(rng.integers(1, (unit.UT if u0 else unit.J) + 3))
    return u0, tau0
