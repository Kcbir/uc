"""Discretised AR(1) price / net-load chains.

The exogenous process is  p_t = mu_t + x_t,  where x_t is a mean-reverting AR(1)
x_{t+1} = rho x_t + eps_{t+1},  discretised on a fixed grid with a time-homogeneous
transition matrix P (Tauchen 1986 or Rouwenhorst 1995).  mu_t is a deterministic
profile, so the chain on (t, x) is time-inhomogeneous only through mu_t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def tauchen(K: int, rho: float, sigma_stat: float, m: float = 3.0):
    """Tauchen discretisation with the stationary s.d. `sigma_stat` held fixed."""
    if K == 1:
        return np.zeros(1), np.ones((1, 1))
    sigma_eps = sigma_stat * np.sqrt(1.0 - rho**2)
    x = np.linspace(-m * sigma_stat, m * sigma_stat, K)
    if sigma_eps == 0.0:
        raise ValueError("rho must be < 1")
    h = x[1] - x[0]
    z = x[None, :] - rho * x[:, None]
    upper = norm.cdf((z + h / 2) / sigma_eps)
    lower = norm.cdf((z - h / 2) / sigma_eps)
    P = upper - lower
    P[:, 0] = upper[:, 0]
    P[:, -1] = 1.0 - lower[:, -1]
    return x, P / P.sum(axis=1, keepdims=True)


def rouwenhorst(K: int, rho: float, sigma_stat: float):
    """Rouwenhorst discretisation; the grid depends only on (K, sigma_stat)."""
    if K == 1:
        return np.zeros(1), np.ones((1, 1))
    p = (1.0 + rho) / 2.0
    P = np.array([[p, 1 - p], [1 - p, p]])
    for n in range(3, K + 1):
        Z = np.zeros((n, n))
        Z[:-1, :-1] += p * P
        Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1] /= 2.0
        P = Z
    psi = np.sqrt(K - 1) * sigma_stat
    return np.linspace(-psi, psi, K), P


def is_stochastically_monotone(P: np.ndarray, tol: float = 1e-12) -> bool:
    """Rows increasing in first-order stochastic dominance:
    sum_{l>=j} P[i,l] is nondecreasing in i for every j."""
    tails = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
    return bool(np.all(np.diff(tails, axis=0) >= -tol))


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    K = P.shape[0]
    A = np.vstack([P.T - np.eye(K), np.ones((1, K))])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def daily_profile(T: int, base: float, amp: float, peak_shift: float = 0.0) -> np.ndarray:
    """Smooth illustrative daily shape: low overnight, morning and evening peaks."""
    h = (np.arange(T) % 24).astype(float)
    shape = (0.55 * np.exp(-0.5 * ((h - 8.0 - peak_shift) / 2.5) ** 2)
             + 1.00 * np.exp(-0.5 * ((h - 18.0 - peak_shift) / 3.0) ** 2)
             - 0.35 * np.exp(-0.5 * ((h - 3.0) / 3.0) ** 2))
    return base + amp * shape


@dataclass
class MarkovChainModel:
    """Exogenous chain p_{t,k} = mu[t] + x[k] with transition matrix P."""
    mu: np.ndarray       # (T,)
    x: np.ndarray        # (K,)
    P: np.ndarray        # (K, K)
    rho: float = 0.0
    sigma_stat: float = 0.0
    method: str = "rouwenhorst"

    @property
    def T(self) -> int:
        return len(self.mu)

    @property
    def K(self) -> int:
        return len(self.x)

    def levels(self) -> np.ndarray:
        """(T, K) array of price (or load) levels."""
        return self.mu[:, None] + self.x[None, :]

    def marginals(self, init: np.ndarray) -> np.ndarray:
        """(T, K) marginal distributions of the chain index, starting from `init` at t=0."""
        out = np.empty((self.T, self.K))
        d = np.asarray(init, dtype=float)
        for t in range(self.T):
            out[t] = d
            d = d @ self.P
        return out

    def simulate(self, n: int, rng: np.random.Generator, init: np.ndarray) -> np.ndarray:
        """(n, T) simulated index paths."""
        cum = np.cumsum(self.P, axis=1)
        idx = np.empty((n, self.T), dtype=np.int64)
        idx[:, 0] = rng.choice(self.K, size=n, p=init)
        for t in range(1, self.T):
            u = rng.random(n)
            idx[:, t] = np.minimum((u[:, None] > cum[idx[:, t - 1]]).sum(axis=1), self.K - 1)
        return idx


def make_chain(mu, K=21, rho=0.9, sigma_stat=8.0, method="rouwenhorst") -> MarkovChainModel:
    if method == "rouwenhorst":
        x, P = rouwenhorst(K, rho, sigma_stat)
    elif method == "tauchen":
        x, P = tauchen(K, rho, sigma_stat)
    else:
        raise ValueError(method)
    return MarkovChainModel(np.asarray(mu, float), x, P, rho, sigma_stat, method)


def deterministic_chain(path) -> MarkovChainModel:
    """Degenerate one-state chain: the price path is mu itself."""
    return MarkovChainModel(np.asarray(path, float), np.zeros(1), np.ones((1, 1)), 0.0, 0.0, "deterministic")
