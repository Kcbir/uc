"""Exact backward induction for a single price-taking unit on the augmented state.

State (t, k, c): price index k, counter index c (status + time-in-status, see model.Counter).
Everything is vectorised over (k, c); counters move deterministically via index arrays.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .model import Counter, Unit, build_counter
from .price_chain import MarkovChainModel


@dataclass
class Solution:
    unit: Unit
    chain: MarkovChainModel
    cnt: Counter
    gamma: float
    V: np.ndarray        # (T+1, K, n)
    Q: np.ndarray        # (T, K, n, 2), -inf where infeasible
    policy: np.ndarray   # (T, K, n) int8, ties broken towards "on"
    r: np.ndarray        # (T, K) per-period on-profit

    @property
    def scale(self) -> float:
        return 1.0 + float(np.max(np.abs(self.V)))

    def advantage(self) -> np.ndarray:
        """Q(on) - Q(off) (inf/-inf where a status is locked)."""
        with np.errstate(invalid="ignore"):
            return self.Q[..., 1] - self.Q[..., 0]


def solve(unit: Unit, chain: MarkovChainModel, gamma: float = 1.0) -> Solution:
    cnt = build_counter(unit)
    T, K, n = chain.T, chain.K, cnt.n
    r = unit.profit(chain.levels())
    V = np.zeros((T + 1, K, n))
    Q = np.empty((T, K, n, 2))
    mask = cnt.feas[None, :, :]
    for t in range(T - 1, -1, -1):
        W = chain.P @ V[t + 1]                       # E[V_{t+1}(k', c') | k]
        q0 = -cnt.sc[None, :, 0] + gamma * W[:, cnt.nxt[:, 0]]
        q1 = r[t][:, None] - cnt.sc[None, :, 1] + gamma * W[:, cnt.nxt[:, 1]]
        Qt = np.stack([q0, q1], axis=-1)
        Qt = np.where(mask, Qt, -np.inf)
        Q[t] = Qt
        V[t] = Qt.max(axis=-1)
    policy = (Q[..., 1] >= Q[..., 0]).astype(np.int8)
    return Solution(unit, chain, cnt, gamma, V, Q, policy, r)


def evaluate_policy(unit: Unit, chain: MarkovChainModel, action_fn, gamma: float = 1.0):
    """Exact expected value of a Markov policy on the augmented state.

    action_fn(t) -> (K, n) int array of desired actions; illegal actions are overridden
    by keeping the current status.  Returns (V (T+1,K,n), actions (T,K,n), overridden (T,K,n))."""
    cnt = build_counter(unit)
    T, K, n = chain.T, chain.K, cnt.n
    r = unit.profit(chain.levels())
    V = np.zeros((T + 1, K, n))
    acts = np.zeros((T, K, n), np.int64)
    over = np.zeros((T, K, n), bool)
    cols = np.arange(n)[None, :]
    for t in range(T - 1, -1, -1):
        a = np.asarray(action_fn(t), np.int64)
        legal = cnt.feas[cols, a]
        keep = cnt.is_on.astype(np.int64)[None, :].repeat(K, 0)
        a_eff = np.where(legal, a, keep)
        W = chain.P @ V[t + 1]
        nxt = cnt.nxt[cols, a_eff]
        V[t] = a_eff * r[t][:, None] - cnt.sc[cols, a_eff] + gamma * np.take_along_axis(W, nxt, axis=1)
        acts[t], over[t] = a_eff, ~legal
    return V, acts, over


def simulate(unit: Unit, chain: MarkovChainModel, action_fn, idx_paths: np.ndarray, c0: int,
             gamma: float = 1.0):
    """Simulate a policy on given price-index paths.  Returns dict with per-path value,
    status matrix, and override counts."""
    cnt = build_counter(unit)
    n_paths, T = idx_paths.shape
    r = unit.profit(chain.levels())
    c = np.full(n_paths, c0, np.int64)
    total = np.zeros(n_paths)
    status = np.zeros((n_paths, T), np.int8)
    overrides = np.zeros(n_paths, np.int64)
    tables = [np.asarray(action_fn(t), np.int64) for t in range(T)]
    for t in range(T):
        k = idx_paths[:, t]
        a = tables[t][k, c]
        legal = cnt.feas[c, a]
        a = np.where(legal, a, cnt.is_on[c].astype(np.int64))
        overrides += ~legal
        total += gamma**t * (a * r[t, k] - cnt.sc[c, a])
        status[:, t] = a
        c = cnt.nxt[c, a]
    return {"value": total, "status": status, "overrides": overrides}


# --------------------------------------------------------------------------------------
# Threshold extraction
# --------------------------------------------------------------------------------------
def start_thresholds(sol: Solution):
    """For each t and each off-counter tau >= DT: smallest price index at which starting is
    optimal (K if never).  Returns (T, J) array with -1 where starting is infeasible."""
    cnt = sol.cnt
    T, K = sol.chain.T, sol.chain.K
    out = -np.ones((T, cnt.J), np.int64)
    for j in range(1, cnt.J + 1):
        if j < cnt.DT:
            continue
        c = cnt.off(j)
        pol = sol.policy[:, :, c]              # (T, K)
        first = np.where(pol.any(axis=1), pol.argmax(axis=1), K)
        out[:, j - 1] = first
    return out


def shutdown_thresholds(sol: Solution):
    """For each t: largest price index at which shutting down (from on, tau=UT) is optimal
    (-1 if never)."""
    c = sol.cnt.on(sol.cnt.UT)
    off = sol.policy[:, :, c] == 0
    K = sol.chain.K
    return np.where(off.any(axis=1), K - 1 - off[:, ::-1].argmax(axis=1), -1)


def single_crossing_violations(G: np.ndarray, tol: float) -> int:
    """Count (t, decision-state) columns whose advantage along the price axis (axis 1)
    goes strictly positive and later strictly negative."""
    s = np.where(G > tol, 1, np.where(G < -tol, -1, 0))
    seen_pos = np.maximum.accumulate(s == 1, axis=1)
    bad = seen_pos[:, :-1] & (s[:, 1:] == -1)
    return int(bad.any(axis=1).sum())


def monotone_violations(G: np.ndarray, tol: float) -> int:
    """Count columns where G decreases in the price index by more than tol."""
    return int((np.diff(G, axis=1) < -tol).any(axis=1).sum())


def decision_advantages(sol: Solution):
    """Return (G_start (T,K,m), G_stay (T,K)) at all decision states:
    G_start[..., i] = Q(start) - Q(stay off) at off counter tau = DT..J,
    G_stay = Q(stay on) - Q(shut down) at on counter tau = UT."""
    A = sol.advantage()
    cnt = sol.cnt
    offs = [cnt.off(j) for j in range(cnt.DT, cnt.J + 1)]
    return A[:, :, offs], A[:, :, cnt.on(cnt.UT)]


def value_differences(sol: Solution):
    """V_t(k, on i) - V_t(k, off j) for all on counters i and off counters j: (T+1, K, UT, J)."""
    cnt = sol.cnt
    Von = sol.V[:, :, :cnt.UT]
    Voff = sol.V[:, :, cnt.UT:]
    return Von[:, :, :, None] - Voff[:, :, None, :]


# --------------------------------------------------------------------------------------
# Independent reference implementations (used only in verification)
# --------------------------------------------------------------------------------------
def _hist_rules(unit: Unit, hist):
    """Feasible actions and startup cost from an explicit status history (no counters)."""
    last = hist[-1]
    run = 0
    for s in reversed(hist):
        if s != last:
            break
        run += 1
    if last == 1:
        return {1: 0.0, 0: 0.0} if run >= unit.UT else {1: 0.0}
    if run < unit.DT:
        return {0: 0.0}
    return {0: 0.0, 1: float(unit.S(run))}


def initial_history(unit: Unit, u0: int, tau0: int):
    """Explicit history whose current run has exact length tau0 in status u0 (uncapped)."""
    return tuple([1 - u0] + [u0] * tau0)


def brute_force_deterministic(unit: Unit, prices: np.ndarray, u0: int, tau0: int):
    """Enumerate all 2^T schedules; feasibility and costs from explicit histories."""
    T = len(prices)
    r = unit.profit(prices)
    best, best_x = -np.inf, None
    for x in itertools.product((0, 1), repeat=T):
        hist = initial_history(unit, u0, tau0)
        val = 0.0
        ok = True
        for t, a in enumerate(x):
            rules = _hist_rules(unit, hist)
            if a not in rules:
                ok = False
                break
            val += a * r[t] - rules[a]
            hist = hist + (a,)
        if ok and val > best:
            best, best_x = val, x
    return best, best_x


def full_history_value(unit: Unit, chain: MarkovChainModel, k0: int, u0: int, tau0: int,
                       gamma: float = 1.0) -> float:
    """Optimal expected value by recursion over (price index, full status history)."""
    r = unit.profit(chain.levels())
    P = chain.P
    T, K = chain.T, chain.K

    def rec(t, k, hist):
        if t == T:
            return 0.0
        best = -np.inf
        for a, S in _hist_rules(unit, hist).items():
            cont = 0.0
            for k2 in range(K):
                if P[k, k2] > 0:
                    cont += P[k, k2] * rec(t + 1, k2, hist + (a,))
            best = max(best, a * r[t, k] - S + gamma * cont)
        return best

    return rec(0, k0, initial_history(unit, u0, tau0))
