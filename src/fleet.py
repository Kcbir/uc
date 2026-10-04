"""Exact fleet unit commitment DP on (net-load index, joint counter).

Cost minimisation:  V_t(k, c) = min_a  SC(c, a) + ED_t(k, a) + gamma * E[V_{t+1}(k', next(c, a))]
where a in {0,1}^N is the joint commitment, c is the joint counter (mixed radix over the
per-unit counters) and ED_t(k, a) is the exact economic-dispatch cost by merit order.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .model import Unit, build_counter
from .price_chain import MarkovChainModel


@dataclass
class FleetTables:
    units: list
    radix: np.ndarray        # (N,) counter sizes
    C: int
    A: int
    status: np.ndarray       # (C, N) current status bits of each joint counter
    nxt: np.ndarray          # (C, A) int32 next joint counter (valid where feasible)
    sc: np.ndarray           # (C, A) startup cost, +inf where infeasible
    legal: np.ndarray        # (C, A) bool
    digits: np.ndarray       # (C, N) per-unit counter index
    cnts: list               # per-unit Counter


def action_bits(N: int) -> np.ndarray:
    """(A, N) matrix: bit g of action a is unit g's status."""
    a = np.arange(2**N)
    return ((a[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int64)


def build_tables(units) -> FleetTables:
    cnts = [build_counter(u) for u in units]
    N = len(units)
    radix = np.array([c.n for c in cnts])
    C = int(np.prod(radix))
    A = 2**N
    bits = action_bits(N)
    digits = np.array(np.unravel_index(np.arange(C), radix)).T   # (C, N), C-order
    strides = np.array([int(np.prod(radix[g + 1:])) for g in range(N)])
    nxt = np.zeros((C, A), np.int64)
    sc = np.zeros((C, A))
    legal = np.ones((C, A), bool)
    status = np.zeros((C, N), np.int8)
    for g, cn in enumerate(cnts):
        cg = digits[:, g][:, None]
        ag = bits[:, g][None, :]
        legal &= cn.feas[cg, ag]
        nxt += cn.nxt[cg, ag] * strides[g]
        sc += cn.sc[cg, ag]
        status[:, g] = cn.is_on[digits[:, g]]
    sc = np.where(legal, sc, np.inf)
    return FleetTables(list(units), radix, C, A, status, nxt.astype(np.int32), sc, legal, digits, cnts)


def joint_index(units, states) -> int:
    """states: list of (u, tau) per unit."""
    cnts = [build_counter(u) for u in units]
    radix = [c.n for c in cnts]
    return int(np.ravel_multi_index([c.index(u, t) for c, (u, t) in zip(cnts, states)], radix))


def dispatch_cost(units, demand: np.ndarray, voll: float, spill: float) -> np.ndarray:
    """Exact economic dispatch cost for every commitment set and every demand level.

    min sum_g (F_g + c_g q_g) + voll*unserved + spill*overgeneration
    s.t. Pmin_g <= q_g <= Pmax_g for committed units.  Because costs are linear, filling
    the committed units in merit order above their Pmin is optimal.
    Returns (len(demand), 2^N)."""
    N = len(units)
    bits = action_bits(N).astype(float)                 # (A, N)
    Pmin = np.array([u.Pmin for u in units])
    Pmax = np.array([u.Pmax for u in units])
    c = np.array([u.c for u in units])
    F = np.array([u.F for u in units])
    order = np.argsort(c, kind="stable")
    d = np.asarray(demand, float)[:, None]              # (D, 1)
    base = bits @ (F + c * Pmin)                        # (A,)
    rem = d - (bits @ Pmin)[None, :]                    # (D, A)
    cost = base[None, :] + spill * np.maximum(-rem, 0.0)
    rem = np.maximum(rem, 0.0)
    for g in order:
        room = bits[:, g] * (Pmax[g] - Pmin[g])
        q = np.minimum(rem, room[None, :])
        cost += c[g] * q
        rem -= q
    return cost + voll * rem


def solve_fleet(units, chain: MarkovChainModel, voll=1000.0, spill=1000.0, gamma=1.0,
                tables: FleetTables | None = None, store_policy=True):
    tb = tables or build_tables(units)
    T, K = chain.T, chain.K
    ED = np.stack([dispatch_cost(units, chain.levels()[t], voll, spill) for t in range(T)])  # (T,K,A)
    V = np.zeros((K, tb.C))
    pol = np.zeros((T, K, tb.C), np.uint8 if tb.A <= 256 else np.int32) if store_policy else None
    for t in range(T - 1, -1, -1):
        W = chain.P @ V
        Vn = np.empty_like(V)
        for k in range(K):
            q = tb.sc + ED[t, k][None, :] + gamma * W[k][tb.nxt]
            a = q.argmin(axis=1)
            Vn[k] = q[np.arange(tb.C), a]
            if store_policy:
                pol[t, k] = a
        V = Vn
    return {"V0": V, "policy": pol, "tables": tb, "ED": ED}


def evaluate_fleet_policy(units, chain: MarkovChainModel, desired, voll=1000.0, spill=1000.0,
                          gamma=1.0, tables: FleetTables | None = None, ED=None):
    """Exact expected cost of a policy given by desired(t) -> (K, C) joint actions; per-unit
    illegal actions are overridden by keeping that unit's status."""
    tb = tables or build_tables(units)
    T, K = chain.T, chain.K
    if ED is None:
        ED = np.stack([dispatch_cost(units, chain.levels()[t], voll, spill) for t in range(T)])
    N = len(units)
    bits = action_bits(N)
    pow2 = (1 << np.arange(N))
    V = np.zeros((K, tb.C))
    rows = np.arange(tb.C)
    for t in range(T - 1, -1, -1):
        W = chain.P @ V
        Vn = np.empty_like(V)
        des = desired(t)
        for k in range(K):
            a = override(tb, bits, pow2, des[k])
            Vn[k] = tb.sc[rows, a] + ED[t, k][a] + gamma * W[k][tb.nxt[rows, a]]
        V = Vn
    return V


def override(tb: FleetTables, bits, pow2, a_des: np.ndarray) -> np.ndarray:
    """Replace each unit's illegal desired action by its current status (vectorised)."""
    rows = np.arange(len(a_des))
    if np.all(tb.legal[rows, a_des]):
        return a_des
    want = bits[a_des]                                  # (C, N)
    eff = want.copy()
    for g, cn in enumerate(tb.cnts):
        digit = tb.digits[:, g]
        ok = cn.feas[digit, want[:, g]]
        eff[:, g] = np.where(ok, want[:, g], cn.is_on[digit].astype(np.int64))
    return eff @ pow2


def simulate_fleet(units, chain: MarkovChainModel, desired, idx_paths, c0, voll=1000.0,
                   spill=1000.0, gamma=1.0, tables: FleetTables | None = None, ED=None):
    tb = tables or build_tables(units)
    n, T = idx_paths.shape
    N = len(units)
    if ED is None:
        ED = np.stack([dispatch_cost(units, chain.levels()[t], voll, spill) for t in range(T)])
    bits = action_bits(N)
    pow2 = (1 << np.arange(N))
    cnts = [build_counter(u) for u in units]
    strides = np.array([int(np.prod(tb.radix[g + 1:])) for g in range(N)])
    c = np.full(n, c0, np.int64)
    total = np.zeros(n)
    overrides = np.zeros(n, np.int64)
    for t in range(T):
        k = idx_paths[:, t]
        des = desired(t)[k, c]
        want = bits[des]
        eff = want.copy()
        for g, cn in enumerate(cnts):
            digit = (c // strides[g]) % tb.radix[g]
            ok = cn.feas[digit, want[:, g]]
            overrides += ~ok
            eff[:, g] = np.where(ok, want[:, g], cn.is_on[digit].astype(np.int64))
        a = eff @ pow2
        total += gamma**t * (tb.sc[c, a] + ED[t, k, a])
        c = tb.nxt[c, a].astype(np.int64)
    return {"cost": total, "overrides": overrides}


def naive_to_true_map(units, naive_units, tb: FleetTables) -> np.ndarray:
    """For each true joint counter, the naive joint counter with the same status vector."""
    naive_cnts = [build_counter(u) for u in naive_units]
    radix = [c.n for c in naive_cnts]
    idx = [np.where(tb.status[:, g] == 1, naive_cnts[g].on(1), naive_cnts[g].off(1)) for g in range(len(units))]
    return np.ravel_multi_index(idx, radix)


def brute_force_fleet(units, demand_path, init_states, voll=1000.0, spill=1000.0):
    """Enumerate all joint schedules (deterministic demand); history-based rules per unit."""
    from .dp import _hist_rules, initial_history
    N, T = len(units), len(demand_path)
    ED = dispatch_cost(units, np.asarray(demand_path, float), voll, spill)   # (T, A)
    pow2 = 1 << np.arange(N)
    best = np.inf
    for sched in itertools.product(range(2**N), repeat=T):
        hists = [initial_history(u, *s) for u, s in zip(units, init_states)]
        cost = 0.0
        ok = True
        for t, a in enumerate(sched):
            for g, u in enumerate(units):
                ag = (a >> g) & 1
                rules = _hist_rules(u, hists[g])
                if ag not in rules:
                    ok = False
                    break
                cost += rules[ag]
                hists[g] = hists[g] + (ag,)
            if not ok:
                break
            cost += ED[t, a]
        if ok:
            best = min(best, cost)
    return best
