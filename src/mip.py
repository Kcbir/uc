"""Deterministic single-unit UC as a MIP (HiGHS via scipy.optimize.milp).

Three-binary (u, v, w) formulation with min up/down "turn-on/turn-off" inequalities and
startup-cost categories selected by the time since the last shutdown, in the style
surveyed by Knueven, Ostrowski & Watson (2020).  Used (i) as an independent check of the
DP on deterministic price paths and (ii) as the "deterministic UC on the mean path" baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds

from .model import Unit


def startup_categories(unit: Unit):
    """Maximal runs of constant S(tau) for tau in [DT, J]; last one extends to infinity.
    Returns list of (lo, hi, cost) with hi = None for the last category."""
    taus = np.arange(unit.DT, unit.J + 1)
    S = unit.S(taus)
    cats = []
    lo = taus[0]
    for i in range(1, len(taus)):
        if S[i] != S[i - 1]:
            cats.append((int(lo), int(taus[i]), float(S[i - 1])))
            lo = taus[i]
    cats.append((int(lo), None, float(S[-1])))
    return cats


def solve_deterministic_mip(unit: Unit, prices, u0: int, tau0: int, time_limit: float = 60.0):
    """Maximise sum_t [(p_t - c) q_t - F u_t] - startup costs.  Returns (value, schedule)."""
    p = np.asarray(prices, float)
    T = len(p)
    cats = startup_categories(unit)
    nc = len(cats)
    # variable layout per t: u, v, w, q, delta_1..delta_nc
    nv = 4 + nc
    N = nv * T
    iu = lambda t: nv * t
    iv = lambda t: nv * t + 1
    iw = lambda t: nv * t + 2
    iq = lambda t: nv * t + 3
    idl = lambda t, s: nv * t + 4 + s

    obj = np.zeros(N)
    for t in range(T):
        obj[iq(t)] = -(p[t] - unit.c)
        obj[iu(t)] = unit.F
        for s, (_, _, cost) in enumerate(cats):
            obj[idl(t, s)] = cost

    rows, lbs, ubs = [], [], []

    def add(coefs, lb, ub):
        row = np.zeros(N)
        for j, a in coefs:
            row[j] += a
        rows.append(row)
        lbs.append(lb)
        ubs.append(ub)

    lb_var = np.zeros(N)
    ub_var = np.ones(N)
    for t in range(T):
        ub_var[iq(t)] = unit.Pmax
    J = unit.J
    tau0c = min(tau0, unit.UT if u0 == 1 else J)
    for t in range(T):
        # logic
        prev = [] if t == 0 else [(iu(t - 1), -1.0)]
        const = u0 if t == 0 else 0.0
        add([(iu(t), 1.0), (iv(t), -1.0), (iw(t), 1.0)] + prev, const, const)
        # output bounds
        add([(iq(t), 1.0), (iu(t), -unit.Pmin)], 0.0, np.inf)
        add([(iq(t), 1.0), (iu(t), -unit.Pmax)], -np.inf, 0.0)
        # min up / min down
        add([(iv(s), 1.0) for s in range(max(0, t - unit.UT + 1), t + 1)] + [(iu(t), -1.0)], -np.inf, 0.0)
        add([(iw(s), 1.0) for s in range(max(0, t - unit.DT + 1), t + 1)] + [(iu(t), 1.0)], -np.inf, 1.0)
        # initial conditions
        if u0 == 1 and t < unit.UT - tau0c:
            lb_var[iu(t)] = 1.0
        if u0 == 0 and t < unit.DT - tau0c:
            ub_var[iu(t)] = 0.0
        # startup categories
        add([(idl(t, s), 1.0) for s in range(nc)] + [(iv(t), -1.0)], 0.0, 0.0)
        for s, (lo, hi, _) in enumerate(cats[:-1]):
            coefs = [(idl(t, s), 1.0)]
            const = 0.0
            for i in range(lo, hi):
                if t - i >= 0:
                    coefs.append((iw(t - i), -1.0))
                elif u0 == 0 and tau0 < J and t - i == -tau0:
                    const += 1.0   # the initial shutdown happened tau0 periods before t = 0
            add(coefs, -np.inf, const)
    integrality = np.ones(N)
    for t in range(T):
        integrality[iq(t)] = 0
    res = milp(obj, integrality=integrality, bounds=Bounds(lb_var, ub_var),
               constraints=LinearConstraint(np.array(rows), np.array(lbs), np.array(ubs)),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if not res.success or res.x is None:
        raise RuntimeError(res.message)
    sched = np.round(res.x[[iu(t) for t in range(T)]]).astype(int)
    return -res.fun, sched


def schedule_value(unit: Unit, prices, sched, u0: int, tau0: int):
    """Profit of a fixed schedule on a price path (raises if infeasible)."""
    from .dp import _hist_rules, initial_history
    r = unit.profit(np.asarray(prices, float))
    hist = initial_history(unit, u0, tau0)
    val = 0.0
    for t, a in enumerate(sched):
        rules = _hist_rules(unit, hist)
        if a not in rules:
            raise ValueError("infeasible schedule")
        val += a * r[t] - rules[a]
        hist = hist + (int(a),)
    return val


def schedule_startup_cost(unit: Unit, sched, u0: int, tau0: int) -> float:
    """Total startup cost of a fixed schedule (raises if infeasible)."""
    from .dp import _hist_rules, initial_history
    hist = initial_history(unit, u0, tau0)
    cost = 0.0
    for a in sched:
        rules = _hist_rules(unit, hist)
        if int(a) not in rules:
            raise ValueError("infeasible schedule")
        cost += rules[int(a)]
        hist = hist + (int(a),)
    return cost
