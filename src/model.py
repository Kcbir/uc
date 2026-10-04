"""Unit model and the minimal counter automaton (augmented state).

Timing convention (used everywhere): the decision state at the start of period t is
(k_t, u_{t-1}, tau_t), where u_{t-1} is the status in the previous period and tau_t is
the number of consecutive periods, ending at t-1, spent in that status (capped).
The action a_t in {0,1} is the status in period t.  A unit started in period t
(u_{t-1}=0, a_t=1) pays S(tau_t) and earns r(p_t) in period t.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Unit:
    name: str
    Pmin: float
    Pmax: float
    c: float          # marginal cost $/MWh
    F: float          # no-load cost $/h
    UT: int           # minimum up time (periods)
    DT: int           # minimum down time (periods)
    S_hot: float
    S_warm: float
    S_cold: float
    T_warm: int       # offline duration at which the warm cost starts
    T_cold: int       # offline duration at which the cold cost starts
    source: str = "illustrative"

    def S(self, tau) -> np.ndarray:
        """Startup cost after tau >= 1 periods offline (nondecreasing, piecewise constant)."""
        tau = np.asarray(tau)
        return np.where(tau >= self.T_cold, self.S_cold,
                        np.where(tau >= self.T_warm, self.S_warm, self.S_hot)).astype(float)

    @property
    def T_cold_eff(self) -> int:
        """Smallest tau >= 1 such that S is constant on [tau, infinity)."""
        L = max(self.T_cold, self.T_warm, 1)
        vals = self.S(np.arange(1, L + 1))
        diff = np.nonzero(vals != vals[-1])[0]
        return 1 if len(diff) == 0 else int(diff[-1]) + 2

    @property
    def J(self) -> int:
        """Cap of the offline counter: max(DT, T_cold)."""
        return max(self.DT, self.T_cold_eff)

    @property
    def n_counter(self) -> int:
        return self.UT + self.J

    def profit(self, p) -> np.ndarray:
        """r(p) = max_{q in [Pmin,Pmax]} (p-c) q - F  (nondecreasing, convex)."""
        p = np.asarray(p, float)
        return np.where(p >= self.c, (p - self.c) * self.Pmax, (p - self.c) * self.Pmin) - self.F

    def break_even(self) -> float:
        """Price at which r(p) = 0."""
        if self.F <= 0:
            return self.c if self.Pmin == 0 else self.c + self.F / max(self.Pmin, 1e-300)
        return self.c + self.F / self.Pmax


@dataclass
class Counter:
    """Deterministic counter automaton on the augmented status component.

    index i in [0, UT)        : on,  tau = i + 1          (tau = UT means ">= UT")
    index UT + j - 1, j=1..J  : off, tau = j              (tau = J  means ">= J")
    """
    UT: int
    J: int
    is_on: np.ndarray    # (n,) bool
    tau: np.ndarray      # (n,) int
    feas: np.ndarray     # (n, 2) bool
    nxt: np.ndarray      # (n, 2) int
    sc: np.ndarray       # (n, 2) startup cost incurred
    DT: int = 1

    @property
    def n(self) -> int:
        return self.UT + self.J

    def on(self, tau: int) -> int:
        return min(tau, self.UT) - 1

    def off(self, tau: int) -> int:
        return self.UT + min(tau, self.J) - 1

    def index(self, u: int, tau: int) -> int:
        return self.on(tau) if u else self.off(tau)

    def decision_states(self):
        """Counter indices where both actions are feasible."""
        return np.nonzero(self.feas.all(axis=1))[0]


def build_counter(unit: Unit) -> Counter:
    UT, DT, J = unit.UT, unit.DT, unit.J
    n = UT + J
    is_on = np.zeros(n, bool)
    is_on[:UT] = True
    tau = np.concatenate([np.arange(1, UT + 1), np.arange(1, J + 1)])
    feas = np.zeros((n, 2), bool)
    nxt = np.zeros((n, 2), np.int64)
    sc = np.zeros((n, 2))
    on_idx = np.arange(UT)
    off_idx = UT + np.arange(J)
    # on states
    feas[on_idx, 1] = True
    nxt[on_idx, 1] = np.minimum(on_idx + 1, UT - 1)
    feas[on_idx, 0] = tau[on_idx] >= UT
    nxt[on_idx, 0] = UT  # off, tau = 1
    # off states
    feas[off_idx, 0] = True
    nxt[off_idx, 0] = UT + np.minimum(np.arange(J) + 1, J - 1)
    feas[off_idx, 1] = tau[off_idx] >= DT
    nxt[off_idx, 1] = 0  # on, tau = 1
    sc[off_idx, 1] = unit.S(tau[off_idx])
    return Counter(UT, J, is_on, tau, feas, nxt, sc, DT)


def naive_unit(unit: Unit, S_const: float) -> Unit:
    """History-blind model: UT = DT = 1 and one constant startup cost."""
    return replace(unit, name=unit.name + "-naive", UT=1, DT=1, S_hot=S_const, S_warm=S_const,
                   S_cold=S_const, T_warm=1, T_cold=1)


# --------------------------------------------------------------------------------------
# RTS-GMLC unit data
# --------------------------------------------------------------------------------------
THERMAL_CATEGORIES = ("Coal", "Gas CC", "Gas CT", "Oil CT", "Oil ST")
RTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gen.csv")


def _ceil_hours(x: float) -> int:
    return max(1, int(math.ceil(float(x) - 1e-9)))


def unit_from_rts_row(r: dict) -> Unit:
    """Build a Unit from one RTS-GMLC gen.csv row.

    Costs: the piecewise-linear heat-rate curve is replaced by its chord between Pmin and
    Pmax (c = slope, F = intercept), times the fuel price.  Startup costs are
    Start Heat {Hot,Warm,Cold} x fuel price + non-fuel start cost.  We interpret
    Start Time {Warm,Cold} Hr as the offline durations at which the warm / cold cost
    applies; all durations are rounded up to whole hours.
    """
    fp = float(r["Fuel Price $/MMBTU"])
    pmax = float(r["PMax MW"])
    pcts = [float(r[f"Output_pct_{i}"]) for i in range(4)]
    P = [p * pmax for p in pcts]
    hr0 = float(r["HR_avg_0"])
    inc = [float(r[f"HR_incr_{i}"]) for i in range(1, 4)]
    fuel = [hr0 * P[0] / 1000.0]
    for i in range(3):
        fuel.append(fuel[-1] + inc[i] * (P[i + 1] - P[i]) / 1000.0)
    cost = [f * fp for f in fuel]
    c = (cost[-1] - cost[0]) / (P[-1] - P[0])
    F = cost[0] - c * P[0]
    nf = float(r["Non Fuel Start Cost $"] or 0.0)
    S = [float(r[k]) * fp + nf for k in ("Start Heat Hot MBTU", "Start Heat Warm MBTU", "Start Heat Cold MBTU")]
    T_warm = _ceil_hours(r["Start Time Warm Hr"])
    T_cold = max(T_warm, _ceil_hours(r["Start Time Cold Hr"]))
    return Unit(name=r["GEN UID"], Pmin=round(P[0], 3), Pmax=pmax, c=round(c, 3), F=round(F, 2),
                UT=_ceil_hours(r["Min Up Time Hr"]), DT=_ceil_hours(r["Min Down Time Hr"]),
                S_hot=round(S[0], 1), S_warm=round(S[1], 1), S_cold=round(S[2], 1),
                T_warm=T_warm, T_cold=T_cold, source="RTS-GMLC")


def load_rts_units(path: str = RTS_PATH):
    """All thermal RTS-GMLC units (Coal, Gas CC, Gas CT, Oil CT, Oil ST).

    Returns (units, categories) or (None, None) if the file is unavailable.
    The single nuclear unit is excluded (must-run; its start-time fields are 9999)."""
    if not os.path.exists(path):
        return None, None
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    units, cats = [], []
    for r in rows:
        if r["Category"] in THERMAL_CATEGORIES:
            units.append(unit_from_rts_row(r))
            cats.append(r["Category"])
    return units, cats


def illustrative_units():
    """Fallback only (used if data/gen.csv is missing): clearly illustrative values."""
    return [
        Unit("coal-illustr", 30, 76, 16.4, 350, 8, 4, 7100, 10300, 11200, 10, 12),
        Unit("ccgt-illustr", 170, 355, 26.8, 210, 8, 5, 12400, 17600, 28000, 1, 2),
        Unit("ct-illustr", 22, 55, 28.9, 490, 3, 3, 1760, 4360, 5670, 1, 1),
    ], ["Coal", "Gas CC", "Gas CT"]


def rts_unit(uid: str) -> Unit:
    units, _ = load_rts_units()
    if units is None:
        raise FileNotFoundError(RTS_PATH)
    for u in units:
        if u.name == uid:
            return u
    raise KeyError(uid)
