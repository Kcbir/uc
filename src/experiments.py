"""All experiments.  Every number that appears in the paper is written by this module to
results/*.json, tables/*.tex or tables/numbers.tex (LaTeX macros)."""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

from . import adversary
from .dp import (brute_force_deterministic, decision_advantages, evaluate_policy, full_history_value,
                 monotone_violations, shutdown_thresholds, simulate, single_crossing_violations, solve,
                 start_thresholds, value_differences)
from .fleet import (brute_force_fleet, build_tables, dispatch_cost, evaluate_fleet_policy, joint_index,
                    naive_to_true_map, simulate_fleet, solve_fleet)
from .instances import random_chain, random_det_path, random_initial, random_unit
from .mip import schedule_startup_cost, schedule_value, solve_deterministic_mip
from .model import (THERMAL_CATEGORIES, Unit, build_counter, illustrative_units, load_rts_units, naive_unit)
from .price_chain import (MarkovChainModel, daily_profile, deterministic_chain, is_stochastically_monotone,
                          make_chain, stationary_distribution)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "tables")

# --------------------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------------------
CHECKS = []          # verification table rows (claims that are proved)
FINDINGS = []        # numerical findings (not claims)
MACROS = {}
CHAINS = {"n": 0, "monotone": 0}
TIMINGS = {}


def reg_chain(ch: MarkovChainModel):
    CHAINS["n"] += 1
    ok = is_stochastically_monotone(ch.P)
    CHAINS["monotone"] += ok
    assert ok, "price chain is not stochastically monotone"
    return ch


def check(name, ref, instances, failures, unit="instances", note=""):
    CHECKS.append(dict(check=name, ref=ref, instances=int(instances), passes=int(instances - failures),
                       failures=int(failures), unit=unit, note=note))


def finding(name, instances, violations, unit="instances", note=""):
    FINDINGS.append(dict(finding=name, instances=int(instances), violations=int(violations), unit=unit, note=note))


def macro(name, value, fmt="{}"):
    assert name.isalpha(), name
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def save_json(name, obj):
    os.makedirs(RES, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Unit):
            return asdict(o)
        raise TypeError(type(o))

    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def close(a, b, rel=1e-7):
    return abs(a - b) <= rel * (1 + abs(a) + abs(b))


def fmt_int(n):
    return f"{int(n):,}".replace(",", "{,}")


# --------------------------------------------------------------------------------------
# base configuration (illustrative price process; unit data from RTS-GMLC when available)
# --------------------------------------------------------------------------------------
BASE = dict(T=48, K=21, rho=0.9, sigma=12.0, method="rouwenhorst", mu_base=18.0, mu_amp=45.0)
SINGLE_IDS = ["101_STEAM_3", "115_STEAM_3", "123_STEAM_3", "107_CC_1", "113_CT_1"]
FLEET_IDS = ["101_STEAM_3", "107_CC_1", "113_CT_1", "115_STEAM_1", "101_CT_1", "113_CT_2"]
FLEET = dict(T=24, K=9, rho=0.9, sigma_frac=0.06, base_frac=0.52, amp_frac=0.33, voll=1000.0, spill=1000.0)


def get_units():
    units, cats = load_rts_units()
    if units is None:
        units, cats = illustrative_units()
        return units, cats, False
    return units, cats, True


def unit_by_id(units, uid):
    for u in units:
        if u.name == uid:
            return u
    raise KeyError(uid)


def base_chain(K=None, rho=None, sigma=None):
    b = BASE
    mu = daily_profile(b["T"], b["mu_base"], b["mu_amp"])
    return reg_chain(make_chain(mu, K=K or b["K"], rho=b["rho"] if rho is None else rho,
                                sigma_stat=sigma or b["sigma"], method=b["method"]))


# --------------------------------------------------------------------------------------
# 1. exactness checks
# --------------------------------------------------------------------------------------
def exp_exactness(n_bf=200, n_hist=30, n_mip=100, seed=1):
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(n_bf):
        u = random_unit(rng, max_ut=5, max_dt=5)
        ch = random_det_path(rng, 10)
        u0, tau0 = random_initial(rng, u)
        s = solve(u, ch)
        v_bf, _ = brute_force_deterministic(u, ch.mu, u0, tau0)
        fails += not close(s.V[0, 0, s.cnt.index(u0, tau0)], v_bf)
    check("DP optimum = exhaustive enumeration of all $2^{10}$ schedules (deterministic prices)", "Thm.~\\ref{thm:markov}", n_bf, fails)
    macro("NBruteForce", n_bf)

    fails = 0
    for _ in range(n_hist):
        u = random_unit(rng, max_ut=3, max_dt=3)
        ch = reg_chain(random_chain(rng, T=6, K=3))
        u0, tau0 = random_initial(rng, u)
        k0 = int(rng.integers(0, 3))
        s = solve(u, ch)
        fails += not close(s.V[0, k0, s.cnt.index(u0, tau0)], full_history_value(u, ch, k0, u0, tau0))
    check("Augmented-state DP = DP over full status histories (stochastic prices)", "Thm.~\\ref{thm:markov}", n_hist, fails)
    macro("NHistory", n_hist)

    fails = fails2 = 0
    for _ in range(n_mip):
        u = random_unit(rng)
        ch = random_det_path(rng, 24)
        u0, tau0 = random_initial(rng, u)
        s = solve(u, ch)
        v_mip, sched = solve_deterministic_mip(u, ch.mu, u0, tau0)
        fails += not close(s.V[0, 0, s.cnt.index(u0, tau0)], v_mip, rel=1e-6)
        fails2 += not close(schedule_value(u, ch.mu, sched, u0, tau0), v_mip, rel=1e-6)
    check("DP optimum = HiGHS MIP optimum ($T=24$, deterministic prices)", "Thm.~\\ref{thm:markov}", n_mip, fails)
    macro("NMip", n_mip)
    return {"n_bf": n_bf, "n_hist": n_hist, "n_mip": n_mip}


# --------------------------------------------------------------------------------------
# 2. minimality of the counter (Moore minimisation + value distinguishability)
# --------------------------------------------------------------------------------------
def moore_classes(cnt) -> int:
    n = cnt.n

    def relabel(sigs):
        ids = {}
        return np.array([ids.setdefault(s, len(ids)) for s in sigs])

    sig0 = [tuple((bool(cnt.feas[c, a]), float(cnt.sc[c, a]) if cnt.feas[c, a] else None) for a in (0, 1))
            for c in range(n)]
    cls = relabel(sig0)
    while True:
        sig = [(int(cls[c]),) + tuple(int(cls[cnt.nxt[c, a]]) if cnt.feas[c, a] else -1 for a in (0, 1))
               for c in range(n)]
        new = relabel(sig)
        if new.max() == cls.max():
            return int(new.max()) + 1
        cls = new


def distinguishing_word(cnt, c1, c2):
    """Shortest action word after which feasibility or incurred cost differs (BFS on pairs)."""
    seen = {(c1, c2)}
    q = deque([(c1, c2, ())])
    while q:
        x, y, w = q.popleft()
        for a in (0, 1):
            fx, fy = cnt.feas[x, a], cnt.feas[y, a]
            if fx != fy or (fx and cnt.sc[x, a] != cnt.sc[y, a]):
                return w + (a,)
            if fx:
                nxt = (cnt.nxt[x, a], cnt.nxt[y, a])
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((*nxt, w + (a,)))
    return None


def exp_minimality(n_moore=2000, n_value=200, seed=2):
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(n_moore):
        u = random_unit(rng, max_ut=10, max_dt=10)
        u = replace(u, S_hot=max(u.S_hot, 1.0), S_warm=max(u.S_warm, 1.0), S_cold=max(u.S_cold, 1.0))
        fails += moore_classes(build_counter(u)) != u.UT + max(u.DT, u.T_cold_eff)
    check("Moore minimisation of the counter automaton leaves $\\UT+\\max(\\DT,T_{\\mathrm{cold}})$ classes",
          "Thm.~\\ref{thm:minimal}", n_moore, fails)
    macro("NMoore", fmt_int(n_moore))
    # the positivity assumption is needed: S = 0, UT = DT = 1 collapses on/off
    u0 = Unit("zero", 10, 50, 20, 0, 1, 1, 0, 0, 0, 1, 1)
    macro("MooreZeroS", moore_classes(build_counter(u0)))

    fails = pairs = 0
    for _ in range(n_value):
        u = random_unit(rng, max_ut=6, max_dt=6)
        u = replace(u, S_hot=max(u.S_hot, 1.0), S_warm=max(u.S_warm, 1.0), S_cold=max(u.S_cold, 1.0),
                    Pmin=max(u.Pmin, 1.0))
        cnt = build_counter(u)
        for c1 in range(cnt.n):
            for c2 in range(c1 + 1, cnt.n):
                w = distinguishing_word(cnt, c1, c2)
                pairs += 1
                if w is None:
                    fails += 1
                    continue
                L = len(w)
                M = 4.0 * (L * u.S_cold + u.F + 1.0) / min(u.Pmin, u.Pmax)
                prices = np.array([u.c + M if a else u.c - M for a in w])
                s = solve(u, deterministic_chain(prices))
                fails += abs(s.V[0, 0, c1] - s.V[0, 0, c2]) <= 1e-11 * s.scale
    check("Every pair of counter values has different optimal value on the constructed price path",
          "Cor.~\\ref{cor:value}", pairs, fails, unit="pairs")
    macro("NValuePairs", fmt_int(pairs))
    macro("NValueUnits", n_value)


# --------------------------------------------------------------------------------------
# 3. threshold structure
# --------------------------------------------------------------------------------------
def b1_checks(s, tol):
    """Checks of Theorem B1 on one UT=DT=1 constant-S solution."""
    u, ch, cnt = s.unit, s.chain, s.cnt
    S = u.S_hot
    on, off = cnt.on(1), cnt.off(1)
    D = s.Q[:, :, on, 1] - s.Q[:, :, on, 0]                   # Delta_t (T, K)
    Dn = np.vstack([D[1:], np.zeros((1, ch.K))])
    rec = s.r + s.gamma * (ch.P @ np.clip(Dn, 0, S).T).T
    ident = np.max(np.abs(D - rec)) <= tol
    mono = monotone_violations(D, tol) == 0
    start = s.policy[:, :, off] == 1
    stay = s.policy[:, :, on] == 1
    upper = np.all(np.diff(start.astype(int), axis=1) >= 0)
    lower = np.all(np.diff(stay.astype(int), axis=1) >= 0)
    hyst = np.all(start <= stay)                  # start => stay on  (p_down <= p_up)
    be_shut = np.all(s.r[~stay] < tol)
    be_start = np.all(s.r[start] >= (1 - s.gamma) * S - tol)
    return dict(ident=ident, mono=mono, thresh=upper and lower, hyst=hyst, be=be_shut and be_start, D=D)


def exp_b1(n=1000, n_flat=500, seed=3):
    rng = np.random.default_rng(seed)
    agg = dict(ident=0, mono=0, thresh=0, hyst=0, be=0)
    for i in range(n):
        u = random_unit(rng, UT=1, DT=1, const_S=True)
        ch = reg_chain(random_chain(rng))
        g = 1.0 if i % 2 == 0 else float(rng.uniform(0.8, 1.0))
        s = solve(u, ch, gamma=g)
        r = b1_checks(s, 1e-9 * s.scale)
        for k in agg:
            agg[k] += not r[k]
    check("$\\Delta_t=r+\\gamma\\,\\mathbb E[\\mathrm{clamp}(\\Delta_{t+1},0,S)]$ (every $t,p$)", "Thm.~\\ref{thm:b1}(a)", n, agg["ident"])
    check("$\\Delta_t(p)$ nondecreasing in $p$ (every $t$)", "Thm.~\\ref{thm:b1}(b)", n, agg["mono"])
    check("Start/shut-down sets are upper/lower price sets", "Thm.~\\ref{thm:b1}(c)", n, agg["thresh"])
    check("Hysteresis $p^{\\downarrow}_t\\le p^{\\uparrow}_t$", "Thm.~\\ref{thm:b1}(c)", n, agg["hyst"])
    check("Shut down only if $r(p)<0$; start only if $r(p)\\ge(1-\\gamma)S$", "Cor.~\\ref{cor:breakeven}", n, agg["be"])
    macro("NBone", fmt_int(n))
    fails = 0
    for _ in range(n_flat):
        u = random_unit(rng, UT=1, DT=1, const_S=True)
        ch = reg_chain(random_chain(rng, flat=True))
        s = solve(u, ch)
        r = b1_checks(s, 1e-9 * s.scale)
        fails += bool(np.any(np.diff(r["D"], axis=0) > 1e-9 * s.scale))
    check("Stationary prices: $\\Delta_t\\ge\\Delta_{t+1}$, thresholds nondecreasing in $t$", "Prop.~\\ref{prop:time}", n_flat, fails)
    macro("NBoneFlat", n_flat)


def analyse_general(s):
    """Per-instance statistics for the general (UT, DT, S(tau)) case."""
    tol = 1e-9 * s.scale
    u, cnt = s.unit, s.cnt
    Gs, Gk = decision_advantages(s)
    sc = sum(single_crossing_violations(Gs[:, :, j], tol) for j in range(Gs.shape[2])) + \
        single_crossing_violations(Gk, tol)
    mono = sum(monotone_violations(Gs[:, :, j], tol) for j in range(Gs.shape[2])) + monotone_violations(Gk, tol)
    D = value_differences(s)
    P_viol = int((np.diff(D, axis=1) < -tol).any())
    # break-even bracketing
    offs = [cnt.off(j) for j in range(cnt.DT, cnt.J + 1)]
    start_any = (s.policy[:, :, offs] == 1)
    r = s.r[:, :, None]
    be_start = bool(np.any(start_any & (r < (1 - s.gamma) * u.S(np.arange(cnt.DT, cnt.J + 1))[None, None, :] - tol)))
    shut = s.policy[:, :, cnt.on(cnt.UT)] == 0
    be_shut = bool(np.any(shut & (s.r > tol)))
    # sawtooth in tau
    th = start_thresholds(s)
    taus = np.arange(cnt.DT, cnt.J + 1)
    Sv = u.S(taus)
    within = across = n_within = n_across = 0
    for a in range(len(taus) - 1):
        k1, k2 = th[:, taus[a] - 1], th[:, taus[a + 1] - 1]
        if Sv[a + 1] == Sv[a]:
            n_within += len(k1)
            within += int(np.sum(k2 > k1))
        else:
            n_across += len(k1)
            across += int(np.sum(k2 < k1))
    # the proved inequality behind the sawtooth: G(j+1) >= G(j) whenever S(j+1) = S(j)
    Gsaw = 0
    for a in range(Gs.shape[2] - 1):
        if Sv[a + 1] == Sv[a]:
            Gsaw += int(np.any(Gs[:, :, a + 1] < Gs[:, :, a] - tol))
    return dict(sc=sc, mono=mono, P=P_viol, be_start=be_start, be_shut=be_shut, within=within,
                n_within=n_within, across=across, n_across=n_across, Gsaw=Gsaw)


def exp_general(n_iid=1000, n_gen=3000, n_flat=1000, seed=4):
    rng = np.random.default_rng(seed)
    fails = cols = 0
    for _ in range(n_iid):
        u = random_unit(rng)
        ch = reg_chain(random_chain(rng, rho=0.0))
        a = analyse_general(solve(u, ch))
        fails += a["sc"] > 0
    check("i.i.d.\\ prices: every decision state has a price threshold", "Thm.~\\ref{thm:iid}", n_iid, fails)
    macro("NIid", fmt_int(n_iid))

    tot = dict(sc=0, mono=0, P=0, be_start=0, be_shut=0, within=0, n_within=0, across=0, n_across=0, Gsaw=0)
    for _ in range(n_gen):
        u = random_unit(rng)
        ch = reg_chain(random_chain(rng))
        a = analyse_general(solve(u, ch))
        for k in tot:
            tot[k] += a[k] if k in ("within", "n_within", "across", "n_across") else int(a[k] > 0)
    check("Start sets nested in $\\tau$ within constant-$S$ segments ($G_t(\\tau{+}1)\\ge G_t(\\tau)$)",
          "Prop.~\\ref{prop:sawtooth}", n_gen, tot["Gsaw"])
    finding("Threshold property (random smooth instances)", n_gen, tot["sc"])
    finding("Advantage $G$ monotone in $p$", n_gen, tot["mono"])
    finding("$V(\\text{on},i)-V(\\text{off},j)$ monotone in $p$ (supermodularity)", n_gen, tot["P"])
    finding("Start only if $r(p)\\ge(1-\\gamma)S(\\tau)$", n_gen, tot["be_start"])
    finding("Shut down only if $r(p)\\le 0$", n_gen, tot["be_shut"])
    finding("B3(i): $p^{\\uparrow}_t(\\tau)$ nondecreasing across a cost step", tot["n_across"], tot["across"], unit="pairs")
    macro("NGen", fmt_int(n_gen))
    macro("GenSC", tot["sc"])
    macro("GenMono", fmt_int(tot["mono"]))
    macro("GenP", fmt_int(tot["P"]))
    macro("GenBEStart", fmt_int(tot["be_start"]))
    macro("GenBEShut", fmt_int(tot["be_shut"]))
    macro("NWithin", fmt_int(tot["n_within"]))
    macro("NAcross", fmt_int(tot["n_across"]))
    macro("AcrossDown", fmt_int(tot["across"]))

    fu = fd = 0
    for _ in range(n_flat):
        u = random_unit(rng)
        ch = reg_chain(random_chain(rng, flat=True))
        s = solve(u, ch)
        th = start_thresholds(s)[:, u.DT - 1:]
        sd = shutdown_thresholds(s)
        fu += bool((np.diff(th, axis=0) < 0).any())
        fd += bool((np.diff(sd) < 0).any())
    finding("B3(ii): stationary prices, start thresholds nondecreasing in $t$", n_flat, fu)
    finding("B3(ii): stationary prices, shut-down thresholds nondecreasing in $t$", n_flat, fd)
    macro("NFlat", fmt_int(n_flat))
    macro("FlatUp", fu)
    macro("FlatDown", fd)
    save_json("general_sweep.json", dict(n_iid=n_iid, n_gen=n_gen, n_flat=n_flat, totals=tot, flat_up=fu, flat_down=fd))


def exp_persistence(n=300, seed=5):
    rng = np.random.default_rng(seed)
    rhos = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    out = {}
    for label, kw in (("B1", dict(UT=1, DT=1, const_S=True)), ("general", {})):
        report = label == "B1"   # general-case counts go to results/persistence.json only
        cnt = {"above": [0, 0], "below": [0, 0], "down": [0, 0]}
        for _ in range(n):
            u = random_unit(rng, **kw)
            base = random_chain(rng, method="rouwenhorst", K=21)
            ths, sds = [], []
            for r in rhos:
                ch = reg_chain(make_chain(base.mu, K=21, rho=r, sigma_stat=base.sigma_stat, method="rouwenhorst"))
                s = solve(u, ch)
                ths.append(start_thresholds(s)[:, u.J - 1])
                sds.append(shutdown_thresholds(s))
            ths, sds = np.array(ths), np.array(sds)
            for t in range(ths.shape[1]):
                if ths[0, t] >= 21 or ths[:, t].max() >= 21:
                    continue            # never starts at some rho: threshold undefined
                key = "above" if ths[0, t] > 10 else "below"
                cnt[key][0] += 1
                cnt[key][1] += bool(np.any(np.diff(ths[:, t]) > 0))
                if sds[:, t].min() >= 0:
                    cnt["down"][0] += 1
                    cnt["down"][1] += bool(np.any(np.diff(sds[:, t]) > 0))
        out[label] = cnt
        if not report:
            continue
        finding(f"B3(iii) [{label}]: $p^{{\\uparrow}}$ nonincreasing in $\\rho$ (threshold above mean)",
                cnt["above"][0], cnt["above"][1], unit="$(\\text{inst},t)$")
        finding(f"B3(iii) [{label}]: $p^{{\\uparrow}}$ nonincreasing in $\\rho$ (threshold below mean)",
                cnt["below"][0], cnt["below"][1], unit="$(\\text{inst},t)$")
        finding(f"B3(iii) [{label}]: $p^{{\\downarrow}}$ nonincreasing in $\\rho$",
                cnt["down"][0], cnt["down"][1], unit="$(\\text{inst},t)$")
    for lab, key in (("B", "B1"), ("G", "general")):
        for part in ("above", "below", "down"):
            c = out[key][part]
            macro(f"Pers{lab}{part.capitalize()}N", fmt_int(c[0]))
            macro(f"Pers{lab}{part.capitalize()}V", fmt_int(c[1]))
            macro(f"Pers{lab}{part.capitalize()}Pct", 100.0 * c[1] / max(c[0], 1), "{:.1f}")
    macro("NPers", n)
    save_json("persistence.json", dict(rhos=rhos, n=n, counts=out))


# --------------------------------------------------------------------------------------
# 4. counterexamples: hand-checkable instances + budgeted adversarial search
# --------------------------------------------------------------------------------------
def counterexample_instances():
    """(name, unit, chain, t, counter index, kind)."""
    I2 = np.eye(2)
    ex = []
    # (a) UT = 2, constant S: shut-down decision non-monotone (deterministic, T = 3)
    u = Unit("cex-UT", 0, 2, 10, 9, 2, 1, 8, 8, 8, 1, 1)
    ex.append(("UT", u, MarkovChainModel(np.array([0.0, 35.0, 5.0]), np.array([0.0, 10.0]), I2), 0, 1, "shutdown"))
    # (b) DT = 2, constant S: start decision non-monotone (deterministic, T = 4)
    u = Unit("cex-DT", 1, 1, 10, 3, 1, 2, 9, 9, 9, 1, 1)
    ex.append(("DT", u, MarkovChainModel(np.array([30.0, -15.0, 25.0, 15.0]), np.array([0.0, 5.0]), I2), 0, 2, "start"))
    # (c) UT = DT = 1, tau-dependent S: start decision non-monotone (stochastic, T = 5)
    u = Unit("cex-S", 0, 1, 10, 7, 1, 1, 0, 5, 17, 3, 4)
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    ex.append(("S", u, MarkovChainModel(np.array([-15.0, 5.0, 10.0, 35.0, 10.0]), np.array([0.0, 15.0]), P), 0, 2, "start"))
    return ex


def exp_counterexamples(budget=20000, seeds=(0, 1, 2, 3), workers=8):
    rows = []
    fails = 0
    for name, u, ch, t, c, kind in counterexample_instances():
        reg_chain(ch)
        s = solve(u, ch)
        G = s.advantage()[t, :, c]
        ok = G[0] > 0 and G[1] < 0
        if ch.P[0, 1] == 0:  # deterministic scenarios: confirm by enumeration
            for k in range(2):
                cnt = s.cnt
                u0, tau0 = (1, int(cnt.tau[c])) if cnt.is_on[c] else (0, int(cnt.tau[c]))
                prices = ch.levels()[:, k]
                v_bf, _ = brute_force_deterministic(u, prices, u0, tau0)
                ok &= close(v_bf, s.V[0, k, c])
        # the Q-values derived by hand in the proof of Prop. (counterexamples)
        hand = {"UT": ((32, 24), (53, 54)), "DT": ((8, 5), (14, 15))}.get(name)
        if hand is not None:
            for k in range(2):
                ok &= close(s.Q[t, k, c, 1], hand[k][0]) and close(s.Q[t, k, c, 0], hand[k][1])
        fails += not ok
        rows.append(dict(name=name, unit=u, mu=ch.mu, x=ch.x, P=ch.P, t=t, tau=int(s.cnt.tau[c]), kind=kind,
                         G=G, V=s.V[0, :, c]))
        macro(f"Cex{name}Glo", G[0], "{:.2f}")
        macro(f"Cex{name}Ghi", G[1], "{:.2f}")
    check("Counterexamples (Prop.~\\ref{prop:cex}) reproduce: $G>0$ at low price, $G<0$ at high price",
          "Prop.~\\ref{prop:cex}", 3, fails)
    save_json("counterexamples.json", rows)

    jobs = [(cls, seed, budget) for cls in adversary.CLASSES for seed in seeds]
    with Pool(workers) as pool:
        res = pool.map(_adv_job, jobs)
    summary = {}
    for cls in adversary.CLASSES:
        rs = [r for r in res if r["cls"] == cls]
        summary[cls] = dict(desc=adversary.CLASSES[cls][1], runs=len(rs), found=sum(r["found"] for r in rs),
                            evals=sum(r["evals"] for r in rs), best=max(r["best_margin"] for r in rs),
                            example=next((r["params"] for r in rs if r["found"]), None))
    save_json("adversary.json", summary)
    macro("AdvBudget", fmt_int(budget))
    macro("AdvSeeds", len(seeds))
    return summary


def _adv_job(args):
    cls, seed, budget = args
    r = adversary.search(adversary.CLASSES[cls][0], seed, max_evals=budget)
    r["cls"] = cls
    return r


# --------------------------------------------------------------------------------------
# 5. base case on RTS-GMLC units, naive vs augmented, deterministic-MIP baseline
# --------------------------------------------------------------------------------------
def path_stats(status, u0=1):
    """Mean number of starts per path and fraction of hours off, from simulated statuses."""
    prev = np.hstack([np.full((status.shape[0], 1), u0, np.int8), status[:, :-1]])
    return float(((status == 1) & (prev == 0)).sum(axis=1).mean()), float((status == 0).mean())


def exp_single_units(units, n_paths=5000, seed=6):
    ch = base_chain()
    init = stationary_distribution(ch.P)
    marg = ch.marginals(init)
    rng = np.random.default_rng(seed)
    paths = ch.simulate(n_paths, rng, init)
    rows = []
    sc_fail = mc_fail = mip_fail = 0
    for uid in SINGLE_IDS:
        u = unit_by_id(units, uid)
        t0 = time.perf_counter()
        s = solve(u, ch)
        dt = time.perf_counter() - t0
        a = analyse_general(s)
        sc_fail += a["sc"] > 0
        c0 = s.cnt.on(u.UT)
        v_opt = float(init @ s.V[0, :, c0])
        row = dict(unit=uid, n=u.n_counter, states=ch.K * u.n_counter, solve_ms=1000 * dt, opt=v_opt,
                   opt_mc=None, naive={})
        sim = simulate(u, ch, lambda t: s.policy[t], paths, c0)
        row["opt_mc"] = float(sim["value"].mean())
        row["opt_starts"], row["opt_off"] = path_stats(sim["status"])
        se = sim["value"].std(ddof=1) / math.sqrt(n_paths)
        mc_fail += abs(row["opt_mc"] - v_opt) > 4 * se
        for lab, S in (("hot", u.S_hot), ("warm", u.S_warm), ("cold", u.S_cold)):
            nu = naive_unit(u, S)
            sn = solve(nu, ch)
            ncnt = sn.cnt
            nidx = np.where(s.cnt.is_on, ncnt.on(1), ncnt.off(1))
            fn = (lambda sn=sn, nidx=nidx: (lambda t: sn.policy[t][:, nidx]))()
            Vn, acts, over = evaluate_policy(u, ch, fn)
            simn = simulate(u, ch, fn, paths, c0)
            v = float(init @ Vn[0, :, c0])
            se = simn["value"].std(ddof=1) / math.sqrt(n_paths)
            mc_fail += abs(simn["value"].mean() - v) > 4 * se
            row["naive"][lab] = dict(value=v, mc=float(simn["value"].mean()), gap=v_opt - v,
                                     gap_pct=100 * (v_opt - v) / abs(v_opt),
                                     override_rate=float(simn["overrides"].sum() / (n_paths * ch.T)),
                                     planned_value=float(init @ sn.V[0, :, ncnt.on(1)]))
            row["naive"][lab]["starts"], row["naive"][lab]["off"] = path_stats(simn["status"])
        # deterministic UC on the mean price path (HiGHS MIP), evaluated under stochastic prices
        mean_path = (marg * ch.levels()).sum(axis=1)
        v_mip, sched = solve_deterministic_mip(u, mean_path, 1, u.UT)
        s_det = solve(u, deterministic_chain(mean_path))
        mip_fail += not close(v_mip, s_det.V[0, 0, s_det.cnt.on(u.UT)], rel=1e-6)
        exp_r = (marg * u.profit(ch.levels())).sum(axis=1)
        starts = [t for t in range(ch.T) if sched[t] == 1 and (sched[t - 1] == 0 if t > 0 else False)]
        # a fixed schedule's profit is linear in r(p_t), so its expectation is exact
        v_fixed = float(np.dot(sched, exp_r)) - schedule_startup_cost(u, sched, 1, u.UT)
        row["mean_path"] = dict(value=float(v_fixed), gap=v_opt - v_fixed, gap_pct=100 * (v_opt - v_fixed) / abs(v_opt),
                                mip_obj=float(v_mip), n_starts=len(starts))
        rows.append(row)
    check("Base-case RTS-GMLC units: every decision state has a price threshold", "(numerical)", len(SINGLE_IDS), sc_fail)
    check("Exact policy evaluation agrees with Monte Carlo ($|\\,\\cdot\\,|\\le 4$ s.e.)", "---", 4 * len(SINGLE_IDS), mc_fail,
          unit="policies")
    check("Mean-path HiGHS MIP = deterministic DP on the same path", "---", len(SINGLE_IDS), mip_fail)
    save_json("single_units.json", dict(config=BASE, n_paths=n_paths, rows=rows))
    return rows


def exp_base_figures(units):
    """Arrays for F1-F5 (coal unit 101_STEAM_3 unless stated)."""
    u = unit_by_id(units, "101_STEAM_3")
    ch = base_chain()
    s = solve(u, ch)
    levels = ch.levels()
    th = start_thresholds(s)
    sd = shutdown_thresholds(s)
    out = dict(unit=u, mu=ch.mu, x=ch.x, levels=levels, DT=u.DT, J=u.J, UT=u.UT,
               start_idx=th, shut_idx=sd, break_even=u.break_even())
    # policy map for an off unit: (T, K, J)
    offs = [s.cnt.off(j) for j in range(1, u.J + 1)]
    out["off_policy"] = s.policy[:, :, offs]
    # B1 variant (UT = DT = 1, constant S = S_cold) for Delta curves
    ub = replace(u, name=u.name + "-B1", UT=1, DT=1, S_hot=u.S_cold, S_warm=u.S_cold, T_warm=1, T_cold=1)
    sb = solve(ub, ch)
    on = sb.cnt.on(1)
    out["delta_B1"] = sb.Q[:, :, on, 1] - sb.Q[:, :, on, 0]
    out["S_B1"] = ub.S_cold
    Gs, Gk = decision_advantages(s)
    out["G_start_cold"] = Gs[:, :, -1]
    out["G_stay"] = Gk
    # simulated path (F5): the gas CT, which cycles within the day
    uc = unit_by_id(units, "113_CT_1")
    scm = solve(uc, ch)
    # the first simulated path (in seed order) on which the unit starts at least once
    init = stationary_distribution(ch.P)
    for seed in range(1000):
        path = ch.simulate(1, np.random.default_rng(seed), init)[0]
        c = scm.cnt.off(uc.J)
        status = []
        for t in range(ch.T):
            a = int(scm.policy[t, path[t], c])
            status.append(a)
            c = scm.cnt.nxt[c, a]
        if any(status):
            break
    macro("PathSeed", seed)
    out["sim_path"] = path
    out["sim_price"] = levels[np.arange(ch.T), path]
    out["sim_status"] = status
    out["sim_break_even"] = uc.break_even()
    np.savez(os.path.join(RES, "base_case.npz"), **{k: np.asarray(v) for k, v in out.items() if k != "unit"})
    macro("BaseUnit", u.name.replace("_", "\\_"))
    return out


def exp_sensitivity(units, K=51):
    u = unit_by_id(units, "101_STEAM_3")
    t_star = 14
    res = {"t": t_star, "K": K}
    scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    ch = base_chain(K=K)
    lv = ch.levels()[t_star]
    up, dn = [], []
    for f in scales:
        uf = replace(u, S_hot=u.S_hot * f, S_warm=u.S_warm * f, S_cold=u.S_cold * f)
        s = solve(uf, ch)
        k_up = start_thresholds(s)[t_star, uf.J - 1]
        k_dn = shutdown_thresholds(s)[t_star]
        up.append(float(lv[k_up]) if k_up < K else float("nan"))
        dn.append(float(lv[k_dn]) if k_dn >= 0 else float("nan"))
    res["S_scale"] = dict(scales=scales, p_up=up, p_down=dn)
    rhos = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    up, dn = [], []
    for r in rhos:
        ch = base_chain(K=K, rho=r)
        s = solve(u, ch)
        k_up = start_thresholds(s)[t_star, u.J - 1]
        k_dn = shutdown_thresholds(s)[t_star]
        up.append(float(lv[k_up]) if k_up < K else float("nan"))
        dn.append(float(lv[k_dn]) if k_dn >= 0 else float("nan"))
    res["rho"] = dict(rhos=rhos, p_up=up, p_down=dn)
    res["break_even"] = u.break_even()
    res["mu_t"] = float(ch.mu[t_star])
    save_json("sensitivity.json", res)


# --------------------------------------------------------------------------------------
# 6. fleet: exact DP, naive vs augmented, deterministic baseline, runtime scaling
# --------------------------------------------------------------------------------------
def fleet_chain(units, T=None, K=None):
    f = FLEET
    cap = sum(u.Pmax for u in units)
    mu = daily_profile(T or f["T"], f["base_frac"] * cap, f["amp_frac"] * cap)
    return reg_chain(make_chain(mu, K=K or f["K"], rho=f["rho"], sigma_stat=f["sigma_frac"] * cap,
                                method="rouwenhorst"))


def fleet_init(units):
    """Base-load units (coal, CC) on and free to switch; the others off and cold."""
    states = []
    for u in units:
        base = u.c < 27.0
        states.append((1, u.UT) if base else (0, u.J))
    return states


def exp_fleet_checks(n_bf=5, n_ed=300, seed=7):
    rng = np.random.default_rng(seed)
    from scipy.optimize import linprog
    fails = 0
    for _ in range(n_ed):
        us = [random_unit(rng) for _ in range(4)]
        d = float(rng.uniform(0, 1.2 * sum(u.Pmax for u in us)))
        a = int(rng.integers(0, 16))
        ed = dispatch_cost(us, np.array([d]), 1000.0, 1000.0)[0, a]
        on = [u for g, u in enumerate(us) if (a >> g) & 1]
        n = len(on)
        res = linprog(np.array([u.c for u in on] + [1000.0, 1000.0]), A_eq=np.array([[1.0] * n + [1.0, -1.0]]),
                      b_eq=[d], bounds=[(u.Pmin, u.Pmax) for u in on] + [(0, None), (0, None)], method="highs")
        fails += not close(res.fun + sum(u.F for u in on), ed, rel=1e-7)
    check("Merit-order dispatch cost = LP optimum (HiGHS)", "---", n_ed, fails)
    fails = 0
    for _ in range(n_bf):
        us = [random_unit(rng, max_ut=3, max_dt=3) for _ in range(2)]
        T = 5
        dem = rng.uniform(0, sum(u.Pmax for u in us), T)
        init = [(int(rng.integers(0, 2)), int(rng.integers(1, 4))) for _ in us]
        out = solve_fleet(us, deterministic_chain(dem))
        fails += not close(out["V0"][0, joint_index(us, init)], brute_force_fleet(us, dem, init))
    check("Fleet DP = enumeration of all joint schedules ($N=2$, $T=5$)", "Thm.~\\ref{thm:markov}", n_bf, fails)


def exp_fleet(units, n_paths=5000, seed=8, n_units=5):
    fleet = [unit_by_id(units, i) for i in FLEET_IDS[:n_units]]
    ch = fleet_chain(fleet)
    tb = build_tables(fleet)
    t0 = time.perf_counter()
    out = solve_fleet(fleet, ch, FLEET["voll"], FLEET["spill"], tables=tb)
    t_solve = time.perf_counter() - t0
    init = stationary_distribution(ch.P)
    marg = ch.marginals(init)
    c0 = joint_index(fleet, fleet_init(fleet))
    v_opt = float(init @ out["V0"][:, c0])
    ED = out["ED"]
    rng = np.random.default_rng(seed)
    paths = ch.simulate(n_paths, rng, init)
    pol = out["policy"]
    sim = simulate_fleet(fleet, ch, lambda t: pol[t].astype(np.int64), paths, c0, FLEET["voll"], FLEET["spill"],
                         tables=tb, ED=ED)
    se = sim["cost"].std(ddof=1) / math.sqrt(n_paths)
    mc_fail = abs(sim["cost"].mean() - v_opt) > 4 * se
    res = dict(units=[u.name for u in fleet], cap=sum(u.Pmax for u in fleet), C=tb.C, K=ch.K, T=ch.T, states=tb.C * ch.K, solve_s=t_solve,
               opt=v_opt, opt_mc=float(sim["cost"].mean()), opt_se=float(se), naive={})
    for lab in ("hot", "warm", "cold"):
        nfleet = [naive_unit(u, {"hot": u.S_hot, "warm": u.S_warm, "cold": u.S_cold}[lab]) for u in fleet]
        ntb = build_tables(nfleet)
        nout = solve_fleet(nfleet, ch, FLEET["voll"], FLEET["spill"], tables=ntb)
        mp = naive_to_true_map(fleet, nfleet, tb)
        npol = nout["policy"]
        des = lambda t, npol=npol, mp=mp: npol[t][:, mp].astype(np.int64)
        V = evaluate_fleet_policy(fleet, ch, des, FLEET["voll"], FLEET["spill"], tables=tb, ED=ED)
        v = float(init @ V[:, c0])
        simn = simulate_fleet(fleet, ch, des, paths, c0, FLEET["voll"], FLEET["spill"], tables=tb, ED=ED)
        se = simn["cost"].std(ddof=1) / math.sqrt(n_paths)
        mc_fail += abs(simn["cost"].mean() - v) > 4 * se
        res["naive"][lab] = dict(value=v, mc=float(simn["cost"].mean()), gap=v - v_opt, gap_pct=100 * (v - v_opt) / v_opt,
                                 override_rate=float(simn["overrides"].sum() / (n_paths * ch.T * len(fleet))),
                                 planned=float(init @ nout["V0"][:, naive_to_true_map(fleet, nfleet, tb)[c0]]))
    # deterministic schedule on the mean load path, evaluated exactly under stochastic load
    mean_path = (marg * ch.levels()).sum(axis=1)
    det = solve_fleet(fleet, deterministic_chain(mean_path), FLEET["voll"], FLEET["spill"], tables=tb)
    c = c0
    cost = 0.0
    for t in range(ch.T):
        a = int(det["policy"][t, 0, c])
        cost += tb.sc[c, a] + float(marg[t] @ ED[t, :, a])
        c = int(tb.nxt[c, a])
    res["mean_path"] = dict(value=cost, gap=cost - v_opt, gap_pct=100 * (cost - v_opt) / v_opt,
                            planned=float(det["V0"][0, c0]))
    check("Fleet: exact policy evaluation agrees with Monte Carlo ($\\le 4$ s.e.)", "---", 4, int(mc_fail), unit="policies")
    save_json("fleet.json", res)
    return res


def exp_runtime(units, max_N=6):
    rows = []
    for N in range(1, max_N + 1):
        fl = [unit_by_id(units, i) for i in FLEET_IDS[:N]]
        ch = fleet_chain(fl)
        t0 = time.perf_counter()
        tb = build_tables(fl)
        t1 = time.perf_counter()
        solve_fleet(fl, ch, FLEET["voll"], FLEET["spill"], tables=tb, store_policy=False)
        t2 = time.perf_counter()
        rows.append(dict(N=N, C=tb.C, states=tb.C * ch.K, actions=tb.A, build_s=t1 - t0, solve_s=t2 - t1,
                         status_only=2**N * ch.K))
    save_json("runtime.json", rows)
    return rows


def exp_statespace(units, cats, D=9):
    rows = []
    for u, c in zip(units, cats):
        rows.append(dict(name=u.name, cat=c, UT=u.UT, DT=u.DT, Tcold=u.T_cold_eff, n=u.n_counter))
    by = {}
    for r in rows:
        by.setdefault(r["cat"], []).append(r)
    log_aug = [0.0]
    for r in rows:
        log_aug.append(log_aug[-1] + math.log10(r["n"]))
    save_json("statespace.json", dict(D=D, rows=rows, log10_cum=log_aug))
    return rows, by


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def write(name, text):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def table_units(units, from_rts):
    ids = list(dict.fromkeys(SINGLE_IDS + FLEET_IDS[:5]))
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"Unit & Type & $P^{\min}$ & $P^{\max}$ & $c$ & $F$ & $\UT$ & $\DT$ & $T_{\mathrm{cold}}$ & $S$ (hot/warm/cold) & $n_g$\\",
             r" & & MW & MW & \$/MWh & \$/h & h & h & h & k\$ & \\", r"\midrule"]
    _, cats = load_rts_units()
    names = [u.name for u in units]
    for uid in ids:
        u = unit_by_id(units, uid)
        cat = cats[names.index(uid)] if cats else ""
        S = "/".join(f"{x / 1000:.1f}" if x >= 100 else f"{x / 1000:.2f}" for x in (u.S_hot, u.S_warm, u.S_cold))
        lines.append(f"{uid.replace('_', chr(92) + '_')} & {cat} & {u.Pmin:.0f} & {u.Pmax:.0f} & {u.c:.1f} & {u.F:.0f} & "
                     f"{u.UT} & {u.DT} & {u.T_cold_eff} & {S} & {u.n_counter}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t1_units.tex", "\n".join(lines) + "\n")
    for uid, key in SHORT.items():
        u = unit_by_id(units, uid)
        macro(key + "UT", u.UT)
        macro(key + "DT", u.DT)
        macro(key + "Tcold", u.T_cold_eff)
        macro(key + "Twarm", u.T_warm)
        macro(key + "Pmax", f"{u.Pmax:.0f}")
        macro(key + "N", u.n_counter)


def table_statespace(by, rows, D=9):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Category & Units & $\UT$ & $\DT$ & $T_{\mathrm{cold}}$ & $n_g$\\",
             r"\midrule"]

    def rng_(v):
        return f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"

    for cat in THERMAL_CATEGORIES:
        if cat not in by:
            continue
        rs = by[cat]
        lines.append(f"{cat} & {len(rs)} & {rng_([r['UT'] for r in rs])} & {rng_([r['DT'] for r in rs])} & "
                     f"{rng_([r['Tcold'] for r in rs])} & {rng_([r['n'] for r in rs])}\\\\")
    N = len(rows)
    log_aug = sum(math.log10(r["n"]) for r in rows) + math.log10(D)
    log_stat = N * math.log10(2) + math.log10(D)
    lines += [r"\midrule",
              f"\\multicolumn{{6}}{{p{{0.93\\linewidth}}}}{{All {N} thermal units, $|\\mathcal D|={D}$: augmented state $|\\mathcal D|\\prod_g n_g \\approx 10^{{{log_aug:.1f}}}$; "
              f"status-only $|\\mathcal D|\\,2^{{N}}\\approx 10^{{{log_stat:.1f}}}$.}}\\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("t2_statespace.tex", "\n".join(lines) + "\n")
    macro("NThermal", N)
    macro("NgMin", min(r["n"] for r in rows))
    macro("NgMax", max(r["n"] for r in rows))
    macro("LogAug", log_aug, "{:.1f}")
    macro("LogStat", log_stat, "{:.1f}")
    macro("LogRatio", log_aug - log_stat, "{:.1f}")


def table_verification():
    lines = [r"\begin{tabular}{p{0.52\linewidth}lrrr}", r"\toprule",
             r"Check & Result & Tested & Pass & Fail\\", r"\midrule"]
    lines.append(f"Price chains stochastically monotone (every chain built) & Assumption & {fmt_int(CHAINS['n'])} & "
                 f"{fmt_int(CHAINS['monotone'])} & {fmt_int(CHAINS['n'] - CHAINS['monotone'])}\\\\")
    for c in CHECKS:
        lines.append(f"{c['check']} & {c['ref']} & {fmt_int(c['instances'])} & {fmt_int(c['passes'])} & {fmt_int(c['failures'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t3_verification.tex", "\n".join(lines) + "\n")
    macro("NChains", fmt_int(CHAINS["n"]))
    macro("TotalCheckFailures", sum(c["failures"] for c in CHECKS) + CHAINS["n"] - CHAINS["monotone"])
    save_json("verification.json", dict(checks=CHECKS, findings=FINDINGS, chains=CHAINS))


def table_findings(adv):
    lines = [r"\begin{tabular}{p{0.50\linewidth}rrr}", r"\toprule",
             r"Property (not claimed as a theorem) & Tested & Violations & Unit\\", r"\midrule"]
    for f in FINDINGS:
        lines.append(f"{f['finding']} & {fmt_int(f['instances'])} & {fmt_int(f['violations'])} & {f['unit']}\\\\")
    lines += [r"\midrule",
              r"\multicolumn{4}{l}{\emph{Adversarial search for threshold violations ("
              + f"{MACROS['AdvSeeds']} hill-climbing runs of {MACROS['AdvBudget']} evaluations per class"
              + r")}}\\",
              r"Structural class & Evaluations & Runs with violation & \\", r"\midrule"]
    for cls, r in adv.items():
        lines.append(f"{r['desc']} & {fmt_int(r['evals'])} & {r['found']}/{r['runs']} & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t5_findings.tex", "\n".join(lines) + "\n")
    for cls, key in (("B1", "AdvBone"), ("iid", "AdvIid"), ("UT>1", "AdvUT"), ("DT>1", "AdvDT"), ("S(tau)", "AdvS"),
                     ("general-rouw", "AdvRouw")):
        macro(key + "Found", adv[cls]["found"])
        macro(key + "Evals", fmt_int(adv[cls]["evals"]))
    macro("AdvRuns", adv["B1"]["runs"])


def table_runtime(rt, single, fleet_res, timings):
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Problem & States per period & Actions & Solve time (s)\\", r"\midrule"]
    st = [r["states"] for r in single]
    tm = [r["solve_ms"] / 1000 for r in single]
    lines.append(f"Single units ({len(single)}) & {fmt_int(min(st))}--{fmt_int(max(st))} & 2 & $\\le${max(tm):.3f}\\\\")
    lines.append(r"\midrule")
    for r in rt:
        lines.append(f"Fleet $N={r['N']}$ & {fmt_int(r['states'])} & {r['actions']} & {r['solve_s']:.3g}\\\\")
    lines.append(r"\midrule")
    lines.append(f"Whole pipeline & & & {timings['Total pipeline']:.0f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t4_runtime.tex", "\n".join(lines) + "\n")


def write_macros():
    lines = ["% generated by src/experiments.py -- do not edit"]
    for k, v in sorted(MACROS.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    write("numbers.tex", "\n".join(lines) + "\n")


SHORT = {"101_STEAM_3": "CoalA", "115_STEAM_3": "CoalB", "123_STEAM_3": "CoalC", "107_CC_1": "CC", "113_CT_1": "CT"}


def macros_from_results(single, fleet, rt, n_paths=5000):
    """Headline numbers quoted in the text, and the cost-of-ignoring-history table (T6)."""
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & Optimal & \multicolumn{3}{c}{Naive $(p,u)$ model, $S\equiv$} & Mean-path & Max.\\",
             r"\cmidrule(lr){3-5}",
             r"Instance & value (\$) & hot & warm & cold & MIP & overrides\\", r"\midrule"]

    def row(label, opt, nv, mp):
        ov = max(100 * nv[k]["override_rate"] for k in nv)
        cells = " & ".join(f"{nv[k]['gap_pct']:.1f}\\%" for k in ("hot", "warm", "cold"))
        return f"{label} & {fmt_int(round(opt))} & {cells} & {mp['gap_pct']:.1f}\\% & {ov:.2f}\\%\\\\"

    gaps = {k: [] for k in ("hot", "warm", "cold")}
    ovs = []
    for r in single:
        lines.append(row(r["unit"].replace("_", chr(92) + "_"), r["opt"], r["naive"], r["mean_path"]))
        key = SHORT[r["unit"]]
        for k in ("hot", "warm", "cold"):
            gaps[k].append(r["naive"][k]["gap_pct"])
            ovs.append(100 * r["naive"][k]["override_rate"])
            macro(key + k.capitalize() + "Gap", r["naive"][k]["gap_pct"], "{:.1f}")
        macro(key + "MeanGap", r["mean_path"]["gap_pct"], "{:.1f}")
        macro(key + "StartsOpt", r["opt_starts"], "{:.1f}")
        macro(key + "HotOv", 100 * r["naive"]["hot"]["override_rate"], "{:.1f}")
        macro(key + "StartsHot", r["naive"]["hot"]["starts"], "{:.1f}")
        macro(key + "OffOpt", 100 * r["opt_off"], "{:.0f}")
        macro(key + "OffHot", 100 * r["naive"]["hot"]["off"], "{:.0f}")
    lines.append(r"\midrule")
    lines.append(row(f"Fleet $N={len(fleet['units'])}$ (cost)", fleet["opt"], fleet["naive"], fleet["mean_path"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("t6_naive.tex", "\n".join(lines) + "\n")
    for k in ("hot", "warm", "cold"):
        macro("Single" + k.capitalize() + "GapMin", min(gaps[k]), "{:.1f}")
        macro("Single" + k.capitalize() + "GapMax", max(gaps[k]), "{:.1f}")
        macro("Fleet" + k.capitalize() + "Gap", fleet["naive"][k]["gap_pct"], "{:.1f}")
        macro("Fleet" + k.capitalize() + "Ov", 100 * fleet["naive"][k]["override_rate"], "{:.2f}")
    macro("SingleOvMax", max(ovs), "{:.2f}")
    macro("FleetOpt", fmt_int(round(fleet["opt"])))
    macro("FleetMeanGap", fleet["mean_path"]["gap_pct"], "{:.1f}")
    macro("FleetC", fmt_int(fleet["C"]))
    macro("FleetStates", fmt_int(fleet["states"]))
    macro("FleetSolve", fleet["solve_s"], "{:.1f}")
    macro("FleetK", fleet["K"])
    macro("FleetPaths", fmt_int(n_paths))
    last = rt[-1]
    macro("RtMaxN", last["N"])
    macro("RtMaxStates", fmt_int(last["states"]))
    macro("RtMaxTime", last["solve_s"], "{:.1f}")
    for r in rt:
        macro("RtTime" + "ABCDEFGH"[r["N"] - 1], r["solve_s"], "{:.3g}")
    macro("BaseK", BASE["K"])
    macro("BaseT", BASE["T"])
    macro("BaseRho", BASE["rho"])
    macro("BaseSigma", f"{BASE['sigma']:.0f}")
    macro("MuBase", f"{BASE['mu_base']:.0f}")
    macro("MuAmp", f"{BASE['mu_amp']:.0f}")
    mu = daily_profile(BASE["T"], BASE["mu_base"], BASE["mu_amp"])
    macro("MuMin", f"{mu.min():.0f}")
    macro("MuMax", f"{mu.max():.0f}")
    macro("FleetCap", f"{fleet['cap']:.0f}")
    mu_d = daily_profile(FLEET["T"], FLEET["base_frac"] * fleet["cap"], FLEET["amp_frac"] * fleet["cap"])
    macro("FleetLoadMin", f"{mu_d.min():.0f}")
    macro("FleetLoadMax", f"{mu_d.max():.0f}")
    macro("FleetRho", FLEET["rho"])
    macro("FleetSigmaPct", f"{100 * FLEET['sigma_frac']:.0f}")
    macro("FleetBasePct", f"{100 * FLEET['base_frac']:.0f}")
    macro("FleetAmpPct", f"{100 * FLEET['amp_frac']:.0f}")
    macro("VOLL", f"{FLEET['voll']:,.0f}".replace(",", "{,}"))
    macro("NPaths", fmt_int(n_paths))
    macro("SensT", 14)
    macro("SensK", 51)
