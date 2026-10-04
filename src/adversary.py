"""Adversarial (hill-climbing) search for violations of the price-threshold property.

The objective for an instance is the largest single-crossing violation margin over all
decision states and periods, normalised by the value scale:
    max_{t, state} max_{k < k'} min(G_t(k), -G_t(k')) / (1 + max|V|),
which is > 0 exactly when some advantage G goes strictly positive and later strictly
negative in the price index.  The search is deterministic given (mode, seed, budget).
"""
from __future__ import annotations

import numpy as np

from .dp import decision_advantages, solve
from .model import Unit
from .price_chain import make_chain

CLASSES = {
    # name: (constraints, LaTeX description)
    "B1": ("ut1 dt1 constS", r"$\UT=\DT=1$, constant $S$ (Thm.~\ref{thm:b1})"),
    "iid": ("iid", r"i.i.d.\ prices, any $\UT,\DT,S(\tau)$ (Thm.~\ref{thm:iid})"),
    "UT>1": ("dt1 constS", r"$\UT\ge1$, $\DT=1$, constant $S$"),
    "DT>1": ("ut1 constS", r"$\UT=1$, $\DT\ge1$, constant $S$"),
    "S(tau)": ("ut1 dt1", r"$\UT=\DT=1$, $\tau$-dependent $S$"),
    "S(tau)-det": ("ut1 dt1 det", r"$\UT=\DT=1$, $\tau$-dependent $S$, deterministic scenarios ($P=I$)"),
    "constS": ("constS", r"any $\UT,\DT$, constant $S$"),
    "general": ("", r"any $\UT,\DT,S(\tau)$"),
    "general-rouw": ("rouw", r"any $\UT,\DT,S(\tau)$; Rouwenhorst, $\rho\le0.95$, $K\ge11$"),
}


def margin(G: np.ndarray) -> float:
    runmax = np.maximum.accumulate(G, axis=1)
    return float(np.minimum(runmax[:, :-1], -G[:, 1:]).max())


def constrain(z: dict, mode: str) -> dict:
    z = dict(z)
    z["sig"] = max(z["sig"], 0.5)
    for key, lo, hi in (("UT", 1, 10), ("DT", 1, 10), ("Tw", 1, 10), ("Tc", 0, 10), ("K", 2, 12)):
        z[key] = int(min(max(z[key], lo), hi))
    if "constS" in mode:
        z["S2"] = z["S3"] = 0.0
    if "ut1" in mode:
        z["UT"] = 1
    if "dt1" in mode:
        z["DT"] = 1
    if "iid" in mode:
        z["rho"] = 0.0
    if "det" in mode:
        z["det"] = True
    if "rouw" in mode:
        z["meth"] = "rouwenhorst"
        z["rho"] = min(z["rho"], 0.95)
        z["K"] = max(z["K"], 11)
    return z


def build(z: dict):
    u = Unit("adv", z["Pmin"], z["Pmin"] + z["dP"], z["c"], z["F"], z["UT"], z["DT"], z["S1"],
             z["S1"] + z["S2"], z["S1"] + z["S2"] + z["S3"], z["Tw"], z["Tw"] + z["Tc"])
    ch = make_chain(np.asarray(z["mu"]), K=z["K"], rho=z["rho"], sigma_stat=z["sig"], method=z["meth"])
    if z.get("det"):
        ch.P = np.eye(ch.K)   # K parallel deterministic price scenarios, ordered by a common shift
    return u, ch


def score(z: dict) -> float:
    u, ch = build(z)
    s = solve(u, ch, gamma=z["g"])
    Gs, Gk = decision_advantages(s)
    m = max([margin(Gk)] + [margin(Gs[:, :, j]) for j in range(Gs.shape[2])])
    return m / s.scale


def _rand(rng):
    T = int(rng.integers(4, 21))
    return dict(Pmin=float(rng.uniform(0, 100)), dP=float(rng.uniform(1, 200)), c=float(rng.uniform(10, 50)),
                F=float(rng.uniform(0, 2000)), UT=int(rng.integers(1, 7)), DT=int(rng.integers(1, 7)),
                S1=float(rng.uniform(0, 5000)), S2=float(rng.uniform(0, 5000)), S3=float(rng.uniform(0, 5000)),
                Tw=int(rng.integers(1, 6)), Tc=int(rng.integers(0, 6)), mu=rng.uniform(0, 60, T),
                K=int(rng.integers(2, 12)), rho=float(rng.uniform(0, 0.99)), sig=float(rng.uniform(1, 30)),
                meth=str(rng.choice(["tauchen", "rouwenhorst"])), g=float(rng.choice([1.0, rng.uniform(0.5, 1)])))


def _perturb(z, rng):
    z = dict(z)
    k = str(rng.choice(list(z.keys())))
    if k == "mu":
        z["mu"] = z["mu"] + rng.normal(0, 5, len(z["mu"]))
    elif k in ("UT", "DT", "Tw", "Tc", "K"):
        z[k] = z[k] + int(rng.choice([-1, 1]))
    elif k == "meth":
        z[k] = "tauchen" if z[k] == "rouwenhorst" else "rouwenhorst"
    elif k == "rho":
        z[k] = float(np.clip(z[k] + rng.normal(0, 0.1), 0, 0.99))
    elif k == "g":
        z[k] = float(np.clip(z[k] + rng.normal(0, 0.1), 0.3, 1))
    else:
        z[k] = float(max(1.0 if k == "dP" else 0.0, z[k] * np.exp(rng.normal(0, 0.3)) + rng.normal(0, 1)))
    return z


def search(mode: str, seed: int, max_evals: int = 3000, restart_every: int = 300, tol: float = 1e-8):
    rng = np.random.default_rng(seed)
    best, best_z, evals = -np.inf, None, 0
    while evals < max_evals:
        z = constrain(_rand(rng), mode)
        f = score(z)
        evals += 1
        for _ in range(restart_every):
            if evals >= max_evals or f > tol:
                break
            z2 = constrain(_perturb(z, rng), mode)
            f2 = score(z2)
            evals += 1
            if f2 >= f:
                z, f = z2, f2
        if f > best:
            best, best_z = f, z
        if best > tol:
            break
    params = None
    if best_z is not None:
        params = {k: (np.round(v, 4).tolist() if k == "mu" else v) for k, v in best_z.items()}
    return {"mode": mode, "seed": seed, "evals": evals, "best_margin": best, "found": bool(best > tol),
            "params": params}
