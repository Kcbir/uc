"""Figures F1-F8 (vector PDF) from results/*.json and results/base_case.npz."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# reference categorical palette (light mode), fixed order; neutral inks for text/axes
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE_LIGHT = "#cde2fb"
LOCKED = "#e9e8e4"
OFF = "#fcfcfb"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W = 7.0  # text width in inches (two-column layout)
COL = 3.4  # column width in inches


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def _price(levels, t, k):
    K = levels.shape[1]
    return np.where((k >= 0) & (k < K), levels[t, np.clip(k, 0, K - 1)], np.nan)


def f1_policy_map(b):
    levels, pol, DT, J = b["levels"], b["off_policy"], int(b["DT"]), int(b["J"])
    ts = [4, 17, 30]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.8), sharey=True)
    cmap = ListedColormap([LOCKED, OFF, BLUE_LIGHT])
    for ax, t in zip(axes, ts):
        img = np.where(np.arange(1, J + 1)[None, :] < DT, 0, 1 + pol[t])       # (K, J)
        y = levels[t]
        dy = y[1] - y[0]
        ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                  extent=[0.5, J + 0.5, y[0] - dy / 2, y[-1] + dy / 2])
        taus = np.arange(DT, J + 1)
        k_up = b["start_idx"][t, taus - 1]
        p_up = np.where(k_up < len(y), y[np.clip(k_up, 0, len(y) - 1)] - dy / 2, np.nan)
        ax.step(np.r_[taus - 0.5, J + 0.5], np.r_[p_up, p_up[-1]], where="post", color=BLUE, lw=1.6)
        ax.axhline(b["break_even"], color=INK2, lw=0.8, ls=":")
        ax.set_title(f"$t={t}$ (hour {t % 24}:00)", color=INK)
        ax.set_xlabel(r"time offline $\tau$ (h)")
        ax.grid(False)
    axes[0].set_ylabel(r"price $p$ (\$/MWh)")
    handles = [Patch(color=BLUE_LIGHT, label="start"), Patch(facecolor=OFF, edgecolor=MUTED, lw=0.5, label="stay off"),
               Patch(color=LOCKED, label=r"locked ($\tau<\mathrm{DT}$)"),
               plt.Line2D([], [], color=BLUE, label=r"threshold $p^{\uparrow}_t(\tau)$"),
               plt.Line2D([], [], color=INK2, lw=0.8, ls=":", label="break-even")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08))
    _save(fig, "f1_policy_map.pdf")


def f2_hysteresis(b):
    levels, J = b["levels"], int(b["J"])
    T = levels.shape[0]
    t = np.arange(T)
    p_up = _price(levels, t, b["start_idx"][:, J - 1])
    p_dn = _price(levels, t, b["shut_idx"])
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.fill_between(t, p_dn, p_up, step="mid", color=BLUE_LIGHT, lw=0, alpha=0.8, label="hysteresis band")
    ax.step(t, p_up, where="mid", color=BLUE, label=r"$p^{\uparrow}_t$ (cold start)")
    ax.step(t, p_dn, where="mid", color=ORANGE, label=r"$p^{\downarrow}_t$ (shut down)")
    ax.plot(t, b["mu"], color=INK2, lw=1.0, ls="--", label=r"$\mu_t$")
    ax.axhline(b["break_even"], color=INK2, lw=0.8, ls=":", label="break-even")
    ax.set_xlabel("hour $t$")
    ax.set_ylabel(r"price (\$/MWh)")
    ax.set_xlim(0, T - 1)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), columnspacing=1.0, handlelength=1.6)
    _save(fig, "f2_hysteresis.pdf")


def f3_delta(b):
    levels = b["levels"]
    ts = [4, 17, 30]
    cols = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    D = b["delta_B1"] / 1000
    for t, c in zip(ts, cols):
        ax.plot(levels[t], D[t], color=c, marker="o", ms=2.5, label=f"$t={t}$")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(float(b["S_B1"]) / 1000, color=INK2, lw=0.8, ls="--")
    ax.text(levels[ts[0]][0], float(b["S_B1"]) / 1000 + 0.6, "$S$", color=INK2)
    ax.set_xlabel(r"price $p$ (\$/MWh)")
    ax.set_ylabel(r"$\Delta_t(p)$ (k\$)")
    ax.legend(loc="upper left")
    _save(fig, "f3_delta.pdf")


def f4_sensitivity():
    s = _load("sensitivity.json")
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.8), sharey=True)
    for ax, key, xl in ((axes[0], "S_scale", "startup-cost multiplier"), (axes[1], "rho", r"AR(1) persistence $\rho$")):
        d = s[key]
        x = d["scales"] if key == "S_scale" else d["rhos"]
        ax.plot(x, d["p_up"], color=BLUE, marker="o", ms=3, label=r"$p^{\uparrow}_t$ (cold start)")
        ax.plot(x, d["p_down"], color=ORANGE, marker="s", ms=3, label=r"$p^{\downarrow}_t$")
        ax.axhline(s["break_even"], color=INK2, lw=0.8, ls=":", label="break-even")
        ax.axhline(s["mu_t"], color=INK2, lw=0.8, ls="--", label=r"$\mu_t$")
        ax.set_xlabel(xl)
        if key == "S_scale":
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in x])
    axes[0].set_ylabel(r"threshold price (\$/MWh)")
    axes[0].set_title(f"(a) thresholds vs startup cost ($t={s['t']}$)")
    axes[1].set_title(f"(b) thresholds vs persistence ($t={s['t']}$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    _save(fig, "f4_sensitivity.pdf")


def f5_path(b):
    T = len(b["sim_price"])
    t = np.arange(T)
    st = np.asarray(b["sim_status"])
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    for i in range(T):
        if st[i]:
            ax.axvspan(i - 0.5, i + 0.5, color=BLUE_LIGHT, lw=0)
    ax.plot(t, b["sim_price"], color=INK, lw=1.2, marker="o", ms=2, label="price $p_t$")
    ax.plot(t, b["mu"], color=INK2, lw=0.9, ls="--", label=r"$\mu_t$")
    ax.axhline(float(b["sim_break_even"]), color=INK2, lw=0.8, ls=":", label="CT break-even")
    prev = np.r_[0, st[:-1]]
    starts = np.where((st == 1) & (prev == 0))[0]
    stops = np.where((st == 0) & (prev == 1))[0]
    ax.scatter(starts, b["sim_price"][starts], marker="^", s=40, color=BLUE, zorder=5, label="start", edgecolor="white", lw=0.8)
    ax.scatter(stops, b["sim_price"][stops], marker="v", s=40, color=ORANGE, zorder=5, label="shut down", edgecolor="white", lw=0.8)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel("hour $t$ (shaded: unit on)")
    ax.set_ylabel(r"price (\$/MWh)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32), columnspacing=1.0, handlelength=1.6)
    _save(fig, "f5_path.pdf")


def f6_statespace():
    s = _load("statespace.json")
    D = s["D"]
    cum = np.array(s["log10_cum"][1:]) + np.log10(D)
    N = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.plot(N, cum, color=BLUE, label=r"augmented $|\mathcal{D}|\prod_g n_g$")
    ax.plot(N, N * np.log10(2) + np.log10(D), color=ORANGE, label=r"status only $|\mathcal{D}|\,2^N$")
    ax.set_xlabel("number of RTS-GMLC thermal units $N$")
    ax.set_ylabel(r"$\log_{10}$ (number of states)")
    ax.legend(loc="upper left")
    ax.text(N[-1], cum[-1], f"$10^{{{cum[-1]:.0f}}}$", ha="right", va="bottom", color=INK2)
    _save(fig, "f6_statespace.pdf")


def f7_naive():
    single = _load("single_units.json")["rows"]
    fleet = _load("fleet.json")
    short = {"101_STEAM_3": "Coal 76 MW", "115_STEAM_3": "Coal 155 MW", "123_STEAM_3": "Coal 350 MW",
             "107_CC_1": "CC 355 MW", "113_CT_1": "CT 55 MW"}
    names = [short.get(r["unit"], r["unit"]) for r in single] + [f"Fleet ($N={len(fleet['units'])}$)"]
    data = {k: [r["naive"][k]["gap_pct"] for r in single] + [fleet["naive"][k]["gap_pct"]] for k in ("hot", "warm", "cold")}
    data["mean"] = [r["mean_path"]["gap_pct"] for r in single] + [fleet["mean_path"]["gap_pct"]]
    labels = {"hot": r"naive $(p,u)$, hot $S$", "warm": r"naive $(p,u)$, warm $S$", "cold": r"naive $(p,u)$, cold $S$",
              "mean": "mean-path schedule"}
    cols = {"hot": BLUE, "warm": ORANGE, "cold": AQUA, "mean": YELLOW}
    x = np.arange(len(names))
    w = 0.19
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for i, k in enumerate(("hot", "warm", "cold", "mean")):
        ax.bar(x + (i - 1.5) * w, data[k], width=w - 0.02, color=cols[k], label=labels[k], zorder=3)
    for i, v in enumerate(data["hot"]):
        if v > 1:
            ax.text(x[i] - 1.5 * w, v + 1, f"{v:.0f}", ha="center", fontsize=6.5, color=INK2)
    for i, v in enumerate(data["mean"]):
        if v > 1:
            ax.text(x[i] + 1.5 * w, v + 1, f"{v:.0f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("loss vs. augmented optimum (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=1, loc="upper left", fontsize=6.8)
    _save(fig, "f7_naive.pdf")


def f8_runtime():
    rt = _load("runtime.json")
    N = [r["N"] for r in rt]
    tm = [r["solve_s"] for r in rt]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.semilogy(N, tm, color=BLUE, marker="o", ms=4, label="exact DP solve time")
    for n, t, r in zip(N, tm, rt):
        ax.text(n, t * 1.6, f"{r['states']:,}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("number of units $N$")
    ax.set_ylabel("seconds (log scale)")
    ax.set_xticks(N)
    ax.set_ylim(min(tm) / 3, max(tm) * 8)
    ax.set_title("labels: states per period", fontsize=7.5, color=INK2)
    _save(fig, "f8_runtime.pdf")


def make_all():
    b = dict(np.load(os.path.join(RES, "base_case.npz")))
    f1_policy_map(b)
    f2_hysteresis(b)
    f3_delta(b)
    f4_sensitivity()
    f5_path(b)
    f6_statespace()
    f7_naive()
    f8_runtime()
