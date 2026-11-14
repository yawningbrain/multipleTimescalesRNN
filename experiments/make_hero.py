"""Build the README hero figure (docs/assets/hero.png).

Runs the notebook's example network once (N=1500, g=2, s=(1, 3), seed 0), saves
the data behind panels (a) and (b) to results/example_*.csv, and combines them
with the parameter scan in results/raw.json for panel (c).

    python experiments/make_hero.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rnn_library import (make_coupling_matrix, population_acfs,  # noqa: E402
                         simulate, timescales)

RESULTS, ASSETS = ROOT / "results", ROOT / "docs" / "assets"
N_VALUES, S_VALUES, G, SEED = [750, 750], [1.0, 3.0], 2.0, 0
DT, T_END, MAX_LAG = 0.1, 2000.0, 8000
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def example_data():
    acf_path, tr_path = RESULTS / "example_acfs.csv", RESULTS / "example_traces.csv"
    if not (acf_path.exists() and tr_path.exists()):
        J = make_coupling_matrix(N_VALUES, S_VALUES, G, rng=SEED)
        t, x = simulate(J, T_END, dt=DT, transient=50.0, rng=SEED)
        acfs = population_acfs(x, N_VALUES, max_lag=MAX_LAG)
        lags = np.arange(MAX_LAG) * DT
        mask = (t >= 200) & (t <= 800)
        i_fast, i_slow = 0, N_VALUES[0]          # one example unit per population
        with open(acf_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["lag_ms", "acf_s1", "acf_s3"])
            w.writerows(zip(lags, acfs[0], acfs[1]))
        with open(tr_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_ms", "x_s1_unit0", "x_s3_unit750"])
            w.writerows(zip(t[mask], x[i_fast, mask], x[i_slow, mask]))
    a = np.loadtxt(acf_path, delimiter=",", skiprows=1)
    tr = np.loadtxt(tr_path, delimiter=",", skiprows=1)
    return a, tr


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, tr = example_data()
    tau = timescales(a[:, 0], a[:, 1:].T)
    rows = json.loads((RESULTS / "raw.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a) traces
    ax = axes[0]
    ax.plot(tr[:, 0], tr[:, 1], color=BLUE, lw=1.4, label=r"$s=1$")
    ax.plot(tr[:, 0], tr[:, 2], color=ORANGE, lw=1.4, label=r"$s=3$")
    ax.set_xlabel("time (ms)"); ax.set_ylabel(r"$x(t)$")
    ax.set_title("a  Two units, two speeds", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    # (b) ACFs
    ax = axes[1]
    for col, color, name, t_ in ((1, BLUE, r"$s=1$", tau[0]), (2, ORANGE, r"$s=3$", tau[1])):
        ax.plot(a[:, 0], a[:, col], color=color, lw=2.4, label=f"{name}:  τ = {t_:.0f} ms")
        ax.plot([t_], [0.5], "o", color=color, ms=7, mec=SURFACE, mew=1.5)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xlim(0, 400); ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r"lag $\tau$ (ms)"); ax.set_ylabel(r"$C(\tau)/C(0)$")
    ax.set_title("b  Autocorrelation timescales", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    # (c) log ratio vs s2, two slow fractions
    ax = axes[2]
    for exp, color, name in (("scan_s", BLUE, r"$n_2=0.5$"), ("scan_s_rare", ORANGE, r"$n_2=0.1$")):
        by = {}
        for r in rows:
            if r["exp"] == exp and r["s"][1] > 1.0 and None not in r["tau"]:
                by.setdefault(r["s"][1], []).append(np.log(r["tau"][1] / r["tau"][0]))
        ks = sorted(by)
        m = np.array([np.mean(by[k]) for k in ks]); sd = np.array([np.std(by[k], ddof=1) for k in ks])
        ax.errorbar(ks, m, yerr=sd, color=color, lw=2.4, marker="o", ms=6, capsize=3, label=name)
    ax.set_xlabel(r"self-coupling $s_2$  ($s_1=1$)"); ax.set_ylabel(r"$\log(\tau_2/\tau_1)$")
    ax.set_title("c  Rare slow units separate more", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    for ax in axes:
        style(ax)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / "hero.png", dpi=160, facecolor=SURFACE)
    print("wrote", ASSETS / "hero.png")


if __name__ == "__main__":
    main()
