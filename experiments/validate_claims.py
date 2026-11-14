"""Numerically check claims of Stern, Istrate & Mazzucato (arXiv:2110.09165).

Experiments
-----------
A. ``scan_s``      With n1 = n2 = 0.5 and s1 = 1: how do tau1, tau2 and the ratio
                   tau2/tau1 depend on s2?
B. ``scan_n``      With s = (1, 3.5): how does the ratio depend on the slow
                   fraction n2?  (The paper's supralinear-in-s statement is for
                   n2/n1 -> 0.)
C. ``three_pops``  card(S) = 3: do we get three ordered timescales, and how do
                   single-unit timescales distribute across populations?
D. ``scan_s_rare`` Rare slow population (n2 = 0.1): how does log(tau2/tau1) depend
                   on s2/s1?  (The paper: supralinear as n2/n1 -> 0.)

Usage::

    python experiments/validate_claims.py            # run everything, save results/
    python experiments/validate_claims.py --plot     # re-plot from results/raw.json

Simulations are independent, so they run in parallel (one process each).
Finished simulations are cached in results/raw.json and are not repeated.
"""
import os

# One BLAS thread per worker: we parallelise over simulations instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rnn_library import (autocorrelation, make_coupling_matrix,  # noqa: E402
                         population_acfs, population_slices, simulate,
                         timescales, half_width_at_half_max)

RESULTS = ROOT / "results"
DT, TRANSIENT, MAX_LAG_MS = 0.1, 200.0, 2000.0
G, T_END = 2.0, 6000.0
N_TOTAL = 1000


def run_job(job):
    """Simulate one network; return population (and optionally unit) timescales."""
    N_values, s_values, seed = job["N"], job["s"], job["seed"]
    J = make_coupling_matrix(N_values, s_values, job.get("g", G), rng=seed)
    t, x = simulate(J, T_END, dt=DT, transient=TRANSIENT, rng=seed)
    n_lags = int(MAX_LAG_MS / DT)
    lags = np.arange(n_lags) * DT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acfs = population_acfs(x, N_values, max_lag=n_lags)
        tau = timescales(lags, acfs)
        out = {**job, "tau": [None if np.isnan(v) else float(v) for v in tau]}
        if job.get("per_unit"):
            units = []
            for start in range(0, x.shape[0], 100):
                a = autocorrelation(np.tanh(x[start:start + 100]), n_lags)
                units += [half_width_at_half_max(lags, r) for r in a]
            out["tau_units"] = [None if np.isnan(v) else float(v) for v in units]
    return out


def build_jobs():
    jobs = []
    for s2 in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):                      # A
        for seed in range(4):
            jobs.append(dict(exp="scan_s", N=[500, 500], s=[1.0, s2], seed=seed))
    for n2 in (0.5, 0.2, 0.1, 0.05):                                # B
        n_slow = int(N_TOTAL * n2)
        for seed in range(4):
            jobs.append(dict(exp="scan_n", N=[N_TOTAL - n_slow, n_slow],
                             s=[1.0, 3.5], seed=seed))
    for s2 in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):                       # D
        for seed in range(4):
            jobs.append(dict(exp="scan_s_rare", N=[900, 100], s=[1.0, s2], seed=seed))
    for seed in range(3):                                           # C
        jobs.append(dict(exp="three_pops", N=[400, 300, 300],
                         s=[1.0, 2.5, 3.5], seed=seed, per_unit=True))
    return jobs


def nanmean_sd(v):
    v = np.array([np.nan if a is None else a for a in v], dtype=float)
    ok = np.isfinite(v)
    return (v[ok].mean(), v[ok].std(ddof=1) if ok.sum() > 1 else 0.0, int((~ok).sum()))


def summarise(rows):
    print(f"\n{'exp':<11}{'N':<18}{'s':<18}{'tau (mean ± sd over seeds)':<44}censored")
    keys = sorted({(r["exp"], tuple(r["N"]), tuple(r["s"])) for r in rows},
                  key=lambda k: (k[0], k[2], k[1]))
    for exp, N, s in keys:
        grp = [r for r in rows if (r["exp"], tuple(r["N"]), tuple(r["s"])) == (exp, N, s)]
        cells, cens = [], 0
        for k in range(len(N)):
            m, sd, c = nanmean_sd([r["tau"][k] for r in grp]); cens += c
            cells.append(f"{m:7.1f}±{sd:<5.1f}")
        print(f"{exp:<11}{str(list(N)):<18}{str(list(s)):<18}{' '.join(cells):<44}{cens}")


# ------------------------------------------------------------------ plotting
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # fixed categorical order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def per_seed(rows, exp, key_fn, pop):
    """{key: array of per-seed tau of population ``pop``} (nan for censored)."""
    out = {}
    for r in rows:
        if r["exp"] == exp:
            v = r["tau"][pop]
            out.setdefault(key_fn(r), []).append(np.nan if v is None else v)
    return {k: np.array(v) for k, v in sorted(out.items())}


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- A: tau vs s2 ------------------------------------------------------
    t1 = per_seed(rows, "scan_s", lambda r: r["s"][1], 0)
    t2 = per_seed(rows, "scan_s", lambda r: r["s"][1], 1)
    s2 = np.array(list(t1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for tau, color, name in ((t1, BLUE, r"$s_1=1$ (fast population)"),
                             (t2, ORANGE, r"$s_2$ population")):
        m = np.array([np.nanmean(v) for v in tau.values()])
        sd = np.array([np.nanstd(v, ddof=1) for v in tau.values()])
        a.errorbar(s2, m, yerr=sd, color=color, lw=2, marker="o", ms=6, capsize=3, label=name)
    a.set_yscale("log"); a.set_xlabel(r"self-coupling $s_2$"); a.set_ylabel("timescale (HWHM, ms)")
    a.set_title(r"Both timescales grow with $s_2$ ($n_1=n_2=0.5$)", loc="left", fontsize=11)
    a.legend(frameon=False)
    ratio = np.array([np.nanmean(t2[k] / t1[k]) for k in t1])
    rsd = np.array([np.nanstd(t2[k] / t1[k], ddof=1) for k in t1])
    b.errorbar(s2, ratio, yerr=rsd, color=INK, lw=2, marker="o", ms=6, capsize=3)
    b.set_xlabel(r"self-coupling $s_2$"); b.set_ylabel(r"$\tau_2/\tau_1$")
    b.set_title(r"...but the ratio stays below 2 over this range", loc="left", fontsize=11)
    for ax in (a, b): style(ax)
    fig.tight_layout(); fig.savefig(RESULTS / "scan_s.png", dpi=150); plt.close(fig)

    # --- B: ratio vs slow fraction ----------------------------------------
    frac = lambda r: r["N"][1] / sum(r["N"])
    u1 = per_seed(rows, "scan_n", frac, 0); u2 = per_seed(rows, "scan_n", frac, 1)
    n2 = np.array(list(u1))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for tau, color, name in ((u1, BLUE, r"$s_1=1$"), (u2, ORANGE, r"$s_2=3.5$")):
        m = np.array([np.nanmean(v) for v in tau.values()])
        sd = np.array([np.nanstd(v, ddof=1) for v in tau.values()])
        a.errorbar(n2, m, yerr=sd, color=color, lw=2, marker="o", ms=6, capsize=3, label=name)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"slow fraction $n_2$"); a.set_ylabel("timescale (HWHM, ms)")
    a.set_title("Fast population is dragged along less\nwhen slow units are rare", loc="left", fontsize=11)
    a.legend(frameon=False)
    ratio = np.array([np.nanmean(u2[k] / u1[k]) for k in u1])
    rsd = np.array([np.nanstd(u2[k] / u1[k], ddof=1) for k in u1])
    b.errorbar(n2, ratio, yerr=rsd, color=INK, lw=2, marker="o", ms=6, capsize=3)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel(r"slow fraction $n_2$"); b.set_ylabel(r"$\tau_2/\tau_1$")
    b.set_title("Separation grows, then plateaus (~3) for $n_2 \\lesssim 0.2$", loc="left", fontsize=11)
    for ax in (a, b): style(ax)
    fig.tight_layout(); fig.savefig(RESULTS / "scan_n.png", dpi=150); plt.close(fig)

    # --- D: rare slow population, log ratio vs s2 -------------------------
    r1 = per_seed(rows, "scan_s_rare", lambda r: r["s"][1], 0)
    r2 = per_seed(rows, "scan_s_rare", lambda r: r["s"][1], 1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=SURFACE)
    for (a1, a2), color, name in (((t1, t2), BLUE, r"$n_2=0.5$"), ((r1, r2), ORANGE, r"$n_2=0.1$")):
        ks = np.array([k for k in a1 if k > 1.0])
        lr = np.array([np.log(np.nanmean(a2[k] / a1[k])) for k in ks])
        sd = np.array([np.nanstd(np.log(a2[k] / a1[k]), ddof=1) for k in ks])
        ax.errorbar(ks, lr, yerr=sd, color=color, lw=2, marker="o", ms=6, capsize=3, label=name)
    ax.set_xlabel(r"self-coupling ratio $s_2/s_1$  ($s_1=1$)")
    ax.set_ylabel(r"$\log(\tau_2/\tau_1)$")
    ax.set_title("Timescale separation vs self-coupling ratio", loc="left", fontsize=11)
    ax.legend(frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(RESULTS / "scan_s_rare.png", dpi=150); plt.close(fig)

    # --- C: three populations, per-unit timescales -------------------------
    c_rows = [r for r in rows if r["exp"] == "three_pops"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    colors = (BLUE, ORANGE, AQUA)
    allv = np.array([v for r in c_rows for v in r["tau_units"] if v is not None])
    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 60)
    for k, sl in enumerate(population_slices(c_rows[0]["N"])):
        vals = np.array([v for r in c_rows for v in r["tau_units"][sl] if v is not None])
        ax.hist(vals, bins=bins, color=colors[k], alpha=0.85, lw=0, density=True,
                label=f"$s={c_rows[0]['s'][k]}$ (median {np.median(vals):.0f} ms)")
    ax.set_xscale("log"); ax.set_xlabel("single-unit timescale (HWHM, ms)"); ax.set_ylabel("density")
    ax.set_title("card(S)=3: single-unit timescales are ordered by population, but overlap", loc="left", fontsize=11)
    ax.legend(frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(RESULTS / "three_pops.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--plot", action="store_true", help="re-plot from results/raw.json only")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    raw = RESULTS / "raw.json"
    if args.plot:
        rows = json.loads(raw.read_text())
    else:
        rows = json.loads(raw.read_text()) if raw.exists() else []
        key = lambda j: (j["exp"], tuple(j["N"]), tuple(j["s"]), j["seed"])
        done = {key(r) for r in rows}
        jobs = [j for j in build_jobs() if key(j) not in done]
        print(f"{len(jobs)} new simulations ({len(rows)} cached) on {args.workers} workers "
              f"(T_END={T_END:g} ms, g={G})", flush=True)
        if jobs:
            with ProcessPoolExecutor(args.workers) as ex:
                rows += list(ex.map(run_job, jobs))
            raw.write_text(json.dumps(rows))
    summarise(rows)
    plot(rows)
    print(f"\nfigures written to {RESULTS}/")


if __name__ == "__main__":
    main()
