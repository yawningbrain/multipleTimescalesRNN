"""Plotting helpers (imported lazily so the core library does not need a display)."""

import matplotlib.pyplot as plt
import numpy as np

from .network import population_slices

# Fixed-order categorical colours (blue, orange, aqua, yellow, magenta, green):
# population k always gets slot k.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot_acfs_with_traces(
    t,
    x,
    lags,
    acfs,
    N_values,
    s_values,
    tau,
    n_traces=5,
    trace_window=(200, 900),
    xlim=None,
    rng=None,
):
    """Population ACFs with their HWHM timescales and an inset of example traces.

    Parameters
    ----------
    t, x : time vector and fields from :func:`rnn_library.simulate`.
    lags, acfs : lag vector and ``(K, n_lags)`` ACFs from
        :func:`rnn_library.population_acfs`.
    tau : HWHM timescale of each population (:func:`rnn_library.timescales`).
    """
    rng = np.random.default_rng(rng)
    fig, ax = plt.subplots(figsize=(10, 8))
    for k, sl in enumerate(population_slices(N_values)):
        color = PALETTE[k % len(PALETTE)]
        ax.plot(
            lags,
            acfs[k],
            lw=3,
            color=color,
            label=rf"$s_{k}$={s_values[k]:g}, $N_{k}$={N_values[k]}, "
            rf"$\tau_{k}$={tau[k]:.1f} ms",
        )
        if np.isfinite(tau[k]):
            ax.hlines(0.5, 0, tau[k], color=color, lw=2, alpha=0.6, ls="--")
            ax.plot([tau[k]], [0.5], "o", color=color)
    ax.axhline(0, color="gray", lw=1, alpha=0.7)
    ax.set_xlabel(r"$\tau$ (ms)")
    ax.set_ylabel(r"$C(\tau)/C(0)$")
    ax.set_xlim(*(xlim or (0, min(lags[-1], 4 * np.nanmax(tau) if np.isfinite(tau).any() else lags[-1]))))
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Autocorrelation functions")
    ax.legend(loc="upper right")

    inset = ax.inset_axes([0.45, 0.3, 0.5, 0.3])
    lo, hi = trace_window
    window = (t >= lo) & (t <= hi)
    for k, sl in enumerate(population_slices(N_values)):
        picks = rng.integers(sl.start, sl.stop, n_traces)
        for i in picks:
            inset.plot(t[window], x[i, window], color=PALETTE[k % len(PALETTE)], lw=0.75)
    inset.set_xlabel("time (ms)")
    inset.set_ylabel(r"$x(t)$")
    inset.set_title("Network behavior", fontsize=10)
    return fig, ax
