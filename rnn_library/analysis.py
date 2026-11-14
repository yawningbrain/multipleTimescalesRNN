"""Autocorrelation and timescale estimation.

The timescale of a unit is defined as in the paper: the half width at half
maximum (HWHM) of its normalised autocorrelation function (ACF)
``C(tau) = <phi[x(t)] phi[x(t + tau)]>_t`` with ``phi = tanh``.
"""

import warnings

import numpy as np

from .network import population_slices


def autocorrelation(y, max_lag=None, normalize=True, axis=-1):
    """Uncentered autocorrelation along ``axis`` computed with the FFT.

    Uses the biased estimator ``C[k] = (1 / T) * sum_t y[t] y[t + k]``, which
    is positive semi-definite and has far less variance at long lags than the
    ``1 / (T - k)`` version. The signal is deliberately *not* mean-subtracted:
    the uncentered ACF is the quantity that appears in the mean-field theory
    (and what the notebook previously computed with ``center=False``).

    Parameters
    ----------
    y : ndarray
        Signal(s). Time runs along ``axis``.
    max_lag : int, optional
        Number of lags to return (``0 .. max_lag - 1``); default: all ``T``.
    normalize : bool
        Divide by the lag-0 value so that ``C[0] == 1``.
    axis : int
        Time axis.
    """
    y = np.moveaxis(np.asarray(y, dtype=float), axis, -1)
    T = y.shape[-1]
    max_lag = T if max_lag is None else min(int(max_lag), T)
    n_fft = 1 << int(np.ceil(np.log2(2 * T)))  # zero-pad: linear, not circular
    spec = np.fft.rfft(y, n=n_fft, axis=-1)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft, axis=-1)[..., :max_lag] / T
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            acf = acf / acf[..., :1]
    return np.moveaxis(acf, -1, axis)


def population_acfs(x, N_values, max_lag=None, chunk=256):
    """Population-averaged ACF of ``tanh(x)``, one row per population.

    Each unit's ACF is normalised to 1 at lag 0 and the ACFs are then averaged
    over the units of a population. Units are processed in chunks so that
    memory stays bounded for large ``N``.

    Returns
    -------
    (K, max_lag) ndarray
    """
    out = []
    for sl in population_slices(N_values):
        n = sl.stop - sl.start
        total = 0.0
        for start in range(sl.start, sl.stop, chunk):
            stop = min(start + chunk, sl.stop)
            total = total + autocorrelation(np.tanh(x[start:stop]), max_lag).sum(axis=0)
        out.append(total / n)
    return np.array(out)


def half_width_at_half_max(lags, acf, fraction=0.5):
    """First lag at which ``acf`` drops below ``fraction * acf[0]``.

    The crossing is located by linear interpolation between samples.
    Returns ``nan`` (and warns) if the ACF never drops that low, which means
    the simulation was too short to resolve the timescale.
    """
    lags = np.asarray(lags, dtype=float)
    acf = np.asarray(acf, dtype=float)
    level = fraction * acf[0]
    below = np.flatnonzero(acf < level)
    if below.size == 0:
        warnings.warn(
            "The ACF did not drop to half of its maximum; "
            "increase the simulation length.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    i = below[0]
    if i == 0:  # already below at the first sample
        return float(lags[0])
    # linear interpolation between samples i-1 (>= level) and i (< level)
    frac = (acf[i - 1] - level) / (acf[i - 1] - acf[i])
    return float(lags[i - 1] + frac * (lags[i] - lags[i - 1]))


def timescales(lags, acfs, fraction=0.5):
    """HWHM timescale of each row of ``acfs`` (see :func:`half_width_at_half_max`)."""
    return np.array([half_width_at_half_max(lags, a, fraction) for a in acfs])
