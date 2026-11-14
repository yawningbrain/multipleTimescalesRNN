import warnings

import numpy as np
import pytest

from rnn_library import (
    autocorrelation,
    half_width_at_half_max,
    make_coupling_matrix,
    population_acfs,
    population_slices,
    simulate,
    timescales,
)
from rnn_library.rnn_functions import FWHM_funct


# ---------------------------------------------------------------- network
def test_diagonal_holds_self_couplings_for_three_populations():
    # Regression: the old notebook set ``temp_pos_var = N_values[idx]`` instead of
    # accumulating, which silently misassigned units when there were K > 2 groups.
    N_values, s_values = [3, 5, 4], [1.0, 2.0, 3.0]
    J = make_coupling_matrix(N_values, s_values, g=1.0, rng=0)
    expected = np.repeat(s_values, N_values)
    np.testing.assert_array_equal(np.diag(J), expected)


def test_population_slices_cover_network_without_gaps():
    sl = population_slices([3, 5, 4])
    assert [(s.start, s.stop) for s in sl] == [(0, 3), (3, 8), (8, 12)]


def test_coupling_statistics():
    N, g = 2000, 2.0
    J = make_coupling_matrix([N], [1.0], g, rng=1)
    off = J[~np.eye(N, dtype=bool)]
    assert abs(off.mean()) < 5e-3
    assert off.std() == pytest.approx(g / np.sqrt(N), rel=0.01)


def test_seed_reproducibility():
    a = make_coupling_matrix([10, 10], [1, 2], 1.5, rng=42)
    b = make_coupling_matrix([10, 10], [1, 2], 1.5, rng=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("N_values,s_values", [([5, 5], [1.0]), ([0, 5], [1.0, 2.0])])
def test_invalid_populations_rejected(N_values, s_values):
    with pytest.raises(ValueError):
        make_coupling_matrix(N_values, s_values, 1.0)


def test_simulate_shapes_and_decay_for_zero_coupling():
    # J = 0 -> dx/dt = -x, exactly solvable: x(t) = x0 * exp(-t).
    J = np.zeros((4, 4))
    x0 = np.array([1.0, -2.0, 0.5, 0.0])
    t, x = simulate(J, t_end=5.0, dt=0.1, x0=x0)
    assert t.shape == (50,) and x.shape == (4, 50)
    assert t[0] == 0.0
    np.testing.assert_allclose(x, x0[:, None] * np.exp(-t)[None, :], atol=1e-5)


def test_simulate_transient_is_discarded():
    J = np.zeros((1, 1))
    t, x = simulate(J, t_end=1.0, dt=0.1, transient=2.0, x0=np.array([1.0]))
    assert t[0] == 0.0
    assert x[0, 0] == pytest.approx(np.exp(-2.0), rel=1e-4)


# --------------------------------------------------------------- analysis
def test_autocorrelation_matches_direct_computation():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    acf = autocorrelation(y, normalize=False)
    direct = np.array([np.dot(y[: len(y) - k], y[k:]) for k in range(len(y))]) / len(y)
    np.testing.assert_allclose(acf, direct, atol=1e-10)


def test_autocorrelation_normalised_and_max_lag():
    y = np.random.default_rng(1).standard_normal((3, 100))
    acf = autocorrelation(y, max_lag=20)
    assert acf.shape == (3, 20)
    np.testing.assert_allclose(acf[:, 0], 1.0)


def test_autocorrelation_axis():
    y = np.random.default_rng(2).standard_normal((100, 3))
    np.testing.assert_allclose(
        autocorrelation(y, axis=0), autocorrelation(y.T, axis=-1).T
    )


def test_hwhm_of_exponential():
    tau = 7.0
    lags = np.arange(0, 100, 0.05)
    # exp(-t/tau) = 1/2  at  t = tau * ln 2
    hwhm = half_width_at_half_max(lags, np.exp(-lags / tau))
    assert hwhm == pytest.approx(tau * np.log(2), abs=1e-3)


def test_hwhm_sees_the_last_sample():
    # Regression: the old implementation compared signs[0:-2] with signs[1:-1],
    # so a crossing between the last two samples was missed.
    lags = np.array([0.0, 1.0, 2.0])
    acf = np.array([1.0, 0.9, 0.1])
    assert half_width_at_half_max(lags, acf) == pytest.approx(1.0 + 0.4 / 0.8)


def test_hwhm_nan_with_warning_when_never_reached():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(half_width_at_half_max(np.arange(5.0), np.ones(5)))


def test_population_acfs_and_timescales_recover_known_ar1_processes():
    # Two populations of AR(1) processes (discretised OU) with different
    # correlation times; tanh is replaced by working in the small-x regime.
    rng = np.random.default_rng(3)
    dt, T = 0.1, 40000
    taus = [2.0, 10.0]
    N_values = [6, 6]
    rows = []
    for tau, n in zip(taus, N_values):
        a = np.exp(-dt / tau)
        noise = rng.standard_normal((n, T)) * 0.05 * np.sqrt(1 - a * a)
        y = np.zeros((n, T))
        for k in range(1, T):
            y[:, k] = a * y[:, k - 1] + noise[:, k]
        rows.append(y)
    x = np.vstack(rows)  # tanh(x) ~ x for |x| << 1
    acfs = population_acfs(x, N_values, max_lag=600, chunk=4)
    assert acfs.shape == (2, 600)
    ts = timescales(np.arange(600) * dt, acfs)
    np.testing.assert_allclose(ts, np.array(taus) * np.log(2), rtol=0.15)
    assert ts[1] > ts[0]


# ----------------------------------------------------------- legacy shims
def test_legacy_FWHM_funct_still_works_and_does_not_raise_on_failure():
    lags = np.arange(0.0, 10.0, 0.1)
    assert FWHM_funct(lags, np.exp(-lags)) == pytest.approx(np.log(2), abs=1e-2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(FWHM_funct(lags, np.ones_like(lags)))
