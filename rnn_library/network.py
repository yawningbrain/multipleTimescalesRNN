"""Random recurrent network with heterogeneous self-couplings.

Model (Stern, Istrate & Mazzucato, arXiv:2110.09165, Eq. 1)::

    dx_i/dt = -x_i + s_i * tanh(x_i) + sum_{j != i} J_ij * tanh(x_j)

with ``J_ij ~ Normal(0, g**2 / N)``. Units are grouped into ``K`` populations;
population ``k`` has ``N_values[k]`` units that all share the self-coupling
``s_values[k]``. Time is dimensionless (the paper reads one unit as 1 ms).
"""

import numpy as np
from scipy.integrate import odeint


def make_coupling_matrix(N_values, s_values, g, rng=None):
    """Build the coupling matrix ``J`` of a network with ``K`` populations.

    Off-diagonal entries are i.i.d. ``Normal(0, g**2 / N)``; the diagonal of
    unit ``i`` is replaced by its self-coupling ``s_i``, so that
    ``dx/dt = -x + J @ tanh(x)`` is the full model.

    Parameters
    ----------
    N_values : sequence of int
        Number of units in each population.
    s_values : sequence of float
        Self-coupling of each population (same length as ``N_values``).
    g : float
        Network gain.
    rng : int, numpy.random.Generator or None
        Seed or generator, for reproducibility.

    Returns
    -------
    J : (N, N) ndarray
    """
    N_values = np.asarray(N_values, dtype=int)
    s_values = np.asarray(s_values, dtype=float)
    if N_values.shape != s_values.shape or N_values.ndim != 1:
        raise ValueError("N_values and s_values must be 1-D and of equal length")
    if np.any(N_values <= 0):
        raise ValueError("every population needs at least one unit")
    rng = np.random.default_rng(rng)
    N = int(N_values.sum())
    J = rng.normal(0.0, g / np.sqrt(N), size=(N, N))
    J[np.diag_indices(N)] = np.repeat(s_values, N_values)
    return J


def population_slices(N_values):
    """Return one ``slice`` per population, indexing units of the network."""
    edges = np.concatenate([[0], np.cumsum(np.asarray(N_values, dtype=int))])
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def simulate(J, t_end, dt=0.1, transient=0.0, x0=None, rng=None):
    """Integrate ``dx/dt = -x + J @ tanh(x)``.

    Parameters
    ----------
    J : (N, N) ndarray
        Coupling matrix, e.g. from :func:`make_coupling_matrix`.
    t_end : float
        Length of the recorded trajectory.
    dt : float
        Sampling interval of the recorded trajectory.
    transient : float
        Extra time integrated *before* recording starts, to forget the
        initial condition. Not included in the returned data.
    x0 : (N,) ndarray, optional
        Initial condition; default is ``Uniform(-2, 2)``.
    rng : int, numpy.random.Generator or None
        Seed or generator used for ``x0``.

    Returns
    -------
    t : (T,) ndarray
        Times ``0, dt, 2 dt, ...`` of the recorded part (``T = t_end / dt``).
    x : (N, T) ndarray
        Recorded local fields.
    """
    N = J.shape[0]
    if x0 is None:
        x0 = np.random.default_rng(rng).uniform(-2.0, 2.0, size=N)

    def rhs(x, _t):
        return -x + J @ np.tanh(x)

    n_transient = int(round(transient / dt))
    n_record = int(round(t_end / dt))
    timeline = np.arange(n_transient + n_record) * dt
    x = odeint(rhs, x0, timeline)
    t = timeline[n_transient:] - timeline[n_transient]
    return t, np.ascontiguousarray(x[n_transient:].T)
