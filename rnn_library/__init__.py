"""Tools for simulating random networks with a reservoir of timescales."""

from .analysis import (
    autocorrelation,
    half_width_at_half_max,
    population_acfs,
    timescales,
)
from .network import make_coupling_matrix, population_slices, simulate

__all__ = [
    "autocorrelation",
    "half_width_at_half_max",
    "make_coupling_matrix",
    "population_acfs",
    "population_slices",
    "simulate",
    "timescales",
]
