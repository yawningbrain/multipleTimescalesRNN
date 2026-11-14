# Roadmap: making this repository more professional

Status of each item is explicit. Effort is a rough estimate for one person.

## Already done (branch `refactor/library-tests-validation`)

- [x] Importable, documented library (`network`, `analysis`, `plotting`) instead of code embedded in notebooks
- [x] Two real bugs fixed, each with a regression test (K>2 population offsets, half-max off-by-one)
- [x] Heavy dependency (TensorFlow) removed; `pyproject.toml`, `.gitignore`, tracked build artefacts removed
- [x] 16 `pytest` tests; GitHub Actions CI on Python 3.10 / 3.12 / 3.13
- [x] Reproducible experiment script with caching; tidy CSV + JSON data and a data dictionary
- [x] README with hero figure, results table and an explicit list of what was *not* checked

## Decisions (resolved)

- License: MIT (`LICENSE`). Author: Nicolae Istrate (`CITATION.cff`, `pyproject.toml`).
- Still open: whether to also upstream to another repository, and a `master` -> `main` rename.

## Phase 1: repository hygiene (about 0.5 day)

1. ~~`LICENSE`, `CITATION.cff`~~ (done).
2. `CHANGELOG.md` (the README's "What changed" section is the seed) and a `v0.2.0` git tag/release.
3. `CONTRIBUTING.md` (how to run tests and experiments, code style).
4. Pre-commit with `ruff` (lint + format) and `nbstripout`/output policy for notebooks, so diffs stay reviewable. Run the same checks in CI.
5. Branch protection on the default branch: require CI and a review.
6. `__version__` in the package (authors, license and URLs in `pyproject.toml` are done).
7. Run the notebooks in CI (e.g. `nbmake` with a reduced `t_end`) so they cannot silently rot.

## Phase 2: scientific credibility (the highest-value work; 2-4 days)

The README lists what has **not** been checked. In priority order:

1. **Numerical convergence.** Show that timescales are insensitive to `dt`, the integrator tolerance, run length and the discarded transient. Currently the only evidence is single settings.
2. **Finite-size scaling with error bars.** Timescales at `N = 250 ... 4000` over many seeds; the README currently says only that estimates depend on `N`, run length and seed, without isolating each.
3. **Mean-field theory vs simulation.** Implement the DMFT self-consistency for the population model and overlay predicted and simulated ACFs/timescales. This is the paper's central object and would turn the repo from "a simulation" into a verification of the theory.
4. **Check the remaining claims**: transition to chaos vs `s_c(g)`, the fixed-point stability criterion `g^2 sum_a n_a <q_a^-1> <= 1`, and the switching-time scaling `log T ~ s^2 - s log s`. The numeric form of `s_c(g)` I was given came from a model-generated summary and is *unverified* — read it from the paper before coding it.
5. **Uncertainty on timescales**: bootstrap or block-bootstrap confidence intervals instead of seed-to-seed s.d. only; handle censored (never-reaches-half) estimates explicitly rather than dropping them.
6. **Compare the new FFT ACF against the old TensorFlow-Probability ACF** on the same trajectory, so the migration is demonstrably result-preserving (TF is not installed in the current environment).
7. Separate estimation noise from true within-population spread for single-unit timescales (e.g. split-half estimates), since the current histogram cannot distinguish them.

## Phase 3: software quality (2-3 days)

1. Command-line entry point (`rnn-timescales run config.yaml`) and config-driven experiments, replacing constants at the top of `validate_claims.py`.
2. **Provenance**: store git commit, package versions, parameters and seeds alongside every result file.
3. Type hints + `mypy`; test coverage report in CI (the current suite has no coverage measurement; edge cases such as `K=1`, very short runs and the plotting module are untested).
4. Performance: the dense `odeint` run dominates cost (about 15 s for N=1500, 20 k steps). Options: fixed-step RK4 with a float32 matvec, `numba`, or JAX on GPU. Measure first; keep the SciPy path as the reference.
5. Property-based tests (e.g. `hypothesis`) for the ACF and HWHM functions.

## Phase 4: presentation and distribution (1-2 days)

1. Documentation site (MkDocs Material) with the model, API reference (from docstrings), the results page, and a "how to reproduce" page.
2. Binder / Colab badge for the notebook so people can run it without installing anything.
3. Publish to PyPI; archive a release on Zenodo for a citable DOI.
4. Consistent figure style module shared by the notebook, experiments and docs (currently the hero and experiment figures share a palette by copy-paste).

## Suggested order

Decisions above -> Phase 1 -> Phase 2 items 1-3 and 6 -> Phase 3 items 1-2 -> the rest. Phase 2 matters most if the goal is to rely on or publish results from this repo; Phases 1 and 4 matter most if the goal is for others to use it.
