# Simulation data

Everything here is produced by `python experiments/validate_claims.py` (and
`experiments/make_hero.py` for the `example_*` files). Re-running is deterministic:
every simulation is seeded, and finished runs are cached in `raw.json`.

**Common settings:** `g = 2`, `N = 1000` units, `dt = 0.1` ms, 200 ms transient discarded,
6000 ms recorded, ACF lags up to 2000 ms. Timescale = half width at half maximum (HWHM)
of the population-averaged, per-unit-normalised autocorrelation of `tanh(x)`.
An empty `tau_hwhm_ms` means the ACF never dropped to half within 2000 ms (censored);
none of the stored runs are censored.

| File | Rows | Description |
|---|---|---|
| `raw.json` | 67 sims | Full records: `exp, N, s, seed, tau` (+ `tau_units`). Source of truth for the CSVs. |
| `timescales.csv` | one per (simulation, population) | `experiment, seed, population, n_units, n_total, frac, s, g, t_end_ms, tau_hwhm_ms` |
| `summary.csv` | one per (setting, population) | Mean and s.d. over seeds: `experiment, population, n_units, n_total, s, tau_mean_ms, tau_sd_ms, n_seeds, n_censored` |
| `unit_timescales.csv` | one per unit (`three_pops` only) | `experiment, seed, unit, population, s, tau_hwhm_ms`. Single-unit HWHM; noisy at 6 s of data. |
| `example_acfs.csv` | 8000 lags | Notebook example (`N = 1500`, `s = 1, 3`, 2000 ms, seed 0): `lag_ms, acf_s1, acf_s3` |
| `example_traces.csv` | 6000 samples | One unit per population, 200–800 ms: `t_ms, x_s1_unit0, x_s3_unit750` |
| `run.log` | | Console summary of the last full run |
| `*.png` | | Figures shown in the main README |

## Experiments

| `experiment` | Populations (`N`) | `s` | Seeds | Question |
|---|---|---|---|---|
| `scan_s` | 500 + 500 | `(1, s2)`, `s2 = 1 ... 3.5` | 4 | Timescales vs `s2` at equal sizes |
| `scan_n` | `(1000 - n, n)`, `n = 500, 200, 100, 50` | `(1, 3.5)` | 4 | Effect of the slow fraction |
| `scan_s_rare` | 900 + 100 | `(1, s2)`, `s2 = 1.5 ... 4` | 4 | `log(tau2/tau1)` vs `s2` with rare slow units |
| `three_pops` | 400 + 300 + 300 | `(1, 2.5, 3.5)` | 3 | `card(S) = 3`, single-unit spread |

Note that identical settings in different experiments (e.g. `scan_s` and `scan_n` at
`N = 500 + 500, s = (1, 3.5)`) give identical numbers, because the seeds are shared.
