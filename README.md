# BookingForge

**Hotel Booking Intelligence & Optimization Research Engine**

An applied AI/ML + software engineering portfolio project exploring hotel booking cancellations, demand forecasting, and constrained decision algorithms. All capacity/cost decisions will be explicitly simulated, not used as real hotel operational policy.

## Current status

**Phase 1: Real dataset acquired, validated, and first provisional ML baseline trained locally.**

- Hotel Booking Demand dataset (António, Almeida & Nunes, 2019); 119,390 records, 32 columns, July 2015–August 2017 arrivals.
- 21 unit tests passed on Windows/Python 3.12 in the first ML run. More tests added since; verify them locally.
- Data quality, time consistency, outcome maturity and booking-creation cohort audit modules available.
- Prior-probability baseline and scikit-learn logistic regression trained on selected *outcome-mature* booking cohorts.
- **Final test cohort has NOT been evaluated.** No deployment or revenue claims.

### First preliminary validation benchmark (local run, 2026-11-02)

| Metric | Prior probability | Logistic regression |
|---|---:|---:|
| ROC AUC | 0.5000 | 0.8315 |
| Average precision | 0.4574 | 0.8381 |
| Brier score (lower better) | 0.2506 | 0.1852 |

Training: **63,334** records; validation: **11,991** records. This benchmark is **not an unbiased estimate** of real-world performance. Labels were filtered to those known by each cohort's historical cutoff; doing so disproportionately selects fast-resolving bookings. Additionally, historical feature snapshots are missing and data were selected by arrival date. Read [baseline methodological limitations](docs/BASELINE_PROTOCOL.md) before interpreting metrics.

## Research principles

1. Define prediction timing before selecting predictors.
2. Never use `is_canceled`, `reservation_status`, or `reservation_status_date` as model predictors.
3. Treat the conservative booking-time feature allowlist as **provisional**; no historical value snapshots exist.
4. Audit label availability and right-truncation before asserting generalization.
5. Keep raw booking rows and all employer/customer data out of Git.
6. Clearly mark simulated hotel capacity, financial loss and alternative decisions as hypothetical.
7. Report negative results, uncertainty and all evaluation limitations.

## Reproduce the current baseline

Python 3.11+ and the publicly available CSV are required. Raw dataset and model artifacts are not stored in this repo.

**Dataset location:** `data/raw/hotels.csv`, sourced from the publicly available [TidyTuesday hotel bookings copy](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/readme.md). Verify source terms before redistribution. Expected dataset SHA256 from local initial audit:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

In PowerShell (repository root):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.data_audit --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.temporal_audit --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.cohort --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.selection_diagnostics --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.train_baseline --csv data/raw/hotels.csv
```

Report JSON files are saved under `reports/`. The test labels are not used by the current training implementation.

## Project roadmap

1. **Foundation and baseline:** dataset contract, audits, unit tests, conservative first ML benchmark (**in progress; diagnostic caveats remain**).
2. **Custom algorithms and ML:** own kNN and NumPy MLP implementations, benchmark against scikit-learn, with leakage-aware evaluation.
3. **Forecasting and deep learning:** seasonal baseline, PyTorch temporal or sequence models where data support them.
4. **Decision algorithms:** Poisson-binomial dynamic programming, Monte Carlo simulations, heuristics and room assignment with synthetic constraints.
5. **Application engineering:** FastAPI backend, dashboard, integration tests, CI, Docker and experiment documentation.

### Documentation

- [Dataset contract](docs/DATA_CONTRACT.md)
- [Booking observation-time contract](docs/BOOKING_TIME_CONTRACT.md)
- [Preliminary baseline protocol and limitations](docs/BASELINE_PROTOCOL.md)
