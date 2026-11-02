# BookingForge

**Hotel Booking Intelligence — Applied Machine Learning and Software Engineering Research**

A reproducible exploration of historical hotel reservation outcomes, original NumPy
algorithms, time-based evaluation, data audits and decision simulation.
**This is not an operational hotel cancellation-risk or overbooking system.**
Any future inventory/revenue analysis must use explicitly synthetic assumptions.

## Current research status

- Public historical dataset: **119,390 rows, 32 columns**, arrivals July
  2015 to August 2017. Individual raw booking rows are not stored in Git.
- **58/58 automated tests passed** locally on **2026-11-02** at commit
  `dd1ac51`. The next four tests for an alternate arrival-cohort protocol
  require local verification.
- Original algorithms: NumPy **exact kNN** and **one-hidden-layer MLP** with
  explicit analytic backpropagation, Adam and finite-difference gradient tests.
- Comparison models: scikit-learn logistic regression and constant
  training-prior probability.
- The **2017 outcome fields remain sealed** in the research benchmark.
- All benchmarks are exploratory. Source dates are not immutable reservation
  creation timestamps, nor are model features verified booking-time snapshots.

## What experiments really showed

Initial, selected validation: logistic ROC AUC **0.8315** on **11,991**
matured labels. This result is *superseded* as a general performance claim:
outcome maturity selected fast-resolving reservations.

Full **19,339-booking retrospective validation** with historical training:
logistic AUC **0.6776**, Brier **0.2460**; custom NumPy kNN AUC **0.6295**,
Brier **0.3312**; custom NumPy MLP AUC **0.6715**, Brier **0.5950**.
A constant prior achieved Brier **0.2370**, better than the three models.

**Cohort-structure investigation** on an earlier internal chronological split:
training had **39,507** known outcomes, inner validation **6,860**.
The constant prior Brier was **0.243547**; all four logistic/MLP
with/without arrival-year configurations were worse (**0.5054–0.6016**).

Why this is important: among bookings whose outcomes were known
before March or August 2016, **every booking arriving in 2017
was canceled or a no-show** (228 historical train, 107 inner
validation). That is a **structural artifact of the label-availability
cutoff**, not evidence that arrival year predicts cancellation at booking
creation. Earlier year-sensitivity experiments also exposed severe
model overconfidence. **We did not hide these failures or tune the
models to make their scores look better.**

Read [all verified experiments, negative results and limits](docs/EXPERIMENT_RESULTS.md).

## Next: separate retrospective arrival-cohort algorithm benchmark

A new, **separately scoped task** avoids the specific future-arrival
maturity artifact without pretending that original booking-time data exist:

- **Fit on:** arrivals July 2015–August 2016 whose eventual status was
  known by September 30, 2016.
- **Gap:** September 2016 arrivals.
- **Score on:** October–December 2016 arrivals, using their final
  outcomes *only retrospectively*.
- **Exclude:** every 2017 arrival before inspecting its final status.
- **Compare:** prior, logistic regression and original NumPy MLP, with
  and without raw arrival year. No row subsampling.

This tests an **arrival-date-separated retrospective classification
algorithm**, **not a valid model for predicting cancellation at original
reservation creation**. It still has historical feature-timing
uncertainty, limited sample coverage and already-used exploratory
information. Do not compare metrics to the incompatible creation-time
experiments as if they measured the same task.

### Run on Windows / PowerShell

Python 3.12 and pinned dependencies in `requirements-ml.txt`.
Place the public [TidyTuesday hotel bookings CSV](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv)
at `data/raw/hotels.csv`; expected SHA256:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.arrival_benchmark --csv data/raw/hotels.csv --output reports/arrival_benchmark.json --mlp-epochs 30
```

**New benchmark metrics are unknown until the local run completes.**
The experiment writes aggregate metrics and counts only to
`reports/arrival_benchmark.json`. Earlier reports are preserved.

## Research and engineering standards

1. Separate historical observation time, available labels, and feature timing.
2. Never fit or select models from final-test outcomes.
3. Never use `is_canceled`, `reservation_status` or
   `reservation_status_date` as predictors.
4. Train-only fitting for preprocessing; explicit reproducible
   data windows, code commits, seeds, metrics and baselines.
5. Record failure modes and selection bias rather than promising deployment.
6. Do not upload real guest data, private hotel company information or
   invented simulated economic results.

## Roadmap

1. Dataset contract, chronological audits and carefully qualified research
   benchmarks (**implemented; known threats to validity**).
2. Custom kNN and NumPy MLP with analytic gradients (**implemented,
   validated on synthetic tests and measured on actual historical cohorts**).
3. Reassess outcome/observation-time contract before adding PyTorch training
   and GPU profiling.
4. Research seasonal demand forecasting with explicit source limitations,
   and separately model simulated stochastic hotel capacity decisions.
5. Create FastAPI/API tests, CI, Docker, an explainable dashboard, and
   reproducible experiment cards.

## Documentation

- [Dataset contract](docs/DATA_CONTRACT.md)
- [Booking-time limitations](docs/BOOKING_TIME_CONTRACT.md)
- [Original retrospective cohort protocol](docs/BASELINE_PROTOCOL.md)
- [Measured experiments, limitations and next protocol](docs/EXPERIMENT_RESULTS.md)
