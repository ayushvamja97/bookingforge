# BookingForge

**Hotel Booking Intelligence — Applied Machine Learning and Software Engineering Research**

A reproducible exploration of historical hotel reservation outcomes, original NumPy
algorithms, time-based evaluation, data audits and decision simulation.
**This is not an operational hotel cancellation-risk or overbooking system.**
Any future inventory/revenue analysis must use explicitly synthetic assumptions.

## Current research status

- Public historical dataset: **119,390 rows, 32 columns**, arrivals July
  2015 to August 2017. Individual raw booking rows are not stored in Git.
- **62/62 automated tests passed** locally on **2026-11-02** at commit
  `2b2e853`; five new reliability tests are committed but not yet verified on Windows.
- Original algorithms: NumPy **exact kNN** and **one-hidden-layer MLP** with
  explicit analytic backpropagation, Adam and finite-difference gradient tests.
- Comparison models: scikit-learn logistic regression and constant
  training-prior probability.
- New experiment pipelines exclude 2017 outcome fields from fitting and scoring. **Important qualification:** earlier repository audit code did read 2017 cohort outcomes in aggregate, so 2017 is **not a completely untouched blind holdout**. No independent final-test performance claim is currently justified.
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

## New result: separate retrospective arrival-cohort algorithm benchmark

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

**Verified local results:** 58,792 training arrivals and 14,517
retrospective validation arrivals, with no random sampling.

| Model | ROC AUC ↑ | AP ↑ | Brier ↓ |
|---|---:|---:|---:|
| Training prior | 0.5000 | 0.3822 | 0.2368 |
| Logistic with year | 0.7182 | 0.6154 | 0.2175 |
| Logistic without year | 0.7172 | 0.6140 | 0.2064 |
| Original NumPy MLP with year | 0.7207 | 0.6383 | **0.2032** |
| Original NumPy MLP without year | **0.7300** | **0.6456** | 0.2072 |

The neural network without year ranks outcomes best, but the neural
network with year has the best probability Brier score. Neither is a
confirmed winner without uncertainty estimates. Both beat the constant
training-prior benchmark by Brier. These results do **not** validate
prediction at original booking creation because historical predictor
snapshots are unavailable; cohorts are defined by arrival and were
created after exploratory investigation of prior validation results.

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

The benchmark writes aggregate metrics and counts to
`reports/arrival_benchmark.json`. The first measured run is recorded
in [the experiment ledger](docs/EXPERIMENT_RESULTS.md). Earlier reports
are preserved. The next planned step is *descriptive* probability
calibration and subpopulation/uncertainty analysis, with no hyperparameter
or feature selection on the same validation set.

## Next verification: reliability and subgroups

A separate **fixed-model reliability audit** is now implemented in
`bookingforge.arrival_reliability`. It repeats the same fit protocol
for the two logistic and two original NumPy neural configurations,
then reports **ten calibration bins**, hotel and arrival-month
performance, and **paired row-bootstrap** uncertainty in the
with-year vs no-year MLP differences. It does **not** tune models,
fit calibrators, alter thresholds or score 2017 arrival outcomes.

Run after `git pull` and passing all expected **67 tests**:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.arrival_reliability --csv data/raw/hotels.csv --output reports/arrival_reliability.json --mlp-epochs 30 --bootstrap-replicates 300
```

Results are **not yet measured**. The bootstrap resamples individual
validation rows while holding trained models fixed; it cannot establish
independent significance or account for clustered bookings and time
dependence. See [the detailed protocol](docs/EXPERIMENT_RESULTS.md).

## Research and engineering standards

1. Separate historical observation time, available labels, and feature timing.
2. Do not fit models using 2017 outcomes; acknowledge that legacy aggregate audits already inspected them, precluding a genuinely blind final-test claim.
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
