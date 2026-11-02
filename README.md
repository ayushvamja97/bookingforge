# BookingForge

**Hotel Booking Intelligence — Applied Machine Learning and Software Engineering Research**

A reproducible exploration of historical hotel reservation outcomes, original NumPy
algorithms, time-based evaluation, data audits and decision simulation.
**This is not an operational hotel cancellation-risk or overbooking system.**
Any future inventory/revenue analysis must use explicitly synthetic assumptions.

## Current research status

- Public historical dataset: **119,390 rows, 32 columns**, arrivals July
  2015 to August 2017. Individual raw booking rows are not stored in Git.
- **67/67 automated tests passed** locally on **2026-11-02** for the fixed-model
  reliability audit. Full-data model export subsequently succeeded with
  **zero save/reload prediction difference for both networks**. New artifact and
  FastAPI tests remain unverified on Windows (expected total **77**).
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
network with year has the best probability Brier score. Exploratory paired row-bootstrap intervals
favor the no-year MLP for ROC AUC and the with-year MLP for Brier,
but neither has independent confirmation. Both beat the constant
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

The first measured run is recorded in the
[experiment ledger](docs/EXPERIMENT_RESULTS.md). Prior reports
remain available locally.

## Verified reliability analysis

**67/67 tests passed**. The unchanged algorithms were refitted and
their scores reproduced on the same 14,517 validation records.
The probability-calibration ECE of the MLP with year was **0.0510**,
versus **0.0906** without year. The constant prior achieved a smaller
ECE (**0.0265**) but has no ranking ability; ECE alone cannot choose
a useful model.

Hotel and month subgroups demonstrated heterogeneous performance.
For example the MLP with year had October AUC **0.6914** versus
November **0.7544**. The paired row-bootstrap (300 resamples)
found an AUC difference **+0.009342** for no-year minus with-year
and a Brier difference **+0.003965** (positive Brier is worse).
These intervals are exploratory, condition on fixed models,
ignore group/time dependence, and cannot support an independent
significance or generalization claim.

A detailed JSON report is in
`reports/arrival_reliability.json` on the local machine.

## Next verification: portable NumPy neural-network artifacts

`bookingforge.model_artifact` now trains the **same original NumPy
MLP with and without year**, exporting each model separately:

- `models/arrival_v1/arrival_mlp_with_year.npz`
- `models/arrival_v1/arrival_mlp_with_year.json`
- `models/arrival_v1/arrival_mlp_without_year.npz`
- `models/arrival_v1/arrival_mlp_without_year.json`

The NumPy archives store only numerical weights; JSON stores train-fitted
scaling, imputation, categorical encodings, and data/protocol metadata.
No pickle is used to load model weights. The loader verifies SHA256,
tensor shapes and feature schema before predicting, including unseen
categories. The exporter compares original versus reloaded predictions
over all historical validation **features**, with tolerance at most
`1e-12`. It does not fit or select models using validation outcomes.

Run after pulling and passing the expected **72 tests**:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = "C:\Users\Ayush\bookingforge\src"
.\.venv\Scripts\python.exe -m bookingforge.model_artifact --csv data/raw/hotels.csv --output-dir models/arrival_v1 --report reports/arrival_artifact_verification.json --variant both --epochs 30
```

No full-dataset artifact-verification result is known yet.
Both model variations remain **research-only historical arrival
classifiers**, not validated booking-time predictions. The
`models/` and `reports/` directories are excluded from Git.

## Next verification: local FastAPI inference

The Python module `bookingforge.api` exposes an **opt-in local-only**
research interface backed by the saved NumPy models. It loads the
two artifacts once on startup, checks their SHA256 integrity,
accepts only validated synthetic 2015–2016 reservation features,
and rejects cancellation/status fields. Prediction requests never
trigger model retraining and never use historical validation targets.

**Optional dependencies:** FastAPI, Uvicorn, and HTTPX are pinned
separately in `requirements-api.txt`. Install the extras before
running the extended test suite:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000
```

Expected test count: **77** (the previous 67 are verified; the five artifact and five API tests still need a complete local test-run log).
The default OpenAPI UI is at `http://127.0.0.1:8000/docs`.
**Do not use public network binding or input actual guest records.**
The API has no authentication and its scores are not validated
original booking-time cancellation probabilities.

**Verified local HTTP milestone (2026-11-02):** The `/health` endpoint
returned `ready` with both saved model variants loaded. A synthetic
City Hotel request to `POST /v1/research/predict` returned **HTTP 200**
and `research_cancellation_score=0.2177707187048443` using the
`without_year` model. Both `research_only=true` and
`calibrated_for_booking_time=false` were present. This confirms
the local saved-model-to-HTTP request integration, not statistical
calibration or a real-world risk estimate.

See [the local API guide](docs/API_RESEARCH.md) for a complete
synthetic PowerShell request, response fields and limitations.

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
- [Local research API guide](docs/API_RESEARCH.md)
- [Original retrospective cohort protocol](docs/BASELINE_PROTOCOL.md)
- [Measured experiments, limitations and next protocol](docs/EXPERIMENT_RESULTS.md)
