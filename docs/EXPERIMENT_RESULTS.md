# BookingForge: verified full-cohort experiments and diagnostic follow-up

This document separates **observed results** from **unrun experiments**. All
results are for the historical Hotel Booking Demand dataset, whose original
reservation-time feature snapshots and complete booking-creation cohorts
are unavailable. All evaluation remains exploratory, not deployment grade.

**Holdout integrity qualification:** Earlier aggregate-audit code
(`bookingforge.cohort.audit_creation_cohorts`) reads final target and
status fields for *all* creation cohorts, including the nominal
2017/test cohort. Later training and benchmarking routines deliberately
exclude those labels from model fitting and model evaluation, but
this repository has **not maintained a perfectly blind final-test
holdout from the outset**. Never describe its final-test outcomes as
globally untouched or claim an independent final-test result. A
new externally sourced time period or appropriately sequestered
data would be required for truly blind confirmation.

## Experiment A — selected validation subset (superseded)

- Training labels known by 2016-08-31: **63,334** records.
- Validation labels known by 2016-12-31: **11,991** records.
- Logistic-regression ROC AUC **0.8315**, Brier **0.1852**, average precision **0.8381**.
- Constant prior ROC AUC **0.5000**, Brier **0.2506**, average precision **0.4574**.

The historical validation cohort actually contained 19,339 records. Selecting
only 11,991 bookings whose outcomes resolved quickly changed the target
population, so Experiment A scores should **not** be interpreted as real
booking-cancellation performance or compared directly with Experiment B.

## Experiment B — full retrospective validation (verified locally)

The same historically known **63,334 training labels**, with **all 19,339**
validation-cohort bookings scored retrospectively using their eventual outcome.
All model transformations were fitted on training features only. The model
runner did not access test labels (older aggregate audits did).

| Model | ROC AUC ↑ | AP ↑ | Brier ↓ |
|---|---:|---:|---:|
| Training prior (constant) | 0.5000 | 0.3832 | **0.2370** |
| Logistic regression | **0.6776** | **0.5155** | 0.2460 |
| Original exact NumPy kNN (k=15) | 0.6295 | 0.4724 | 0.3312 |

Observed in a local full-dataset run on **2026-11-02**. Automated suite:
**34 tests passed** at commit `b0ccc4b`.

Interpretation: the logistic model discriminates better than a constant prior
but its uncalibrated probabilities are **worse by Brier score** than that simple
prior. kNN shows worse rank discrimination and probability error than
logistic regression. The 0.8315 vs 0.6776 ROC AUC difference is driven by a
change in evaluated population, so it is not a controlled model improvement
or deterioration claim.

## Experiment C — original NumPy neural net (verified on local device)

The model in `src/bookingforge/mlp_numpy.py` uses a single ReLU hidden
layer, sigmoid output, binary cross-entropy, exact analytic backpropagation,
mini-batch Adam and L2 weight penalty. There are **no PyTorch or TensorFlow**
training operations in this original baseline.

Predeclared configuration:
- 32 hidden units, 30 epochs, batch size 256
- Adam learning rate 0.001, L2 weight penalty 0.0001
- Seed 42, train-only imputation/scaling/one-hot encoding
- No validation-driven early stopping, parameter search or threshold tuning
- No random subsampling of training or validation cohorts

The `compare_knn` runner trained **all four** algorithms using the same
feature matrix. Verified local run on **2026-11-02**: **38/38 tests passed**;
63,334 training bookings and 19,339 validation bookings processed. The MLP's
training objective fell from **0.563358 (epoch 1)** to **0.487512 (epoch 30)**.
NumPy backpropagation also passed finite-difference gradient tests.

| Model | ROC AUC ↑ | AP ↑ | Brier ↓ |
|---|---:|---:|---:|
| Training prior | 0.5000 | 0.3832 | **0.2370** |
| Logistic regression | **0.6776** | **0.5155** | 0.2460 |
| Original NumPy kNN (k=15) | 0.6295 | 0.4724 | 0.3312 |
| Original NumPy MLP (32 hidden units, 30 epochs) | 0.6715 | 0.5151 | **0.5950** |

The MLP's Brier score is concerning: it is *much worse* than the constant prior,
although ROC AUC and AP suggest some ranking ability. Falling training loss
cannot establish out-of-time probability calibration. We must investigate
predicted-probability distributions, saturation, and subgroup behavior rather
than assume more training or larger architecture will fix the issue.

**Diagnostic next step (does not tune models):** Run
`python -m bookingforge.probability_diagnostics --csv data/raw/hotels.csv`
from the repo root with `PYTHONPATH=src`. The diagnostic re-fits the same
logistic/MLP configurations on the same train cohort and reports aggregate
train/validation probability summaries and per-hotel validation breakdowns.
It neither scores test labels nor chooses new hyperparameters.

PowerShell from the repository root:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.compare_knn --csv data/raw/hotels.csv --output reports/full_four_model_comparison.json --k 15 --batch-size 64 --mlp-hidden 32 --mlp-epochs 30 --mlp-batch-size 256 --mlp-learning-rate 0.001
```

Do not overwrite the earlier `reports/full_knn_comparison.json` run;
it is useful as an independent reproducibility comparison.

## Experiment D — probability diagnostic (verified locally)

Local run on **2026-11-02**: **42/42 tests passed**; recreated the same
training run on 63,334 selected training records and the full 19,339-record
retrospective validation cohort.

| Model | Training mean probability | Validation mean probability | Validation Brier |
|---|---:|---:|---:|
| Logistic regression | 0.408 | 0.554 | 0.2460 |
| Original NumPy MLP | 0.409 | **0.986** | **0.5950** |

The actual cancellation rate in full retrospective validation was **0.383**.
The NumPy MLP assigned **P(canceled) > 0.99 to 67.5%** of validation records
(while only 7.6% of training records received that score). Its subgroup
validation means were **0.989** for City Hotel and **0.978** for Resort Hotel,
despite observed rates **0.441** and **0.262**, respectively. Logistic
regression also overestimated validation risk, but less dramatically.

This is consistent with severe covariate shift/extrapolation, overfitting,
or both. A data/feature bug is not ruled out. The result alone does **not**
prove which feature caused the failure. We must not assert that training
longer or adding GPU capacity will fix probability calibration.

## Experiment E — input feature drift (verified locally)

Local audit on **2026-11-02**: **46/46 automated tests passed**. The
63,334-record historically eligible training cohort and the full 19,339-record
retrospective validation cohort show:

| Feature | Train mean | Validation mean | Shift in training SD units |
|---|---:|---:|---:|
| Arrival year | 2015.666072 | 2016.543151 | +1.757087 |
| Lead time | 107.687987 | 93.562749 | -0.127611 |
| Week nights | 2.462058 | 2.569781 | +0.057863 |
| Weekend nights | 0.902548 | 0.930038 | +0.027724 |

No new categories were seen for hotel, arrival month or reserved room type.
The arrival-year shift is much larger than the other numeric shifts. A larger
share of later arrival years may cause the neural model to extrapolate strongly,
even when those years technically occur in the historical training support.
This **does not establish** which feature caused extreme predictions.

## Experiment F — prediction sensitivity (verified locally)

Local result on **2026-11-02**: **50/50 tests passed**. Repeating the original
30-epoch NumPy MLP and fixed logistic regression confirmed that changing
**only** `arrival_date_year` in validation features to the training median
of **2016** moved MLP average cancellation predictions from **0.986 to 0.684**.

For validation records arriving in **2017** (n=10,504), average neural
predictions moved from **0.998 to 0.442** under that synthetic replacement.
For **2016** arrivals (n=8,835), the original mean **0.971** did not change,
because their year was already 2016. This makes arrival-year dependency
a substantial contributor to the extreme 2017 predictions, **not the only
source of overconfidence**.

Replacing `lead_time` with its training median (72 days) moved the
overall MLP mean **0.986 to 0.993**. This was a much smaller model
response, with no claim that it improves real-world predictions.

These are **label-blind, synthetic input interventions**, not new Brier
scores or valid production feature transformations. The original
outer-validation diagnostics have already been inspected to generate
hypotheses, so that cohort can no longer be described as independent
for future model selection.

## Experiment G — inner chronological arrival-year ablation (verified locally)

On **2026-11-02**, the new local experiment passed **54/54 automated tests**.
Inner model fitting used **39,507** historically label-mature bookings; the
inner chronological evaluation covered **6,860** bookings. Each MLP ran
30 epochs under its original fixed training configuration.

| Model | ROC AUC (higher) | AP (higher) | Brier (lower) | Mean predicted P |
|---|---:|---:|---:|---:|
| Logistic WITH arrival year | 0.5789 | 0.4982 | 0.6016 | 0.991 |
| Logistic WITHOUT arrival year | 0.4231 | 0.3652 | 0.5181 | 0.808 |
| Original NumPy MLP WITH arrival year | **0.6281** | **0.5381** | 0.5820 | 0.977 |
| Original NumPy MLP WITHOUT arrival year | 0.4413 | 0.3782 | **0.5054** | 0.814 |

Removing year reduced the average predictions and Brier errors, but
**both models then scored ROC AUC less than 0.5**, meaning their ranking
was worse than a random ranking on these selected evaluation records.
Even the reduced Brier score is still concerning. None of these models
is established as calibrated. Do not assert the no-year feature set
is superior, let alone ready for deployment.

The label-maturity-selected historical construction and arrival-bounded
source can generate substantial cohort shift. Also, the year-removal
hypothesis was motivated by already-inspected outer validation. Therefore
the outer validation cohort is **not independent for claims about year
ablation**, despite not being used for fitting in this inner experiment.

## Experiment H — inner-cohort composition and constant baseline (verified locally)

The local diagnostic passed **58/58 tests** and completed on
**2026-11-02**. It used **39,507 historical training** and **6,860
inner validation** bookings. The training outcome prevalence was
**46.653%**, while validation prevalence was **38.688%**.

**Constant training-prior baseline on inner validation:**
ROC AUC **0.5000**, average precision **0.38688**, Brier **0.243547**,
log loss **0.680224**. This Brier result beats ALL four feature-ablation
models above by a wide margin.

| Arrival year | Historical training records | Canceled/no-show fraction | Inner validation records | Canceled/no-show fraction |
|---|---:|---:|---:|---:|
| 2015 | 21,996 | 37.016% | 0 | — |
| 2016 | 17,283 | 58.213% | 6,753 | 37.717% |
| 2017 | 228 | **100%** | 107 | **100%** |

**Critical structural artifact:** Under a historical cutoff of
2016-03-31 for train and 2016-08-31 for inner validation,
a stay arriving in **2017 cannot yet have checked out**. If its
final status is already known before those cutoffs, the booking can
only belong to an early-terminating category (such as cancellation).
Therefore filtering to status-mature outcomes creates a near-tautological
relationship between future arrival year and outcome in those rows,
rather than revealing a legitimate future risk factor.

The problem is deeper than a single feature, trainable architecture,
or insufficient epochs. No model with or without year achieved
better Brier score than the constant prior. Retain these negative
results rather than tuning them away.

The source has no immutable original booking-time feature history and
is arrival-bounded. Even a corrected retrospective model is not
evidence of a prospectively deployable reservation-risk tool.

## Experiment I — arrival-separated retrospective benchmark (verified locally)

**Recorded:** 2026-11-02; **62/62 tests passed** at source commit
`2b2e853`. This benchmark is a **different, specifically retrospective
algorithmic research task**, not a booking-creation-time prediction
experiment. Do NOT directly compare its metrics with the earlier
booking-creation cohorts as if they were the same statistical task.

- Training: **58,792** records arriving from 2015-07-01 to 2016-08-31,
  all final outcomes known by **2016-09-30**. Training outcome prevalence:
  **35.6%**.
- Excluded transition: all September 2016 arrivals.
- Validation: **14,517** arrivals from 2016-10-01 to 2016-12-31,
  retrospectively using eventual final outcomes only to score, never fit.
  Validation outcome prevalence: **38.2%**.
- **393** validation final outcomes were recorded after 2016-12-31:
  retrospectively scored, not assumed known during that period.
- **46,081** arrivals outside either period excluded by design.
  2017 arrivals were skipped before the benchmark reader accesses
  outcomes (legacy aggregate audit previously examined nominal 2017
  outcomes, so this is not a globally blind holdout).

| Model | ROC AUC ↑ | Average precision ↑ | Brier ↓ | Mean predicted P |
|---|---:|---:|---:|---:|
| Constant training prior | 0.5000 | 0.3822 | 0.2368 | 0.356 |
| Logistic with raw arrival year | 0.7182 | 0.6154 | 0.2175 | 0.278 |
| Logistic without raw arrival year | 0.7172 | 0.6140 | 0.2064 | 0.373 |
| Original NumPy MLP with year | 0.7207 | 0.6383 | **0.2032** | 0.347 |
| Original NumPy MLP without year | **0.7300** | **0.6456** | 0.2072 | 0.353 |

All four models achieved better ROC AUC and Brier scores than the
constant training-prior baseline on this retrospective October–December
2016 cohort. The **MLP without year** had the best ranking metrics;
the **MLP with year** had the best Brier score. These modest differences
have not been given confidence intervals, so there is **no established
statistically significant winner**.

Both MLP variants ran the specified **30 epochs** on the user's local
CPU; the with-year MLP objective fell from **0.580277 to 0.527097**,
the no-year MLP objective from **0.581752 to 0.532010**.
No row subsampling, no outer-test model fitting, no early stopping,
and no additional hyperparameter search.

**Limits:** this model sees columns that may represent their final
post-booking values. It cannot be described as a deployed or reliable
booking-time risk model. Validation's eventual outcomes are obtained
retrospectively. The historical source is selected by arrival date, and
this protocol was developed after previous results had been inspected.
This is not an independent, prespecified prospective test.
Uncertainty intervals, grouped subgroup evaluation, calibration
curves, and a new independent external cohort are needed to justify
stronger claims.

Reproduce with:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = "C:\Users\Ayush\bookingforge\src"
.\.venv\Scripts\python.exe -m bookingforge.arrival_benchmark --csv data/raw/hotels.csv --output reports/arrival_benchmark.json --mlp-epochs 30
```

**Next experiment:** Add *descriptive only* hotel/month subgroups,
reliability bins and uncertainty estimates for fixed models. Avoid
choosing features, hyperparameters or decision thresholds based on
repeated inspection of this same validation set.

## Experiment J — fixed-model reliability and uncertainty diagnostics (verified locally)

**Local execution:** 2026-11-02, **67/67 automated tests passed**.
Re-ran unchanged logistic-regression and original NumPy-MLP fits,
with and without the raw arrival year, on **58,792 training records**;
scored the same **14,517 retrospective October–December 2016 arrivals**.
All MLP runs completed 30 epochs and exactly reproduced the original
training-objective progression.

| Model | ROC AUC ↑ | Brier ↓ | ECE (10 fixed bins) ↓ |
|---|---:|---:|---:|
| Training prior | 0.5000 | 0.236820 | **0.026529** |
| Logistic with year | 0.718235 | 0.217525 | 0.120609 |
| Logistic without year | 0.717186 | 0.206408 | 0.056390 |
| Original MLP with year | 0.720688 | **0.203197** | 0.051002 |
| Original MLP without year | **0.730030** | 0.207161 | 0.090582 |

The constant predictor is closest to average prevalence in these fixed
bins, but has no ranking ability; **ECE alone is not a sufficient
model-quality criterion**. The with-year MLP has the lowest Brier score;
the no-year MLP ranks best by AUC. The MLPs' average validation
probabilities and high-confidence rates are recorded by Experiment I.

### Retrospective subgroup heterogeneity

| Group | Records | MLP with year: Brier / AUC | MLP without year: Brier / AUC |
|---|---:|---|---|
| City Hotel | 9,819 | 0.218498 / 0.712210 | 0.222339 / 0.716399 |
| Resort Hotel | 4,698 | 0.171216 / 0.702828 | 0.175440 / 0.704804 |
| October | 6,203 | 0.220952 / 0.691429 | 0.232385 / 0.686699 |
| November | 4,454 | 0.182547 / 0.754362 | 0.174502 / 0.785073 |
| December | 3,860 | 0.198492 / 0.706424 | 0.204312 / 0.726841 |

October has weaker ranking performance than November for both
fixed MLPs; hotel-level Brier differs materially. These are
descriptive differences across non-identical populations, not
controlled explanations of difficulty or proof of seasonal drift.

### Paired row-resampling, fixed models

**Difference is WITHOUT-year MLP minus WITH-year MLP**, using
300 reproducible paired bootstrap replicates (seed 2026).

| Quantity | Observed difference | Descriptive 95% percentile interval |
|---|---:|---|
| ROC AUC | +0.009342 | [+0.003742, +0.015165] |
| Brier (lower is better) | +0.003965 | [+0.002091, +0.005907] |

A positive AUC difference favors the no-year model; a positive
Brier difference favors the with-year model. The row-bootstrap
assumes independently sampled records and holds the fitted models
fixed. It omits model-refit variance, time-series and hotel/guest
clustering, historical cohort selection, and repeated exploratory
comparisons. **Do not treat intervals excluding zero as definitive
statistical significance or proof of generalization.** Neither
configuration is selected or tuned by this report.

No calibrator was fitted, no classification threshold optimized,
no raw individual rows exported, and no 2017 arrival outcomes
read by the new diagnostic. Legacy aggregate reports did
previously inspect nominal test outcomes, which remain
ineligible for claims of independent external confirmation.

See locally generated aggregate JSON:
`reports/arrival_reliability.json`.

## Experiment K — portable NumPy model export (verified on local full dataset)

`bookingforge.model_artifact` now trains and exports the two **unchanged
and separately labeled** original NumPy MLP configurations:
`with_year` and `without_year`. By default it trains **both**,
rather than claiming one is the winner. They use identical 2015–Aug
2016 eligible arrival-based training data, the 2016-09-30 training
outcome cutoff, seed 42, 32 hidden units, mini-batch size 256,
Adam learning rate 0.001, L2 0.0001, and 30 epochs.

The module saves, under the ignored `models/arrival_v1/` directory:

- `arrival_mlp_with_year.npz` and `arrival_mlp_with_year.json`
- `arrival_mlp_without_year.npz` and
  `arrival_mlp_without_year.json`

Each NumPy archive stores **only** W1, b1, W2, b2 arrays. JSON stores
the expected feature schema, train-fitted median imputations,
standardization means/scales, one-hot categorical support, training
cohort/method settings, dataset SHA256, and the NumPy archive SHA256.
The inference loader checks archive integrity, model tensor shapes,
feature allowlist, and the schema, loads with `allow_pickle=False`,
and reproduces sklearn one-hot encoding (including unseen categories
mapped to all-zero indicator vectors) without needing the fitted
scikit-learn preprocessing object.

**Model reproducibility test:** The trainer refits the declared
configurations using ONLY training labels, serializes them, reloads
each artifact, and compares raw cancellation probabilities to the
original in-memory model over all **14,517 validation input rows**.
It requires a maximum absolute prediction discrepancy no greater than
`1e-12`, and does not use retrospective validation labels to train
or select either configuration. It does not inspect 2017 arrival
outcomes; older aggregate audits already accessed nominal
2017 test labels.

No per-booking data or validation predictions are saved. A separate
aggregate JSON report records the verification and file hashes.
**An exactly reproducible research model is not a calibrated or
approved booking-time predictor.** All source and predictor-timing
uncertainties from the preceding experiments remain in force.

Windows PowerShell:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = "C:\Users\Ayush\bookingforge\src"
.\.venv\Scripts\python.exe -m bookingforge.model_artifact --csv data/raw/hotels.csv --output-dir models/arrival_v1 --report reports/arrival_artifact_verification.json --variant both --epochs 30
```

**Local full-dataset export verified on 2026-11-02:** both fixed
30-epoch original NumPy MLP variants trained on **58,792** historical
arrival records. With-year training objective ended **0.527097**;
without-year objective ended **0.532010**. Both variants were saved
to the four specified `.npz`/`.json` model files, reloaded, and
achieved **maximum absolute prediction difference = 0** on every
retrospective validation *feature* row, as required.

Verification metadata saved locally to
`reports/arrival_artifact_verification.json`. The specific
SHA256 values have not been supplied in the user's log, so no
hash values are invented.

**Test verification caveat:** the user's message verified the
full export, not the newly extended 72-test suite. Five artifact
unit tests have been committed but have **not been shown as passed**
on the user's device. The prior 67-test suite is verified.

## Experiment L — local-only FastAPI inference (pending verification)

The new `bookingforge.api` is an **optional**, unauthenticated
local research/demo service intended for `127.0.0.1` only.
It does **not** retrain, tune, recalibrate, score historical
outcomes, or make hotel operating decisions. At startup it loads
the existing SHA256-verified research NumPy artifacts, failing
closed if the files are missing, tampered or incompatible.

Routes:

- `GET /health` — confirms both original models are loaded.
- `GET /v1/research/models` — exposes schema, training-row count,
  dataset fingerprint, and arrival-year inclusion for each variant.
- `POST /v1/research/predict` — accepts strict synthetic
  2015–2016 booking-like inputs and a required `with_year` or
  `without_year` variant; returns a *research score* plus
  warnings. Unknown fields, outcome/status columns, malformed
  dates and 2017/future arrival dates must receive HTTP 422.

The endpoint requires no real guest data, makes no claims of
booking-time calibration and is unsuitable for public network
deployment. **Do not bind to 0.0.0.0.** No production
authentication, authorization, rate limiting, guest privacy
assurance, or service hardening is claimed.

Its optional packages are pinned separately in
`requirements-api.txt` (FastAPI, Uvicorn, HTTPX).
Five synthetic HTTP tests check model-file startup, repeatable
prediction parity with loaded artifacts, strict input rejection,
and fail-closed startup. An end-to-end PowerShell walkthrough
using fabricated input is in [the research API guide](API_RESEARCH.md).

Expected full suite after five artifact tests and five API tests:
**77 tests**, pending local verification. HTTP inference
output/latency/performance has not yet been observed; no
specific prediction score is reported.

## Limitations and next scientific steps

1. The arrival-bounded source is not a complete cohort of bookings made
   in any chosen creation period.
2. Historical labels resolved by a cutoff are a selected subset; training
   on them does not correct for censoring or outcome-dependent observation.
3. We do not have immutable feature snapshots. Temporal feature availability
   is plausible but unverified.
4. The full retrospective validation cohort has eventual outcomes and may
   fairly compare algorithms **on this dataset**, but it is not a
   prospective deployment evaluation.
5. Repeatedly tuning hyperparameters on the same validation cohort can
   overfit selection. Freeze the experiment design; when optimization
   becomes necessary, create an *inner* training-only time split.
6. Nominal 2017 test outcomes have already been inspected by legacy aggregate audits; independent confirmation requires a new, properly sequestered data period or source.
7. Confidence intervals, lead-time/hotel subgroup errors, and probability
   calibration should be assessed in separate documented experiments.
8. Forecasting, Poisson-binomial DP, Monte Carlo and optimization must
   separately declare their own observation times and simulation assumptions.
