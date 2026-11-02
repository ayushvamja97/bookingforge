# BookingForge: verified full-cohort experiments and diagnostic follow-up

This document separates **observed results** from **unrun experiments**. All
results are for the historical Hotel Booking Demand dataset, whose original
reservation-time feature snapshots and complete booking-creation cohorts
are unavailable. All evaluation remains exploratory, not deployment grade.

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
All model transformations were fitted on training features only. Final test
cohort outcomes remain sealed.

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

## Experiment I — alternative arrival-cohort retrospective benchmark (pending)

To investigate whether algorithms can learn cancellation patterns
without the **matured future-arrival structural artifact**, the new
`bookingforge.arrival_benchmark` uses a deliberately different
observation/evaluation design:

- **Fit cohort:** arrivals from **2015-07-01 through 2016-08-31**;
  include only final statuses recorded **by 2016-09-30**.
- **Gap:** September 2016 arrivals excluded.
- **Retrospective scoring cohort:** all bookings arriving
  **2016-10-01 through 2016-12-31**, using eventual final outcomes
  only for scoring (never model fitting).
- **Sealed:** arrivals in 2017 and beyond are skipped *before*
  reading outcome dates and targets.
- Compare prior probability, original fixed logistic regression and
  original NumPy MLP, with and without numeric arrival year.
  Both feature configurations use all eligible rows; train-only
  imputation/scaling/encoding, 32 hidden units, 30 epochs, seed 42.
- Report class prevalence, row counts, missing/late label counts,
  ROC AUC, average precision, Brier, log loss, and mean confidence.
- **This is NOT booking-creation-time prediction.** It is an explicitly
  retrospective, *arrival-date-separated* algorithmic comparison.
  Historical predictor snapshots remain unavailable, training has a
  remaining small label-availability filter, and exploratory choices
  have already been influenced by prior validation results. Do not
  advertise the results as independently confirmed or production-ready.

Reproduce in PowerShell:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = "C:\Users\Ayush\bookingforge\src"
.\.venv\Scripts\python.exe -m bookingforge.arrival_benchmark --csv data/raw/hotels.csv --output reports/arrival_benchmark.json --mlp-epochs 30
```

Results of this experiment are NOT YET observed. Do not copy
metrics from an earlier incompatible cohort as if they belonged
to this benchmark.

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
6. A later fixed-protocol final-test evaluation may be appropriate after
   resolving or explicitly restricting the deployment claim; not before.
7. Confidence intervals, lead-time/hotel subgroup errors, and probability
   calibration should be assessed in separate documented experiments.
8. Forecasting, Poisson-binomial DP, Monte Carlo and optimization must
   separately declare their own observation times and simulation assumptions.
