# BookingForge: validated experiments and next neural baseline

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

## Experiment C — original NumPy neural net (to be run)

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

The `compare_knn` runner now trains **all four** algorithms using the
same feature matrix and reports ROC AUC, average precision, Brier, log loss,
accuracy and F1. It saves the MLP training objective by epoch. NumPy MLP
backpropagation was independently checked against finite-difference
gradients on synthetic inputs. The full-cohort results are **not yet known**.

PowerShell from the repository root:

```powershell
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.compare_knn --csv data/raw/hotels.csv --output reports/full_four_model_comparison.json --k 15 --batch-size 64 --mlp-hidden 32 --mlp-epochs 30 --mlp-batch-size 256 --mlp-learning-rate 0.001
```

Do not overwrite the earlier `reports/full_knn_comparison.json` run;
it is useful as an independent reproducibility comparison.

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
