# BookingForge: retrospective full-validation protocol (v2)

**Status:** exploratory research benchmark, not a deployed or statistically unbiased hotel-cancellation predictor. The previous v1 subset-only validation score must not be compared directly to v2 scores.

## Target

Predict whether a booking ultimately ends as a cancellation or no-show (binary `is_canceled`), from a conservative list of booking-time **proxy** attributes. The source lacks immutable booking-creation timestamps and historical feature snapshots. Approximate the booking date as **arrival minus lead_time**.

## Chronology and outcome maturity

- **Training booking-creation cohort:** booking date on/before **2016-08-31**.
- **Training labels:** only those with final status date on/before **2016-08-31**. This yields 63,334 eligible records in the initial local audit. No outcomes finalized later are used for training.
- **Validation booking-creation cohort:** booked from **2016-09-01** through **2016-12-31**. **Include every record with a valid eventual final outcome**, even if its outcome became known in 2017. This yields 19,339 records in the local audit. Retrospective validation may score them, but may **never** fit model weights, scalers, imputers, categories, thresholds, or hyperparameters on their labels.
- **Nominal final test booking-creation cohort:** created from **2017-01-01** through **2017-08-31**. The newer model-fitting and scoring routines do not access its outcomes. **Caveat:** legacy aggregate diagnostic code in `cohort.py` did inspect outcome fields across all cohorts, including this cohort, so the dataset cannot be described as completely blind or untouched.
- The historical training fit date is August 31, 2016. Validation scores represent a *retrospective backtest when eventual outcomes become known*, not a contemporaneous score obtainable on Dec 31, 2016.
- Temporal validation allows comparing the models on the *same* 19,339 retrospective validation rows without selecting only fast-resolving outcomes.

## Models and comparison

- Baseline: constant prior probability estimated from **training labels only**.
- Logistic regression: scikit-learn, fit on **all** eligible training records.
- Custom k-nearest neighbors: exact Euclidean squared distance in NumPy using batches to constrain memory, uniform vote with fixed `k=15`, and **all** eligible training records for neighbor lookup. We explicitly do not use scikit-learn KNeighborsClassifier for predictions.
- Same numeric median imputer, standard scaler and categorical one-hot encoder fit **only** on training; validation is transformed without fitting.
- No random subsampling of training or validation rows.
- Model comparison uses ROC AUC, average precision, Brier score, log loss, accuracy and F1 at a **fixed** 0.5 classification threshold. No hyperparameters are tuned in this initial comparison.
- Reproducible inputs: raw CSV SHA256, source-code version in Git, environment versions, kNN k, batch size, and resulting aggregate JSON report. No raw rows are committed.

## Remaining scientific limitations

1. **Historical training is label-selected.** Training on only outcomes resolved by the August 2016 cutoff overrepresents early-resolving bookings. Do not report model quality as an unbiased estimate of overall booking risk.
2. **Arrival-window selection / right truncation.** The dataset is collected by arrivals between July 2015 and August 2017, so reconstructed creation-time cohorts do not represent all bookings created in those periods.
3. **Feature-timing uncertainty.** Predictor values are not proven to be the values recorded at the original booking date; even fields with plausible names can be edited later.
4. **Shared booking records / duplicate structures.** No unique event log or booking IDs permit a robust independence/group-level leakage audit.
5. **No fully blind test cohort.** Model training and scored evaluation did not use nominal final-test outcomes, but previous aggregate audits inspected them. Independent model confirmation will require a new, properly sequestered holdout.
6. **No real hotel policy.** Future capacity/cost experiments must be labeled simulated, with assumed constraints and costs.

**Interpretation:** This is a legitimate fully computed exploratory algorithmic comparison, with documented threats to validity. It is *not* yet evidence for production cancellation forecasting. Better-captured booking snapshots or a time-to-event/censoring-aware research formulation would strengthen deployment claims.

## Commands

From the project root (PowerShell):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m bookingforge.train_baseline --csv data/raw/hotels.csv --output reports/baseline_validation_v2.json
.\.venv\Scripts\python.exe -m bookingforge.compare_knn --csv data/raw/hotels.csv --output reports/full_knn_comparison.json --k 15 --batch-size 64
```

The second experiment performs its own logistic/prior baseline on exactly the same full training and validation sets as custom kNN; do not compare its metrics to the older 0.8315 ROC AUC v1 result.
