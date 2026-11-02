# Baseline experiment v1: methodological contract and limitations

This is an **experimental engineering baseline**, not a validated hotel-cancellation prediction service.

## Research question
Can a logistic regression classifier improve discrimination/calibration over a constant prior-probability predictor using a conservative set of booking-time proxy variables?

## Observation and split
- Reconstruct reservation creation time as `arrival_date - lead_time`. These timestamps are **not verified booking event logs**.
- Train creation cohort: created on/before **2016-08-31**, with final-status date on/before **2016-08-31**.
- Validation creation cohort: created between **2016-09-01** and **2016-12-31**, with final-status date on/before **2016-12-31**.
- Test creation cohort: **2017-01-01** through **2017-08-31**. **Not evaluated or tuned on in this first experiment.**
- Never use `reservation_status`, `reservation_status_date`, or `is_canceled` as model predictors. Dates of final outcomes are checked solely for cohort eligibility.
- Model selection for this first benchmark is limited to the single specified logistic baseline. The test set must remain sealed while we study the validity of the benchmark.

## Predictor features
`hotel`, `lead_time`, arrival year/month/day, weekday/weekend stay lengths, adults/children/babies, and `reserved_room_type`.

Feature names are also validated against `src/bookingforge/cohort.py` on module import. All are **provisional booking-time proxies**. We do NOT know whether the stored values are their original values from reservation creation.

## Preprocessing and metrics
- On training data only, fit numeric median imputation, scaling, and categorical one-hot encoding with unknown-category handling.
- Fit dummy prior-probability classifier and logistic regression.
- On selected validation records, report ROC AUC, average precision, Brier score, log loss, accuracy and F1 at an **arbitrary 0.5 threshold**.
- Save input SHA256, cohort sizes, selected feature names, result metrics, and limitations into `reports/baseline_validation.json`.
- No test-set metrics are computed. No model is deployed or production-calibrated.

## Why results can be biased
1. **Outcome-dependent selection.** Filtering to statuses known by a cutoff drops still-open bookings. The validation cohort in particular overrepresents fast-resolving outcomes; these are not a representative sample of all booked reservations.
2. **Arrival-bounded extraction.** The dataset only contains arrivals in 2015-07 through 2017-08; training/validation by reconstructed creation date is not a complete population of bookings created then. E.g. a booking created in 2016 with arrival beyond Aug 2017 is absent.
3. **Feature drift.** We lack booking-time snapshots and may unknowingly use later-updated fields, despite the conservative allowlist.
4. **No contemporaneous deployment inference.** Performance on selected historical resolved records is not evidence of a hotel's real expected performance, revenue or decision improvement.

## Run
From the repository root in PowerShell:

```powershell
python -m pip install -r requirements-ml.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m unittest discover -s tests -v
python -m bookingforge.train_baseline --csv data/raw/hotels.csv --output reports/baseline_validation.json
```

The raw CSV is ignored by Git. Do not commit it. Do not publish raw booking records. Aggregate metrics are intended for documentation after local verification.

## Next research task after baseline
Evaluate an *observation-window/survival-aware* formulation or alternative dataset, accounting for unsettled outcomes rather than discarding them without modeling censoring. Only then can a credible benchmark support claims about generalization.
