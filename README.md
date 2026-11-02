# BookingForge

**Hotel Booking Intelligence & Optimization Research Engine**

An applied AI/ML and software-engineering portfolio project researching cancellation prediction, demand forecasting, and constrained decision algorithms. All future hotel inventory/cost decisions are strictly **simulations**, not operational advice.

## Status

**Phase 1:** Data acquisition, provenance, quality/temporal audits, historical-cutoff protocol, and first exploratory machine-learning baseline.

**Phase 2 in progress:** Original NumPy kNN and a from-scratch NumPy neural network, compared on **all eligible train and validation rows**.

- Real publicly available dataset: 119,390 records, 32 columns, arrivals July 2015–August 2017.
- **50 automated tests passed locally** at commit `358c273` on 2026-11-02. Four new inner-chronological ablation tests await local verification.
- Historical training cohort contains **63,334** bookings with outcomes known by 2016-08-31.
- The **full retrospective validation creation cohort contains 19,339 bookings** with recorded eventual outcomes. These labels are used only for later scoring, never model fitting.
- Final test creation cohort remains **sealed**. No deployment claims or simulated revenue claims.

## Earlier benchmark — superseded, not comparable to the new protocol

The original outcome-mature-only validation run produced:

| Metric | Constant training prior | Logistic regression |
|---|---:|---:|
| ROC AUC | 0.5000 | 0.8315 |
| Average precision | 0.4574 | 0.8381 |
| Brier score (lower is better) | 0.2506 | 0.1852 |

This run used **63,334 training** and **only 11,991 selected validation records**. Retrospective diagnosis showed the omitted 7,348 validation bookings had different eventual cancellation rates. Therefore **0.8315 must not be presented as general population performance or compared directly with the newer validation protocol**. The selection audit found that only 62.00% of validation records had final outcomes by the validation cutoff.

## New protocol: full retrospective validation

We train only on outcomes available by 2016-08-31 and evaluate **all 19,339** later-created validation bookings using their eventual final outcomes. Historical training selection bias, arrival-window selection, and unavailable feature snapshots remain limitations. See [experiment methods and caveats](docs/BASELINE_PROTOCOL.md).

The comparison includes:
- Constant prior-probability predictor trained only on historical training labels.
- Logistic regression with preprocessing fit **only** on historical training features.
- **From-scratch NumPy kNN** using exact squared-Euclidean nearest neighbors, batches to limit peak memory, a fixed `k=15`, and progress reporting. No sampling of the dataset.

**Verified full-cohort results:** constant prior AUC 0.5000, AP 0.3832, Brier 0.2370; logistic AUC 0.6776, AP 0.5155, Brier 0.2460; original kNN AUC 0.6295, AP 0.4724, Brier 0.3312. The new four-model comparison is also verified: original NumPy MLP ROC AUC **0.6715**, AP **0.5151**, but Brier **0.5950**, a serious probability-quality failure under investigation. The saved earlier kNN report is `reports/full_knn_comparison.json`.

## Reproduce

Use Python 3.12 on Windows (tested dependency versions are pinned in `requirements-ml.txt`). The raw CSV is **not included** in this repository.

Download the Hotel Booking Demand data copy from the [TidyTuesday hotel dataset](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-02-11/hotels.csv), place it at `data/raw/hotels.csv` and verify its SHA256 fingerprint:

`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`

Run from the repository root using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.data_audit --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.temporal_audit --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.cohort --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.selection_diagnostics --csv data/raw/hotels.csv
.\.venv\Scripts\python.exe -m bookingforge.train_baseline --csv data/raw/hotels.csv --output reports/baseline_validation_v2.json
.\.venv\Scripts\python.exe -m bookingforge.compare_knn --csv data/raw/hotels.csv --output reports/full_four_model_comparison.json --k 15 --batch-size 64 --mlp-hidden 32 --mlp-epochs 30
```

**Verified probability diagnostics:** Logistic regression predicted an average cancellation probability of **0.554** in validation; the original NumPy MLP predicted **0.986** although actual validation prevalence was **0.383**. The MLP assigned >99% cancellation probability to **67.5%** of validation records. The cause remains under investigation. Run `python -m bookingforge.feature_drift --csv data/raw/hotels.csv` with `PYTHONPATH=src` to inspect label-blind feature shifts (not yet verified locally).

**Verified sensitivity audit:** replacing validation arrival year with the training median (2016) changed the original NumPy MLP's overall mean predicted cancellation probability from 98.6% to 68.4%; for the 10,504 bookings arriving in 2017, mean probability fell from 99.8% to 44.2%. This synthetic intervention is NOT a valid real-world feature edit or a new accuracy score. The next experiment uses a strictly inner chronological split, with and without the raw arrival year. Run `python -m bookingforge.inner_year_ablation --csv data/raw/hotels.csv --output reports/inner_year_ablation.json --mlp-epochs 30` after pulling, testing, and setting `PYTHONPATH=src`. The final test set remains sealed.

The exact kNN computation is significantly more expensive than logistic regression. It processes the full validation population in batches rather than reducing its size.

## Methodological boundaries

1. Booking-creation dates are approximated by arrival date minus lead time; no immutable booking creation events exist in the dataset.
2. Source data are bounded by **arrival** dates, resulting in incomplete and potentially biased **creation** cohorts.
3. Historical training selects only outcome-mature labels, changing the training distribution.
4. Feature values are not immutable original booking snapshots; even conservative predictors are provisional.
5. Label, reservation status and final-status date are never model predictors.
6. Validate the baseline and any future model on identical evaluation rows, report uncertainty, and reserve final test evaluation.
7. Never commit raw individual booking records, invent model results, or claim real hotel deployment.

## Roadmap

1. **Data quality and experimental protocol:** audits, missingness, label maturity, and reproducible baseline (completed for exploratory research; scientific limitations remain).
2. **Custom algorithms:** original NumPy kNN (full-cohort benchmark verified); original NumPy MLP/backprop with gradient checks (awaiting full-cohort experiment).
3. **Deep learning & demand:** temporal forecasting, seasonal baselines, then PyTorch sequence models where justified.
4. **Optimization & simulations:** Poisson-binomial DP, exact/heuristic resource matching, Monte Carlo under explicit synthetic assumptions.
5. **Application engineering:** FastAPI, Next.js UI, Docker, CI, systematic benchmarks and design documentation.

## Documentation

- [Dataset contract](docs/DATA_CONTRACT.md)
- [Booking-time contract](docs/BOOKING_TIME_CONTRACT.md)
- [Updated full-cohort protocol](docs/BASELINE_PROTOCOL.md)
- [Verified results and predeclared NumPy MLP experiment](docs/EXPERIMENT_RESULTS.md)
