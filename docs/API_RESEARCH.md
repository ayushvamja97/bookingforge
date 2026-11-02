# BookingForge local-only research inference API

**Status: experimental demo; not approved for live hotel operations.**

The API loads the two existing, saved original NumPy neural-network
classifiers; it **never retrains on an HTTP request**. The supported
historical booking-like inputs are deliberately limited to synthetic
arrivals from **July 2015 through December 2016**. No guest PII,
cancellation labels, payment data or final-status dates are accepted.

There is **no authentication**, so this service MUST bind only to
`127.0.0.1`. Never use `--host 0.0.0.0`, expose a port, or deploy it
publicly without a separately designed and reviewed security layer.
Its outputs are **retrospective research scores**, not validated
probabilities at original reservation creation. The historical source
has no immutable original booking-time feature snapshots.

## One-time setup

Run from the repository root using Windows PowerShell:

```powershell
cd "C:\Users\Ayush\bookingforge"
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected test count: **77** (72 previous tests, not yet confirmed
in user logs, plus 5 new HTTP API tests). The pinned optional
dependencies for the API are FastAPI, Uvicorn and HTTPX. The test
suite imports FastAPI, so install these extras **before** running
all tests.

## Required model artifacts

The API refuses to start if the saved artifacts are missing,
tampered or incompatible. Do not commit weights or raw booking data.

Expected files, created by `bookingforge.model_artifact`:

- `models/arrival_v1/arrival_mlp_with_year.npz`
- `models/arrival_v1/arrival_mlp_with_year.json`
- `models/arrival_v1/arrival_mlp_without_year.npz`
- `models/arrival_v1/arrival_mlp_without_year.json`

If they do not exist, first run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m bookingforge.model_artifact --csv data/raw/hotels.csv --output-dir models/arrival_v1 --report reports/arrival_artifact_verification.json --variant both --epochs 30
```

**This model-export workflow was verified on the user's computer:
both versions trained on 58,792 historical records, and both had
exactly zero discrepancy between in-memory and saved/reloaded
model predictions over all 14,517 validation feature rows.**
This does not certify booking-time validity.

## Start the local server

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000
```

In another PowerShell terminal, view:

- `http://127.0.0.1:8000/health` — verifies both files were loaded
- `http://127.0.0.1:8000/v1/research/models` — lists strict model
  input features and which variant uses arrival year
- `http://127.0.0.1:8000/docs` — local interactive OpenAPI docs

## Example synthetic prediction request

In a separate PowerShell terminal, use only fabricated data:

```powershell
$body = @{
    variant = "without_year"
    booking = @{
        hotel = "City Hotel"
        lead_time = 72
        arrival_date_year = 2016
        arrival_date_month = "November"
        arrival_date_day_of_month = 12
        stays_in_weekend_nights = 1
        stays_in_week_nights = 2
        adults = 2
        children = 0
        babies = 0
        reserved_room_type = "A"
    }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/research/predict" -Method POST -ContentType "application/json" -Body $body
```

The response returns `research_cancellation_score` as a number
between 0 and 1, `variant`, whether the arrival-year column was
used by the algorithm, and explicit research-only / non-calibration
warnings. The value will depend on the already-trained model: no
example prediction result is fabricated here.

For `variant = "without_year"`, the date is still required as
**historical input context and a validity check**, even though
`arrival_date_year` is deliberately omitted from the encoded
predictors. For `variant = "with_year"`, that field is included.

Only known categorical fields and nonnegative integer quantities
are accepted. Unknown fields, including `is_canceled`,
`reservation_status`, and `reservation_status_date`, return
HTTP **422**. Invalid calendar dates and 2017/future arrivals
also return **422**.

To stop the server, press **Ctrl+C**.

## Scientific and security boundaries

- A working API does not establish clinical/statistical/operational
  trust in its results. It only demonstrates **real software
  integration with reproducible ML model artifacts**.
- Model selection and calibration must not be performed through repeated
  requests against already-inspected historical validation labels.
- The existing 2017 nominal test holdout is not pristine: legacy
  aggregate audits previously read its outcomes, although this API
  never reads outcome columns.
- No frontend, external access, authentication, tenant isolation,
  rate limiting, or production audit logging has been implemented.
- Do not submit a real guest's booking or personal data to this demo.

See [the experiment ledger](EXPERIMENT_RESULTS.md) for measured AUC,
average precision, Brier scores, calibration and robustness limits.
