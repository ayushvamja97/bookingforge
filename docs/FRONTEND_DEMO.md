# BookingForge · Full-Stack Research Dashboard

Next.js / React / TypeScript, paired with FastAPI and the original NumPy neural networks. Synthetic data only; retrospective research output, not a production hotel cancellation-risk tool.

## Start locally (Windows)

Terminal 1, FastAPI:

```powershell
cd "C:\Users\Ayush\bookingforge"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000
```

Terminal 2, Next.js (Node.js 22+):

```powershell
cd "C:\Users\Ayush\bookingforge\frontend"
npm install
npm run test
npm run typecheck
npm run build
npm run dev
```

Visit http://127.0.0.1:3000. The UI checks API health, accepts historical synthetic booking-like attributes, and sends them to Next.js's same-origin server endpoint. This endpoint validates and forwards the request to **both** fixed NumPy models loaded by FastAPI. It does not train or adjust weights.

## Optional local Docker Compose

Stop both servers, and ensure `models/arrival_v1` contains your **already trained** `.npz`/`.json` model pairs. At repo root:

```powershell
docker compose up --build
```

The model directory is mounted read-only. The browser and API ports are published to **127.0.0.1 only**; the containers communicate internally. Stop with Ctrl+C and `docker compose down`. Never expose this unauthenticated demo to a public address.

## Verification

The previous Python API milestone had 77 passing tests. `frontend/npm run test` checks the booking input and model response validation. `npm run typecheck` and `npm run build` verify Next.js. GitHub Actions provides Python and frontend jobs on each push/PR. Docker/Next builds require Node dependency access and local testing; the repository currently has no committed frontend npm lockfile.

## Statistical limitations

The 0.7300 retrospective ROC AUC and 0.2032 retrospective Brier refer to distinct original NumPy MLP configurations on 14,517 arrivals from Oct–Dec 2016. These metrics are not independent prospective booking-time performance. The historical hotel source has no immutable booking-creation snapshots; older exploratory audits already inspected nominal 2017 test outcomes. Use only fabricated demo data and no real guest bookings.

### Local full-stack smoke test

With the Next.js dashboard and FastAPI both running, open another PowerShell window:

```powershell
cd "C:\Users\Ayush\bookingforge\frontend"
npm run smoke
```

This sends a single **synthetic** booking through the Next.js proxy and both loaded NumPy models. It requires the local saved artifacts and services, so it is not an automatic GitHub Actions step.
