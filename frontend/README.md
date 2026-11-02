# BookingForge · Research Dashboard

A polished **local-only Next.js 16 / React 19 / TypeScript** frontend for the existing BookingForge NumPy-MLP research API. No live guest data, no training in the browser, no claimed booking-time risk probability.

## Start both services on Windows

Terminal 1 (existing FastAPI):

```powershell
cd "C:\Users\Ayush\bookingforge"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
.\.venv\Scripts\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Next.js):

```powershell
cd "C:\Users\Ayush\bookingforge\frontend"
npm install
npm run test
npm run typecheck
npm run dev
```

Open **http://127.0.0.1:3000**. Click **Compare research models** to receive two scores from the trained NumPy networks. The UI uses an internal **same-origin Next.js route** to proxy requests to localhost FastAPI; no browser CORS changes are needed. Click the status pill to retry the API health check.

The dashboard includes a synthetic booking form, comparison of both model variants, historical benchmark metrics, architecture explanation, accessible input validation, and raw score export via clipboard. A model with no arrival year still requires a valid historical arrival date as a demo input constraint.

**Warnings:** Scores are retrospective experimental outputs, **not operational cancellation probabilities**. The interface accepts only fabricated hotel reservations dated July 2015 through December 2016. Do not enter actual guest data. Do not bind either service to a public host. This app is unauthenticated.

To build for distribution: `npm run build`. A separate Docker Compose configuration from the project root can run both services with localhost-only published ports, with existing research weights mounted read-only.

The frontend uses **Node.js 22+** and Next.js 16.4.3. Unit tests use the Node 22 type-stripping test runner to exercise the shared request/response guards.
