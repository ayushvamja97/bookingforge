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

### Full-stack smoke test

When **both** local servers are running, open a third PowerShell terminal in `frontend` and execute `npm run smoke`. It checks the Next.js health proxy and sends one fabricated booking through **both** saved NumPy models. It is deliberately excluded from offline unit tests/CI, since it needs the local model weights and running services.

### Troubleshooting: smoke test reports HTTP 503

A 503 from `/api/research/status` means the Next.js proxy is running,
but could not confirm the FastAPI `/health` response and both saved
model variants. It is **not** a model-accuracy or training failure.

Keep **three PowerShell terminals** open at the same time:

1. **Terminal A — FastAPI from the repository root:**

   ```powershell
   cd "C:\Users\Ayush\bookingforge"
   $env:PYTHONPATH = (Join-Path (Get-Location) "src")
   .\.venv\Scripts\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000
   ```

2. **Terminal B — Next.js:**

   ```powershell
   cd "C:\Users\Ayush\bookingforge\frontend"
   npm run dev
   ```

3. **Terminal C — check the backend, the frontend proxy, then smoke:**

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health
   Invoke-RestMethod http://127.0.0.1:3000/api/research/status
   npm run smoke
   ```

First health response must show `status: ready` and both
`with_year` and `without_year`. Second must show
`connected: true` and `models: 2`.

If FastAPI responds directly but Next still returns 503, check
`frontend/.env.local` or `BOOKINGFORGE_API_URL`: the **local**
Next.js server should target `http://127.0.0.1:8000`. Docker
Compose instead sets `http://api:8000` inside its container.
After changing Next.js environment variables, restart `npm run dev`.

The smoke runner now checks FastAPI **before** Next.js and prints
specific connection instructions rather than a bare
`503 !== 200` assertion.
