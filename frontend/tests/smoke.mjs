/** End-to-end smoke test. Requires FastAPI :8000 AND Next.js :3000 running. */
import assert from "node:assert/strict";

const frontend = process.env.BOOKINGFORGE_DASHBOARD_URL ?? "http://127.0.0.1:3000";
const backend = process.env.BOOKINGFORGE_API_URL ?? "http://127.0.0.1:8000";

async function request(url, options, failureHint) {
  try {
    return await fetch(url, options);
  } catch (error) {
    throw new Error(`${failureHint}\nURL: ${url}\nCause: ${error?.message ?? String(error)}`, { cause: error });
  }
}

function healthReady(data) {
  return data?.status === "ready" &&
    data?.research_only === true &&
    Array.isArray(data.loaded_variants) &&
    data.loaded_variants.includes("with_year") &&
    data.loaded_variants.includes("without_year");
}

console.log("Checking FastAPI backend and saved models...");
const apiHealthUrl = new URL("/health", backend);
const apiHealth = await request(
  apiHealthUrl,
  { signal: AbortSignal.timeout(10000) },
  "FastAPI is not reachable. In a SEPARATE PowerShell window run from the repository root:\n" +
  '  $env:PYTHONPATH = "C:\\Users\\Ayush\\bookingforge\\src"\n' +
  '  .\\.venv\\Scripts\\python.exe -m uvicorn bookingforge.api:app --host 127.0.0.1 --port 8000\n' +
  "Keep the server window running, then retry npm run smoke."
);
if (!apiHealth.ok) {
  throw new Error(`FastAPI /health responded with HTTP ${apiHealth.status}; check the FastAPI terminal for startup errors.`);
}
let apiStatus;
try {
  apiStatus = await apiHealth.json();
} catch {
  throw new Error("FastAPI /health did not return JSON.");
}
assert.ok(healthReady(apiStatus),
  "FastAPI health response does not confirm both saved models. Check models/arrival_v1 and restart FastAPI.");
console.log("FastAPI OK: both saved models loaded.");

console.log("Checking Next.js proxy...");
const uiHealthUrl = new URL("/api/research/status", frontend);
const uiHealth = await request(
  uiHealthUrl,
  { signal: AbortSignal.timeout(10000) },
  "Next.js is not reachable. In another PowerShell window run:\n" +
  '  cd "C:\\Users\\Ayush\\bookingforge\\frontend"\n' +
  "  npm run dev\n" +
  "Keep both servers running, then retry npm run smoke."
);
if (!uiHealth.ok) {
  const responseText = (await uiHealth.text()).slice(0, 500);
  throw new Error(
    `Next.js /api/research/status returned HTTP ${uiHealth.status}, although FastAPI is reachable.\n` +
    "Check frontend/.env.local BOOKINGFORGE_API_URL (local default: http://127.0.0.1:8000).\n" +
    "Restart npm run dev after changing any environment file.\n" +
    `Response: ${responseText}`
  );
}
const status = await uiHealth.json();
assert.deepEqual(status, { connected: true, models: 2 });

const booking = {
  hotel: "City Hotel",
  lead_time: 72,
  arrival_date_year: 2016,
  arrival_date_month: "November",
  arrival_date_day_of_month: 12,
  stays_in_weekend_nights: 1,
  stays_in_week_nights: 2,
  adults: 2,
  children: 0,
  babies: 0,
  reserved_room_type: "A",
};
const compareUrl = new URL("/api/research/compare", frontend);
const response = await request(compareUrl, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ booking }),
  signal: AbortSignal.timeout(30000),
}, "Next.js comparison request failed. Check both server terminal windows.");
if (!response.ok) {
  throw new Error(
    `Next.js /api/research/compare returned HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`
  );
}
const data = await response.json();
assert.equal(data.research_only, true);
for (const variant of ["with_year", "without_year"]) {
  const model = data.models?.[variant];
  assert.equal(model?.variant, variant);
  assert.equal(model?.research_only, true);
  assert.equal(model?.calibrated_for_booking_time, false);
  assert.ok(model.research_cancellation_score >= 0 && model.research_cancellation_score <= 1);
  console.log(`${variant}: raw research score = ${model.research_cancellation_score.toFixed(6)}`);
}
console.log("BookingForge local Next.js → FastAPI → saved NumPy models: PASS");
