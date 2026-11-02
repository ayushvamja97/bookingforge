/** End-to-end smoke test; requires local Next.js (3000) and FastAPI (8000). */
import assert from "node:assert/strict";

const base = process.env.BOOKINGFORGE_DASHBOARD_URL ?? "http://127.0.0.1:3000";
const health = await fetch(new URL("/api/research/status", base), { signal: AbortSignal.timeout(10000) });
assert.equal(health.status, 200, "Next.js must reach both local saved models");
const status = await health.json();
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
const response = await fetch(new URL("/api/research/compare", base), {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ booking }),
  signal: AbortSignal.timeout(30000),
});
assert.equal(response.status, 200, "Both saved research models must respond");
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
