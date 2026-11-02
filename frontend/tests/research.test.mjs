import assert from "node:assert/strict";
import test from "node:test";
import { DEMO_BOOKING, parseBooking, parseModelResponse, formatScore } from "../lib/research.ts";

test("valid synthetic historical booking is accepted without mutation", () => {
  const before = structuredClone(DEMO_BOOKING);
  assert.deepEqual(parseBooking(DEMO_BOOKING), before);
  assert.deepEqual(DEMO_BOOKING, before);
});

test("reject outcome fields, unknown fields and absent data", () => {
  assert.equal(parseBooking({ ...DEMO_BOOKING, is_canceled: 1 }), null);
  assert.equal(parseBooking({ ...DEMO_BOOKING, reservation_status_date: "2016-11-13" }), null);
  const { lead_time, ...withoutLead } = DEMO_BOOKING;
  assert.equal(parseBooking(withoutLead), null);
  assert.equal(parseBooking([]), null);
});

test("reject future dates, non-calendar dates, non-finite and invalid quantities", () => {
  for (const b of [
    { ...DEMO_BOOKING, arrival_date_year: 2017 },
    { ...DEMO_BOOKING, arrival_date_year: 2016, arrival_date_month: "February", arrival_date_day_of_month: 30 },
    { ...DEMO_BOOKING, arrival_date_year: 2015, arrival_date_month: "January", arrival_date_day_of_month: 5 },
    { ...DEMO_BOOKING, lead_time: Number.NaN },
    { ...DEMO_BOOKING, lead_time: -1 },
    { ...DEMO_BOOKING, children: 1.5 },
    { ...DEMO_BOOKING, hotel: "Actual Hotel" },
  ]) assert.equal(parseBooking(b), null);
});

test("only valid research-model responses are accepted", () => {
  const valid = {
    research_cancellation_score: 0.2177707187048443,
    variant: "without_year",
    arrival_year_used_by_model: false,
    research_only: true,
    calibrated_for_booking_time: false,
    warning: "Research-only model",
  };
  assert.deepEqual(parseModelResponse(valid, "without_year"), valid);
  assert.equal(parseModelResponse({ ...valid, research_cancellation_score: 2 }, "without_year"), null);
  assert.equal(parseModelResponse({ ...valid, calibrated_for_booking_time: true }, "without_year"), null);
  assert.equal(parseModelResponse(valid, "with_year"), null);
  assert.equal(formatScore(valid.research_cancellation_score), "21.78%");
});
