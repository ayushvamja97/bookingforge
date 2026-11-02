/** Shared, dependency-free validation contract for the synthetic local demo. */
export type Variant = "with_year" | "without_year";

export interface ResearchBooking {
  hotel: "City Hotel" | "Resort Hotel";
  lead_time: number;
  arrival_date_year: 2015 | 2016;
  arrival_date_month: string;
  arrival_date_day_of_month: number;
  stays_in_weekend_nights: number;
  stays_in_week_nights: number;
  adults: number;
  children: number;
  babies: number;
  reserved_room_type: string;
}

export interface ModelResponse {
  research_cancellation_score: number;
  variant: Variant;
  arrival_year_used_by_model: boolean;
  research_only: true;
  calibrated_for_booking_time: false;
  warning: string;
}

export const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
] as const;
export const ROOM_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"] as const;

export const DEMO_BOOKING: ResearchBooking = {
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

const KEYS = Object.keys(DEMO_BOOKING);
const countFields = [
  "lead_time", "arrival_date_day_of_month", "stays_in_weekend_nights",
  "stays_in_week_nights", "adults", "children", "babies",
] as const;

export function parseBooking(value: unknown): ResearchBooking | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Record<string, unknown>;
  if (Object.keys(row).length !== KEYS.length ||
      Object.keys(row).some((key) => !KEYS.includes(key))) return null;
  if (row.hotel !== "City Hotel" && row.hotel !== "Resort Hotel") return null;
  if (row.arrival_date_year !== 2015 && row.arrival_date_year !== 2016) return null;
  if (typeof row.arrival_date_month !== "string" || !MONTHS.includes(row.arrival_date_month as typeof MONTHS[number])) return null;
  if (typeof row.reserved_room_type !== "string" || !ROOM_TYPES.includes(row.reserved_room_type as typeof ROOM_TYPES[number])) return null;
  for (const key of countFields) {
    const v = row[key];
    const max = key === "lead_time" ? 10000 : key === "arrival_date_day_of_month" ? 31 : 1000;
    if (typeof v !== "number" || !Number.isSafeInteger(v) || v < (key === "arrival_date_day_of_month" ? 1 : 0) || v > max) return null;
  }
  const month = MONTHS.indexOf(row.arrival_date_month as typeof MONTHS[number]);
  const year = row.arrival_date_year as number;
  const day = row.arrival_date_day_of_month as number;
  const date = new Date(Date.UTC(year, month, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month || date.getUTCDate() !== day) return null;
  if (date.getTime() < Date.UTC(2015, 6, 1) || date.getTime() > Date.UTC(2016, 11, 31)) return null;
  return row as unknown as ResearchBooking;
}

export function parseModelResponse(value: unknown, variant: Variant): ModelResponse | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  if (r.variant !== variant ||
      r.arrival_year_used_by_model !== (variant === "with_year") ||
      r.research_only !== true ||
      r.calibrated_for_booking_time !== false ||
      typeof r.warning !== "string" || !r.warning ||
      typeof r.research_cancellation_score !== "number" ||
      !Number.isFinite(r.research_cancellation_score) ||
      r.research_cancellation_score < 0 || r.research_cancellation_score > 1) return null;
  return r as unknown as ModelResponse;
}

export function formatScore(raw: number) {
  return `${(raw * 100).toFixed(2)}%`;
}

export const RESEARCH_WARNING =
  "Historical retrospective model score only. Not a validated or calibrated probability at booking creation; never use for real hotel decisions.";
