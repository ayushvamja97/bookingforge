import { NextResponse } from "next/server";
import { parseBooking, parseModelResponse, type Variant } from "@/lib/research";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const API = process.env.BOOKINGFORGE_API_URL || "http://127.0.0.1:8000";
const VARIANTS: Variant[] = ["with_year", "without_year"];

export async function POST(request: Request) {
  const contentLength = Number(request.headers.get("content-length") || 0);
  if (contentLength > 8192) return NextResponse.json({ error: "Request is too large." }, { status: 413 });
  let body: unknown;
  try {
    const raw = await request.text();
    if (raw.length > 8192) return NextResponse.json({ error: "Request is too large." }, { status: 413 });
    body = JSON.parse(raw);
  } catch {
    return NextResponse.json({ error: "Enter a valid synthetic booking." }, { status: 400 });
  }
  if (body === null || typeof body !== "object" || Array.isArray(body) ||
      Object.keys(body).length !== 1 || !Object.hasOwn(body, "booking")) {
    return NextResponse.json({ error: "Expected only a booking object." }, { status: 400 });
  }
  const booking = parseBooking((body as { booking?: unknown }).booking);
  if (!booking) return NextResponse.json({ error: "Use valid, historical synthetic booking fields only." }, { status: 422 });

  try {
    const results = await Promise.all(VARIANTS.map(async (variant) => {
      const response = await fetch(`${API}/v1/research/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ variant, booking }),
        cache: "no-store",
        signal: AbortSignal.timeout(15000),
      });
      if (!response.ok) throw new Error(`The model service returned ${response.status}`);
      const data: unknown = await response.json();
      const parsed = parseModelResponse(data, variant);
      if (!parsed) throw new Error("The model service returned an unexpected format");
      return parsed;
    }));
    return NextResponse.json({
      models: { with_year: results[0], without_year: results[1] },
      research_only: true,
    }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({
      error: "The local model service is unavailable. Start FastAPI on port 8000, then retry.",
    }, { status: 502, headers: { "Cache-Control": "no-store" } });
  }
}
