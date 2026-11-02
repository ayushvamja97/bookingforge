import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const API = process.env.BOOKINGFORGE_API_URL || "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${API}/health`, {
      cache: "no-store", signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error("Backend unhealthy");
    const data: unknown = await response.json();
    if (data === null || typeof data !== "object" || Array.isArray(data)) throw new Error("Invalid backend response");
    const health = data as Record<string, unknown>;
    if (health.status !== "ready" || health.research_only !== true ||
        !Array.isArray(health.loaded_variants) ||
        !health.loaded_variants.includes("with_year") ||
        !health.loaded_variants.includes("without_year")) throw new Error("Models not ready");
    return NextResponse.json({ connected: true, models: 2 }, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return NextResponse.json({ connected: false, models: 0 }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
