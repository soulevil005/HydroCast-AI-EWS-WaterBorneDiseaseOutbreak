import { NextResponse } from "next/server";
import { getForecastPayload } from "../../../../lib/server/dashboard-data";

export async function GET() {
  const payload = await getForecastPayload();
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
