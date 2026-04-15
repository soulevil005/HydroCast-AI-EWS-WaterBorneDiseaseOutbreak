import { NextResponse } from "next/server";
import { getSummaryPayload } from "../../../../lib/server/dashboard-data";

export async function GET() {
  const payload = await getSummaryPayload();
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
