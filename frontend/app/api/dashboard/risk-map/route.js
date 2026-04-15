import { NextResponse } from "next/server";
import { getRiskMapPayload } from "../../../../lib/server/dashboard-data";

export async function GET() {
  const payload = await getRiskMapPayload();
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
