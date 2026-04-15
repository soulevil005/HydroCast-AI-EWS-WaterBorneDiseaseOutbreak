import { NextResponse } from "next/server";
import { getShapPayload } from "../../../../lib/server/dashboard-data";

export async function GET() {
  const payload = await getShapPayload();
  return NextResponse.json(payload, {
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
