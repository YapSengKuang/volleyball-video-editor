import { queueWithCourt, requireJob } from "@/lib/jobs";
import { assertUuid, errorResponse, HttpError, json, readJson } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    await requireJob(id);
    const body = (await readJson(req, 8 * 1024)) as { corners?: unknown };
    if (!Array.isArray(body.corners) || body.corners.length !== 4) {
      throw new HttpError("Click all four corners of your court.", 400);
    }
    const corners = body.corners.map((point) => {
      if (!point || typeof point !== "object") throw new HttpError("A corner is invalid.", 400);
      const row = point as { x?: unknown; y?: unknown };
      const x = Number(row.x);
      const y = Number(row.y);
      if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || x > 1 || y < 0 || y > 1) {
        throw new HttpError("A corner is outside the picture.", 400);
      }
      return { x, y };
    });
    const queued = await queueWithCourt(id, corners);
    if (!queued) throw new HttpError("Rallies can be found again once this step is finished.", 409);
    return json({ id, status: "queued" });
  } catch (error) {
    return errorResponse(error);
  }
}
