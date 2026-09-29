import { config } from "@/lib/config";
import { getSegments, normalizeSegments, replaceSegments, requireJob } from "@/lib/jobs";
import { hitLimit } from "@/lib/limits";
import { assertUuid, errorResponse, HttpError, json, readJson } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function PUT(req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status !== "ready" && job.status !== "done") {
      throw new HttpError("The timeline can be edited once rallies are ready, and not during export.", 409);
    }
    if (job.duration_seconds == null) throw new HttpError("This game has no duration yet.", 409);
    if (await hitLimit(`job:${id}`, "segment_write", 60, config.segmentWritesPerMinute)) {
      throw new HttpError("Too many timeline edits. Wait a minute.", 429, 60);
    }
    const body = (await readJson(req, config.maxSegmentBodyBytes)) as { segments?: unknown };
    const segments = normalizeSegments(body.segments, Number(job.duration_seconds));
    const saved = await replaceSegments(id, segments);
    return json({ segments: saved });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    await requireJob(id);
    return json({ segments: await getSegments(id) });
  } catch (error) {
    return errorResponse(error);
  }
}
