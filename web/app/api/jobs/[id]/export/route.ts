import { config } from "@/lib/config";
import { claimExport, getSegments, requireJob } from "@/lib/jobs";
import { hitLimit, peekLimit } from "@/lib/limits";
import { assertUuid, errorResponse, HttpError, json } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status === "export_queued" || job.status === "exporting") {
      throw new HttpError("This game is already being exported.", 409);
    }
    if (job.status !== "ready" && job.status !== "done") {
      throw new HttpError("Find the rallies before exporting.", 409);
    }
    const segments = await getSegments(id);
    if (!segments.some((segment) => segment.keep)) {
      throw new HttpError("Mark at least one stretch of play to keep.", 400);
    }
    const used = await peekLimit(`job:${id}`, "export", 60 * 60);
    if (used >= config.exportsPerJobPerHour) {
      throw new HttpError("This game was already exported twice this hour.", 429, 3600);
    }
    const claimed = await claimExport(id);
    if (!claimed) throw new HttpError("This game is already being exported.", 409);
    await hitLimit(`job:${id}`, "export", 60 * 60, config.exportsPerJobPerHour);
    return json({ id, status: "export_queued" });
  } catch (error) {
    return errorResponse(error);
  }
}
