import { config } from "@/lib/config";
import { requireJob } from "@/lib/jobs";
import { signGet } from "@/lib/s3";
import { assertUuid, errorResponse, HttpError, json } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(_req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status === "uploading" || job.status === "failed") {
      throw new HttpError("This game is not ready to play.", 409);
    }
    const url = await signGet(job.source_key, config.playbackUrlSeconds);
    return json({ url });
  } catch (error) {
    return errorResponse(error);
  }
}
