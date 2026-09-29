import { config } from "@/lib/config";
import { requireJob } from "@/lib/jobs";
import { signGet } from "@/lib/s3";
import { assertUuid, errorResponse, HttpError, json } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status !== "done" || !job.output_key) {
      throw new HttpError("The cut video is not ready yet.", 409);
    }
    const base = job.filename.replace(/\.[^.]+$/, "") || "game";
    const format = new URL(req.url).searchParams.get("format");
    if (format === "game") {
      if (!job.merged_key) {
        throw new HttpError("Export again to build the full game.", 409);
      }
      const url = await signGet(job.merged_key, config.downloadUrlSeconds, `${base}-full.mp4`, "video/mp4");
      return json({ url });
    }
    const url = await signGet(job.output_key, config.downloadUrlSeconds, `${base}-rallies.zip`, "application/zip");
    return json({ url });
  } catch (error) {
    return errorResponse(error);
  }
}
