import { markCalibrating, markFailed, requireJob } from "@/lib/jobs";
import { completeUpload, deleteObject, objectSize } from "@/lib/s3";
import { assertUuid, errorResponse, HttpError, json, readJson } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status !== "uploading" || !job.upload_id) {
      throw new HttpError("This upload is already finished.", 409);
    }
    const body = (await readJson(req, 256 * 1024)) as {
      parts?: { partNumber?: unknown; etag?: unknown }[];
    };
    if (!Array.isArray(body.parts) || body.parts.length === 0) {
      throw new HttpError("The upload is missing its parts.", 400);
    }
    const parts = body.parts.map((part) => ({
      PartNumber: Number(part.partNumber),
      ETag: typeof part.etag === "string" ? part.etag : "",
    }));
    if (parts.some((part) => !Number.isInteger(part.PartNumber) || !part.ETag)) {
      throw new HttpError("A file part is missing its id.", 400);
    }
    parts.sort((a, b) => a.PartNumber - b.PartNumber);
    await completeUpload(job.source_key, job.upload_id, parts);
    const actual = await objectSize(job.source_key);
    if (actual !== Number(job.size_bytes)) {
      await deleteObject(job.source_key);
      await markFailed(id, "The uploaded file size did not match. It was deleted.");
      throw new HttpError("The uploaded file size did not match. It was deleted.", 400);
    }
    const queued = await markCalibrating(id);
    if (!queued) throw new HttpError("This upload is already finished.", 409);
    return json({ id, status: "calibrating" });
  } catch (error) {
    return errorResponse(error);
  }
}
