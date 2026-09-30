import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { config } from "@/lib/config";
import { attachBlob, markCalibrating, requireJob } from "@/lib/jobs";
import { HttpError } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json()) as HandleUploadBody;
  try {
    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        const payload = clientPayload ? (JSON.parse(clientPayload) as { jobId?: string }) : {};
        if (!payload.jobId) throw new HttpError("Missing game id.", 400);
        const job = await requireJob(payload.jobId);
        if (job.status !== "uploading" || job.upload_id !== "blob") {
          throw new HttpError("This upload is not waiting for a file.", 409);
        }
        if (!pathname.startsWith(`sources/${job.id}`)) {
          throw new HttpError("Unexpected upload path.", 400);
        }
        return {
          allowedContentTypes: [
            "video/mp4",
            "video/quicktime",
            "video/x-matroska",
            "application/octet-stream",
          ],
          maximumSizeInBytes: config.maxUploadBytes,
          addRandomSuffix: false,
          allowOverwrite: true,
          tokenPayload: JSON.stringify({ jobId: job.id }),
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        const payload = tokenPayload ? (JSON.parse(tokenPayload) as { jobId?: string }) : {};
        if (!payload.jobId || !blob.url) return;
        await attachBlob(payload.jobId, blob.url);
        await markCalibrating(payload.jobId);
      },
    });
    return Response.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Upload failed.";
    const status = error instanceof HttpError ? error.status : 400;
    return Response.json({ error: message }, { status });
  }
}
