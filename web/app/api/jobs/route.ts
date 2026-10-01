import { randomUUID } from "crypto";
import { acceptedContentType, config, contentTypeFor, extensionOf, sanitizeFilename } from "@/lib/config";
import { hasInflightUpload, insertUploadingJob, setUploadId, deleteJob, usedBytes } from "@/lib/jobs";
import { hitLimit } from "@/lib/limits";
import { createUpload } from "@/lib/s3";
import { clientIp, errorResponse, HttpError, json, readJson } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const body = (await readJson(req, 16 * 1024)) as {
      filename?: unknown;
      size?: unknown;
      contentType?: unknown;
    };
    const filename = typeof body.filename === "string" ? sanitizeFilename(body.filename) : "";
    const size = Number(body.size);
    const contentType = typeof body.contentType === "string" ? body.contentType : "";
    if (!filename || !Number.isFinite(size) || size <= 0) {
      throw new HttpError("Choose a video file.", 400);
    }
    const ext = extensionOf(filename);
    if (!ext) throw new HttpError("Use an mp4 or mov file. PHP and JSP files are not accepted.", 400);
    if (!acceptedContentType(ext, contentType)) {
      throw new HttpError("That file type is not a video.", 400);
    }
    if (size > config.maxUploadBytes) {
      throw new HttpError("That file is over the 8 GB limit.", 413);
    }
    const ip = clientIp(req);
    if (await hasInflightUpload(ip)) {
      throw new HttpError("Finish the upload already in progress before starting another.", 429, 60);
    }
    if (await hitLimit(`ip:${ip}`, "create_job", 60 * 60, config.jobsPerIpPerHour)) {
      throw new HttpError("Too many new games from this network. Try again in about an hour.", 429, 3600);
    }
    const used = await usedBytes();
    if (used + size > config.storageCapBytes) {
      throw new HttpError(
        "Storage is full. Wait for older games to expire, 24 hours after they finish.",
        429,
        3600,
      );
    }

    const id = randomUUID();
    const sourceKey = `sources/${id}.${ext}`;
    const storedType = contentTypeFor(ext);
    await insertUploadingJob({
      id,
      filename,
      contentType: storedType,
      sizeBytes: size,
      sourceKey,
      clientIp: ip,
    });
    if (config.useBlob) {
      await setUploadId(id, "blob");
      return json({ id, mode: "blob" });
    }
    try {
      const uploadId = await createUpload(sourceKey, storedType);
      await setUploadId(id, uploadId);
      const partCount = Math.ceil(size / config.partSize);
      return json({ id, partSize: config.partSize, partCount });
    } catch (error) {
      await deleteJob(id);
      throw error;
    }
  } catch (error) {
    return errorResponse(error);
  }
}
