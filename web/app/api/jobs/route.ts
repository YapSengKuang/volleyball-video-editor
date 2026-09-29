import { randomUUID } from "crypto";
import { config, contentTypeFor, extensionOf } from "@/lib/config";
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
    const filename = typeof body.filename === "string" ? body.filename.split(/[/\\]/).pop()?.slice(0, 200) : "";
    const size = Number(body.size);
    if (!filename || !Number.isFinite(size) || size <= 0) {
      throw new HttpError("Choose a video file.", 400);
    }
    const ext = extensionOf(filename);
    if (!ext) throw new HttpError("Use an mp4, mov, or mkv file.", 400);
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
    const sourceKey = `sources/${id}`;
    const contentType = contentTypeFor(ext);
    await insertUploadingJob({
      id,
      filename,
      contentType,
      sizeBytes: size,
      sourceKey,
      clientIp: ip,
    });
    try {
      const uploadId = await createUpload(sourceKey, contentType);
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
