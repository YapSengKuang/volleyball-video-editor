import { config } from "@/lib/config";
import { requireJob } from "@/lib/jobs";
import { hitLimit } from "@/lib/limits";
import { signPart } from "@/lib/s3";
import { assertUuid, clientIp, errorResponse, HttpError, json, readJson } from "@/lib/http";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(req: Request, context: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await context.params;
    assertUuid(id);
    const job = await requireJob(id);
    if (job.status !== "uploading" || !job.upload_id) {
      throw new HttpError("This upload is not waiting for file parts.", 409);
    }
    const body = (await readJson(req, 16 * 1024)) as { partNumbers?: unknown };
    if (!Array.isArray(body.partNumbers) || body.partNumbers.length === 0) {
      throw new HttpError("Choose which parts to sign.", 400);
    }
    if (body.partNumbers.length > config.maxPartsPerSign) {
      throw new HttpError("Ask for fewer parts at a time.", 400);
    }
    const partCount = Math.ceil(Number(job.size_bytes) / config.partSize);
    const partNumbers = body.partNumbers.map((value) => Number(value));
    if (partNumbers.some((part) => !Number.isInteger(part) || part < 1 || part > partCount)) {
      throw new HttpError("A part number is out of range.", 400);
    }
    const ip = clientIp(req);
    if (await hitLimit(`ip:${ip}`, "presign", 10 * 60, config.presignsPerIpPer10Min)) {
      throw new HttpError("Too many upload requests. Wait a few minutes and try again.", 429, 600);
    }
    const urls = await Promise.all(
      partNumbers.map(async (partNumber) => ({
        partNumber,
        url: await signPart(job.source_key, job.upload_id as string, partNumber),
      })),
    );
    return json({ urls });
  } catch (error) {
    return errorResponse(error);
  }
}
