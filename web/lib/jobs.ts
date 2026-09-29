import { randomUUID } from "crypto";
import { config } from "./config";
import { pool } from "./db";
import { HttpError } from "./http";
import { signGet } from "./s3";

export type JobStatus =
  | "uploading"
  | "queued"
  | "analyzing"
  | "ready"
  | "export_queued"
  | "exporting"
  | "done"
  | "failed";

export type SegmentRow = {
  id: string;
  start: number;
  end: number;
  keep: boolean;
  origin: "auto" | "user";
};

type JobRecord = {
  id: string;
  status: JobStatus;
  filename: string;
  content_type: string;
  size_bytes: string;
  output_bytes: string | null;
  source_key: string;
  output_key: string | null;
  merged_key: string | null;
  court_corners: string | null;
  upload_id: string | null;
  duration_seconds: number | null;
  progress: number;
  phase: string | null;
  error: string | null;
  warning: string | null;
  eta_seconds: number | null;
  created_at: Date;
  expires_at: Date | null;
};

const BUSY = new Set<JobStatus>(["queued", "analyzing", "export_queued", "exporting"]);

export function isBusy(status: JobStatus): boolean {
  return BUSY.has(status);
}

function mapSegment(row: {
  id: string;
  start_seconds: number;
  end_seconds: number;
  keep: boolean;
  origin: "auto" | "user";
}): SegmentRow {
  return {
    id: row.id,
    start: Number(row.start_seconds),
    end: Number(row.end_seconds),
    keep: row.keep,
    origin: row.origin,
  };
}

export async function getJob(id: string): Promise<JobRecord | null> {
  const result = await pool.query<JobRecord>("SELECT * FROM jobs WHERE id = $1", [id]);
  return result.rows[0] ?? null;
}

export async function requireJob(id: string): Promise<JobRecord> {
  const job = await getJob(id);
  if (!job) throw new HttpError("Unknown game.", 404);
  return job;
}

export async function getSegments(jobId: string): Promise<SegmentRow[]> {
  const result = await pool.query<{
    id: string;
    start_seconds: number;
    end_seconds: number;
    keep: boolean;
    origin: "auto" | "user";
  }>(
    `SELECT id, start_seconds, end_seconds, keep, origin
     FROM segments WHERE job_id = $1 ORDER BY position`,
    [jobId],
  );
  return result.rows.map(mapSegment);
}

export async function usedBytes(): Promise<number> {
  const result = await pool.query<{ used: string }>(
    `SELECT (COALESCE(SUM(size_bytes), 0) + COALESCE(SUM(output_bytes), 0) + COALESCE(SUM(merged_bytes), 0))::bigint AS used
     FROM jobs WHERE status <> 'failed'`,
  );
  return Number(result.rows[0]?.used ?? 0);
}

export async function hasInflightUpload(ip: string): Promise<boolean> {
  const result = await pool.query(
    "SELECT 1 FROM jobs WHERE client_ip = $1 AND status = 'uploading' LIMIT 1",
    [ip],
  );
  return (result.rowCount ?? 0) > 0;
}

export async function insertUploadingJob(input: {
  id: string;
  filename: string;
  contentType: string;
  sizeBytes: number;
  sourceKey: string;
  clientIp: string;
}): Promise<void> {
  try {
    await pool.query(
      `INSERT INTO jobs (
         id, status, filename, content_type, size_bytes, source_key, client_ip, phase, progress
       ) VALUES ($1, 'uploading', $2, $3, $4, $5, $6, 'Uploading', 0)`,
      [input.id, input.filename, input.contentType, input.sizeBytes, input.sourceKey, input.clientIp],
    );
  } catch (error) {
    if (typeof error === "object" && error && "code" in error && error.code === "23505") {
      throw new HttpError("Finish the upload already in progress before starting another.", 429, 60);
    }
    throw error;
  }
}

export async function setUploadId(id: string, uploadId: string): Promise<void> {
  await pool.query("UPDATE jobs SET upload_id = $2, updated_at = now() WHERE id = $1", [id, uploadId]);
}

export async function deleteJob(id: string): Promise<void> {
  await pool.query("DELETE FROM jobs WHERE id = $1", [id]);
}

export async function markCalibrating(id: string): Promise<boolean> {
  const result = await pool.query(
    `UPDATE jobs
     SET status = 'calibrating', upload_id = NULL, phase = 'Mark your court', progress = 0, updated_at = now()
     WHERE id = $1 AND status = 'uploading'`,
    [id],
  );
  return (result.rowCount ?? 0) > 0;
}

export async function queueWithCourt(
  id: string,
  corners: { x: number; y: number }[],
): Promise<boolean> {
  const result = await pool.query(
    `UPDATE jobs
     SET court_corners = $2,
         status = 'queued',
         phase = 'Waiting to analyze',
         progress = 0,
         error = NULL,
         updated_at = now()
     WHERE id = $1 AND status IN ('calibrating', 'ready', 'done')`,
    [id, JSON.stringify(corners)],
  );
  return (result.rowCount ?? 0) > 0;
}

export async function markFailed(id: string, message: string): Promise<void> {
  await pool.query(
    `UPDATE jobs
     SET status = 'failed', error = $2, phase = 'Failed', size_bytes = 0,
         progress = 0, updated_at = now(), finished_at = now(),
         expires_at = now() + interval '1 hour'
     WHERE id = $1`,
    [id, message.slice(0, 500)],
  );
}

export async function replaceSegments(
  jobId: string,
  segments: { start: number; end: number; keep: boolean }[],
): Promise<SegmentRow[]> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("DELETE FROM segments WHERE job_id = $1", [jobId]);
    for (let position = 0; position < segments.length; position += 1) {
      const segment = segments[position];
      await client.query(
        `INSERT INTO segments (id, job_id, start_seconds, end_seconds, keep, origin, position)
         VALUES ($1, $2, $3, $4, $5, 'user', $6)`,
        [randomUUID(), jobId, segment.start, segment.end, segment.keep, position],
      );
    }
    await client.query(
      `INSERT INTO corrections (id, job_id, segments)
       VALUES ($1, $2, $3::jsonb)`,
      [randomUUID(), jobId, JSON.stringify(segments)],
    );
    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
  return getSegments(jobId);
}

export async function claimExport(id: string): Promise<boolean> {
  const result = await pool.query(
    `UPDATE jobs
     SET status = 'export_queued', progress = 0, phase = 'Waiting to export', error = NULL, updated_at = now()
     WHERE id = $1 AND status IN ('ready', 'done')`,
    [id],
  );
  return (result.rowCount ?? 0) > 0;
}

export async function publicJob(job: JobRecord, segments: SegmentRow[]) {
  const playable = job.status !== "uploading" && job.status !== "failed";
  let playbackUrl: string | null = null;
  if (playable) {
    playbackUrl = await signGet(job.source_key, config.playbackUrlSeconds);
  }
  return {
    job: {
      id: job.id,
      status: job.status,
      filename: job.filename,
      sizeBytes: Number(job.size_bytes),
      outputBytes: job.output_bytes == null ? null : Number(job.output_bytes),
      durationSeconds: job.duration_seconds == null ? null : Number(job.duration_seconds),
      progress: Number(job.progress),
      phase: job.phase,
      error: job.error,
      warning: job.warning,
      etaSeconds: job.eta_seconds == null ? null : Number(job.eta_seconds),
      createdAt: job.created_at,
      expiresAt: job.expires_at,
      playbackUrl,
      hasFullGame: Boolean(job.merged_key),
      courtCorners: parseCorners(job.court_corners),
    },
    segments,
  };
}

function parseCorners(raw: string | null): { x: number; y: number }[] | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as { x?: unknown; y?: unknown }[];
    if (!Array.isArray(value) || value.length !== 4) return null;
    return value.map((point) => ({ x: Number(point.x), y: Number(point.y) }));
  } catch {
    return null;
  }
}

export function normalizeSegments(
  input: unknown,
  duration: number,
): { start: number; end: number; keep: boolean }[] {
  if (!Array.isArray(input) || input.length > config.maxSegments) {
    throw new HttpError("The clip list is invalid.", 400);
  }
  if (input.length === 0) return [];
  const parsed = input.map((item) => {
    if (!item || typeof item !== "object") throw new HttpError("A clip is invalid.", 400);
    const row = item as { start?: unknown; end?: unknown; keep?: unknown };
    const start = Number(row.start);
    const end = Number(row.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || typeof row.keep !== "boolean") {
      throw new HttpError("A clip is invalid.", 400);
    }
    return { start: Math.max(0, start), end: Math.min(duration, end), keep: row.keep };
  });
  parsed.sort((a, b) => a.start - b.start);
  for (let index = 0; index < parsed.length; index += 1) {
    const clip = parsed[index];
    if (!(clip.end > clip.start + 0.2)) throw new HttpError("A clip is too short.", 400);
    if (clip.start < -0.05 || clip.end > duration + 0.05) {
      throw new HttpError("A clip sits outside the game.", 400);
    }
    if (index > 0 && clip.start < parsed[index - 1].end - 0.05) {
      throw new HttpError("Two clips overlap.", 400);
    }
  }
  return parsed;
}
