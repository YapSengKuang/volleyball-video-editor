import "server-only";

function numberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

const EIGHT_GB = 8 * 1024 * 1024 * 1024;
const TEN_GB = 10 * 1024 * 1024 * 1024;

function r2Endpoint(): string {
  const account = (process.env.R2_ACCOUNT_ID ?? "").trim();
  return account ? `https://${account}.r2.cloudflarestorage.com` : "";
}

const r2 = r2Endpoint();
const usingR2 = Boolean(r2) || (process.env.S3_ENDPOINT ?? "").includes("r2.cloudflarestorage.com");

export const config = {
  databaseUrl:
    process.env.DATABASE_URL ??
    "postgres://volleyball:volleyball@localhost:5432/volleyball",
  s3Endpoint: r2 || process.env.S3_ENDPOINT || "http://localhost:9000",
  s3PublicEndpoint: r2 || process.env.S3_PUBLIC_ENDPOINT || process.env.S3_ENDPOINT || "http://localhost:9000",
  s3Bucket: process.env.R2_BUCKET || process.env.S3_BUCKET || "videos",
  s3AccessKey: process.env.R2_ACCESS_KEY_ID || process.env.S3_ACCESS_KEY || "minio",
  s3SecretKey: process.env.R2_SECRET_ACCESS_KEY || process.env.S3_SECRET_KEY || "minio-secret-key",
  s3Region: usingR2 ? process.env.S3_REGION || "auto" : process.env.S3_REGION || "us-east-1",
  useBlob: Boolean(process.env.BLOB_READ_WRITE_TOKEN) && !usingR2,
  maxUploadBytes: Math.min(numberEnv("MAX_UPLOAD_BYTES", EIGHT_GB), EIGHT_GB),
  maxDurationSeconds: numberEnv("MAX_DURATION_SECONDS", 2.5 * 60 * 60),
  storageCapBytes: Math.min(numberEnv("STORAGE_CAP_BYTES", TEN_GB), TEN_GB),
  jobsPerIpPerHour: numberEnv("JOBS_PER_IP_PER_HOUR", 3),
  presignsPerIpPer10Min: numberEnv("PRESIGNS_PER_IP_PER_10_MIN", 30),
  segmentWritesPerMinute: numberEnv("SEGMENT_WRITES_PER_MINUTE", 60),
  exportsPerJobPerHour: numberEnv("EXPORTS_PER_JOB_PER_HOUR", 2),
  playbackUrlSeconds: numberEnv("PLAYBACK_URL_SECONDS", 15 * 60),
  downloadUrlSeconds: numberEnv("DOWNLOAD_URL_SECONDS", 60 * 60),
  partSize: numberEnv("PART_SIZE", 64 * 1024 * 1024),
  maxSegmentBodyBytes: numberEnv("MAX_SEGMENT_BODY_BYTES", 256 * 1024),
  retentionHours: numberEnv("RETENTION_HOURS", 24),
  maxPartsPerSign: 8,
  maxSegments: 2000,
};

export const allowedExtensions = ["mp4", "mov"] as const;

const blockedScript = /\.(php\d?|phtml|phar|jspx?|jsw|jsv)(\.|$)/i;

export function sanitizeFilename(filename: string): string {
  const base = filename.split(/[/\\]/).pop() ?? "";
  return base.replace(/[<>"'&]/g, "").replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 200);
}

export function extensionOf(filename: string): string | null {
  const base = sanitizeFilename(filename);
  if (!base || blockedScript.test(base)) return null;
  const match = /\.([a-z0-9]+)$/i.exec(base);
  if (!match) return null;
  const ext = match[1].toLowerCase();
  return allowedExtensions.includes(ext as (typeof allowedExtensions)[number]) ? ext : null;
}

export function contentTypeFor(ext: string): string {
  if (ext === "mov") return "video/quicktime";
  return "video/mp4";
}

export function acceptedContentType(ext: string, contentType: string): boolean {
  const normalized = contentType.toLowerCase().split(";")[0].trim();
  if (!normalized || normalized === "application/octet-stream") return true;
  if (ext === "mov") return normalized === "video/quicktime" || normalized === "video/mp4";
  return normalized === "video/mp4";
}
