function numberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

export const config = {
  databaseUrl:
    process.env.DATABASE_URL ??
    "postgres://volleyball:volleyball@localhost:5432/volleyball",
  s3Endpoint: process.env.S3_ENDPOINT ?? "http://localhost:9000",
  s3PublicEndpoint: process.env.S3_PUBLIC_ENDPOINT ?? "http://localhost:9000",
  s3Bucket: process.env.S3_BUCKET ?? "videos",
  s3AccessKey: process.env.S3_ACCESS_KEY ?? "minio",
  s3SecretKey: process.env.S3_SECRET_KEY ?? "minio-secret-key",
  s3Region: process.env.S3_REGION ?? "us-east-1",
  useBlob: Boolean(process.env.BLOB_READ_WRITE_TOKEN),
  maxUploadBytes: numberEnv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024 * 1024),
  maxDurationSeconds: numberEnv("MAX_DURATION_SECONDS", 2.5 * 60 * 60),
  storageCapBytes: numberEnv("STORAGE_CAP_BYTES", 30 * 1024 * 1024 * 1024),
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

export const allowedExtensions = ["mp4", "mov", "mkv"] as const;

export function extensionOf(filename: string): string | null {
  const base = filename.split(/[/\\]/).pop() ?? "";
  const match = /\.([a-z0-9]+)$/i.exec(base);
  if (!match) return null;
  const ext = match[1].toLowerCase();
  return allowedExtensions.includes(ext as (typeof allowedExtensions)[number]) ? ext : null;
}

export function contentTypeFor(ext: string): string {
  if (ext === "mov") return "video/quicktime";
  if (ext === "mkv") return "video/x-matroska";
  return "video/mp4";
}
