import { config } from "@/lib/config";
import { json } from "@/lib/http";

export const dynamic = "force-dynamic";

export function GET() {
  return json({
    maxUploadBytes: config.maxUploadBytes,
    maxDurationSeconds: config.maxDurationSeconds,
    jobsPerIpPerHour: config.jobsPerIpPerHour,
    retentionHours: config.retentionHours,
    extensions: ["mp4", "mov", "mkv"],
  });
}
