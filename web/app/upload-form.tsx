"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { formatBytes, formatHours } from "@/lib/time";

type Limits = {
  maxUploadBytes: number;
  maxDurationSeconds: number;
  jobsPerIpPerHour: number;
  retentionHours: number;
};

const fallbackLimits: Limits = {
  maxUploadBytes: 8 * 1024 * 1024 * 1024,
  maxDurationSeconds: 2.5 * 60 * 60,
  jobsPerIpPerHour: 3,
  retentionHours: 24,
};

async function mapPool<T>(items: T[], limit: number, fn: (item: T) => Promise<void>) {
  let index = 0;
  async function worker() {
    while (index < items.length) {
      const current = items[index];
      index += 1;
      await fn(current);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
}

export function UploadForm() {
  const router = useRouter();
  const [limits, setLimits] = useState<Limits>(fallbackLimits);
  const [progress, setProgress] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetch("/api/limits")
      .then((res) => res.json())
      .then((data: Limits) => setLimits(data))
      .catch(() => undefined);
  }, []);

  async function onFile(file: File | null) {
    if (!file || progress !== null) return;
    setError(null);
    setMessage(file.size > 1024 ** 3 ? "Large game. Uploading and finding rallies will take a while." : null);
    setProgress(0);
    try {
      const created = await postJson("/api/jobs", {
        filename: file.name,
        size: file.size,
        contentType: file.type || "application/octet-stream",
      });
      const partSize = created.partSize as number;
      const partCount = created.partCount as number;
      const id = created.id as string;
      const etags: { partNumber: number; etag: string }[] = [];
      let uploaded = 0;
      const parts = Array.from({ length: partCount }, (_, index) => index + 1);
      while (parts.length) {
        const partNumbers = parts.splice(0, 8);
        const signed = await postJson(`/api/jobs/${id}/parts`, { partNumbers });
        const urls = signed.urls as { partNumber: number; url: string }[];
        await mapPool(urls, 3, async ({ partNumber, url }) => {
          const start = (partNumber - 1) * partSize;
          const slice = file.slice(start, Math.min(start + partSize, file.size), "application/octet-stream");
          let etag = "";
          let lastError = "Upload part failed.";
          for (let attempt = 0; attempt < 3 && !etag; attempt += 1) {
            const res = await fetch(url, { method: "PUT", body: slice });
            if (res.ok) {
              etag = res.headers.get("ETag") ?? res.headers.get("etag") ?? "";
              break;
            }
            lastError = `Part ${partNumber} was rejected.`;
          }
          if (!etag) throw new Error(lastError);
          etags.push({ partNumber, etag });
          uploaded += slice.size;
          setProgress(uploaded / file.size);
        });
      }
      etags.sort((a, b) => a.partNumber - b.partNumber);
      await postJson(`/api/jobs/${id}/complete`, { parts: etags });
      router.push(`/jobs/${id}`);
    } catch (err) {
      setProgress(null);
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  return (
    <section className="panel">
      <h1>Upload a game</h1>
      <p className="lede">
        The editor makes a small 480p preview, finds rallies from movement inside the court you mark, then cuts those
        times out of the original as separate clips or one highlight reel.
      </p>
      <p className="note">
        Club gyms are often noisy and may not use a whistle. After the upload you mark your four court corners so the
        courts beside you are left out. A rally is a few seconds of play that keeps moving, with a little time before
        the serve and after the ball dies.
      </p>
      <div className="limits" aria-label="Limits">
        <span>Up to {formatBytes(limits.maxUploadBytes)}</span>
        <span>Up to {formatHours(limits.maxDurationSeconds)}</span>
        <span>{limits.jobsPerIpPerHour} new games per hour from this network</span>
        <span>Deleted {limits.retentionHours} hours after processing finishes</span>
      </div>
      <label className="drop">
        <input
          type="file"
          accept=".mp4,.mov,.mkv,video/mp4,video/quicktime,video/x-matroska"
          disabled={progress !== null}
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            event.target.value = "";
            void onFile(file);
          }}
        />
        <strong>{progress === null ? "Choose an mp4, mov, or mkv" : "Uploading…"}</strong>
        <p>One game can be uploading at a time. You get one file per rally, not one long cut.</p>
      </label>
      {progress !== null && (
        <div className="bar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)} role="progressbar">
          <span style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
      )}
      {message && <p className="note">{message}</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

async function postJson(url: string, body: unknown): Promise<Record<string, unknown>> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await res.json().catch(() => ({}))) as { error?: string };
  if (!res.ok) throw new Error(data.error || "Request failed.");
  return data;
}
