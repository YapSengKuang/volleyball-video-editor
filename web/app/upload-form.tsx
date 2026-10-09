"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { upload } from "@vercel/blob/client";
import { describeFailure, requestJson } from "@/lib/request";
import { formatBytes, formatHours, formatRemaining } from "@/lib/time";

type Limits = {
  maxUploadBytes: number;
  maxDurationSeconds: number;
  jobsPerIpPerHour: number;
  retentionHours: number;
  storageCapBytes: number;
};

const fallbackLimits: Limits = {
  maxUploadBytes: 8 * 1024 * 1024 * 1024,
  maxDurationSeconds: 2.5 * 60 * 60,
  jobsPerIpPerHour: 3,
  retentionHours: 24,
  storageCapBytes: 10 * 1024 * 1024 * 1024,
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
  const [uploadEta, setUploadEta] = useState<number | null>(null);
  const uploadStarted = useRef(0);
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
    const lower = file.name.toLowerCase();
    if (!/\.(mp4|mov)$/.test(lower) || /\.(php|jsp)/.test(lower)) {
      setError("Use an mp4 or mov file. PHP and JSP files are not accepted.");
      return;
    }
    if (file.size > limits.maxUploadBytes) {
      setError("That file is over the 8 GB limit.");
      return;
    }
    setError(null);
    setMessage(file.size > 1024 ** 3 ? "Large game. Uploading and finding rallies will take a while." : null);
    setProgress(0);
    setUploadEta(null);
    uploadStarted.current = performance.now();
    try {
      const created = await postJson("/api/jobs", {
        filename: file.name,
        size: file.size,
        contentType: file.type || "application/octet-stream",
      });
      const id = created.id as string;
      if (created.mode === "blob") {
        const ext = file.name.split(".").pop()?.toLowerCase() || "mp4";
        let blob: { url: string };
        try {
          blob = await upload(`sources/${id}.${ext}`, file, {
            access: "private",
            handleUploadUrl: "/api/blob",
            clientPayload: JSON.stringify({ jobId: id }),
            multipart: true,
            onUploadProgress: ({ percentage }) => {
              const ratio = percentage / 100;
              setProgress(ratio);
              const elapsed = (performance.now() - uploadStarted.current) / 1000;
              if (ratio > 0.05 && elapsed > 2) {
                setUploadEta((elapsed * (1 - ratio)) / ratio);
              }
            },
          });
        } catch (error) {
          throw new Error(describeFailure(error, "Upload the video to storage", "/api/blob"));
        }
        await postJson(`/api/jobs/${id}/complete`, { blobUrl: blob.url });
        router.push(`/jobs/${id}`);
        return;
      }
      const partSize = created.partSize as number;
      const partCount = created.partCount as number;
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
          let lastError = `Upload part ${partNumber} failed.`;
          let storageHost = url;
          try {
            storageHost = new URL(url).host;
          } catch {
            storageHost = url;
          }
          for (let attempt = 0; attempt < 3 && !etag; attempt += 1) {
            let res: Response;
            try {
              res = await fetch(url, { method: "PUT", body: slice });
            } catch (error) {
              lastError = describeFailure(error, `Upload part ${partNumber} to ${storageHost}`, url);
              continue;
            }
            if (res.ok) {
              etag = res.headers.get("ETag") ?? res.headers.get("etag") ?? "";
              if (!etag) lastError = `Upload part ${partNumber} to ${storageHost} returned ${res.status} without an ETag.`;
              break;
            }
            const body = await res.text().catch(() => "");
            lastError = `Upload part ${partNumber} to ${storageHost} returned ${res.status}: ${body.replace(/\s+/g, " ").slice(0, 160) || res.statusText}`;
          }
          if (!etag) throw new Error(lastError);
          etags.push({ partNumber, etag });
          uploaded += slice.size;
          const ratio = uploaded / file.size;
          setProgress(ratio);
          const elapsed = (performance.now() - uploadStarted.current) / 1000;
          if (ratio > 0.05 && elapsed > 2) {
            setUploadEta((elapsed * (1 - ratio)) / ratio);
          }
        });
      }
      etags.sort((a, b) => a.partNumber - b.partNumber);
      await postJson(`/api/jobs/${id}/complete`, { parts: etags });
      router.push(`/jobs/${id}`);
    } catch (err) {
      setProgress(null);
      setUploadEta(null);
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  return (
    <section className="panel">
      <h1>Upload a game</h1>
      <p className="lede">
        Click the four corners of your court. Movement outside that box is ignored, so a game in the background is
        left out. The cut is made from movement inside the court, with sound only as a hint.
      </p>
      <p className="note">
        Clips keep a couple of seconds before the serve and after the point. Trim one if it still runs long.
      </p>
      <div className="limits" aria-label="Limits">
        <span>Up to {formatBytes(limits.maxUploadBytes)}</span>
        <span>Up to {formatHours(limits.maxDurationSeconds)}</span>
        <span>{limits.jobsPerIpPerHour} new games per hour from this network</span>
        <span>{formatBytes(limits.storageCapBytes)} stored</span>
        <span>Deleted {limits.retentionHours} hours after processing finishes</span>
      </div>
      <label className="drop">
        <input
          type="file"
          accept=".mp4,.mov,video/mp4,video/quicktime"
          disabled={progress !== null}
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            event.target.value = "";
            void onFile(file);
          }}
        />
        <strong>{progress === null ? "Choose an mp4 or mov" : "Uploading…"}</strong>
        <p>One game can be uploading at a time. You get one file per rally, not one long cut.</p>
      </label>
      {progress !== null && (
        <>
          <div className="bar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)} role="progressbar">
            <span style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <p className="note">
            Uploading — {Math.round(progress * 100)}%
            {uploadEta != null ? ` · ${formatRemaining(uploadEta)}` : " · Estimating time…"}
          </p>
        </>
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
  return requestJson(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}
