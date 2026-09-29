"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { formatClock, formatRemaining } from "@/lib/time";
import { CourtPicker } from "./court-picker";

type Segment = {
  id: string;
  start: number;
  end: number;
  keep: boolean;
  origin: "auto" | "user";
};

type Job = {
  id: string;
  status: string;
  filename: string;
  durationSeconds: number | null;
  progress: number;
  phase: string | null;
  etaSeconds: number | null;
  error: string | null;
  warning: string | null;
  expiresAt: string | null;
  playbackUrl: string | null;
  hasFullGame: boolean;
  courtCorners: { x: number; y: number }[] | null;
};

const busy = new Set(["queued", "analyzing", "export_queued", "exporting"]);

export function JobView({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [pickingCourt, setPickingCourt] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const clockRef = useRef<HTMLSpanElement>(null);
  const timeRef = useRef(0);
  const segmentsRef = useRef(segments);
  const dirtyRef = useRef(false);
  const saveToken = useRef(0);
  const clipEnd = useRef<number | null>(null);
  const pendingSeek = useRef<number | null>(null);
  segmentsRef.current = segments;

  const applyPayload = useCallback((payload: { job: Job; segments: Segment[] }) => {
    setJob(payload.job);
    if (payload.job.playbackUrl) setPlaybackUrl(payload.job.playbackUrl);
    if (!dirtyRef.current) setSegments(payload.segments);
  }, []);

  const refresh = useCallback(async () => {
    const res = await fetch(`/api/jobs/${id}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not load this game.");
    applyPayload(data);
  }, [applyPayload, id]);

  useEffect(() => {
    let stop = false;
    refresh().catch((error: Error) => {
      if (!stop) setLoadError(error.message);
    });
    return () => {
      stop = true;
    };
  }, [refresh]);

  useEffect(() => {
    if (!job || !busy.has(job.status)) return;
    const timer = window.setTimeout(() => {
      refresh().catch((error: Error) => setLoadError(error.message));
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [job, refresh]);

  useEffect(() => {
    if (!job?.durationSeconds) return;
    let frame = 0;
    const tick = () => {
      const video = videoRef.current;
      if (video) {
        timeRef.current = video.currentTime;
        if (clockRef.current) clockRef.current.textContent = formatClock(video.currentTime);
        if (clipEnd.current != null && video.currentTime >= clipEnd.current - 0.05) {
          video.pause();
          clipEnd.current = null;
        }
      }
      frame = requestAnimationFrame(tick);
      return;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [job?.durationSeconds]);

  useEffect(() => {
    if (!playbackUrl) return;
    const timer = window.setTimeout(() => {
      pendingSeek.current = videoRef.current?.currentTime ?? 0;
      void fetch(`/api/jobs/${id}/playback`)
        .then((res) => res.json())
        .then((data: { url?: string }) => {
          if (data.url) setPlaybackUrl(data.url);
        })
        .catch(() => undefined);
    }, 10 * 60 * 1000);
    return () => window.clearTimeout(timer);
  }, [id, playbackUrl]);

  async function persist(next: Segment[]) {
    const problem = clipProblem(next, job?.durationSeconds ?? 0);
    if (problem) {
      setActionError(problem);
      setSaveState("error");
      return;
    }
    const token = ++saveToken.current;
    dirtyRef.current = true;
    setSaveState("saving");
    setActionError(null);
    setSegments(next);
    const res = await fetch(`/api/jobs/${id}/segments`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        segments: next.map((segment) => ({ start: segment.start, end: segment.end, keep: segment.keep })),
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (token !== saveToken.current) return;
    if (!res.ok) {
      setSaveState("error");
      setActionError(data.error || "Could not save the clips.");
      return;
    }
    dirtyRef.current = false;
    setSegments(data.segments);
    setSaveState("saved");
  }

  function playClip(clip: Segment) {
    const video = videoRef.current;
    if (!video) return;
    clipEnd.current = clip.end;
    video.currentTime = clip.start;
    void video.play();
  }

  function addClip() {
    const duration = job?.durationSeconds ?? 0;
    const start = Math.min(timeRef.current, Math.max(0, duration - 1));
    const end = Math.min(duration, start + 12);
    const clip: Segment = {
      id: crypto.randomUUID(),
      start,
      end,
      keep: true,
      origin: "user",
    };
    const next = [...segmentsRef.current, clip].sort((a, b) => a.start - b.start);
    void persist(next);
  }

  async function exportCut() {
    setActionError(null);
    const res = await fetch(`/api/jobs/${id}/export`, { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setActionError(data.error || "Export did not start.");
      return;
    }
    await refresh();
  }

  async function download(format: "clips" | "game") {
    const res = await fetch(`/api/jobs/${id}/download?format=${format}`);
    const data = await res.json();
    if (!res.ok) {
      setActionError(data.error || "Download is not ready.");
      return;
    }
    window.location.href = data.url;
  }

  if (loadError) return <p className="error">{loadError}</p>;
  if (!job) return <p className="note">Loading the game…</p>;
  if (job.status === "calibrating" || pickingCourt) {
    return (
      <section className="panel">
        <h1>{job.filename}</h1>
        <CourtPicker
          playbackUrl={playbackUrl}
          onSave={async (corners) => {
            const res = await fetch(`/api/jobs/${id}/court`, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ corners }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error || "Could not save the court.");
            setPickingCourt(false);
            await refresh();
          }}
        />
        {job.status !== "calibrating" && (
          <button type="button" className="secondary" onClick={() => setPickingCourt(false)}>
            Cancel
          </button>
        )}
      </section>
    );
  }
  if (job.status === "failed") {
    return (
      <section className="panel">
        <h1>{job.filename}</h1>
        <p className="error">{job.error || "This game could not be processed."}</p>
      </section>
    );
  }
  if (job.status === "uploading" || job.status === "queued" || job.status === "analyzing") {
    return (
      <section className="panel">
        <h1>{job.filename}</h1>
        <p>{job.phase || "Working…"}</p>
        <div className="bar" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
          <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
        </div>
        <p className="note">
          {Math.round(job.progress * 100)}%
          {job.etaSeconds != null ? ` · ${formatRemaining(job.etaSeconds)}` : job.status === "analyzing" ? " · Estimating time…" : ""}
        </p>
        <p className="note">
          The ball scan runs on this computer&apos;s processor, a few times a second. Giving Docker more CPU cores in
          Docker Desktop settings makes it finish sooner. A graphics card does not speed up this install.
        </p>
      </section>
    );
  }

  const duration = job.durationSeconds ?? 0;
  const included = segments.filter((segment) => segment.keep);
  const kept = included.reduce((sum, segment) => sum + (segment.end - segment.start), 0);
  const locked = job.status === "export_queued" || job.status === "exporting" || saveState === "saving";
  const exporting = job.status === "export_queued" || job.status === "exporting";

  return (
    <section className="stage">
      <div className="panel">
        <h1>{job.filename}</h1>
        <p className="note">
          Rallies follow the ball while it is moving inside your court. Serve setup and walking the ball back are left
          out. Each clip has a little time before the first touch and after the last one.
        </p>
        {job.warning && <p className="warning">{job.warning}</p>}
        {job.error && <p className="error">{job.error}</p>}
        {actionError && <p className="error">{actionError}</p>}
        <video
          ref={videoRef}
          src={playbackUrl ?? undefined}
          controls
          playsInline
          onLoadedData={() => {
            const video = videoRef.current;
            if (video && pendingSeek.current != null) {
              video.currentTime = pendingSeek.current;
              pendingSeek.current = null;
            }
          }}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <span className="stat">{included.length} clips</span>
          <span className="stat">{formatClock(kept)} kept</span>
          <span className="clock" ref={clockRef}>
            0:00
          </span>
          <button type="button" className="secondary" onClick={addClip} disabled={locked || duration <= 0}>
            New clip at playhead
          </button>
          <button type="button" className="secondary" onClick={() => setPickingCourt(true)} disabled={locked}>
            Mark court again
          </button>
        </div>
        <div className="clips">
          {segments.length === 0 && <p className="note">No rallies yet. Play the game and add a clip at the playhead.</p>}
          {segments.map((clip, index) => (
            <article key={clip.id} className={clip.keep ? "clip" : "clip off"}>
              <button type="button" onClick={() => playClip(clip)} disabled={duration <= 0}>
                Play
              </button>
              <div>
                <strong>Rally {index + 1}</strong>
                <span className="when">
                  {formatClock(clip.start)} – {formatClock(clip.end)} · {formatClock(clip.end - clip.start)}
                </span>
              </div>
              <div className="times">
                <label>
                  <input
                    type="checkbox"
                    checked={clip.keep}
                    disabled={locked}
                    onChange={() => {
                      const next = segmentsRef.current.map((item) =>
                        item.id === clip.id ? { ...item, keep: !item.keep, origin: "user" as const } : item,
                      );
                      void persist(next);
                    }}
                  />
                  Include
                </label>
                <input
                  aria-label={`Rally ${index + 1} start seconds`}
                  type="number"
                  min={0}
                  max={duration}
                  step={0.1}
                  defaultValue={clip.start.toFixed(1)}
                  key={`${clip.id}-start-${clip.start.toFixed(2)}`}
                  disabled={locked}
                  onBlur={(event) => {
                    const start = Number(event.target.value);
                    const next = segmentsRef.current.map((item) =>
                      item.id === clip.id ? { ...item, start, origin: "user" as const } : item,
                    );
                    void persist(next);
                  }}
                />
                <input
                  aria-label={`Rally ${index + 1} end seconds`}
                  type="number"
                  min={0}
                  max={duration}
                  step={0.1}
                  defaultValue={clip.end.toFixed(1)}
                  key={`${clip.id}-end-${clip.end.toFixed(2)}`}
                  disabled={locked}
                  onBlur={(event) => {
                    const end = Number(event.target.value);
                    const next = segmentsRef.current.map((item) =>
                      item.id === clip.id ? { ...item, end, origin: "user" as const } : item,
                    );
                    void persist(next);
                  }}
                />
              </div>
            </article>
          ))}
        </div>
        <div className="row" style={{ marginTop: 14 }}>
          <button type="button" onClick={() => void exportCut()} disabled={locked || included.length === 0 || saveState === "error"}>
            {exporting ? "Exporting…" : "Export full-resolution clips"}
          </button>
          {job.status === "done" && (
            <>
              <button type="button" onClick={() => void download("clips")}>
                Download clips
              </button>
              <button type="button" onClick={() => void download("game")} disabled={!job.hasFullGame}>
                Download full game
              </button>
            </>
          )}
          <span className="note">{saveState === "saving" ? "Saving…" : saveState === "saved" ? "Saved" : ""}</span>
        </div>
        {exporting && (
          <>
            <div className="bar" role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </div>
            <p className="note">
              {job.phase}
              {job.etaSeconds != null ? ` · ${formatRemaining(job.etaSeconds)}` : " · Estimating time…"}
            </p>
          </>
        )}
        {job.expiresAt && <p className="note">The game and the clips are deleted 24 hours after processing finishes.</p>}
      </div>
    </section>
  );
}

function clipProblem(clips: Segment[], duration: number): string | null {
  const sorted = [...clips].sort((a, b) => a.start - b.start);
  for (let index = 0; index < sorted.length; index += 1) {
    const clip = sorted[index];
    if (!(clip.end > clip.start + 0.2)) return "A clip is too short.";
    if (clip.start < 0 || clip.end > duration + 0.05) return "A clip sits outside the game.";
    if (index > 0 && clip.start < sorted[index - 1].end - 0.05) return "Two clips overlap.";
  }
  return null;
}
