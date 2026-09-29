"use client";

import { useState, type MouseEvent } from "react";

type Point = { x: number; y: number };

export function CourtPicker({
  playbackUrl,
  onSave,
}: {
  playbackUrl: string | null;
  onSave: (corners: Point[]) => Promise<void>;
}) {
  const [points, setPoints] = useState<Point[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  function addPoint(event: MouseEvent<SVGSVGElement>) {
    if (points.length >= 4) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    setPoints([
      ...points,
      {
        x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
        y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
      },
    ]);
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      await onSave(points);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the court.");
      setSaving(false);
    }
  }

  const polygon = points.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");

  return (
    <div>
      <p className="lede">
        Click the four corners of your court. Games on the courts beside you stay outside the box, so their movement
        is not treated as your rally.
      </p>
      <div className="stage-video">
        <video src={playbackUrl ?? undefined} controls playsInline />
        <svg className="court-overlay" viewBox="0 0 100 100" preserveAspectRatio="none" onClick={addPoint}>
          {points.length >= 2 && <polyline points={polygon} fill="none" stroke="#8ddea8" strokeWidth="0.6" />}
          {points.length === 4 && <polygon points={polygon} fill="rgba(47,158,87,0.28)" stroke="#8ddea8" strokeWidth="0.6" />}
          {points.map((point, index) => (
            <circle key={index} cx={point.x * 100} cy={point.y * 100} r="1.4" fill="#f4fff6" />
          ))}
        </svg>
      </div>
      <p className="note">{points.length} of 4 corners. Use the video controls to find a frame that shows the whole court.</p>
      {error && <p className="error">{error}</p>}
      <div className="row">
        <button type="button" className="secondary" onClick={() => setPoints((current) => current.slice(0, -1))} disabled={points.length === 0 || saving}>
          Undo
        </button>
        <button type="button" className="secondary" onClick={() => setPoints([])} disabled={points.length === 0 || saving}>
          Clear
        </button>
        <button type="button" onClick={() => void save()} disabled={points.length !== 4 || saving}>
          {saving ? "Starting…" : "Find rallies"}
        </button>
      </div>
    </div>
  );
}
