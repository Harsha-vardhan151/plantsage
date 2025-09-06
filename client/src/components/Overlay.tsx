// client/src/components/ResultsView.tsx
import React from "react";
import type { AnalyzeResp } from "../api";

export default function ResultsView({
  data,
  previewUrl
}: {
  data: AnalyzeResp | null;
  previewUrl: string | null;
}) {
  if (!data) return null;

  const lowConfidence = data.species.length > 0 && data.species[0].confidence < 0.35;

  return (
    <>
      {previewUrl && (
        <div className="card">
          <img src={previewUrl} alt="preview" style={{ width: "100%", borderRadius: 16 }} />
        </div>
      )}

      <div className="card">
        <h3>Top species</h3>
        <ul className="list">
          {data.species.map((s, i) => (
            <li key={i}>
              <b>{s.name}</b>
              <span>{(s.confidence * 100).toFixed(1)}%</span>
            </li>
          ))}
        </ul>
        <div className="meta">
          <span>latency: {data.metadata.latency_ms.toFixed(1)} ms</span>
          <span>mode: {data.metadata.mode}</span>
          <span>device: {data.metadata.device}</span>
        </div>
      </div>

      {lowConfidence && (
        <div className="card warn">
          🤔 Low confidence. Try: <br />
          • add another photo (top + underside) <br />
          • move closer in good light <br />
          • include leaf + stem
        </div>
      )}
    </>
  );
}
