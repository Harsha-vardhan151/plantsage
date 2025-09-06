// client/src/pages/DiseasePage.tsx
import React, { useMemo, useState } from "react";
import { analyzeImage, type AnalyzeResp } from "../api";
import CameraCapture from "../components/CameraCapture";

export default function DiseasePage() {
  const [file, setFile] = useState<File|null>(null);
  const [preview, setPreview] = useState<string|null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string|null>(null);
  const [data, setData] = useState<AnalyzeResp|null>(null);

  const doAnalyze = async (f: File) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setBusy(true); setError(null); setData(null);
    try {
      const res = await analyzeImage(f, 5, navigator.language ?? "en", "disease");
      setData(res);
    } catch (e:any) { setError(e?.message ?? "Failed"); }
    finally { setBusy(false); }
  };

  const issues = data?.issues ?? [];
  const boxes = useMemo(()=> data?.boxes ?? [], [data]);

  return (
    <>
      <div className="card">
        <p className="muted">Detect disease/deficiency signs. Add a clear leaf/lesion photo.</p>
        <div className="row">
          <button className="btn outline" onClick={()=>document.getElementById("picker-dis")?.click()}>📁 Pick image</button>
          <input id="picker-dis" type="file" accept="image/*" hidden onChange={(e)=> e.target.files?.[0] && doAnalyze(e.target.files[0])}/>
          <span className="spacer" />
          <a className="btn" href="#dis-camera">📸 Camera</a>
        </div>
      </div>

      {busy && <div className="card">⏳ analyzing…</div>}
      {error && <div className="card warn">⚠️ {error}</div>}

      {preview && (
        <div className="card">
          {/* show boxes overlaid for symptomatic regions */}
          <div className="overlay-wrap">
            <img src={preview!} alt="preview" />
            <canvas id="overlay" />
          </div>
        </div>
      )}

      {preview && boxes.length>0 && (
        <div className="card">
          <h3>Detected regions</h3>
          <div className="muted small">Server returned {boxes.length} box(es)</div>
        </div>
      )}

      {issues.length>0 ? (
        <div className="card">
          <h3>Top issues</h3>
          <ul className="list">
            {issues.map((it, i)=>(
              <li key={i}>
                <b>{it.name}</b>
                <span>
                  {(it.confidence*100).toFixed(1)}%
                  {typeof it.severity === "number" ? ` • sev ${it.severity.toFixed(2)}` : ""}
                </span>
              </li>
            ))}
          </ul>
          {typeof data?.severity === "number" && (
            <div className="muted">Overall severity: {data!.severity!.toFixed(2)}</div>
          )}
          <div className="meta" style={{marginTop:8}}>
            <span>latency: {data!.metadata.latency_ms.toFixed(1)} ms</span>
            <span>mode: {data!.metadata.mode}</span>
            <span>device: {data!.metadata.device}</span>
          </div>
        </div>
      ) : (
        preview && !busy && <div className="card warn">No issues detected with confidence. Try closer, well-lit lesion area.</div>
      )}

      <section id="dis-camera"><CameraCapture onCapture={doAnalyze} /></section>
    </>
  );
}
