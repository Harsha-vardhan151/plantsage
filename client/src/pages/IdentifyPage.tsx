// client/src/pages/IdentifyPage.tsx
import React, { useMemo, useState } from "react";
import { analyzeImage, type AnalyzeResp } from "../api";
import FilePicker from "../components/FilePicker";
import CameraCapture from "../components/CameraCapture";
import Overlay from "../components/Overlay";



export default function IdentifyPage() {
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
      const res = await analyzeImage(f, 5, navigator.language ?? "en", "identify");
      setData(res);
    } catch (e:any) { setError(e?.message ?? "Failed"); }
    finally { setBusy(false); }
  };

  const top = data?.species ?? [];
  const boxes = useMemo(()=> data?.boxes ?? [], [data]);

  return (
    <>
      <div className="card">
        <p className="muted">Identify the plant species. Upload or capture a photo.</p>
        <div className="row">
          <button className="btn outline" onClick={()=>document.getElementById("picker-id")?.click()}>📁 Pick image</button>
          <input id="picker-id" type="file" accept="image/*" hidden onChange={(e)=> e.target.files?.[0] && doAnalyze(e.target.files[0])}/>
          <span className="spacer" />
          <a className="btn" href="#id-camera">📸 Camera</a>
        </div>
      </div>

      {busy && <div className="card">⏳ analyzing…</div>}
      {error && <div className="card warn">⚠️ {error}</div>}

      {preview && boxes.length>0 && <div className="card"><Overlay src={preview} boxes={boxes} /></div>}

      {preview && <div className="card"><img src={preview} style={{width:"100%",borderRadius:16}}/></div>}

      {top.length>0 && (
        <div className="card">
          <h3>Top species</h3>
          <ul className="list">
            {top.map((s,i)=>(
              <li key={i}><b>{s.name}</b><span>{(s.confidence*100).toFixed(1)}%</span></li>
            ))}
          </ul>
          <div className="meta">
            <span>latency: {data!.metadata.latency_ms.toFixed(1)} ms</span>
            <span>mode: {data!.metadata.mode}</span>
            <span>device: {data!.metadata.device}</span>
          </div>
          {top[0].confidence < 0.35 && (
            <div className="card warn" style={{marginTop:12}}>
              🤔 Low confidence. Try another angle, better light, include leaf + stem.
            </div>
          )}
        </div>
      )}

      <section id="id-camera"><CameraCapture onCapture={doAnalyze} /></section>
    </>
  );
}
