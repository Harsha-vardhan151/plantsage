// client/src/components/CameraCapture.tsx  (unchanged helper)
import React, { useEffect, useRef, useState } from "react";
type Props = { onCapture: (file: File) => void };

export default function CameraCapture({ onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setStreaming(true);
        }
      } catch (e:any) { setError(e?.message ?? "Camera not available"); }
    })();
    return () => {
      const v = videoRef.current;
      const s = v?.srcObject as MediaStream | undefined;
      s?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const snap = () => {
    const v = videoRef.current!, c = canvasRef.current!;
    c.width = v.videoWidth; c.height = v.videoHeight;
    const ctx = c.getContext("2d")!;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob((blob) => { if (blob) onCapture(new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" })); }, "image/jpeg", 0.92);
  };

  if (error) return <div className="card warn">⚠️ {error}</div>;

  return (
    <div className="card">
      <video ref={videoRef} playsInline muted style={{ width: "100%", borderRadius: 16 }} />
      <div className="row">
        <button className="btn" onClick={snap} disabled={!streaming}>📸 Capture</button>
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}
