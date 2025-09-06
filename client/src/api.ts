// client/src/api.ts  (send optional 'task' flag)
export type SpeciesPred = { name: string; confidence: number };
export type Issue = { name: string; confidence: number; severity?: number };
export type Box = { x1: number; y1: number; x2: number; y2: number; label?: string; score?: number };

export type AnalyzeResp = {
  species: SpeciesPred[];
  issues: Issue[];
  severity: number | null;
  boxes: Box[];
  metadata: {
    latency_ms: number;
    mode: string;
    temperature: number;
    versions: Record<string, string>;
    device: string;
  };
};

export async function analyzeImage(
  file: File,
  topk = 5,
  locale = "en",
  task?: "identify" | "disease"
): Promise<AnalyzeResp> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("topk", String(topk));
  fd.append("locale", locale);
  if (task) fd.append("task", task); // backend may ignore; harmless

  const res = await fetch("/api/v1/analyze", { method: "POST", body: fd });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}
