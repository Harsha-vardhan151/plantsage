// gateway/server.js (ESM)
import express from "express";
import compression from "compression";
import { createProxyMiddleware } from "http-proxy-middleware";
import serveStatic from "serve-static";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 8080;
const FASTAPI = process.env.FASTAPI_BASE || "http://localhost:8000";

const app = express();
app.use(compression());

// Proxy /api/* → FastAPI (pathRewrite removes /api prefix)
app.use(
  "/api",
  createProxyMiddleware({
    target: FASTAPI,
    changeOrigin: true,
    pathRewrite: { "^/api": "" }
  })
);

// Serve the built React app
const distDir = path.resolve(__dirname, "../client/dist");
app.use(serveStatic(distDir, { index: ["index.html"] }));
app.get("*", (_req, res) => res.sendFile(path.join(distDir, "index.html")));

app.listen(PORT, () => {
  console.log(`[gateway] http://localhost:${PORT}  (API → ${FASTAPI})`);
});
