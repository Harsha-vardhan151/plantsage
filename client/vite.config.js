// client/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // when running UI alone, proxy to gateway or directly to FastAPI
      "/api": {
        target: process.env.VITE_API_BASE ?? "http://localhost:8080",
        changeOrigin: true
      }
    }
  },
  build: { outDir: "dist", sourcemap: true }
});
