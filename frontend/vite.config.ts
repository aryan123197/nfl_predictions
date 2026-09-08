import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api to the FastAPI server so the browser makes same-origin
// requests. This mirrors how the app should be deployed (one origin, API
// behind a path) and means no absolute API URL gets baked into a build.
const apiProxy = {
  "/api": {
    target: "http://127.0.0.1:8000",
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ""),
  },
};

export default defineConfig({
  plugins: [react()],
  // `server` is the dev server; `preview` serves the production build and
  // does NOT inherit server.proxy -- it needs its own copy, or every /api
  // call 404s when checking a build locally.
  server: { port: 5173, proxy: apiProxy },
  preview: { port: 5173, proxy: apiProxy },
});
