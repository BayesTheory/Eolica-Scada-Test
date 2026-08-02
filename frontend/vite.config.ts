import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
// `defineConfig` do vitest/config, e não do vite: é a variante que conhece a
// chave `test`. Importar do vite e adicionar `test` mesmo assim é erro de tipo.
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    // Proxy em desenvolvimento: o browser fala com a mesma origem, então não há
    // CORS em dev nem em produção (onde o FastAPI serve o build estático).
    // CORS ligado só em dev é a receita clássica de quebrar no deploy.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
