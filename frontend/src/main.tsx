import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { ApiError } from "./api/client";
import "./styles/tokens.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      // Retentar um 404 é desperdício: o dia não vai passar a existir. Só erros
      // de infraestrutura merecem nova tentativa — a mesma distinção que o
      // backend faz entre 4xx e 5xx.
      retry: (failureCount, error) =>
        error instanceof ApiError ? error.isRetryable && failureCount < 2 : failureCount < 2,
    },
  },
});

const container = document.getElementById("root");
if (!container) {
  throw new Error("Elemento #root não encontrado");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
