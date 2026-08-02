/**
 * Cliente HTTP tipado pelo contrato do backend.
 *
 * Os tipos vêm de `schema.ts`, **gerado** a partir do `openapi.json` que o
 * FastAPI produz. Nenhuma interface é digitada à mão aqui.
 *
 * É o mesmo princípio do feature store do backend: quando dois lados precisam
 * concordar sobre um formato, ter duas descrições dele é convite a divergirem
 * em silêncio. Renomear um campo no `DailyReportResponse` passa a quebrar
 * `npm run build`, em vez de virar `undefined` na tela do operador.
 */

import type { components, paths } from "./schema";

export type DailyReport = components["schemas"]["DailyReportResponse"];
export type CoverageTimeline = components["schemas"]["CoverageTimelineResponse"];
export type DayCoverage = components["schemas"]["DayCoverageEntry"];
export type DriftReport = components["schemas"]["DriftResponse"];
export type FeatureDrift = components["schemas"]["FeatureDrift"];
export type Readiness = components["schemas"]["ReadinessResponse"];
export type HealthStatus = components["schemas"]["HealthStatus"];
export type DriftSeverity = components["schemas"]["DriftSeverity"];

type CoverageQuery = NonNullable<
  paths["/api/v1/coverage"]["get"]["parameters"]["query"]
>;

/** Corpo de erro em RFC 9457, como o backend emite. */
export interface ProblemDetail {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
  code: string;
  [key: string]: unknown;
}

/**
 * Erro de API que preserva o Problem Detail.
 *
 * O `code` é o contrato estável para decidir o que mostrar; `title` e `detail`
 * são texto humano e podem mudar de redação. A UI ramifica pelo `code`.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly problem: ProblemDetail | null;

  constructor(status: number, problem: ProblemDetail | null, fallback: string) {
    super(problem?.detail ?? fallback);
    this.name = "ApiError";
    this.status = status;
    this.code = problem?.code ?? "unknown";
    this.problem = problem;
  }

  /** Um 404 aqui é estado legítimo do acervo, não falha do sistema. */
  get isExpectedAbsence(): boolean {
    return this.code === "not-found";
  }

  /** Dado existe mas não dá para analisar — também não é falha. */
  get isInsufficientData(): boolean {
    return this.code === "insufficient-data";
  }

  get isRetryable(): boolean {
    return this.status >= 500 || this.status === 503;
  }
}

/**
 * Vazio por padrão: em produção o FastAPI serve o build estático na mesma
 * origem, e em desenvolvimento o proxy do Vite faz o mesmo. Nenhum dos dois
 * precisa de CORS — configurar CORS só em dev é a receita clássica de quebrar
 * no deploy.
 */
const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "";

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { Accept: "application/json" },
    ...(signal ? { signal } : {}),
  });

  if (!response.ok) {
    // O backend emite `application/problem+json` em todo caminho de erro, mas
    // um proxy ou balanceador pode devolver HTML. Falhar ao parsear não pode
    // mascarar o status real.
    let problem: ProblemDetail | null = null;
    try {
      problem = (await response.json()) as ProblemDetail;
    } catch {
      problem = null;
    }
    throw new ApiError(response.status, problem, `HTTP ${response.status}`);
  }

  return (await response.json()) as T;
}

function toQueryString(params: Record<string, string | undefined>): string {
  const entries = Object.entries(params).filter(
    (entry): entry is [string, string] => entry[1] !== undefined,
  );
  return entries.length > 0 ? `?${new URLSearchParams(entries).toString()}` : "";
}

export const api = {
  dailyReport: (day: string, signal?: AbortSignal) =>
    request<DailyReport>(`/api/v1/reports/${day}`, signal),

  coverage: (query: CoverageQuery = {}, signal?: AbortSignal) =>
    request<CoverageTimeline>(
      `/api/v1/coverage${toQueryString({
        start: query.start ?? undefined,
        end: query.end ?? undefined,
      })}`,
      signal,
    ),

  drift: (signal?: AbortSignal) => request<DriftReport>("/api/v1/drift", signal),

  readiness: (signal?: AbortSignal) => request<Readiness>("/health/ready", signal),
};
