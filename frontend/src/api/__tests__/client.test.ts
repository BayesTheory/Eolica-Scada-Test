/**
 * O cliente traduz Problem Details em decisões — e a decisão que mais importa
 * é não tratar um 404 como incidente.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "../client";

function mockResponse(status: number, body: unknown): void {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ApiError", () => {
  it("preserva o code do Problem Detail", async () => {
    mockResponse(404, {
      type: "…#not-found",
      title: "Recurso não encontrado",
      status: 404,
      detail: "Telemetria não encontrado para '2030-01-01'",
      instance: "/api/v1/reports/2030-01-01",
      code: "not-found",
    });

    await expect(api.dailyReport("2030-01-01")).rejects.toMatchObject({
      code: "not-found",
      status: 404,
    });
  });

  it("classifica ausência esperada como não-incidente", async () => {
    mockResponse(404, { code: "not-found", detail: "x", status: 404 } as never);
    const error = await api.dailyReport("2030-01-01").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isExpectedAbsence).toBe(true);
    expect((error as ApiError).isRetryable).toBe(false);
  });

  it("classifica dado insuficiente separadamente de ausência", async () => {
    mockResponse(422, { code: "insufficient-data", detail: "x", status: 422 } as never);
    const error = (await api.dailyReport("2022-01-20").catch((e: unknown) => e)) as ApiError;
    expect(error.isInsufficientData).toBe(true);
    expect(error.isExpectedAbsence).toBe(false);
  });

  it("marca falha de infraestrutura como retentável", async () => {
    mockResponse(503, { code: "model-unavailable", detail: "x", status: 503 } as never);
    const error = (await api.drift().catch((e: unknown) => e)) as ApiError;
    expect(error.isRetryable).toBe(true);
  });

  it("não mascara o status quando o corpo não é JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: async () => {
          throw new Error("resposta HTML de um proxy");
        },
      }),
    );
    const error = (await api.drift().catch((e: unknown) => e)) as ApiError;
    expect(error.status).toBe(502);
    expect(error.code).toBe("unknown");
  });
});

describe("montagem de URL", () => {
  it("omite a query quando não há período", async () => {
    mockResponse(200, { days: [] });
    await api.coverage({});
    expect(fetch).toHaveBeenCalledWith("/api/v1/coverage", expect.anything());
  });

  it("inclui apenas os parâmetros informados", async () => {
    mockResponse(200, { days: [] });
    await api.coverage({ start: "2022-01-14" });
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/coverage?start=2022-01-14",
      expect.anything(),
    );
  });
});
