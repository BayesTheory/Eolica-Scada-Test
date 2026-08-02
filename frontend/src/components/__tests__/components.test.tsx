/**
 * Testes de comportamento da interface.
 *
 * O que se verifica aqui não é aparência — é que a interface honra as mesmas
 * distinções que o backend faz. Um 404 é estado do acervo, não falha; cor de
 * status nunca aparece sem rótulo; ausência de informação não vira zero.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DailyReport, DayCoverage, FeatureDrift } from "../../api/client";
import { CoverageTimeline } from "../CoverageTimeline";
import { DailyReportCard } from "../DailyReportCard";
import { DriftBars } from "../DriftBars";

function day(overrides: Partial<DayCoverage> = {}): DayCoverage {
  return {
    day: "2022-01-20",
    readings: 144,
    expected_readings: 144,
    completeness: 1,
    segments: 1,
    longest_segment: 144,
    faulted_readings: 0,
    is_fragmented: false,
    is_absent: false,
    ...overrides,
  };
}

function report(overrides: Partial<DailyReport> = {}): DailyReport {
  return {
    day: "2022-01-20",
    health: {
      status: "OK",
      reason: "Nenhuma janela acima do limiar.",
      exceedances: 0,
      sustained_anomalies: 0,
      evaluated_windows: 85,
      persistence_window: 6,
      previous_period_anomalies: 0,
      previous_period_known: true,
      threshold: { value: 6.8221, method: "percentile", parameter: 99.5 },
    },
    coverage: {
      readings: 144,
      expected_readings: 144,
      completeness: 1,
      analysed_segments: 1,
      discarded_readings: 0,
      is_fragmented: false,
    },
    forecast: {
      power_kw: 0,
      power_kw_measured: -0.02,
      target_time: "2022-01-21T00:00:00Z",
      model_version: "moving-average-6@1",
    },
    forecast_unavailable_reason: null,
    data_range: { start: "2022-01-14T00:00:00Z", end: "2022-01-27T23:50:00Z" },
    ...overrides,
  } as DailyReport;
}

describe("DailyReportCard", () => {
  it("mostra o veredito com rótulo textual, não só cor", () => {
    render(<DailyReportCard report={report()} />);
    expect(screen.getByText("OK")).toBeInTheDocument();
  });

  it("expõe o limiar usado, para a decisão ser auditável", () => {
    render(<DailyReportCard report={report()} />);
    expect(screen.getByText("6.8221")).toBeInTheDocument();
  });

  it("mostra a previsão já com o clamp aplicado pelo servidor", () => {
    render(<DailyReportCard report={report()} />);
    expect(screen.getByText("0.000 kW")).toBeInTheDocument();
  });

  it("avisa quando o dia veio fragmentado", () => {
    const fragmented = report({
      coverage: {
        readings: 63,
        expected_readings: 144,
        completeness: 0.4375,
        analysed_segments: 2,
        discarded_readings: 3,
        is_fragmented: true,
      },
    });
    render(<DailyReportCard report={fragmented} />);
    expect(screen.getByText(/2 trechos separados/)).toBeInTheDocument();
    expect(screen.getByText("43.8%")).toBeInTheDocument();
  });

  it("diz explicitamente quando não há dado da véspera", () => {
    const unknown = report({
      health: { ...report().health, previous_period_known: false, previous_period_anomalies: null },
    });
    render(<DailyReportCard report={unknown} />);
    expect(screen.getByText(/não está sendo tratada como ausência de anomalia/)).toBeInTheDocument();
  });

  it("mostra o motivo quando a previsão falha, em vez de um número falso", () => {
    const noForecast = report({
      forecast: null,
      forecast_unavailable_reason: "São necessárias no mínimo 6 observações, mas só há 2",
    });
    render(<DailyReportCard report={noForecast} />);
    expect(screen.getByText("indisponível")).toBeInTheDocument();
    expect(screen.getByText(/no mínimo 6 observações/)).toBeInTheDocument();
  });
});

describe("CoverageTimeline", () => {
  const days = [
    day({ day: "2022-01-14" }),
    day({ day: "2022-01-15", completeness: 0.4375, readings: 63, segments: 2, is_fragmented: true }),
    day({ day: "2022-01-16", completeness: 0, readings: 0, segments: 0, is_absent: true }),
  ];

  it("resume fragmentação e ausência no subtítulo", () => {
    render(<CoverageTimeline days={days} title="Cobertura" />);
    expect(screen.getByText(/3 dias · 1 fragmentados · 1 sem medição/)).toBeInTheDocument();
  });

  it("descreve o gráfico para leitores de tela", () => {
    render(<CoverageTimeline days={days} title="Cobertura" />);
    expect(screen.getByRole("img", { name: /1 dias fragmentados, 1 sem medição/ })).toBeInTheDocument();
  });

  it("oferece a tabela como visão alternativa", () => {
    render(<CoverageTimeline days={days} title="Cobertura" />);
    expect(screen.getByRole("button", { name: /ver como tabela/ })).toBeInTheDocument();
  });
});

describe("DriftBars", () => {
  const features: FeatureDrift[] = [
    { feature: "generator_temperature", score: 0.42, method: "psi", severity: "severe" },
    { feature: "wind_speed", score: 0.14, method: "psi", severity: "moderate" },
    { feature: "power", score: 0.02, method: "psi", severity: "none" },
  ];

  it("acompanha cada barra de rótulo textual — cor nunca sozinha", () => {
    render(<DriftBars features={features} />);
    expect(screen.getAllByText("agir").length).toBeGreaterThan(0);
    expect(screen.getAllByText("investigar").length).toBeGreaterThan(0);
    expect(screen.getAllByText("estável").length).toBeGreaterThan(0);
  });

  it("mostra o valor numérico de cada feature", () => {
    render(<DriftBars features={features} />);
    expect(screen.getAllByText("0.420").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/PSI 0\.140/).length).toBeGreaterThan(0);
  });

  it("explica a convenção dos limiares", () => {
    render(<DriftBars features={features} />);
    expect(screen.getByText(/Abaixo de 0,10 estável/)).toBeInTheDocument();
  });
});
