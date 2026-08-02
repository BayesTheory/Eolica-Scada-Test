import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError, api } from "./api/client";
import { CoverageTimeline } from "./components/CoverageTimeline";
import { DailyReportCard } from "./components/DailyReportCard";
import { DriftBars } from "./components/DriftBars";

const DEFAULT_DAY = "2022-01-20";

/**
 * Mensagem por estado de erro, ramificando pelo `code` do Problem Detail.
 *
 * A distinção importa: um 404 aqui é estado legítimo do acervo — o dia não
 * existe — e não uma falha do sistema. Mostrá-lo como "erro" seria repetir o
 * problema da v1, onde data ausente virava 500 e o operador era informado de
 * que os dados estavam "corrompidos".
 */
function ErrorNotice({ error }: { error: unknown }) {
  if (error instanceof ApiError && error.isExpectedAbsence) {
    return (
      <div style={styles.notice}>
        <strong>Sem telemetria para esta data.</strong>
        <p style={styles.noticeBody}>
          O dia não está no acervo — o que é diferente de haver falha no sistema. Escolha uma
          data dentro do período coberto.
        </p>
      </div>
    );
  }
  if (error instanceof ApiError && error.isInsufficientData) {
    return (
      <div style={styles.notice}>
        <strong>Dado insuficiente para analisar este dia.</strong>
        <p style={styles.noticeBody}>
          Existem leituras, mas nenhum trecho contíguo é longo o bastante para a janela do
          modelo. O dia está fragmentado demais.
        </p>
      </div>
    );
  }
  const message = error instanceof Error ? error.message : "Erro desconhecido";
  return (
    <div style={{ ...styles.notice, borderLeftColor: "var(--status-critical)" }}>
      <strong>Não foi possível carregar.</strong>
      <p style={styles.noticeBody}>{message}</p>
    </div>
  );
}

function Panel({ children }: { children: React.ReactNode }) {
  return <section style={styles.panel}>{children}</section>;
}

export function App() {
  const [day, setDay] = useState(DEFAULT_DAY);

  const readiness = useQuery({
    queryKey: ["readiness"],
    queryFn: ({ signal }) => api.readiness(signal),
  });

  const report = useQuery({
    queryKey: ["report", day],
    queryFn: ({ signal }) => api.dailyReport(day, signal),
    enabled: day.length === 10,
  });

  const coverage = useQuery({
    queryKey: ["coverage"],
    queryFn: ({ signal }) => api.coverage({}, signal),
  });

  const drift = useQuery({
    queryKey: ["drift"],
    queryFn: ({ signal }) => api.drift(signal),
  });

  return (
    <main style={styles.page}>
      <header style={styles.masthead}>
        <div>
          <h1 style={styles.title}>Eólica SCADA</h1>
          <p style={styles.subtitle}>
            Monitoramento preditivo de turbina a partir de telemetria SCADA — Aventa AV-7, 6,2 kW
          </p>
        </div>
        <div style={styles.readiness}>
          <span
            style={{
              ...styles.readinessDot,
              background: readiness.data?.ready ? "var(--status-good)" : "var(--status-critical)",
            }}
            aria-hidden="true"
          />
          <span>{readiness.data?.ready ? "serviço pronto" : "serviço indisponível"}</span>
        </div>
      </header>

      <div style={styles.controls}>
        <label htmlFor="day" style={styles.label}>
          Dia a analisar
        </label>
        <input
          id="day"
          type="date"
          value={day}
          onChange={(event) => setDay(event.target.value)}
          style={styles.input}
        />
        {coverage.data && (
          <span style={styles.range} className="tabular">
            acervo: {coverage.data.start} → {coverage.data.end}
          </span>
        )}
      </div>

      <Panel>
        {report.isPending && <p style={styles.loading}>carregando relatório…</p>}
        {report.isError && <ErrorNotice error={report.error} />}
        {report.data && <DailyReportCard report={report.data} />}
      </Panel>

      <Panel>
        {coverage.isPending && <p style={styles.loading}>carregando cobertura…</p>}
        {coverage.isError && <ErrorNotice error={coverage.error} />}
        {coverage.data && (
          <>
            <CoverageTimeline
              days={coverage.data.days}
              title="Cobertura de telemetria, dia a dia"
            />
            <p style={styles.footnote}>
              Cobertura média de {(coverage.data.mean_completeness * 100).toFixed(1)}% ao longo
              de {coverage.data.days.length} dias. {coverage.data.fragmented_days} dias vieram em
              mais de um trecho contíguo — cada um desses é analisado por segmento, e não como um
              bloco contínuo.
            </p>
          </>
        )}
      </Panel>

      <Panel>
        {drift.isPending && <p style={styles.loading}>calculando drift…</p>}
        {drift.isError && <ErrorNotice error={drift.error} />}
        {drift.data && (
          <>
            <DriftBars features={drift.data.features} />
            <p style={styles.footnote}>
              Pior feature: <strong>{drift.data.worst_feature}</strong>.{" "}
              {drift.data.requires_action
                ? "Drift severo — o modelo provavelmente precisa de retreino."
                : "Nenhuma feature em drift severo."}
            </p>
          </>
        )}
      </Panel>
    </main>
  );
}

const styles = {
  page: { maxWidth: 1040, margin: "0 auto", padding: "32px 20px 64px" },
  masthead: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 16,
    flexWrap: "wrap",
    marginBottom: 24,
  },
  title: { margin: 0, fontSize: 24, fontWeight: 650, letterSpacing: -0.3 },
  subtitle: { margin: "4px 0 0", fontSize: 13.5, color: "var(--text-secondary)" },
  readiness: {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    fontSize: 12.5,
    color: "var(--text-secondary)",
  },
  readinessDot: { width: 8, height: 8, borderRadius: "50%", display: "inline-block" },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
    marginBottom: 20,
  },
  label: { fontSize: 12.5, color: "var(--text-secondary)" },
  input: {
    fontFamily: "inherit",
    fontSize: 13,
    padding: "6px 10px",
    borderRadius: "var(--radius)",
    border: "1px solid var(--border)",
    background: "var(--surface-1)",
    color: "var(--text-primary)",
  },
  range: { fontSize: 12, color: "var(--text-muted)" },
  panel: {
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-card)",
    padding: 20,
    marginBottom: 20,
  },
  loading: { margin: 0, fontSize: 13, color: "var(--text-muted)" },
  footnote: {
    margin: "16px 0 0",
    fontSize: 12.5,
    color: "var(--text-secondary)",
    maxWidth: "76ch",
  },
  notice: {
    padding: "12px 14px",
    borderLeft: "3px solid var(--status-warning)",
    background: "var(--surface-page)",
    borderRadius: "0 var(--radius) var(--radius) 0",
    fontSize: 13,
  },
  noticeBody: { margin: "4px 0 0", color: "var(--text-secondary)", maxWidth: "62ch" },
} as const satisfies Record<string, React.CSSProperties>;
