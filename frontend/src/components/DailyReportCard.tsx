/**
 * Relatório de um dia: veredito, cobertura, limiar e previsão.
 *
 * O veredito é a informação principal, então vira número-herói com rótulo — não
 * um badge colorido. Cor de status aqui também vem sempre acompanhada de texto.
 *
 * A cobertura aparece ao lado do veredito de propósito: "OK" com 43,8% do dia
 * medido significa bem menos que "OK" com 100%, e a v1 não dava ao operador
 * como notar a diferença.
 */

import type { DailyReport } from "../api/client";

const STATUS_TEXT: Record<string, string> = {
  OK: "OK",
  ALERTA: "ALERTA",
  EM_MANUTENCAO: "EM MANUTENÇÃO",
};

const STATUS_COLOR: Record<string, string> = {
  OK: "var(--status-good)",
  ALERTA: "var(--status-critical)",
  EM_MANUTENCAO: "var(--status-warning)",
};

function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  // `| undefined` explícito: com `exactOptionalPropertyTypes`, "ausente" e
  // "presente com undefined" são tipos diferentes — e aqui o segundo acontece,
  // porque `forecast?.model_version` pode ser undefined.
  hint?: string | undefined;
}) {
  return (
    <div style={styles.tile}>
      <span style={styles.tileLabel}>{label}</span>
      <span style={styles.tileValue} className="tabular">
        {value}
      </span>
      {hint && <span style={styles.tileHint}>{hint}</span>}
    </div>
  );
}

export function DailyReportCard({ report }: { report: DailyReport }) {
  const status = report.health.status;
  const coverage = report.coverage;

  return (
    <section style={styles.card} aria-labelledby="daily-report-heading">
      <header style={styles.header}>
        <div>
          <h2 id="daily-report-heading" style={styles.day} className="tabular">
            {report.day}
          </h2>
          <p style={styles.reason}>{report.health.reason}</p>
        </div>
        <div style={styles.statusBlock}>
          <span
            style={{ ...styles.statusDot, background: STATUS_COLOR[status] ?? "var(--text-muted)" }}
            aria-hidden="true"
          />
          <span style={styles.statusText}>{STATUS_TEXT[status] ?? status}</span>
        </div>
      </header>

      <div style={styles.tiles}>
        <StatTile
          label="cobertura do dia"
          value={`${(coverage.completeness * 100).toFixed(1)}%`}
          hint={`${coverage.readings}/${coverage.expected_readings} leituras · ${coverage.analysed_segments} trecho(s)`}
        />
        <StatTile
          label="janelas acima do limiar"
          value={String(report.health.exceedances)}
          hint={`${report.health.sustained_anomalies} sustentadas em ${report.health.evaluated_windows} avaliadas`}
        />
        <StatTile
          label="limiar em vigor"
          value={report.health.threshold.value.toFixed(4)}
          hint={`${report.health.threshold.method} p=${report.health.threshold.parameter}`}
        />
        <StatTile
          label="previsão"
          value={
            report.forecast ? `${report.forecast.power_kw.toFixed(3)} kW` : "indisponível"
          }
          hint={report.forecast?.model_version ?? report.forecast_unavailable_reason ?? undefined}
        />
      </div>

      {coverage.is_fragmented && (
        <p style={styles.notice}>
          Este dia veio em {coverage.analysed_segments} trechos separados por descontinuidade.
          Cada trecho foi analisado isoladamente; {coverage.discarded_readings} leitura(s) em
          trechos curtos demais para a janela do modelo foram descartadas.
        </p>
      )}

      {!report.health.previous_period_known && (
        <p style={styles.notice}>
          Sem dado do dia anterior: não é possível concluir manutenção em curso. Ausência de
          informação não está sendo tratada como ausência de anomalia.
        </p>
      )}
    </section>
  );
}

const styles = {
  card: {
    background: "var(--surface-1)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-card)",
    padding: 20,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 16,
    flexWrap: "wrap",
  },
  day: { margin: 0, fontSize: 20, fontWeight: 600 },
  reason: { margin: "4px 0 0", fontSize: 13, color: "var(--text-secondary)", maxWidth: "52ch" },
  statusBlock: { display: "inline-flex", alignItems: "center", gap: 8 },
  statusDot: { width: 10, height: 10, borderRadius: "50%", display: "inline-block" },
  statusText: { fontSize: 14, fontWeight: 600, letterSpacing: 0.2 },
  tiles: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
    gap: 12,
    marginTop: 20,
  },
  tile: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
    padding: "12px 14px",
    borderRadius: "var(--radius)",
    border: "1px solid var(--gridline)",
  },
  tileLabel: { fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" },
  tileValue: { fontSize: 22, fontWeight: 600, color: "var(--text-primary)" },
  tileHint: { fontSize: 11, color: "var(--text-secondary)" },
  notice: {
    marginTop: 16,
    marginBottom: 0,
    padding: "10px 12px",
    borderLeft: "3px solid var(--status-warning)",
    background: "var(--surface-page)",
    fontSize: 12.5,
    color: "var(--text-secondary)",
    borderRadius: "0 var(--radius) var(--radius) 0",
  },
} as const satisfies Record<string, React.CSSProperties>;
