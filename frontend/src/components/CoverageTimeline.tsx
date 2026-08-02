/**
 * Cobertura de telemetria dia a dia.
 *
 * É a visualização central do painel, e a razão é histórica: a v1 do sistema
 * analisava dias fragmentados como se fossem contínuos, e reportava um veredito
 * com a mesma confiança de um dia completo. Este gráfico torna a fragmentação
 * impossível de ignorar.
 *
 * Decisões de codificação:
 *
 * - **Uma série, uma cor.** Magnitude ao longo do tempo. Série única dispensa
 *   caixa de legenda — o título nomeia o que está sendo mostrado.
 * - **Fragmentação não é codificada por cor.** Dias partidos em vários trechos
 *   recebem uma marca própria acima da barra. Cor sozinha nunca carrega
 *   informação.
 * - **Dias ausentes recebem hachura**, não uma barra de altura zero — "não
 *   medimos" e "medimos zero" são estados diferentes e precisam parecer
 *   diferentes.
 * - **Tabela alternativa** disponível, para leitura por leitor de tela e para
 *   quem precisa do número exato.
 */

import { useId, useMemo, useState } from "react";

import type { DayCoverage } from "../api/client";
import { Tooltip, type TooltipAnchor } from "./Tooltip";

const HEIGHT = 180;
const PADDING = { top: 16, right: 8, bottom: 28, left: 44 };
const BAR_GAP = 2;
const CORNER = 4;
const FRAGMENT_MARKER = 4;

interface Props {
  days: DayCoverage[];
  title: string;
}

export function CoverageTimeline({ days, title }: Props) {
  const [hovered, setHovered] = useState<{ day: DayCoverage; anchor: TooltipAnchor } | null>(
    null,
  );
  const [showTable, setShowTable] = useState(false);
  const hatchId = useId();

  const width = Math.max(320, days.length * 6 + PADDING.left + PADDING.right);
  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;
  const slot = days.length > 0 ? plotWidth / days.length : plotWidth;
  const barWidth = Math.max(1, slot - BAR_GAP);

  const stats = useMemo(
    () => ({
      fragmented: days.filter((day) => day.is_fragmented).length,
      absent: days.filter((day) => day.is_absent).length,
    }),
    [days],
  );

  return (
    <figure style={styles.figure}>
      <figcaption style={styles.caption}>
        <h3 style={styles.title}>{title}</h3>
        <p style={styles.subtitle}>
          {days.length} dias · {stats.fragmented} fragmentados · {stats.absent} sem medição
        </p>
      </figcaption>

      <div style={styles.plotWrapper}>
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`Cobertura diária de telemetria em ${days.length} dias. ${stats.fragmented} dias fragmentados, ${stats.absent} sem medição.`}
          style={styles.svg}
        >
          <defs>
            {/* Hachura para "sem medição" — o canal que não depende de cor. */}
            <pattern
              id={hatchId}
              width={4}
              height={4}
              patternTransform="rotate(45)"
              patternUnits="userSpaceOnUse"
            >
              <line x1={0} y1={0} x2={0} y2={4} stroke="var(--baseline)" strokeWidth={1.5} />
            </pattern>
          </defs>

          {/* Grade recessiva: só 50% e 100%, as duas leituras que importam. */}
          {[0.5, 1].map((tick) => {
            const y = PADDING.top + plotHeight * (1 - tick);
            return (
              <g key={tick}>
                <line
                  x1={PADDING.left}
                  y1={y}
                  x2={width - PADDING.right}
                  y2={y}
                  stroke="var(--gridline)"
                  strokeWidth={1}
                />
                <text x={PADDING.left - 8} y={y + 4} textAnchor="end" style={styles.tick}>
                  {tick * 100}%
                </text>
              </g>
            );
          })}

          {days.map((day, index) => {
            const x = PADDING.left + index * slot;
            const barHeight = Math.max(1, plotHeight * day.completeness);
            const y = PADDING.top + plotHeight - barHeight;

            return (
              <g
                key={day.day}
                onMouseEnter={(event) =>
                  setHovered({
                    day,
                    anchor: {
                      x: event.currentTarget.getBoundingClientRect().left,
                      y: event.currentTarget.getBoundingClientRect().top,
                    },
                  })
                }
                onMouseLeave={() => setHovered(null)}
              >
                {/* Alvo de hover maior que a marca: barras de 4px são difíceis
                    de acertar com o mouse. */}
                <rect
                  x={x - 2}
                  y={PADDING.top}
                  width={slot + 4}
                  height={plotHeight}
                  fill="transparent"
                />
                {day.is_absent ? (
                  <rect
                    x={x}
                    y={PADDING.top}
                    width={barWidth}
                    height={plotHeight}
                    fill={`url(#${hatchId})`}
                    opacity={0.5}
                  />
                ) : (
                  <rect
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    rx={Math.min(CORNER, barWidth / 2)}
                    fill="var(--series-1)"
                    opacity={hovered && hovered.day.day !== day.day ? 0.45 : 1}
                  />
                )}
                {day.is_fragmented && (
                  <circle
                    cx={x + barWidth / 2}
                    cy={PADDING.top - 6}
                    r={FRAGMENT_MARKER / 2}
                    fill="var(--status-warning)"
                    stroke="var(--surface-1)"
                    strokeWidth={1}
                  />
                )}
              </g>
            );
          })}

          <line
            x1={PADDING.left}
            y1={PADDING.top + plotHeight}
            x2={width - PADDING.right}
            y2={PADDING.top + plotHeight}
            stroke="var(--baseline)"
            strokeWidth={1}
          />
        </svg>
      </div>

      <div style={styles.legend}>
        <span style={styles.legendItem}>
          <span style={{ ...styles.swatch, background: "var(--status-warning)" }} />
          dia fragmentado (mais de um trecho contíguo)
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.swatch, background: "var(--baseline)" }} />
          sem medição
        </span>
        <button type="button" onClick={() => setShowTable((open) => !open)} style={styles.toggle}>
          {showTable ? "ocultar tabela" : "ver como tabela"}
        </button>
      </div>

      {hovered && (
        <Tooltip anchor={hovered.anchor}>
          <strong>{hovered.day.day}</strong>
          <br />
          cobertura <span className="tabular">{(hovered.day.completeness * 100).toFixed(1)}%</span>
          <br />
          <span className="tabular">{hovered.day.readings}</span> de{" "}
          <span className="tabular">{hovered.day.expected_readings}</span> leituras
          <br />
          {hovered.day.is_absent
            ? "sem medição"
            : `${hovered.day.segments} trecho(s) contíguo(s)`}
        </Tooltip>
      )}

      {showTable && <CoverageTable days={days} />}
    </figure>
  );
}

function CoverageTable({ days }: { days: DayCoverage[] }) {
  return (
    <div style={styles.tableWrapper}>
      <table style={styles.table}>
        <caption className="visually-hidden">Cobertura de telemetria por dia</caption>
        <thead>
          <tr>
            <th style={styles.th}>dia</th>
            <th style={{ ...styles.th, textAlign: "right" }}>cobertura</th>
            <th style={{ ...styles.th, textAlign: "right" }}>leituras</th>
            <th style={{ ...styles.th, textAlign: "right" }}>trechos</th>
          </tr>
        </thead>
        <tbody>
          {days.map((day) => (
            <tr key={day.day}>
              <td style={styles.td}>{day.day}</td>
              <td style={{ ...styles.td, textAlign: "right" }} className="tabular">
                {(day.completeness * 100).toFixed(1)}%
              </td>
              <td style={{ ...styles.td, textAlign: "right" }} className="tabular">
                {day.readings}
              </td>
              <td style={{ ...styles.td, textAlign: "right" }} className="tabular">
                {day.segments}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const styles = {
  figure: { margin: 0 },
  caption: { marginBottom: 12 },
  title: { margin: 0, fontSize: 15, fontWeight: 600, color: "var(--text-primary)" },
  subtitle: { margin: "2px 0 0", fontSize: 13, color: "var(--text-secondary)" },
  plotWrapper: { overflowX: "auto", maxWidth: "100%" },
  svg: { display: "block" },
  tick: { fontSize: 11, fill: "var(--text-muted)", fontVariantNumeric: "tabular-nums" },
  legend: {
    display: "flex",
    flexWrap: "wrap",
    gap: 16,
    alignItems: "center",
    marginTop: 8,
    fontSize: 12,
    color: "var(--text-secondary)",
  },
  legendItem: { display: "inline-flex", alignItems: "center", gap: 6 },
  swatch: { width: 8, height: 8, borderRadius: 2, display: "inline-block" },
  toggle: {
    marginLeft: "auto",
    background: "none",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "4px 10px",
    fontSize: 12,
    color: "var(--text-secondary)",
    cursor: "pointer",
    fontFamily: "inherit",
  },
  tableWrapper: { maxHeight: 280, overflowY: "auto", marginTop: 12 },
  table: { width: "100%", borderCollapse: "collapse", fontSize: 13 },
  th: {
    textAlign: "left",
    padding: "6px 8px",
    borderBottom: "1px solid var(--border)",
    color: "var(--text-secondary)",
    fontWeight: 500,
    position: "sticky",
    top: 0,
    background: "var(--surface-1)",
  },
  td: { padding: "5px 8px", borderBottom: "1px solid var(--gridline)" },
} as const satisfies Record<string, React.CSSProperties>;
