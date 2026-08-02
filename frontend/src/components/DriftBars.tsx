/**
 * PSI por feature, com as faixas de severidade da convenção do setor.
 *
 * Aqui a cor é de **status**, não de série — e status tem uma regra própria: o
 * validador de palette confirma que o amarelo de aviso fica abaixo de 3:1 no
 * fundo claro, por construção. A mitigação obrigatória é que a cor nunca
 * carregue significado sozinha.
 *
 * Por isso cada barra vem com o rótulo textual da severidade e o valor
 * numérico visíveis. Um leitor com daltonismo, uma impressão em preto e branco
 * ou o modo de alto contraste do sistema continuam legíveis.
 */

import type { FeatureDrift } from "../api/client";

const MODERATE = 0.1;
const SEVERE = 0.25;
const SCALE_MAX = 0.5;
const BAR_HEIGHT = 14;
const ROW_HEIGHT = 34;
const LABEL_WIDTH = 168;
const VALUE_WIDTH = 132;
const CORNER = 4;

const SEVERITY_LABEL: Record<string, string> = {
  none: "estável",
  moderate: "investigar",
  severe: "agir",
};

const SEVERITY_COLOR: Record<string, string> = {
  none: "var(--status-good)",
  moderate: "var(--status-warning)",
  severe: "var(--status-critical)",
};

/** Ícone por severidade — o canal redundante à cor. */
function SeverityIcon({ severity }: { severity: string }) {
  const color = SEVERITY_COLOR[severity] ?? "var(--text-muted)";
  if (severity === "severe") {
    return (
      <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true">
        <path d="M6 1 11.2 10.5H0.8Z" fill={color} />
        <rect x={5.35} y={4} width={1.3} height={3.4} fill="var(--surface-1)" />
        <rect x={5.35} y={8.1} width={1.3} height={1.3} fill="var(--surface-1)" />
      </svg>
    );
  }
  if (severity === "moderate") {
    return (
      <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true">
        <circle cx={6} cy={6} r={5.2} fill={color} />
        <rect x={5.35} y={3} width={1.3} height={3.6} fill="var(--surface-1)" />
        <rect x={5.35} y={7.5} width={1.3} height={1.3} fill="var(--surface-1)" />
      </svg>
    );
  }
  return (
    <svg width={12} height={12} viewBox="0 0 12 12" aria-hidden="true">
      <circle cx={6} cy={6} r={5.2} fill={color} />
      <path
        d="M3.4 6.2 5.2 8 8.6 4.4"
        stroke="var(--surface-1)"
        strokeWidth={1.4}
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function DriftBars({ features }: { features: FeatureDrift[] }) {
  const width = LABEL_WIDTH + 320 + VALUE_WIDTH;
  const trackWidth = 320;
  const height = features.length * ROW_HEIGHT + 24;

  const position = (value: number) => Math.min(1, value / SCALE_MAX) * trackWidth;

  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>
          Drift por feature (PSI)
        </h3>
        <p style={{ margin: "2px 0 0", fontSize: 13, color: "var(--text-secondary)" }}>
          Referência × período recente. Abaixo de 0,10 estável; até 0,25 investigar; acima, agir.
        </p>
      </figcaption>

      <div style={{ overflowX: "auto" }}>
        <svg width={width} height={height} role="img" aria-label="PSI por feature">
          {/* Limiares como linhas tracejadas recessivas, rotuladas. */}
          {[MODERATE, SEVERE].map((threshold) => (
            <g key={threshold}>
              <line
                x1={LABEL_WIDTH + position(threshold)}
                y1={4}
                x2={LABEL_WIDTH + position(threshold)}
                y2={features.length * ROW_HEIGHT + 4}
                stroke="var(--baseline)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <text
                x={LABEL_WIDTH + position(threshold)}
                y={features.length * ROW_HEIGHT + 18}
                textAnchor="middle"
                style={{ fontSize: 10, fill: "var(--text-muted)" }}
                className="tabular"
              >
                {threshold.toFixed(2)}
              </text>
            </g>
          ))}

          {features.map((feature, index) => {
            const y = index * ROW_HEIGHT + 6;
            const barLength = Math.max(2, position(feature.score));
            const color = SEVERITY_COLOR[feature.severity] ?? "var(--text-muted)";

            return (
              <g key={feature.feature}>
                <text
                  x={LABEL_WIDTH - 12}
                  y={y + BAR_HEIGHT - 2}
                  textAnchor="end"
                  style={{ fontSize: 12, fill: "var(--text-primary)" }}
                >
                  {feature.feature}
                </text>
                <rect
                  x={LABEL_WIDTH}
                  y={y}
                  width={trackWidth}
                  height={BAR_HEIGHT}
                  fill="var(--gridline)"
                  rx={2}
                />
                <rect
                  x={LABEL_WIDTH}
                  y={y}
                  width={barLength}
                  height={BAR_HEIGHT}
                  rx={Math.min(CORNER, barLength / 2)}
                  fill={color}
                />
                {/* Valor e severidade em tinta de texto, nunca na cor da marca. */}
                <text
                  x={LABEL_WIDTH + trackWidth + 12}
                  y={y + BAR_HEIGHT - 2}
                  style={{ fontSize: 12, fill: "var(--text-primary)" }}
                  className="tabular"
                >
                  {feature.score.toFixed(3)}
                </text>
                <text
                  x={LABEL_WIDTH + trackWidth + 68}
                  y={y + BAR_HEIGHT - 2}
                  style={{ fontSize: 12, fill: "var(--text-secondary)" }}
                >
                  {SEVERITY_LABEL[feature.severity] ?? feature.severity}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Lista redundante ao SVG: acessível a leitor de tela e com ícone. */}
      <ul style={{ listStyle: "none", padding: 0, margin: "12px 0 0" }}>
        {features.map((feature) => (
          <li
            key={feature.feature}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 12,
              color: "var(--text-secondary)",
              padding: "2px 0",
            }}
          >
            <SeverityIcon severity={feature.severity} />
            <span style={{ color: "var(--text-primary)" }}>{feature.feature}</span>
            <span className="tabular">PSI {feature.score.toFixed(3)}</span>
            <span>— {SEVERITY_LABEL[feature.severity] ?? feature.severity}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
