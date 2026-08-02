import type { ReactNode } from "react";

export interface TooltipAnchor {
  x: number;
  y: number;
}

/**
 * Tooltip posicionado em coordenadas de viewport.
 *
 * `pointer-events: none` é essencial: sem isso o tooltip rouba o hover da
 * própria marca que o gerou e a interface pisca.
 */
export function Tooltip({ anchor, children }: { anchor: TooltipAnchor; children: ReactNode }) {
  return (
    <div
      role="tooltip"
      style={{
        position: "fixed",
        left: Math.min(anchor.x + 12, window.innerWidth - 200),
        top: Math.max(anchor.y - 8, 8),
        pointerEvents: "none",
        background: "var(--surface-1)",
        color: "var(--text-primary)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "8px 10px",
        fontSize: 12,
        lineHeight: 1.5,
        boxShadow: "0 4px 16px rgba(0,0,0,0.16)",
        zIndex: 40,
        maxWidth: 220,
      }}
    >
      {children}
    </div>
  );
}
