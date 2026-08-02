"""Regime operacional: em que condição a turbina está operando.

Uma turbina não tem *uma* assinatura de operação normal — tem várias, uma por
faixa de vento. Abaixo do cut-in ela não gera e o rotor está parado; na faixa
parcial ela persegue o máximo de potência; acima da nominal ela limita por
pitch; acima do cut-out ela para para se proteger.

Tratar tudo isso como uma distribuição só é o defeito que a medição do backtest
expôs: com 34% de taxa de alarme falso, o detector estava reportando *mudança de
vento* como *mudança de saúde*.

As fronteiras vêm do envelope declarado pelo fabricante (`TurbineSpec`), não de
quantis dos dados — assim o regime significa a mesma coisa antes e depois de um
retreino, e um dataset com vento atipicamente calmo não desloca as faixas.
"""

from __future__ import annotations

from enum import StrEnum

from eolica.domain.turbine.value_objects import TurbineSpec, WindSpeed

# Fração da faixa útil (cut-in → cut-out) que separa carga parcial de plena.
# Aproxima a velocidade nominal, que a folha de dados da Aventa não declara.
RATED_WIND_FRACTION = 0.55


class OperatingRegime(StrEnum):
    """Faixa de operação, determinada pelo vento incidente."""

    BELOW_CUT_IN = "below_cut_in"
    """Vento insuficiente para gerar. Rotor parado ou em rotação livre."""

    PARTIAL_LOAD = "partial_load"
    """Faixa de perseguição de máxima potência. É onde a máquina passa a maior
    parte do tempo produtivo e onde a assinatura é mais informativa."""

    FULL_LOAD = "full_load"
    """Acima da velocidade nominal: potência limitada por controle de pitch."""

    ABOVE_CUT_OUT = "above_cut_out"
    """Vento excessivo. A turbina para por proteção — e parar aqui é o
    comportamento correto, não anomalia."""

    @classmethod
    def of(cls, wind: WindSpeed, spec: TurbineSpec) -> OperatingRegime:
        if wind.mps < spec.cut_in_mps:
            return cls.BELOW_CUT_IN
        if wind.mps > spec.cut_out_mps:
            return cls.ABOVE_CUT_OUT

        rated = spec.cut_in_mps + RATED_WIND_FRACTION * (spec.cut_out_mps - spec.cut_in_mps)
        return cls.PARTIAL_LOAD if wind.mps < rated else cls.FULL_LOAD

    @property
    def is_productive(self) -> bool:
        """True nas faixas em que se espera geração."""
        return self in {OperatingRegime.PARTIAL_LOAD, OperatingRegime.FULL_LOAD}
