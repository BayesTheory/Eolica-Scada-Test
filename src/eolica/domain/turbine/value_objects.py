"""Value objects do subdomínio `turbine`: grandezas físicas com invariantes.

Cada grandeza carrega sua própria validação. A alternativa — passar `float` cru
por toda parte — é o que permitiu ao v1 propagar um `NaN` de vento por três
camadas até virar uma previsão silenciosamente errada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from eolica.shared.errors import InvalidValueError

ABSOLUTE_ZERO_CELSIUS = -273.15


def _require_finite(value: float, subject: str) -> None:
    """Rejeita NaN e infinito.

    Vale para todas as grandezas: `NaN` num sensor não é "valor desconhecido
    que dá pra tocar o barco", é ausência de medição, e precisa ser tratada
    explicitamente pelo pipeline de ingestão — não absorvida por um value object.
    """
    if not math.isfinite(value):
        raise InvalidValueError(
            f"{subject} deve ser uma grandeza finita", value=repr(value), subject=subject
        )


@dataclass(frozen=True, slots=True, order=True)
class WindSpeed:
    """Velocidade do vento no anemômetro da nacele, em m/s."""

    mps: float

    def __post_init__(self) -> None:
        _require_finite(self.mps, "Velocidade do vento")
        if self.mps < 0:
            raise InvalidValueError("Velocidade do vento não pode ser negativa", value=self.mps)


@dataclass(frozen=True, slots=True, order=True)
class PowerKw:
    """Potência ativa no conversor, em kW.

    Valores negativos são fisicamente reais: com vento abaixo do cut-in a
    turbina consome da rede para manter eletrônica e controle de pitch. O
    dataset tem 24 linhas assim só no recorte de duas semanas.

    Por isso a distinção entre `kw` (a medida, preservada) e `for_display()`
    (o número que vai ao operador, com clamp em zero). No v1 essa regra estava
    no *prompt do LLM* — o que significa que ela valia só quando o operador
    perguntava via chat, e que uma mudança de modelo de linguagem podia
    revogá-la sem ninguém notar.
    """

    kw: float

    def __post_init__(self) -> None:
        _require_finite(self.kw, "Potência")

    @property
    def is_parasitic(self) -> bool:
        """True quando a turbina está consumindo em vez de gerar."""
        return self.kw < 0

    def for_display(self) -> float:
        """Potência apresentada ao operador: nunca negativa."""
        return max(0.0, self.kw)


@dataclass(frozen=True, slots=True, order=True)
class RotorSpeed:
    """Rotação do rotor, em RPM."""

    rpm: float

    def __post_init__(self) -> None:
        _require_finite(self.rpm, "Rotação do rotor")
        if self.rpm < 0:
            raise InvalidValueError("Rotação do rotor não pode ser negativa", value=self.rpm)


@dataclass(frozen=True, slots=True, order=True)
class Temperature:
    """Temperatura do estator do gerador, em graus Celsius."""

    celsius: float

    def __post_init__(self) -> None:
        _require_finite(self.celsius, "Temperatura")
        if self.celsius <= ABSOLUTE_ZERO_CELSIUS:
            raise InvalidValueError(
                "Temperatura não pode estar no zero absoluto ou abaixo", value=self.celsius
            )


@dataclass(frozen=True, slots=True, order=True)
class PitchAngle:
    """Ângulo de passo das pás, em graus."""

    degrees: float

    def __post_init__(self) -> None:
        _require_finite(self.degrees, "Ângulo de pitch")
        if not -180.0 <= self.degrees <= 180.0:
            raise InvalidValueError(
                "Ângulo de pitch fora da faixa física [-180, 180]", value=self.degrees
            )


class OperatingStatus(IntEnum):
    """Estado operacional reportado pelo canal `SERVER.TurSt`.

    Só dois códigos têm semântica que este projeto consegue *defender* a partir
    de evidência:

    - ``10`` — operação normal. Era o filtro `STATUS_OPERACAO` do pipeline v1 e
      é o conjunto sobre o qual o autoencoder foi treinado.
    - ``13`` — falha. Era o `STATUS_FAULT` do script de análise de features.

    O dataset também contém os códigos 8, 9, 11, 12 e 305, que **não constam do
    metadado do fabricante** (`data/metadata/scada_channels.csv` documenta o
    canal, não o dicionário de valores). Inventar um significado para eles seria
    pior que admitir a lacuna: viraria regra de negócio baseada em chute.

    Por isso ``UNKNOWN``. O código 9 sozinho responde por 38% do dataset — um
    "não sei" honesto aqui é informação, não omissão.
    """

    PRODUCING = 10
    FAULT = 13
    UNKNOWN = -1

    @classmethod
    def from_code(cls, code: float | int) -> OperatingStatus:
        """Converte o código bruto do SCADA, sem nunca levantar exceção.

        O CSV traz o status como float (`10.0`) porque o resample de 10 minutos
        tira a média da coluna — o que, aliás, torna um status "10.4" possível
        e é a razão do arredondamento a montante.
        """
        try:
            return cls(int(code))
        except (ValueError, OverflowError):
            return cls.UNKNOWN

    @property
    def is_healthy_operation(self) -> bool:
        """True apenas para operação normal comprovada."""
        return self is OperatingStatus.PRODUCING


@dataclass(frozen=True, slots=True)
class TurbineSpec:
    """Envelope operacional da turbina, vindo da folha de dados do fabricante.

    Os valores default são os da Aventa AV-7 do IET-OST, lidos de
    `data/metadata/turbine_metadata.json`.
    """

    rated_power_kw: float
    cut_in_mps: float
    cut_out_mps: float

    def __post_init__(self) -> None:
        _require_finite(self.rated_power_kw, "Potência nominal")
        _require_finite(self.cut_in_mps, "Velocidade de cut-in")
        _require_finite(self.cut_out_mps, "Velocidade de cut-out")
        if self.rated_power_kw <= 0:
            raise InvalidValueError("Potência nominal deve ser positiva", value=self.rated_power_kw)
        if self.cut_in_mps >= self.cut_out_mps:
            raise InvalidValueError(
                "Velocidade de cut-in deve ser menor que a de cut-out",
                cut_in=self.cut_in_mps,
                cut_out=self.cut_out_mps,
            )

    @classmethod
    def aventa_av7(cls) -> TurbineSpec:
        """Aventa AV-7 — 6.2 kW nominais, cut-in 2.0 m/s, cut-out 12.0 m/s.

        Fonte: `assembly.rated_power` (6200 W) e `control.supervisory.Vin/Vout`
        do metadado do fabricante.
        """
        return cls(rated_power_kw=6.2, cut_in_mps=2.0, cut_out_mps=12.0)

    def expects_production(self, wind: WindSpeed) -> bool:
        """True se, para este vento, se espera geração de energia.

        Fora da faixa a turbina legitimamente entrega ~0 kW: abaixo do cut-in
        não há torque suficiente, acima do cut-out ela se protege e para. Sem
        essa noção, um detector de anomalia acusa toda noite sem vento como
        falha — que é exatamente o tipo de falso positivo que faz operador
        desligar o alarme.
        """
        return self.cut_in_mps <= wind.mps <= self.cut_out_mps
