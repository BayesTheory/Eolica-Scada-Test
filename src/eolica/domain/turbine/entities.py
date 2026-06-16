"""Entidades do subdomínio `turbine`: a leitura e a janela de leituras."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from eolica.domain.turbine.value_objects import (
    OperatingStatus,
    PitchAngle,
    PowerKw,
    RotorSpeed,
    Temperature,
    WindSpeed,
)
from eolica.shared.errors import InsufficientDataError, InvalidValueError


@dataclass(frozen=True, slots=True)
class TurbineReading:
    """Uma observação SCADA já agregada na grade de 10 minutos.

    `GeneratorSpeed` está deliberadamente ausente. O metadado do fabricante
    marca esse canal como ``Reliable Measurement = FALSE`` ("0-10V Signal from
    Generator (not an exact measurement)"), e mesmo assim ele aparecia na lista
    de features do v1. Um canal que o próprio fabricante diz ser impreciso não
    entra num modelo de detecção de anomalia: ele *vira* a anomalia.
    """

    timestamp: datetime
    wind_speed: WindSpeed
    power: PowerKw
    rotor_speed: RotorSpeed
    generator_temperature: Temperature
    pitch: PitchAngle
    status: OperatingStatus

    def feature(self, name: str) -> float:
        """Valor escalar de uma feature, pelo nome canônico."""
        try:
            accessor = _FEATURE_ACCESSORS[name]
        except KeyError:
            raise InvalidValueError(
                f"Feature desconhecida: '{name}'",
                requested=name,
                available=sorted(_FEATURE_ACCESSORS),
            ) from None
        return accessor(self)


# Nomes canônicos das features escalares. Esta é a única tradução entre o
# vocabulário do domínio e os nomes usados por modelos e feature store.
_FEATURE_ACCESSORS: Mapping[str, Callable[[TurbineReading], float]] = {
    "wind_speed": lambda r: r.wind_speed.mps,
    "power": lambda r: r.power.kw,
    "rotor_speed": lambda r: r.rotor_speed.rpm,
    "generator_temperature": lambda r: r.generator_temperature.celsius,
    "pitch": lambda r: r.pitch.degrees,
}

FEATURE_NAMES: tuple[str, ...] = tuple(sorted(_FEATURE_ACCESSORS))


def _validate_chronology(readings: tuple[TurbineReading, ...]) -> None:
    """Exige timestamps estritamente crescentes."""
    for previous, current in itertools.pairwise(readings):
        if current.timestamp <= previous.timestamp:
            raise InvalidValueError(
                "Leituras devem estar em ordem cronológica estrita",
                previous=previous.timestamp.isoformat(),
                current=current.timestamp.isoformat(),
            )


@dataclass(frozen=True, slots=True)
class ReadingWindow:
    """Sequência **contígua** de leituras — a unidade de entrada dos modelos.

    A contiguidade é a razão de esta classe existir. Um LSTM autoencoder assume
    que os passos da janela são igualmente espaçados; se um buraco de 24h passa
    despercebido no meio, o erro de reconstrução dispara e o sistema reporta
    "anomalia na turbina" quando o que houve foi uma anomalia *no coletor*.

    O dataset real tem 30 descontinuidades, duas delas maiores que um dia. O v1
    montava janelas com `iloc[-n:]`, que não tem como saber disso. Aqui é
    impossível construir uma janela furada: ou você usa `of()` e recebe um erro,
    ou usa `split_on_gaps()` e recebe os pedaços válidos.
    """

    readings: tuple[TurbineReading, ...]
    expected_interval: timedelta

    @classmethod
    def of(
        cls, readings: Iterable[TurbineReading], *, expected_interval: timedelta
    ) -> ReadingWindow:
        """Constrói uma janela, exigindo contiguidade.

        Levanta `InvalidValueError` se houver qualquer salto maior que
        `expected_interval`.
        """
        items = tuple(readings)
        if not items:
            raise InsufficientDataError(required=1, available=0, subject="leituras")

        _validate_chronology(items)

        for previous, current in itertools.pairwise(items):
            delta = current.timestamp - previous.timestamp
            if delta > expected_interval:
                raise InvalidValueError(
                    "A janela atravessa uma descontinuidade temporal",
                    gap=str(delta),
                    expected=str(expected_interval),
                    at=current.timestamp.isoformat(),
                )

        return cls(readings=items, expected_interval=expected_interval)

    @classmethod
    def split_on_gaps(
        cls,
        readings: Iterable[TurbineReading],
        *,
        expected_interval: timedelta,
        min_length: int = 1,
    ) -> list[ReadingWindow]:
        """Fatia uma sequência furada em janelas contíguas.

        Esta é a recuperação correta diante de um gap: não abortar o dia inteiro
        (o que descartaria dado bom), nem ignorá-lo (o que geraria alarme falso),
        mas analisar cada trecho íntegro separadamente.

        Segmentos com menos de `min_length` leituras são descartados — não dá
        para rodar uma janela de 60 passos sobre um trecho de 4.
        """
        items = tuple(readings)
        if not items:
            return []

        _validate_chronology(items)

        segments: list[list[TurbineReading]] = [[items[0]]]
        for previous, current in itertools.pairwise(items):
            if current.timestamp - previous.timestamp > expected_interval:
                segments.append([current])
            else:
                segments[-1].append(current)

        return [
            cls(readings=tuple(segment), expected_interval=expected_interval)
            for segment in segments
            if len(segment) >= min_length
        ]

    def __len__(self) -> int:
        return len(self.readings)

    @property
    def start(self) -> datetime:
        return self.readings[0].timestamp

    @property
    def end(self) -> datetime:
        return self.readings[-1].timestamp

    def series(self, name: str) -> tuple[float, ...]:
        """A série temporal de uma única feature ao longo da janela."""
        return tuple(reading.feature(name) for reading in self.readings)

    def matrix(self, feature_names: Iterable[str]) -> tuple[tuple[float, ...], ...]:
        """Matriz (passos × features) na ordem exata pedida.

        A ordem das colunas é responsabilidade de quem chama e precisa casar com
        a ordem usada no treino. O feature store é quem garante isso — ver
        `eolica.infrastructure.featurestore`.
        """
        names = tuple(feature_names)
        return tuple(tuple(reading.feature(name) for name in names) for reading in self.readings)
