"""Adaptador de `ScadaRepository` sobre CSV.

CSV é a fonte deste projeto hoje. A porta existe justamente para que trocá-lo
por TimescaleDB ou Parquet particionado não toque domínio nem caso de uso —
basta outro adaptador satisfazendo o mesmo `Protocol`.
"""

from __future__ import annotations

import bisect
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path

import pandas as pd

from eolica.domain.turbine import (
    OperatingStatus,
    PitchAngle,
    PowerKw,
    RotorSpeed,
    Temperature,
    TurbineReading,
    WindSpeed,
)
from eolica.infrastructure.persistence.schemas import (
    COLUMN_MAPPING,
    INDEX_NAME,
    validate_scada_frame,
)
from eolica.shared.errors import (
    ContractViolationError,
    DataSourceError,
    InsufficientDataError,
    NotFoundError,
)


class CsvScadaRepository:
    """Telemetria em memória, indexada por timestamp.

    O dataset inteiro são ~65 mil linhas (25 MB): cabe na memória com folga e
    dispensa banco. O que **não** se repete do v1 é carregar isso no import do
    módulo — aqui a construção é explícita e acontece no `lifespan` da aplicação,
    onde uma falha vira readiness probe vermelho em vez de `sys.exit(1)` durante
    a coleta de testes.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        validate_scada_frame(frame)
        self._frame = frame
        # `validate_scada_frame` já garantiu o tipo do índice; guardá-lo numa
        # variável tipada estreita `Index[Any]` para `DatetimeIndex` de uma vez
        # só, em vez de espalhar `cast` por todo acesso a `.tz` e `.date`.
        self._index: pd.DatetimeIndex = _as_datetime_index(frame)
        self._timestamps: list[datetime] = self._index.to_pydatetime().tolist()
        self._days = frozenset(self._index.date)

    @classmethod
    def from_path(cls, path: Path, *, timezone: str = "UTC") -> CsvScadaRepository:
        """Carrega e normaliza um CSV de telemetria já reamostrada.

        Os timestamps do arquivo são ingênuos; o metadado do fabricante diz
        explicitamente "UTC Time". Localizar aqui, na fronteira, evita que
        comparação entre aware e naive estoure lá na frente.
        """
        if not path.exists():
            raise DataSourceError(f"Arquivo de telemetria não encontrado: {path}", path=str(path))
        try:
            frame = pd.read_csv(path, index_col=INDEX_NAME, parse_dates=True)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise DataSourceError(
                f"Falha ao ler o CSV de telemetria: {exc}", path=str(path)
            ) from exc

        index = _as_datetime_index(frame)
        if index.tz is None:
            frame.index = index.tz_localize(timezone)
        return cls(frame.sort_index())

    # ── porta ScadaRepository ────────────────────────────────────────────────

    def readings_for_day(self, day: date) -> list[TurbineReading]:
        if day not in self._days:
            raise NotFoundError("Telemetria", day.isoformat())
        start = datetime.combine(day, time.min, tzinfo=self._tzinfo())
        end = datetime.combine(day, time.max, tzinfo=self._tzinfo())
        return self.readings_between(start, end)

    def readings_between(self, start: datetime, end: datetime) -> list[TurbineReading]:
        left = bisect.bisect_left(self._timestamps, start)
        right = bisect.bisect_right(self._timestamps, end)
        return self._materialise(left, right)

    def readings_before(self, moment: datetime, *, limit: int) -> list[TurbineReading]:
        cut = bisect.bisect_right(self._timestamps, moment)
        left = max(0, cut - limit)
        available = cut - left
        if available < limit:
            raise InsufficientDataError(required=limit, available=available, subject="observações")
        return self._materialise(left, cut)

    def available_range(self) -> tuple[datetime, datetime]:
        if not self._timestamps:
            raise NotFoundError("Acervo", "vazio")
        return self._timestamps[0], self._timestamps[-1]

    # ── extras usados pela calibração e pelo monitoramento ───────────────────

    def normal_operation_readings(self) -> list[TurbineReading]:
        """Apenas leituras em operação normal (status 10).

        É o conjunto de referência para calibrar o limiar de anomalia e para ser
        a baseline do drift — o mesmo recorte sobre o qual o autoencoder é
        treinado.
        """
        mask = self._frame[_source_column("status")].round() == OperatingStatus.PRODUCING.value
        # `nonzero` sobre a máscara dá as posições direto, sem procurar cada
        # timestamp na lista — o que era O(n²) sobre 32 mil leituras.
        positions = mask.to_numpy().nonzero()[0]
        return [self._reading_at(int(position)) for position in positions]

    def feature_series(self, feature: str, *, since: datetime | None = None) -> list[float]:
        """Série bruta de uma feature canônica, para cálculo de drift."""
        frame = self._frame if since is None else self._frame.loc[since:]
        return [float(value) for value in frame[_source_column(feature)]]

    def __len__(self) -> int:
        return len(self._frame)

    # ── internos ─────────────────────────────────────────────────────────────

    def _tzinfo(self) -> tzinfo | None:
        return self._index.tz

    def _materialise(self, left: int, right: int) -> list[TurbineReading]:
        return [self._reading_at(position) for position in range(left, right)]

    def _reading_at(self, position: int) -> TurbineReading:
        row = self._frame.iloc[position]
        return TurbineReading(
            timestamp=self._timestamps[position],
            wind_speed=WindSpeed(float(row["WindSpeed"])),
            power=PowerKw(float(row["PowerOutput"])),
            rotor_speed=RotorSpeed(float(row["RotorSpeed"])),
            generator_temperature=Temperature(float(row["GeneratorTemperature"])),
            pitch=PitchAngle(float(row["PitchDeg"])),
            status=OperatingStatus.from_code(row["Status_rounded"]),
        )


def _as_datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Estreita `Index[Any]` para `DatetimeIndex`, falhando alto se não for.

    O pandas tipa `DataFrame.index` como `Index[Any]`, então `.tz` e `.date` não
    são visíveis ao verificador de tipos. Fazer a checagem uma vez, num ponto
    só, é melhor que espalhar `cast()` — e transforma um `AttributeError` obscuro
    em erro de contrato nomeado.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ContractViolationError(
            contract="scada.index.v1",
            violations=[f"o índice deve ser DatetimeIndex, veio {type(index).__name__}"],
        )
    return index


_REVERSE_MAPPING = {canonical: source for source, canonical in COLUMN_MAPPING.items()}


def _source_column(canonical: str) -> str:
    try:
        return _REVERSE_MAPPING[canonical]
    except KeyError:
        raise DataSourceError(
            f"Feature canônica sem coluna correspondente: '{canonical}'",
            available=sorted(_REVERSE_MAPPING),
        ) from None


DEFAULT_SAMPLING_INTERVAL = timedelta(minutes=10)
