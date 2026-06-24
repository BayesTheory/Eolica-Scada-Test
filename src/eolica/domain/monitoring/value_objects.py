"""Value objects do subdomínio `monitoring`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from eolica.shared.errors import InsufficientDataError

# Convenção consolidada em risco de crédito e adotada em MLOps para PSI.
MODERATE_DRIFT_THRESHOLD = 0.10
SEVERE_DRIFT_THRESHOLD = 0.25


class DriftMethod(StrEnum):
    """Estatística usada para comparar duas distribuições."""

    PSI = "psi"
    KS = "kolmogorov_smirnov"


class DriftSeverity(StrEnum):
    """Quanto a distribuição atual se afastou da de referência."""

    NONE = "none"
    MODERATE = "moderate"
    SEVERE = "severe"

    @classmethod
    def classify(cls, value: float) -> DriftSeverity:
        """Classifica um PSI pela convenção de mercado.

        - ``< 0.10`` — estável, nada a fazer.
        - ``0.10–0.25`` — investigar; pode ser sazonalidade.
        - ``> 0.25`` — agir; o modelo provavelmente precisa de retreino.
        """
        if value >= SEVERE_DRIFT_THRESHOLD:
            return cls.SEVERE
        if value >= MODERATE_DRIFT_THRESHOLD:
            return cls.MODERATE
        return cls.NONE


@dataclass(frozen=True, slots=True)
class DriftScore:
    """O drift medido para uma feature."""

    value: float
    method: DriftMethod

    @property
    def severity(self) -> DriftSeverity:
        return DriftSeverity.classify(self.value)


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Drift de todas as features monitoradas, com veredito agregado."""

    scores: Mapping[str, DriftScore]

    @classmethod
    def of(cls, scores: Mapping[str, DriftScore]) -> DriftReport:
        if not scores:
            raise InsufficientDataError(required=1, available=0, subject="features monitoradas")
        return cls(scores=dict(scores))

    @property
    def worst_feature(self) -> str:
        """A feature que mais se deslocou — por onde começar a investigação."""
        return max(self.scores, key=lambda name: self.scores[name].value)

    @property
    def severity(self) -> DriftSeverity:
        """A severidade do relatório é a da pior feature.

        Agregar por média esconderia exatamente o caso que interessa: uma única
        feature colapsando enquanto as outras seguem estáveis.
        """
        return self.scores[self.worst_feature].severity

    @property
    def requires_action(self) -> bool:
        """True apenas em drift severo.

        `MODERATE` sinaliza investigação, não retreino automático: sazonalidade
        de vento move distribuição todo trimestre sem que o modelo tenha
        piorado.
        """
        return self.severity is DriftSeverity.SEVERE
