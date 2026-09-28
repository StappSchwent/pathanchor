"""Typed interfaces between the five agents.

Ref: Sec. III-D ("They interact with well-defined objects but never with strings
of free text").

Every agent consumes and produces a frozen dataclass. Nothing crosses an agent
boundary as prose, so the pipeline's intermediate state can be type-checked,
counted and replayed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pathanchor.cohort.schema import PatientRecord
from pathanchor.conformal.split import PredictionSet, SplitConformal
from pathanchor.guideline.feasibility import FeasibilityResult
from pathanchor.paths.enumerate import ProposalStats
from pathanchor.paths.signatures import EvidencePath


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    patient_id: str
    anchors: tuple[int, ...]
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    patient_id: str
    proposals: Mapping[str, tuple[EvidencePath, ...]]
    anchor_count: int
    stats: ProposalStats


@dataclass(frozen=True, slots=True)
class ValidationResult:
    patient_id: str
    validated: Mapping[str, tuple[EvidencePath, ...]]
    rejected: int
    rejection_reasons: Mapping[str, int]
    coverage: float


@dataclass(frozen=True, slots=True)
class AdjudicationResult:
    patient_id: str
    feasibility: FeasibilityResult


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    patient_id: str
    weighted: Mapping[str, tuple[tuple[EvidencePath, float], ...]]
    downweighted: int
    non_discriminative_signatures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    patient_id: str
    option: str
    score: float
    calibrated: float
    prediction_set: PredictionSet
    abstains: bool


@dataclass(frozen=True, slots=True)
class Recommendation:
    patient_id: str
    option: str
    calibrated_response: float
    evidence_paths: tuple[EvidencePath, ...]
    prediction_set: PredictionSet
    deferred: bool
    feasible: bool

    def signature(self) -> str:
        return f"{self.patient_id}:{self.option}:{len(self.evidence_paths)}"


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """The per-patient record of what each agent returned."""

    recommendation: Recommendation
    feasibility: FeasibilityResult
    stats: ProposalStats
    rejected_paths: int
    retained_paths: int
    downweighted_paths: int


@dataclass(slots=True)
class AgentLedger:
    """Counters the pipeline publishes alongside its results."""

    patients: int = 0
    proposals: int = 0
    validated: int = 0
    rejected: int = 0
    downweighted: int = 0
    deferred: int = 0
    empty_feasible: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, name: str, count: int = 1) -> None:
        self.reasons[name] = self.reasons.get(name, 0) + count

    def as_dict(self) -> dict[str, int]:
        return {
            "patients": self.patients,
            "proposals": self.proposals,
            "validated": self.validated,
            "rejected": self.rejected,
            "downweighted": self.downweighted,
            "deferred": self.deferred,
            "empty_feasible": self.empty_feasible,
        }


@runtime_checkable
class RetrieverProtocol(Protocol):
    def run(self, request: RetrievalRequest) -> RetrievalResult: ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    def run(self, result: RetrievalResult) -> ValidationResult: ...


@runtime_checkable
class AdjudicatorProtocol(Protocol):
    def adjudicate(self, record: PatientRecord) -> AdjudicationResult: ...


@runtime_checkable
class CriticProtocol(Protocol):
    def run(
        self,
        patient_id: str,
        anchors: Sequence[int],
        options: Sequence[str],
        validated: Mapping[str, tuple[EvidencePath, ...]],
    ) -> CritiqueResult: ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    def run(
        self,
        patient_id: str,
        option: str,
        score: float,
    ) -> CalibrationResult: ...


def summarise_options(
    weighted: Mapping[str, tuple[tuple[EvidencePath, float], ...]],
) -> dict[str, int]:
    return {option: len(items) for option, items in weighted.items()}


def conformal_summary(layer: SplitConformal) -> dict[str, float]:
    return {
        "alpha": layer.alpha,
        "tau_abs": layer.tau_abs,
        "quantile": layer.quantile,
        "n_calibration": float(layer.n_calibration),
        "nominal_level": layer.nominal_level(),
    }
