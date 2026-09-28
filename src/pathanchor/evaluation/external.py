"""External validation protocol.

Ref: Sec. III-F ("an additional validation on the 265 samples ... from four
available immune-checkpoint-inhibitor melanoma samples"), Sec. IV-E (AUROC 0.724
against 0.711 for the best comparator, ECE rising to 0.042), Sec. IV-E (the
modality-ablation ladder 0.702 / 0.741 / 0.756 / 0.765).

The four transcriptomic series are public, but this release does not redistribute
them: the module reads a caller-supplied profile table and reports what it finds,
and it marks the cohort-level value unmoved when the table is absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from pathanchor.metrics.calibration import expected_calibration_error
from pathanchor.metrics.discrimination import auroc
from pathanchor.models.encoder import MODALITY_ORDER
from pathanchor.utils.types import FloatArray, IntArray

EXTERNAL_SERIES: tuple[tuple[str, int, str], ...] = (
    ("Hugo 2017, pre-treatment melanoma under anti-PD-1 checkpoint inhibition", 28, "GSE78220"),
    ("Riaz 2017, tumour mutational and microenvironmental sculpting by ICB", 109, "GSE91061"),
    ("Auslander 2018, robust response prediction to ICB in metastatic melanoma", 37, "GSE115821"),
    (
        "Gide 2019, immune cell populations defining response to anti-PD-1 and combined therapy",
        91,
        "PRJEB23709",
    ),
)

SERIES_TOTAL: int = sum(count for _, count, _ in EXTERNAL_SERIES)

MODALITY_LADDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Clinical variables alone", ("clinical",)),
    ("+ molecular features", ("clinical", "molecular")),
    ("+ histopathology", ("clinical", "molecular", "histology")),
    ("Full multimodal input", ("clinical", "molecular", "histology", "image")),
)


@dataclass(frozen=True, slots=True)
class ExternalProfileTable:
    """Labels and expression profiles of one harmonised external cohort."""

    labels: IntArray
    profiles: FloatArray
    series: tuple[str, ...]

    def validate(self) -> None:
        if self.labels.shape[0] != self.profiles.shape[0]:
            raise ValueError("external labels and profiles disagree in length")
        if not bool(np.all((self.labels == 0) | (self.labels == 1))):
            raise ValueError("external labels must be binary")


@dataclass(frozen=True, slots=True)
class ExternalResult:
    n: int
    framework_auroc: float
    comparator_auroc: float
    framework_ece: float
    comparator_ece: float

    def summary(self) -> Mapping[str, object]:
        return {
            "n": self.n,
            "framework_auroc": self.framework_auroc,
            "comparator_auroc": self.comparator_auroc,
            "framework_ece": self.framework_ece,
            "comparator_ece": self.comparator_ece,
        }


def evaluate_external(
    table: ExternalProfileTable,
    framework_scores: FloatArray,
    comparator_scores: FloatArray,
) -> ExternalResult:
    table.validate()
    return ExternalResult(
        n=int(table.labels.shape[0]),
        framework_auroc=auroc(table.labels, framework_scores),
        comparator_auroc=auroc(table.labels, comparator_scores),
        framework_ece=expected_calibration_error(table.labels, _probabilities(framework_scores)),
        comparator_ece=expected_calibration_error(table.labels, _probabilities(comparator_scores)),
    )


def degradation(
    internal_auroc: float, internal_ece: float, result: ExternalResult
) -> dict[str, float]:
    """How far the external values sit below the internal ones.

    Sec. IV-E reports 4.1 points of discrimination and 1.1 points of calibration
    against the internal test split; the function reports the measured equivalent
    rather than the manuscript's figures.
    """
    return {
        "discrimination_points": (internal_auroc - result.framework_auroc) * 100.0,
        "calibration_points": (result.framework_ece - internal_ece) * 100.0,
    }


def series_census() -> dict[str, int]:
    return {name: count for name, count, _ in EXTERNAL_SERIES}


def series_accessions() -> dict[str, str]:
    return {name: accession for name, _, accession in EXTERNAL_SERIES}


def ladder_report(
    table: ExternalProfileTable,
    score_by_modality: Mapping[str, FloatArray],
) -> list[dict[str, object]]:
    """The four-rung modality ladder of Sec. IV-E.

    Each rung is a named subset of the modality order, so a caller cannot report a
    ladder whose rungs are not nested.
    """
    for _, modalities in MODALITY_LADDER:
        missing = [item for item in modalities if item not in MODALITY_ORDER]
        if missing:
            raise ValueError(f"the ladder names unknown modalities: {missing}")
    rows: list[dict[str, object]] = []
    for name, modalities in MODALITY_LADDER:
        if name not in score_by_modality:
            continue
        rows.append(
            {
                "rung": name,
                "modalities": len(modalities),
                "auroc": auroc(table.labels, score_by_modality[name]),
                "n": int(table.labels.shape[0]),
            }
        )
    return rows


def withheld_modality_delta(
    full_auroc: float,
    withheld_auroc: float,
) -> float:
    """Points lost when one modality is withheld (Sec. IV-E reports 0.6 points)."""
    return (full_auroc - withheld_auroc) * 100.0


def _probabilities(values: FloatArray) -> FloatArray:
    shifted = values - float(np.max(values))
    return 1.0 / (1.0 + np.exp(-shifted))


@dataclass(slots=True)
class LeaveOneSiteOut:
    """Retraining schedule that holds out one contributing site at a time."""

    sites: tuple[str, ...] = ("Site A", "Site B", "Site C", "Site D")
    results: dict[str, float] = field(default_factory=dict)

    def record(self, site: str, auroc_value: float) -> None:
        if site not in self.sites:
            raise ValueError(f"unknown site '{site}'")
        self.results[site] = float(auroc_value)

    def summary(self) -> Mapping[str, object]:
        values = [value for value in self.results.values() if np.isfinite(value)]
        return {
            "sites": len(self.results),
            "min": float(min(values)) if values else float("nan"),
            "max": float(max(values)) if values else float("nan"),
            "complete": len(self.results) == len(self.sites),
        }
