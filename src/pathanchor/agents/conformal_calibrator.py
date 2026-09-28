"""Conformal calibrator agent.

Ref: Sec. III-E and Eq. (2); Sec. III-D ("Conformal calibrator (CP) creates the
timed interval and the choice not to act").

Abstention is driven by the width of the prediction set rather than by a
confidence cut chosen by hand, so the quantity that triggers it carries a
finite-sample coverage guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.agents.contracts import CalibrationResult, Recommendation
from pathanchor.conformal.split import PredictionSet, SplitConformal
from pathanchor.paths.signatures import EvidencePath
from pathanchor.utils.types import FloatArray, IntArray


@dataclass(slots=True)
class ConformalCalibrator:
    """Turns a path-marginal score into a calibrated value and a decision."""

    layer: SplitConformal

    def run(self, patient_id: str, option: str, score: float) -> CalibrationResult:
        values = np.asarray([score], dtype=np.float64)
        calibrated = float(self.layer.apply_transform(values)[0])
        low, high = self.layer.interval(values)
        prediction_set = PredictionSet(float(low[0]), float(high[0]))
        return CalibrationResult(
            patient_id=patient_id,
            option=option,
            score=float(score),
            calibrated=calibrated,
            prediction_set=prediction_set,
            abstains=prediction_set.width > self.layer.tau_abs,
        )

    def decide(
        self,
        patient_id: str,
        option: str,
        score: float,
        evidence: Sequence[EvidencePath],
        feasible: bool,
    ) -> Recommendation:
        result = self.run(patient_id, option, score)
        return Recommendation(
            patient_id=patient_id,
            option=result.option,
            calibrated_response=result.calibrated,
            evidence_paths=tuple(evidence),
            prediction_set=result.prediction_set,
            deferred=result.abstains,
            feasible=feasible,
        )

    def calibrate_batch(
        self,
        patient_ids: Sequence[str],
        options: Sequence[str],
        scores: FloatArray,
    ) -> list[CalibrationResult]:
        return [
            self.run(patient_ids[index], options[index], float(scores[index]))
            for index in range(len(patient_ids))
        ]

    def widths(self, scores: FloatArray) -> FloatArray:
        low, high = self.layer.interval(scores)
        return high - low

    def defer_mask(self, scores: FloatArray) -> np.ndarray:
        return self.widths(scores) > self.layer.tau_abs


def coverage_report(
    layer: SplitConformal, scores: FloatArray, labels: IntArray
) -> dict[str, float]:
    widths = layer.mean_width(scores)
    return {
        "coverage": layer.coverage(scores, labels),
        "nominal": layer.nominal_level(),
        "mean_width": widths,
        "quantile": layer.quantile,
        "n_calibration": float(layer.n_calibration),
    }


def abstention_profile(recommendations: Sequence[Recommendation]) -> dict[str, int]:
    counts = {"deferred": 0, "recommended": 0}
    for item in recommendations:
        if item.deferred:
            counts["deferred"] += 1
        else:
            counts["recommended"] += 1
    return counts


def width_quantiles(scores: FloatArray, layer: SplitConformal) -> dict[str, float]:
    low, high = layer.interval(scores)
    span = high - low
    if span.size == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(np.min(span)),
        "median": float(np.median(span)),
        "max": float(np.max(span)),
    }
