"""Assembly of a split's records into a differentiable evidence store.

Ref: Algorithm 1 (the five-agent pipeline over one record), Eq. (1) (the
projected arg max), Sec. III-C (retained path sets).

Enumerating a record's paths is the expensive step, so it happens once per split
and the result is kept as a flat path table with a segment index. Training and
evaluation then read the same store, which is what makes the reported split
values comparable across arms: every arm scores the same retained sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from pathanchor.agents.conformal_calibrator import ConformalCalibrator
from pathanchor.agents.contracts import AgentLedger, RetrievalRequest
from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.agents.retriever import Retriever
from pathanchor.cohort.schema import PatientRecord, TreatmentClass
from pathanchor.conformal.split import SplitConformal
from pathanchor.graph.grounding import (
    EntityDirectory,
    anchor_indices,
    ground_record,
    ground_treatments,
)
from pathanchor.graph.index import CompactGraph
from pathanchor.models.encoder import FEATURE_DIM, MODALITY_ORDER
from pathanchor.models.estimator import PathAnchorEstimator
from pathanchor.paths.enumerate import ProposalBudget
from pathanchor.paths.marginal import segment_log_sum_exp, segment_mean
from pathanchor.paths.scorer import encode_paths
from pathanchor.paths.signatures import EvidencePath
from pathanchor.utils.types import FloatArray, IntArray

TREATMENT_OPTIONS: tuple[str, ...] = tuple(item.value for item in TreatmentClass)


@dataclass(slots=True)
class PatientEvidence:
    """One record's retained path sets, one segment per feasible option."""

    patient_id: str
    features: FloatArray
    modality: FloatArray
    label: int
    is_braf_v600: bool
    options: tuple[str, ...]
    paths: tuple[EvidencePath, ...]
    segments: IntArray
    option_weights: FloatArray
    deferred_evidence: bool


@dataclass(slots=True)
class EvidenceStore:
    records: tuple[PatientRecord, ...]
    items: tuple[PatientEvidence, ...]
    ledger: AgentLedger
    budget: ProposalBudget
    key_index: Mapping[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.items)

    def labels(self) -> IntArray:
        return np.asarray([item.label for item in self.items], dtype=np.int64)

    def features(self) -> FloatArray:
        return np.asarray([item.features for item in self.items], dtype=np.float64)

    def modality(self) -> FloatArray:
        return np.asarray([item.modality for item in self.items], dtype=np.float64)

    def braf_mask(self) -> np.ndarray:
        return np.asarray([item.is_braf_v600 for item in self.items], dtype=bool)

    def option_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            for option in item.options:
                counts[option] = counts.get(option, 0) + 1
        return dict(sorted(counts.items()))

    def path_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            for segment, option in enumerate(item.options):
                counts[option] = counts.get(option, 0) + int(np.sum(item.segments == segment))
        return dict(sorted(counts.items()))

    def flat_paths(self) -> list[EvidencePath]:
        out: list[EvidencePath] = []
        for item in self.items:
            out.extend(item.paths)
        return out

    def offset_segments(self) -> IntArray:
        offsets: list[int] = []
        base = 0
        for item in self.items:
            offsets.append(base)
            base += len(item.paths)
        return np.asarray(offsets, dtype=np.int64)


def _record_features(record: PatientRecord) -> tuple[FloatArray, FloatArray]:
    present = set(record.observed_modalities())
    modality = np.asarray(
        [1.0 if name in present else 0.0 for name in MODALITY_ORDER], dtype=np.float64
    )
    return (np.asarray(record.feature_vector(), dtype=np.float64), modality)


def build_evidence_store(
    records: Sequence[PatientRecord],
    graph: CompactGraph,
    directory: EntityDirectory,
    budget: ProposalBudget,
    adjudicator: GuidelineAdjudicator,
    validator: OntologyValidator,
    critic: CounterfactualCritic,
    min_paths: int = 1,
    restrict_to_feasible: bool = True,
) -> EvidenceStore:
    """Run the retriever, validator and critic over every record of a split.

    ``restrict_to_feasible`` is what separates the deployed system from the
    leave-one-out arm of Table II Panel B: with the projection removed the proposal
    covers every treatment class, so the arg max can land outside C(p) and the
    guideline-discordance rate becomes measurable instead of being zero by
    construction.
    """
    targets = ground_treatments(graph, directory, TREATMENT_OPTIONS)
    retriever = Retriever(graph=graph, targets=targets, budget=budget)
    lookup = {key: index for index, key in enumerate(graph.keys())}
    items: list[PatientEvidence] = []
    ledger = AgentLedger()
    for record in records:
        subgraph = ground_record(graph, directory, record)
        anchors = anchor_indices(graph, subgraph)
        adjudication = adjudicator.adjudicate(record)
        feasible = adjudication.feasibility.options()
        scored = feasible if restrict_to_feasible else TREATMENT_OPTIONS
        ledger.patients += 1
        if not feasible:
            ledger.empty_feasible += 1
        request = RetrievalRequest(
            patient_id=record.patient_id, anchors=tuple(anchors), options=tuple(scored)
        )
        retrieval = retriever.run(request)
        ledger.proposals += retrieval.stats.proposals
        validation = validator.run(retrieval)
        ledger.rejected += validation.rejected
        ledger.validated += sum(len(paths) for paths in validation.validated.values())
        critique = critic.run(record.patient_id, anchors, scored, validation.validated)
        ledger.downweighted += critique.downweighted
        options: list[str] = []
        paths: list[EvidencePath] = []
        segments: list[int] = []
        weights: list[float] = []
        for option in scored:
            kept = critique.weighted.get(option, ())
            if len(kept) < min_paths:
                continue
            segment = len(options)
            options.append(option)
            for path, factor in kept:
                paths.append(path)
                segments.append(segment)
                weights.append(float(factor) * float(np.mean(np.abs(path.weights()))))
        features, modality = _record_features(record)
        items.append(
            PatientEvidence(
                patient_id=record.patient_id,
                features=features,
                modality=modality,
                label=int(record.response),
                is_braf_v600=record.is_braf_v600,
                options=tuple(options),
                paths=tuple(paths),
                segments=np.asarray(segments, dtype=np.int64),
                option_weights=np.asarray(weights, dtype=np.float64),
                deferred_evidence=len(options) == 0,
            )
        )
    ledger.deferred = sum(1 for item in items if item.deferred_evidence)
    return EvidenceStore(
        records=tuple(records),
        items=tuple(items),
        ledger=ledger,
        budget=budget,
        key_index=lookup,
    )


@dataclass(frozen=True, slots=True)
class OptionScoreMatrix:
    """Per-patient, per-option scores with the padding mask."""

    scores: Tensor
    weights: Tensor
    options: tuple[tuple[str, ...], ...]


def score_options(
    store: EvidenceStore,
    graph: CompactGraph,
    model: PathAnchorEstimator,
    use_mean_pool: bool = False,
) -> OptionScoreMatrix:
    """Score every (record, retained option) pair with the estimator.

    One forwards pass per record: the record's paths for all its options travel as
    a single padded batch, the patient embedding conditions the readout, and the
    per-option marginal is a segmented log-sum-exp over the same score vector.
    """
    estimator = model
    width = max(len(item.options) for item in store.items) if store.items else 1
    rows = np.full((len(store.items), width), -60.0, dtype=np.float64)
    mask = np.zeros((len(store.items), width), dtype=np.float64)
    option_names: list[tuple[str, ...]] = []
    for row, item in enumerate(store.items):
        option_names.append(item.options)
        if not item.paths:
            continue
        batch = encode_paths(graph, item.paths, store.budget.max_hops, store.key_index)
        features = torch.as_tensor(item.features, dtype=torch.float32).unsqueeze(0)
        modality = torch.as_tensor(item.modality, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded = estimator.encode(features, modality)
            path_scores = estimator.path_scores(batch, encoded)
        segments = torch.as_tensor(item.segments, dtype=torch.long)
        if use_mean_pool:
            aggregated = segment_mean(path_scores, segments, len(item.options))
        else:
            aggregated = segment_log_sum_exp(path_scores, segments, len(item.options))
        values = [float(value) for value in aggregated.detach().tolist()]
        for segment, _name in enumerate(item.options):
            if segment < width:
                rows[row, segment] = values[segment]
                mask[row, segment] = 1.0
    return OptionScoreMatrix(
        scores=torch.as_tensor(rows, dtype=torch.float32),
        weights=torch.as_tensor(mask, dtype=torch.float32),
        options=tuple(option_names),
    )


def selected_options(matrix: OptionScoreMatrix) -> list[str]:
    """The arg max over the feasible set of Eq. (1), per record."""
    chosen: list[str] = []
    for row, names in enumerate(matrix.options):
        if not names:
            chosen.append("")
            continue
        values = matrix.scores[row].detach().tolist()
        best = max(
            range(len(names)),
            key=lambda index: (
                values[index] if matrix.weights[row, index] > 0 else -np.inf,
                names[index],
            ),
        )
        chosen.append(names[best])
    return chosen


def feature_width() -> int:
    return FEATURE_DIM


def calibrate_store(
    store: EvidenceStore,
    matrix: OptionScoreMatrix,
    conformal: SplitConformal,
) -> list[dict[str, object]]:
    """Apply the conformal layer to each record's selected option."""
    calibrator = ConformalCalibrator(layer=conformal)
    chosen = selected_options(matrix)
    rows: list[dict[str, object]] = []
    for row, item in enumerate(store.items):
        option = chosen[row]
        if not option:
            rows.append({"patient_id": item.patient_id, "option": "", "deferred": True})
            continue
        score = float(matrix.scores[row, matrix.options[row].index(option)])
        decision = calibrator.decide(item.patient_id, option, score, item.paths, feasible=True)
        rows.append(
            {
                "patient_id": item.patient_id,
                "option": decision.option,
                "score": score,
                "calibrated": decision.calibrated_response,
                "low": decision.prediction_set.low,
                "high": decision.prediction_set.high,
                "width": decision.prediction_set.width,
                "deferred": decision.deferred,
            }
        )
    return rows
