"""Component and leave-one-out arms.

Ref: Table II Panel A (additive contributions against a multimodal feature-scoring
base) and Panel B (leave-one-out from the integrated system), Sec. IV-B.

Every arm scores the same retained path sets produced by the same enumerator, so a
difference between arms is a difference in the component under test rather than in
the proposal. The retriever is never an arm: removing it removes the quantity the
whole comparison is about.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import torch

from pathanchor.agents.counterfactual_critic import CounterfactualCritic
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.cohort.schema import PatientRecord
from pathanchor.evaluation.pipeline import (
    EvidenceStore,
    OptionScoreMatrix,
    build_evidence_store,
    score_options,
    selected_options,
)
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.metrics.calibration import expected_calibration_error
from pathanchor.metrics.discrimination import auroc
from pathanchor.models.estimator import EstimatorConfig, PathAnchorEstimator
from pathanchor.paths.enumerate import ProposalBudget
from pathanchor.utils.types import FloatArray


@dataclass(frozen=True, slots=True)
class AblationFlags:
    """The switches an arm may flip."""

    use_base: bool = True
    use_graph_paths: bool = True
    typed_paths: bool = True
    use_ontology_validator: bool = True
    use_counterfactual_critic: bool = True
    projection_enabled: bool = True
    conformal_enabled: bool = True
    agent_team: bool = True
    mean_pool_paths: bool = False

    def estimator_config(self, base: EstimatorConfig) -> EstimatorConfig:
        return replace(
            base,
            typed_paths=self.typed_paths,
            use_graph_paths=self.use_graph_paths,
            use_base=self.use_base,
        )

    def disabled(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.as_dict().items() if not value)

    def as_dict(self) -> Mapping[str, bool]:
        return {
            "base_arm": self.use_base,
            "graph_paths": self.use_graph_paths,
            "typed_paths": self.typed_paths,
            "ontology_validator": self.use_ontology_validator,
            "counterfactual_critic": self.use_counterfactual_critic,
            "projection": self.projection_enabled,
            "conformal": self.conformal_enabled,
            "agent_team": self.agent_team,
            "log_sum_exp": not self.mean_pool_paths,
        }


INTEGRATED = AblationFlags()

PANEL_A_ARMS: tuple[tuple[str, AblationFlags], ...] = (
    (
        "Multimodal feature-scoring base (no graph, no agents)",
        AblationFlags(
            use_graph_paths=False,
            use_ontology_validator=False,
            use_counterfactual_critic=False,
            projection_enabled=False,
            conformal_enabled=False,
            agent_team=False,
        ),
    ),
    (
        "+ typed multimodal knowledge-graph paths",
        AblationFlags(
            use_ontology_validator=False,
            use_counterfactual_critic=False,
            projection_enabled=False,
            conformal_enabled=False,
            agent_team=False,
        ),
    ),
    (
        "+ ontology validator alone",
        AblationFlags(
            use_counterfactual_critic=False,
            projection_enabled=False,
            conformal_enabled=False,
            agent_team=False,
        ),
    ),
    (
        "+ counterfactual critic alone",
        AblationFlags(
            use_ontology_validator=False,
            projection_enabled=False,
            conformal_enabled=False,
            agent_team=False,
        ),
    ),
    (
        "+ constrained projection alone",
        AblationFlags(
            use_ontology_validator=False,
            use_counterfactual_critic=False,
            conformal_enabled=False,
            agent_team=False,
        ),
    ),
    (
        "+ conformal calibration alone",
        AblationFlags(
            use_ontology_validator=False,
            use_counterfactual_critic=False,
            projection_enabled=False,
            agent_team=False,
        ),
    ),
    ("PATHANCHOR (integrated)", INTEGRATED),
)

PANEL_B_ARMS: tuple[tuple[str, AblationFlags], ...] = (
    ("PATHANCHOR (full)", INTEGRATED),
    (
        "- knowledge-graph edges (features only)",
        AblationFlags(use_graph_paths=False, projection_enabled=False, agent_team=False),
    ),
    ("- ontology validator", replace(INTEGRATED, use_ontology_validator=False)),
    ("- guideline projection (unconstrained)", replace(INTEGRATED, projection_enabled=False)),
    ("- counterfactual critic", replace(INTEGRATED, use_counterfactual_critic=False)),
    ("- typed paths (free-text rationale)", replace(INTEGRATED, typed_paths=False)),
    ("- conformal calibration", replace(INTEGRATED, conformal_enabled=False)),
    ("- agent team (single agent)", replace(INTEGRATED, agent_team=False)),
    ("mean-pool paths (no log-sum-exp)", replace(INTEGRATED, mean_pool_paths=True)),
)


@dataclass(frozen=True, slots=True)
class ArmResult:
    name: str
    disabled: tuple[str, ...]
    auroc: float
    ece: float
    discordance: float
    retained_paths: int
    scored_pairs: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "arm": self.name,
            "disabled": ",".join(self.disabled) if self.disabled else "-",
            "auroc": self.auroc,
            "ece": self.ece,
            "discordance": self.discordance,
            "retained_paths": self.retained_paths,
            "scored_pairs": self.scored_pairs,
        }


def build_store_for_arm(
    flags: AblationFlags,
    records: Sequence[object],
    graph: CompactGraph,
    directory: EntityDirectory,
    budget: ProposalBudget,
    adjudicator: GuidelineAdjudicator,
) -> EvidenceStore:
    """Build the arm's evidence store with only the agents it keeps."""
    validator = OntologyValidator(graph=graph, enforce_signatures=flags.use_ontology_validator)
    critic = CounterfactualCritic()
    store = build_evidence_store(
        list(records),  # type: ignore[arg-type]
        graph,
        directory,
        budget,
        adjudicator,
        validator,
        critic,
        restrict_to_feasible=flags.projection_enabled,
    )
    if not flags.use_counterfactual_critic:
        for item in store.items:
            item.option_weights = np.ones_like(item.option_weights)
    return store


def evaluate_arm(
    name: str,
    flags: AblationFlags,
    records: Sequence[object],
    graph: CompactGraph,
    directory: EntityDirectory,
    budget: ProposalBudget,
    adjudicator: GuidelineAdjudicator,
    estimator: PathAnchorEstimator,
    store: EvidenceStore,
) -> ArmResult:
    """Score one arm's store and report its three trust quantities."""
    if flags.use_base and not flags.use_graph_paths:
        matrix = score_base_only(store, estimator)
    else:
        matrix = score_options(store, graph, estimator, use_mean_pool=flags.mean_pool_paths)
    labels = store.labels()
    values = row_scores(matrix)
    probabilities = _probabilities(values)
    chosen = list(selected_options(matrix))
    discordant = _discordance(store, chosen, adjudicator)
    return ArmResult(
        name=name,
        disabled=flags.disabled(),
        auroc=float(auroc(labels, values)),
        ece=float(expected_calibration_error(labels, probabilities)),
        discordance=discordant,
        retained_paths=int(sum(len(item.paths) for item in store.items)),
        scored_pairs=int(np.sum(matrix.weights.numpy() > 0)),
    )


def row_scores(matrix: OptionScoreMatrix) -> FloatArray:
    """The patient-level score: the strongest admissible option."""
    values = matrix.scores.detach().numpy().astype(np.float64)
    mask = matrix.weights.detach().numpy() > 0
    masked = np.where(mask, values, -np.inf)
    out = np.max(masked, axis=1)
    return np.where(np.isfinite(out), out, -60.0)


def score_base_only(store: EvidenceStore, estimator: PathAnchorEstimator) -> OptionScoreMatrix:
    """The feature-only arm: the graph is not read at all."""
    width = max(len(item.options) for item in store.items) if store.items else 1
    rows = np.full((len(store.items), width), -60.0, dtype=np.float64)
    mask = np.zeros((len(store.items), width), dtype=np.float64)
    names: list[tuple[str, ...]] = []
    for row, item in enumerate(store.items):
        names.append(item.options)
        if not item.options:
            continue
        features = torch.as_tensor(item.features, dtype=torch.float32).unsqueeze(0)
        modality = torch.as_tensor(item.modality, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            encoded = estimator.encode(features, modality)
            logit = float(estimator.base_logit(encoded)[0])
        for column in range(len(item.options)):
            rows[row, column] = logit
            mask[row, column] = 1.0
    return OptionScoreMatrix(
        scores=torch.as_tensor(rows, dtype=torch.float32),
        weights=torch.as_tensor(mask, dtype=torch.float32),
        options=tuple(names),
    )


def _probabilities(values: FloatArray) -> FloatArray:
    shifted = values - float(np.max(values))
    return 1.0 / (1.0 + np.exp(-shifted))


def _discordance(
    store: EvidenceStore,
    chosen: Sequence[str],
    adjudicator: GuidelineAdjudicator,
) -> float:
    """Share of recommendations that leave C(p).

    The deployed system reports zero by construction; an arm that removed the
    projection can leave the feasible set, and this measures by how much rather
    than assuming.
    """
    if not store.items:
        return 0.0
    outside = 0
    for item, option in zip(store.items, chosen, strict=True):
        if not option:
            continue
        record = record_for(store, item.patient_id)
        feasibility = adjudicator.adjudicate(record).feasibility
        if not feasibility.is_feasible(option):
            outside += 1
    return outside / len(store.items)


def record_for(store: EvidenceStore, patient_id: str) -> PatientRecord:
    for record in store.records:
        if record.patient_id == patient_id:
            return record
    raise KeyError(patient_id)


def panel_rows(results: Sequence[ArmResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(dict(result.as_dict()))
    return rows


def marginal_gains(results: Sequence[ArmResult]) -> dict[str, float]:
    """Difference against the previous row, in AUROC percentage points."""
    return {
        current.name: (current.auroc - previous.auroc) * 100.0
        for previous, current in pairwise(results)
    }


def leave_one_out_deltas(results: Sequence[ArmResult]) -> dict[str, float]:
    if not results:
        return {}
    reference = results[0].auroc
    return {item.name: (item.auroc - reference) * 100.0 for item in results[1:]}


def synergy_surplus(panel_a: Sequence[ArmResult]) -> float:
    """Integrated gain minus the sum of the independent marginal gains."""
    if len(panel_a) < 3:
        return 0.0
    base = panel_a[0].auroc
    independent = sum((item.auroc - base) for item in panel_a[1:-1])
    integrated = panel_a[-1].auroc - base
    return (integrated - independent) * 100.0
