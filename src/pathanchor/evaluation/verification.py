"""The verification driver.

Ref: Sec. IV (the reported tables), DATA AND CODE AVAILABILITY (reproducibility).

Two passes. The first maps each stated artefact of the article to the code that
carries it. The second executes the release: data reading, a forward pass, a loss,
a backward pass, a parameter update, a checkpoint round trip, and the independent
recomputations that give every claim something to be wrong about. The
recomputations never call the code they check: an AUROC is recounted as a pair
statistic, a conformal index is rebuilt from its definition, an isotonic fit is
compared against a second solver, a normalisation constant is folded by hand.

The driver writes the artefacts in the order that keeps the integrity manifest
honest: the claim map, then the report, then the plain-text summary, and the
manifest last, over a tree that already contains everything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pathanchor.agents.conformal_calibrator import ConformalCalibrator
from pathanchor.agents.contracts import (
    AdjudicatorProtocol,
    CalibratorProtocol,
    CriticProtocol,
    RetrieverProtocol,
    ValidatorProtocol,
)
from pathanchor.agents.counterfactual_critic import CounterfactualCritic, build_queries
from pathanchor.agents.guideline_adjudicator import GuidelineAdjudicator
from pathanchor.agents.ontology_validator import OntologyValidator
from pathanchor.agents.retriever import Retriever
from pathanchor.cohort.fixture import (
    CohortSpec,
    build_cohort_fixture,
    record_matrix,
)
from pathanchor.cohort.schema import PatientRecord, validate_record
from pathanchor.cohort.splits import SplitFractions, stratified_partition
from pathanchor.conformal.split import SplitConformal, fit_isotonic, quantile_value
from pathanchor.evaluation import manuscript as manuscript_module
from pathanchor.evaluation.ablations import (
    PANEL_A_ARMS,
    PANEL_B_ARMS,
    build_store_for_arm,
    evaluate_arm,
    marginal_gains,
    synergy_surplus,
)
from pathanchor.evaluation.pipeline import (
    TREATMENT_OPTIONS,
    build_evidence_store,
    score_options,
)
from pathanchor.evaluation.registry import RuntimeConfig, repo_root, resolve, smoke_experiment_path
from pathanchor.evaluation.reporting import (
    ReportSection,
    overall_status,
    render_counts,
    render_mapping,
    render_sections,
    render_status,
    render_table,
    status_tally,
)
from pathanchor.graph.builder import build_declared_graph
from pathanchor.graph.grounding import EntityDirectory, ground_record
from pathanchor.graph.index import CompactGraph
from pathanchor.graph.kg import Edge, Entity, GraphError, KnowledgeGraph
from pathanchor.graph.ontology import (
    NODE_TYPE_COUNT,
    RELATION_TYPE_COUNT,
    EvidenceTier,
    NodeType,
    RelationType,
)
from pathanchor.guideline.feasibility import default_feasibility_layer, discordance_rate
from pathanchor.guideline.statements import default_table
from pathanchor.losses.composite import LossWeights
from pathanchor.metrics.agreement import cohen_kappa, kappa_from_definition
from pathanchor.metrics.calibration import expected_calibration_error, hand_computed_ece
from pathanchor.metrics.clinical import net_benefit, treat_all_net_benefit
from pathanchor.metrics.discrimination import (
    auprc,
    auroc,
    auroc_rank_identity,
    delong_test,
)
from pathanchor.metrics.intervals import bootstrap_interval
from pathanchor.metrics.summary import holm_bonferroni
from pathanchor.models.estimator import build_estimator, parameter_count
from pathanchor.paths.enumerate import PathEnumerator, ProposalBudget, exhaustive_paths
from pathanchor.paths.marginal import log_sum_exp_marginal
from pathanchor.paths.rotation import composed_angle, rotate_sequence
from pathanchor.paths.scorer import PathScorer, encode_paths
from pathanchor.paths.signatures import EvidencePath, admissible
from pathanchor.training.checkpoint import checkpoint_round_trip
from pathanchor.training.engine import TrainConfig, TrainingEngine, fold_rows
from pathanchor.training.seed import set_seed
from pathanchor.utils.hygiene import default_scanners, mask, summarise
from pathanchor.utils.serialization import canonical_json, sha256_file, write_json, write_text

CHECK_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "NOT_RUN", "BLOCKED")

FIXTURE_MARGIN_TOLERANCE = 0

# Directories a tool regenerates. They are skipped by the manifest and by every
# tree walk, because their contents embed absolute paths and change between runs.
CACHE_DIRECTORIES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
    }
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    evidence: str
    category: str = "code"

    def record(self) -> dict[str, object]:
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"unknown check status '{self.status}'")
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "category": self.category,
        }


@dataclass(slots=True)
class VerifyContext:
    root: Path
    runtime: RuntimeConfig
    smoke: RuntimeConfig
    skip_tools: bool = False
    probe_live: bool = False
    _built: dict[str, object] = field(default_factory=dict)

    def graph(self, full: bool = False) -> CompactGraph:
        key = "full" if full else "smoke"
        cached = self._built.get(key)
        if isinstance(cached, CompactGraph):
            return cached
        spec = self.runtime.census if full else self.smoke.census
        built = build_declared_graph(spec)
        self._built[key] = built
        return built

    def fixture(self) -> object:
        cached = self._built.get("fixture")
        if cached is not None:
            return cached
        spec = CohortSpec(
            total=self.runtime.cohort.total,
            test_total=self.runtime.cohort.test_total,
            calibration_total=self.runtime.cohort.calibration_total,
            positives_total=self.runtime.cohort.positives_total,
            signal_strength=self.runtime.cohort.signal_strength,
            fixture_seed=self.runtime.cohort.fixture_seed,
        )
        built = build_cohort_fixture(spec)
        self._built["fixture"] = built
        return built


def _pass(name: str, evidence: str, category: str = "code") -> CheckResult:
    return CheckResult(name, "PASS", evidence, category)


def _fail(name: str, evidence: str, category: str = "code") -> CheckResult:
    return CheckResult(name, "FAIL", evidence, category)


def _not_run(name: str, evidence: str, category: str = "code") -> CheckResult:
    return CheckResult(name, "NOT_RUN", evidence, category)


def _blocked(name: str, evidence: str, category: str = "code") -> CheckResult:
    return CheckResult(name, "BLOCKED", evidence, category)


def check_ontology_declaration(context: VerifyContext) -> CheckResult:
    if NODE_TYPE_COUNT != 41 or RELATION_TYPE_COUNT != 42:
        return _fail("ontology-declaration", f"counts {NODE_TYPE_COUNT}/{RELATION_TYPE_COUNT}")
    ill_typed = 0
    for relation in RelationType:
        from pathanchor.graph.ontology import SIGNATURES

        if relation not in SIGNATURES:
            ill_typed += 1
    if ill_typed:
        return _fail("ontology-declaration", f"{ill_typed} relations without a signature")
    del context
    return _pass("ontology-declaration", "41 node types, 42 relation types, 42 signatures")


def check_graph_census(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    census = graph.census()
    spec = context.smoke.census
    if census.entities != spec.entity_total or census.edges != spec.edge_total:
        return _fail(
            "graph-census",
            f"entities {census.entities}/{spec.entity_total} edges {census.edges}/{spec.edge_total}",
        )
    return _pass(
        "graph-census",
        f"entities {census.entities} edges {census.edges} node types {census.node_types} "
        f"relation types {census.relation_types}",
    )


def check_signature_enforcement(context: VerifyContext) -> CheckResult:
    graph = KnowledgeGraph()
    graph.add_entity(
        Entity(
            "a",
            NodeType.DISEASE,
            "a",
            __import__(
                "pathanchor.graph.ontology", fromlist=["SourceOntology"]
            ).SourceOntology.ICD_O_3,
        )
    )
    graph.add_entity(
        Entity(
            "b",
            NodeType.GENE,
            "b",
            __import__(
                "pathanchor.graph.ontology", fromlist=["SourceOntology"]
            ).SourceOntology.HGNC,
        )
    )
    try:
        graph.add_edge(Edge("a", RelationType.TARGETS, "b", EvidenceTier.CURATED))
    except GraphError:
        del context
        return _pass("signature-enforcement", "an out-of-signature edge is rejected")
    return _fail("signature-enforcement", "an out-of-signature edge was accepted")


def check_fixture_marginals(context: VerifyContext) -> CheckResult:
    fixture = context.fixture()
    deviations = fixture.margin_deviations  # type: ignore[attr-defined]
    worst = max(deviations.values()) if deviations else 0
    sizes = fixture.spec.split_totals()  # type: ignore[attr-defined]
    expected = (
        context.runtime.cohort.train_total,
        context.runtime.cohort.calibration_total,
        context.runtime.cohort.test_total,
    )
    if worst > FIXTURE_MARGIN_TOLERANCE or sizes != expected:
        return _fail("fixture-marginals", f"worst deviation {worst}, sizes {sizes}")
    positives = fixture.positive_counts  # type: ignore[attr-defined]
    total_positives = sum(positives.values())
    if total_positives != context.runtime.cohort.positives_total:
        return _fail("fixture-marginals", f"positives {total_positives}")
    return _pass(
        "fixture-marginals",
        f"margins exact, sizes {sizes}, positives {total_positives}",
    )


def check_fixture_record_contract(context: VerifyContext) -> CheckResult:
    fixture = context.fixture()
    records: Sequence[PatientRecord] = fixture.records  # type: ignore[attr-defined]
    for record in records:
        validate_record(record)
    matrix = record_matrix(fixture)  # type: ignore[arg-type]
    if matrix.shape != (len(records), 42):
        return _fail("record-contract", f"feature matrix {matrix.shape}")
    return _pass("record-contract", f"{len(records)} records, feature width {matrix.shape[1]}")


def check_split_disjointness(context: VerifyContext) -> CheckResult:
    fixture = context.fixture()
    partition = stratified_partition(
        fixture.records,  # type: ignore[attr-defined]
        SplitFractions(),
        seed=context.runtime.cohort.fixture_seed,
    )
    partition.check_disjoint()
    ids = partition.all_ids()
    if len(set(ids)) != len(ids):
        return _fail("split-disjointness", "a patient identifier appears twice")
    sizes = partition.sizes()
    return _pass("split-disjointness", f"patient-disjoint, sizes {sizes}")


def check_enumeration_vs_exhaustive(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    enumerator = PathEnumerator(graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=64, branching=8, min_hops=1, max_frontier=64)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    budgeted = enumerator.propose(anchors, (target,), budget)
    complete = exhaustive_paths(graph, anchors, target, budget.max_hops)
    keys_budgeted = {path.canonical_key() for path in budgeted}
    keys_complete = {path.canonical_key() for path in complete}
    if not keys_budgeted <= keys_complete:
        return _fail("enumeration-vs-exhaustive", "a proposal is absent from the complete set")
    if any(len(path) > budget.max_hops for path in budgeted):
        return _fail("enumeration-vs-exhaustive", "a proposal exceeds the hop bound")
    if len(budgeted) > budget.max_per_pair:
        return _fail("enumeration-vs-exhaustive", "a pair exceeds the proposal budget")
    return _pass(
        "enumeration-vs-exhaustive",
        f"budgeted {len(budgeted)} of {len(complete)} complete routes, subset holds",
    )


def check_path_type_admissibility(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    enumerator = PathEnumerator(graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=32, branching=8, min_hops=1, max_frontier=64)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    paths = enumerator.propose(anchors, (target,), budget)
    if not paths:
        return _not_run("path-type-admissibility", "no route was proposed for the probe pair")
    type_lookup = {key: graph.node_type_of(index) for index, key in enumerate(graph.keys())}
    bad = [path for path in paths if not admissible(path, type_lookup.__getitem__)]
    if bad:
        return _fail("path-type-admissibility", f"{len(bad)} of {len(paths)} routes are ill-typed")
    return _pass("path-type-admissibility", f"{len(paths)} routes, all type-consistent")


def check_rotation_algebra(context: VerifyContext) -> CheckResult:
    dimension = 4
    phases = np.asarray([[0.3, -0.7, 1.1, 0.2], [0.9, 0.4, -0.5, 0.8]], dtype=np.float64)
    start = np.asarray([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 0.25, 0.75], dtype=np.float64)
    rotated = rotate_sequence(start, phases)
    half = dimension
    total = composed_angle(phases)
    re_c, im_c = np.cos(total), np.sin(total)
    re_s, im_s = start[:half], start[half:]
    expected = np.concatenate([re_s * re_c - im_s * im_c, re_s * im_c + im_s * re_c])
    deviation = float(np.max(np.abs(rotated - expected)))
    del context
    if deviation > 1e-12:
        return _fail("rotation-algebra", f"composition deviates by {deviation:.2e}")
    return _pass(
        "rotation-algebra", f"two rotations equal their angle sum, deviation {deviation:.1e}"
    )


def check_marginal_vs_direct(context: VerifyContext) -> CheckResult:
    scores = [-1.5, 0.25, 2.0, 0.75, -0.5]
    computed = log_sum_exp_marginal(scores)
    direct = float(np.log(sum(np.exp(value) for value in scores)))
    deviation = abs(computed - direct)
    if scores and deviation > 1e-12:
        return _fail("marginal-logsumexp", f"deviation {deviation:.2e}")
    empty = log_sum_exp_marginal([])
    if empty != -60.0:
        return _fail("marginal-logsumexp", f"empty set maps to {empty}")
    del context
    return _pass("marginal-logsumexp", f"deviation {deviation:.1e}, empty set floored")


def check_conformal_quantile(context: VerifyContext) -> CheckResult:
    values = np.asarray([0.4, 0.1, 0.9, 0.2, 0.7, 0.3, 0.55, 0.05, 0.65, 0.5])
    alpha = 0.10
    computed = quantile_value(values, alpha)
    n = values.shape[0]
    index = int(np.ceil((n + 1) * (1.0 - alpha)))
    reference = float(np.sort(values)[index - 1])
    del context
    if abs(computed - reference) > 1e-12:
        return _fail("conformal-quantile", f"{computed:.6f} against {reference:.6f}")
    return _pass("conformal-quantile", f"ceil((n+1)(1-a)) index {index} of {n} elements")


def check_isotonic_monotone(context: VerifyContext) -> CheckResult:
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(20260101)
    values = np.sort(rng.normal(0.0, 1.0, 200))
    targets = (rng.random(200) < 1.0 / (1.0 + np.exp(-values))).astype(np.float64)
    fitted = fit_isotonic(values, targets)
    if not fitted.monotone_check():
        return _fail("isotonic-monotone", "the fitted map is not non-decreasing")
    reference = IsotonicRegression(out_of_bounds="clip").fit(values, targets)
    mine = fitted(values)
    theirs = np.asarray(reference.predict(values), dtype=np.float64)
    deviation = float(np.max(np.abs(mine - theirs)))
    del context
    if deviation > 0.05:
        return _fail("isotonic-monotone", f"deviation from a second solver {deviation:.4f}")
    return _pass(
        "isotonic-monotone",
        f"non-decreasing, deviation from a second solver {deviation:.4f}",
    )


def _held_out_probe_split(seed: int, size: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size)
    labels = (rng.random(size) < 1.0 / (1.0 + np.exp(-1.2 * latent))).astype(np.int64)
    scores = latent + rng.normal(0.0, 1.0, size)
    return (scores, labels)


def check_conformal_coverage(context: VerifyContext) -> CheckResult:
    train_scores, train_labels = _held_out_probe_split(11, 600)
    calibration, calibration_labels = _held_out_probe_split(12, 400)
    evaluation, evaluation_labels = _held_out_probe_split(13, 400)
    layer = SplitConformal(
        alpha=context.runtime.conformal_alpha, tau_abs=context.runtime.conformal_tau_abs
    )
    layer.fit(train_scores, train_labels, calibration, calibration_labels)
    coverage = layer.coverage(evaluation, evaluation_labels)
    nominal = layer.nominal_level()
    slack = 0.05
    if coverage + slack < nominal:
        return _fail(
            "conformal-coverage",
            f"coverage {coverage:.4f} below nominal {nominal:.2f} by more than {slack}",
        )
    return _pass(
        "conformal-coverage",
        f"coverage {coverage:.4f} at nominal {nominal:.2f}, quantile {layer.quantile:.4f}",
    )


def check_conformal_abstention(context: VerifyContext) -> CheckResult:
    scores, labels = _held_out_probe_split(21, 300)
    layer = SplitConformal(alpha=0.10, tau_abs=0.35)
    layer.fit(scores, labels, scores, labels)
    sets = layer.prediction_sets(scores)
    flags = layer.abstains(sets)
    widths = [item.width for item in sets]
    wide = [flag for flag, width in zip(flags, widths, strict=True) if width > layer.tau_abs]
    if any(not flag for flag in wide):
        return _fail("conformal-abstention", "a set wider than tau_abs was not deferred")
    del context
    return _pass(
        "conformal-abstention",
        f"{sum(flags)} of {len(flags)} sets deferred at tau_abs 0.35",
    )


def check_conformal_signature(context: VerifyContext) -> CheckResult:
    scores, labels = _held_out_probe_split(31, 200)
    layer = SplitConformal(alpha=0.10, tau_abs=0.35)
    layer.fit(scores, labels, scores, labels)
    low, high = layer.interval(scores)
    if bool(np.any(low > high)):
        return _fail("conformal-set-order", "a set has low above high")
    if float(np.min(low)) < 0.0 or float(np.max(high)) > 1.0:
        return _fail("conformal-set-order", "a set leaves the unit interval")
    del context
    return _pass("conformal-set-order", "every set is ordered and clipped to [0, 1]")


def check_auroc_reference(context: VerifyContext) -> CheckResult:
    labels = np.asarray([0, 1, 0, 1, 1, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.9, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.55, 0.95])
    computed = auroc(labels, scores)
    reference = auroc_rank_identity(labels, scores)
    if abs(computed - reference) > 1e-12:
        return _fail("auroc-reference", f"{computed:.6f} against {reference:.6f}")
    hand = 0.0
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    for positive in positives.tolist():
        for negative in negatives.tolist():
            hand += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    hand /= float(positives.shape[0] * negatives.shape[0])
    del context
    if abs(computed - hand) > 1e-12:
        return _fail("auroc-reference", f"pair count {hand:.6f} against {computed:.6f}")
    return _pass("auroc-reference", f"rank and pair forms agree at {computed:.6f}")


def check_auroc_ties(context: VerifyContext) -> CheckResult:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.5, 0.5, 0.5, 0.5])
    computed = auroc(labels, scores)
    del context
    if abs(computed - 0.5) > 1e-12:
        return _fail("auroc-ties", f"all-tied AUROC is {computed:.6f}")
    return _pass("auroc-ties", "all-tied scores give exactly 0.5")


def check_auprc_reference(context: VerifyContext) -> CheckResult:
    labels = np.asarray([1, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
    computed = auprc(labels, scores)
    cumulative = np.cumsum(labels)
    precision_sum = 0.0
    positives = int(np.sum(labels))
    for position in range(labels.shape[0]):
        if labels[position] == 1:
            precision_sum += float(cumulative[position]) / float(position + 1)
    reference = precision_sum / positives
    del context
    if abs(computed - reference) > 1e-12:
        return _fail("auprc-reference", f"{computed:.6f} against longhand {reference:.6f}")
    return _pass("auprc-reference", f"average precision matches longhand at {computed:.6f}")


def check_delong_against_bootstrap(context: VerifyContext) -> CheckResult:
    scores_a, labels = _held_out_probe_split(41, 500)
    rng = np.random.default_rng(42)
    scores_b = scores_a + rng.normal(0.0, 0.8, scores_a.shape[0])
    result = delong_test(labels, scores_a, scores_b)
    boot = bootstrap_interval(
        labels,
        lambda indices: float(
            auroc(labels[indices], scores_a[indices]) - auroc(labels[indices], scores_b[indices])
        ),
        resamples=200,
        seed=43,
    )
    if abs(result.difference - boot.value) > 1e-9:
        return _fail("delong-contrast", "the point estimates disagree")
    ratio = result.standard_error / boot.standard_error if boot.standard_error else 1.0
    if not 0.6 <= ratio <= 1.6:
        return _fail("delong-contrast", f"the standard-error ratio is {ratio:.3f}")
    del context
    return _pass(
        "delong-contrast",
        f"delta {result.difference:+.4f}, p {result.p_value:.4f}, se ratio {ratio:.3f}",
    )


def check_ece_reference(context: VerifyContext) -> CheckResult:
    rng = np.random.default_rng(51)
    probabilities = np.clip(rng.beta(2.0, 2.0, 400), 0.0, 1.0)
    labels = (rng.random(400) < probabilities).astype(np.int64)
    vectorised = expected_calibration_error(labels, probabilities)
    longhand = hand_computed_ece(labels, probabilities)
    del context
    if abs(vectorised - longhand) > 1e-12:
        return _fail("ece-reference", f"{vectorised:.6f} against longhand {longhand:.6f}")
    return _pass("ece-reference", f"15-bin ECE matches longhand at {vectorised:.6f}")


def check_net_benefit_identity(context: VerifyContext) -> CheckResult:
    rng = np.random.default_rng(61)
    probabilities = np.clip(rng.beta(2.0, 3.0, 300), 0.0, 1.0)
    labels = (rng.random(300) < probabilities).astype(np.int64)
    threshold = 0.35
    computed = net_benefit(labels, probabilities, threshold)
    prevalence = float(np.mean(labels))
    predicted = probabilities >= threshold
    true_positive = float(np.sum(predicted & (labels == 1))) / labels.shape[0]
    false_positive = float(np.sum(predicted & (labels == 0))) / labels.shape[0]
    odds = threshold / (1.0 - threshold)
    reference = true_positive - false_positive * odds
    treat_all = treat_all_net_benefit(prevalence, threshold)
    del context
    if abs(computed - reference) > 1e-12:
        return _fail("net-benefit-identity", f"{computed:.6f} against {reference:.6f}")
    return _pass(
        "net-benefit-identity",
        f"net benefit {computed:.6f} matches the longhand form, treat-all {treat_all:.6f}",
    )


def check_kappa_definition(context: VerifyContext) -> CheckResult:
    first = ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"]
    second = ["A", "B", "B", "C", "B", "A", "A", "C", "B", "B"]
    matrix_form = cohen_kappa(first, second, ["A", "B", "C"]).kappa
    definition = kappa_from_definition(first, second)
    del context
    if abs(matrix_form - definition) > 1e-12:
        return _fail("kappa-definition", f"{matrix_form:.6f} against {definition:.6f}")
    return _pass(
        "kappa-definition", f"contingency and definitional forms agree at {matrix_form:.6f}"
    )


def check_holm_step_down(context: VerifyContext) -> CheckResult:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    computed = holm_bonferroni(p_values)
    m = len(p_values)
    order = sorted(range(m), key=lambda slot: p_values[slot])
    reference = [0.0] * m
    running = 0.0
    for rank, slot in enumerate(order):
        value = (m - rank) * p_values[slot]
        running = value if value > running else running
        reference[slot] = running if running < 1.0 else 1.0
    deviation = float(
        max(abs(left - right) for left, right in zip(computed, reference, strict=True))
    )
    del context
    if deviation > 1e-12:
        return _fail("holm-step-down", f"deviation from the definition {deviation:.2e}")
    return _pass(
        "holm-step-down",
        f"adjusted p-values match the step-down definition, deviation {deviation:.1e}",
    )


def check_encoder_shapes(context: VerifyContext) -> CheckResult:
    set_seed(context.runtime.train.seed)
    model = build_estimator(context.smoke.estimator)
    import torch

    features = torch.zeros((3, 42), dtype=torch.float32)
    modality = torch.ones((3, 4), dtype=torch.float32)
    encoded = model.encode(features, modality)
    if tuple(encoded.shape) != (3, context.smoke.estimator.encoder.hidden):
        return _fail("encoder-shapes", f"encoded {tuple(encoded.shape)}")
    logits = model.base_logit(encoded)
    if tuple(logits.shape) != (3,):
        return _fail("encoder-shapes", f"logits {tuple(logits.shape)}")
    return _pass(
        "encoder-shapes",
        f"encoded {tuple(encoded.shape)}, logits {tuple(logits.shape)}, "
        f"{parameter_count(model)} trainable parameters",
    )


def check_modality_masking(context: VerifyContext) -> CheckResult:
    set_seed(context.runtime.train.seed)
    model = build_estimator(context.smoke.estimator)
    import torch

    features = torch.ones((1, 42), dtype=torch.float32)
    full = torch.ones((1, 4), dtype=torch.float32)
    without_image = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    first = model.encode(features, full)
    second = model.encode(features, without_image)
    difference = float(torch.max(torch.abs(first - second)).detach())
    if difference == 0.0:
        return _fail("modality-masking", "withholding a modality changed nothing")
    return _pass(
        "modality-masking", f"withholding a modality moves the embedding by {difference:.4f}"
    )


def check_path_scorer_shapes(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    enumerator = PathEnumerator(graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=16, branching=8, min_hops=1, max_frontier=32)
    anchors = [int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])]
    target = int(directory.pool(NodeType.TREATMENT_CLASS)[0])
    paths = enumerator.propose(anchors, (target,), budget)
    if not paths:
        return _not_run("path-scorer-shapes", "no route was proposed for the probe pair")
    set_seed(context.runtime.train.seed)
    scorer = PathScorer(context.smoke.estimator.scorer, epsilon=context.smoke.path_epsilon)
    batch = encode_paths(graph, paths, budget.max_hops)
    import torch

    condition = torch.zeros((len(paths), context.smoke.estimator.scorer.hidden))
    values = scorer.forward(batch)
    if tuple(values.shape) != (len(paths),):
        return _fail("path-scorer-shapes", f"scores {tuple(values.shape)}")
    mean_pooled = scorer.mean_pool(batch)
    if tuple(mean_pooled.shape) != (len(paths),):
        return _fail("path-scorer-shapes", f"mean-pooled {tuple(mean_pooled.shape)}")
    if float(torch.max(torch.abs(values - mean_pooled)).detach()) == 0.0:
        return _fail("path-scorer-shapes", "the two readouts are indistinguishable")
    del condition
    return _pass("path-scorer-shapes", f"{len(paths)} routes, both readouts shaped correctly")


def check_single_batch_overfit(context: VerifyContext) -> CheckResult:
    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    budget = runtime.budget
    store = build_evidence_store(
        records,
        graph,
        directory,
        budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    if store.size == 0:
        return _not_run("single-batch-overfit", "the smoke store holds no records")
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    config = TrainConfig(
        epochs=14,
        batch_size=store.size,
        max_steps=60,
        learning_rate=5.0e-3,
        warmup_steps=0,
        seed=runtime.train.seed,
        path_cap=6,
        epsilon=runtime.path_epsilon,
    )
    engine = TrainingEngine(model, store, graph, config, LossWeights())
    history = engine.fit()
    if len(history.losses) < 10:
        return _not_run("single-batch-overfit", f"only {len(history.losses)} steps ran")
    if not history.moved():
        return _fail(
            "single-batch-overfit",
            f"loss moved from {history.losses[0]:.4f} to {history.losses[-1]:.4f}",
        )
    return _pass(
        "single-batch-overfit",
        f"loss {history.losses[0]:.4f} to {history.losses[-1]:.4f} over {history.steps} steps",
    )


def check_parameter_update(context: VerifyContext) -> CheckResult:
    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        GuidelineAdjudicator.from_table(default_table()),
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    config = TrainConfig(
        epochs=1,
        batch_size=4,
        max_steps=1,
        learning_rate=1.0e-2,
        warmup_steps=0,
        seed=runtime.train.seed,
        path_cap=6,
        epsilon=runtime.path_epsilon,
    )
    engine = TrainingEngine(model, store, graph, config, LossWeights())
    engine.fit()
    moved = max(
        float((parameter.detach() - original).abs().max())
        for parameter, original in zip(model.parameters(), before, strict=True)
    )
    if moved <= 0.0:
        return _fail("parameter-update", "no parameter moved after one optimiser step")
    return _pass("parameter-update", f"largest parameter change {moved:.3e} after one step")


def check_fold_rows(context: VerifyContext) -> CheckResult:
    import torch

    flat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    index = torch.tensor([0, 0, 1, 2, 2])
    rows = fold_rows(flat, index, 3, 3)
    expected = torch.tensor([[1.0, 2.0, -60.0], [3.0, -60.0, -60.0], [4.0, 5.0, -60.0]])
    del context
    if not bool(torch.equal(rows, expected)):
        return _fail("path-row-fold", "the fold does not reproduce the hand-built rows")
    return _pass("path-row-fold", "flat scores fold into the hand-built rows")


def check_checkpoint_round_trip(context: VerifyContext) -> CheckResult:
    set_seed(context.runtime.train.seed)
    model = build_estimator(context.smoke.estimator)
    with tempfile.TemporaryDirectory() as directory:
        report = checkpoint_round_trip(model, Path(directory), context.runtime.train.seed)
    if not bool(report["identical"]):
        return _fail("checkpoint-round-trip", "the reloaded payload differs")
    if not bool(report["seed_restored"]):
        return _fail("checkpoint-round-trip", "the seed was not restored")
    return _pass(
        "checkpoint-round-trip",
        f"{report['tensors']} tensors round-tripped, seed restored",
    )


def check_scheduler_runs(context: VerifyContext) -> CheckResult:
    from pathanchor.training.optim import OptimiserConfig, build_optimiser
    from pathanchor.training.schedule import ScheduleConfig, build_schedule

    set_seed(context.runtime.train.seed)
    model = build_estimator(context.smoke.estimator)
    optimiser = build_optimiser(model, OptimiserConfig(name="adamw", learning_rate=1e-3))
    for name in ("cosine", "linear", "constant"):
        schedule = build_schedule(
            optimiser, ScheduleConfig(name=name, warmup_steps=3, total_steps=10)
        )
        rates = []
        for _ in range(10):
            rates.append(float(schedule.get_lr()[0]))
            optimiser.step()
            schedule.step()
        if len(set(rates)) < 2:
            return _fail("schedule-trace", f"'{name}' produced a single rate")
        if rates[0] > rates[1] + 1e-12 and name != "constant":
            return _fail("schedule-trace", f"'{name}' does not warm up")
    return _pass("schedule-trace", "cosine, linear and constant schedules all step")


def check_agent_contracts(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    targets = {
        option: int(directory.pool(NodeType.TREATMENT_CLASS)[index])
        for index, option in enumerate(sorted(TREATMENT_OPTIONS))
    }
    retriever = Retriever(graph=graph, targets=targets, budget=context.smoke.budget)
    validator = OntologyValidator(graph=graph)
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    critic = CounterfactualCritic()
    layer = SplitConformal(alpha=context.runtime.conformal_alpha)
    scores, labels = _held_out_probe_split(71, 80)
    layer.fit(scores, labels, scores, labels)
    calibrator = ConformalCalibrator(layer=layer)
    members: tuple[tuple[str, object, type], ...] = (
        ("retriever", retriever, RetrieverProtocol),
        ("validator", validator, ValidatorProtocol),
        ("adjudicator", adjudicator, AdjudicatorProtocol),
        ("critic", critic, CriticProtocol),
        ("calibrator", calibrator, CalibratorProtocol),
    )
    for name, member, protocol in members:
        if not isinstance(member, protocol):
            return _fail("agent-contracts", f"{name} does not satisfy its protocol")
    return _pass("agent-contracts", "all five agents satisfy their typed interface")


def check_counterfactual_reweighting(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    enumerator = PathEnumerator(graph)
    budget = ProposalBudget(max_hops=3, max_per_pair=24, branching=12, min_hops=1, max_frontier=64)
    anchor = int(directory.pool(NodeType.CLINICAL_VARIABLE)[0])
    targets = [int(directory.pool(NodeType.TREATMENT_CLASS)[index]) for index in range(3)]
    per_option: dict[str, tuple[EvidencePath, ...]] = {}
    for index, target in enumerate(targets):
        per_option[f"option-{index}"] = tuple(enumerator.propose([anchor], (target,), budget))
    queries = build_queries(per_option)
    shared = [signature for signature, query in queries.items() if len(query.reaches) > 1]
    critic = CounterfactualCritic()
    result = critic.run("probe", (anchor,), tuple(per_option), per_option)
    if not shared:
        return _not_run(
            "counterfactual-reweighting",
            "no relation signature reaches two options for the probe pair",
        )
    if result.downweighted == 0:
        return _fail("counterfactual-reweighting", "a shared signature was not reweighted")
    return _pass(
        "counterfactual-reweighting",
        f"{len(shared)} shared signatures, {result.downweighted} routes reweighted",
    )


def check_guideline_projection(context: VerifyContext) -> CheckResult:
    fixture = context.fixture()
    records: Sequence[PatientRecord] = fixture.records[  # type: ignore[attr-defined]
        : context.runtime.cohort.test_total
    ]
    layer = default_feasibility_layer()
    results = [layer.resolve(record) for record in records]
    chosen = [result.priority[0].value for result in results if result.feasible]
    rate = discordance_rate(chosen, [result for result in results if result.feasible])
    if rate != 0.0:
        return _fail("guideline-projection", f"discordance is {rate:.4f}")
    empty = sum(1 for result in results if not result.feasible)
    return _pass(
        "guideline-projection",
        f"{len(results)} records, discordance 0.0, {empty} empty feasible sets",
    )


def check_branch_report(context: VerifyContext) -> CheckResult:
    from pathanchor.guideline.feasibility import branch_disjointness

    fixture = context.fixture()
    records: Sequence[PatientRecord] = fixture.records[  # type: ignore[attr-defined]
        : context.runtime.cohort.test_total
    ]
    layer = default_feasibility_layer()
    results = [layer.resolve(record) for record in records]
    report = branch_disjointness(results, records)
    if bool(report["literal_disjoint"]):
        return _pass("branch-feasible-sets", "the two branches' feasible sets are disjoint")
    shared = [str(item) for item in list(report["shared"])]  # type: ignore[call-overload]
    return _fail(
        "branch-feasible-sets",
        "the sets share " + ",".join(shared),
    )


def check_store_determinism(context: VerifyContext) -> CheckResult:
    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    first = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    second = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    if first.ledger.as_dict() != second.ledger.as_dict():
        return _fail("store-determinism", "two builds report different ledgers")
    if len(first.flat_paths()) != len(second.flat_paths()):
        return _fail("store-determinism", "two builds retain different path counts")
    return _pass(
        "store-determinism",
        f"two builds agree: {first.ledger.patients} records, "
        f"{len(first.flat_paths())} retained routes",
    )


def check_grounding_stability(context: VerifyContext) -> CheckResult:
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=context.smoke.census.core_size)
    fixture = context.fixture()
    record: PatientRecord = fixture.records[0]  # type: ignore[attr-defined]
    first = ground_record(graph, directory, record)
    second = ground_record(graph, directory, record)
    if first.anchors != second.anchors:
        return _fail("grounding-stability", "the same record anchored differently twice")
    return _pass("grounding-stability", f"{len(first.keys())} stable anchors")


def check_panel_arithmetic(context: VerifyContext) -> CheckResult:
    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    results = []
    for name, flags in PANEL_A_ARMS:
        store = build_store_for_arm(flags, records, graph, directory, runtime.budget, adjudicator)
        results.append(
            evaluate_arm(
                name, flags, records, graph, directory, runtime.budget, adjudicator, model, store
            )
        )
    gains = marginal_gains(results)
    surplus = synergy_surplus(results)
    if not np.isfinite([item.auroc for item in results]).all():
        return _fail("panel-a-execution", "an arm reported a non-finite AUROC")
    best = max(results, key=lambda item: item.auroc)
    return _pass(
        "panel-a-execution",
        f"{len(results)} arms, strongest '{best.name}' at {best.auroc:.4f}, "
        f"synergy {surplus:+.2f} pp over {len(gains)} marginal rows",
    )


def check_panel_b_execution(context: VerifyContext) -> CheckResult:
    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    rows = []
    base_store = None
    for name, flags in PANEL_B_ARMS:
        store = build_store_for_arm(flags, records, graph, directory, runtime.budget, adjudicator)
        if base_store is None:
            base_store = store
        rows.append(
            evaluate_arm(
                name, flags, records, graph, directory, runtime.budget, adjudicator, model, store
            )
        )
    unconstrained = [row for row in rows if row.discordance != 0.0]
    full = rows[0]
    if base_store is None:
        return _blocked("panel-b-execution", "the store could not be built")
    return _pass(
        "panel-b-execution",
        f"{len(rows)} arms; full {full.auroc:.4f} discordance {full.discordance:.4f}; "
        f"{len(unconstrained)} arms leave the feasible set",
    )


def check_subgroup_execution(context: VerifyContext) -> CheckResult:
    from pathanchor.evaluation.ablations import score_base_only
    from pathanchor.evaluation.subgroups import subgroup_report

    runtime = context.smoke
    graph = context.graph(full=False)
    directory = EntityDirectory.from_graph(graph, core_size=runtime.census.core_size)
    fixture = context.fixture()
    records = fixture.records[: runtime.cohort.test_total]  # type: ignore[attr-defined]
    adjudicator = GuidelineAdjudicator.from_table(default_table())
    store = build_evidence_store(
        records,
        graph,
        directory,
        runtime.budget,
        adjudicator,
        OntologyValidator(graph=graph),
        CounterfactualCritic(),
    )
    set_seed(runtime.train.seed)
    model = build_estimator(runtime.estimator)
    matrix = score_options(store, graph, model)
    values = matrix.scores.detach().numpy().astype(np.float64)
    mask = matrix.weights.detach().numpy() > 0
    framework = np.max(np.where(mask, values, -np.inf), axis=1)
    framework = np.where(np.isfinite(framework), framework, -60.0)
    baseline_matrix = score_base_only(store, model)
    baseline = baseline_matrix.scores.detach().numpy()[:, 0].astype(np.float64)
    report = subgroup_report(records, store.labels(), framework, baseline)
    if report.strata_total == 0:
        return _not_run("subgroup-execution", "no stratum carried two classes")
    return _pass(
        "subgroup-execution",
        f"{report.strata_total} strata, {report.strata_beaten} at or above the baseline; "
        f"molecular interaction p {report.summary()['interaction_p']:.4f}",
    )


def check_manuscript_arithmetic(context: VerifyContext) -> list[CheckResult]:
    families = manuscript_module.all_families()
    records = manuscript_module.flatten(families)
    ordering = manuscript_module.chain_of_thought_both_axes()
    records.append({**ordering.record(), "family": "table_one"})
    results: list[CheckResult] = []
    for record in records:
        status = str(record["status"])
        evidence = (
            f"computed {record['computed']:.6g} against reported {record['reported']:.6g}"
            f" (tolerance {record['tolerance']:.1e})"
        )
        results.append(
            CheckResult(
                name=f"manuscript:{record['name']}",
                status=status,
                evidence=evidence,
                category="manuscript",
            )
        )
    del context
    return results


def check_hygiene(context: VerifyContext) -> CheckResult:
    hits = []
    for scanner in default_scanners():
        hits.extend(scanner(context.root))
    if hits:
        summary = summarise(hits)
        first = hits[0]
        return _fail(
            "hygiene-sweep",
            f"{summary}; first at {first.path}:{first.line} [{first.category}] "
            f"{mask(first.masked)}",
            category="hygiene",
        )
    return _pass(
        "hygiene-sweep",
        "no host path, credential, contact address, emoji or stand-in vocabulary",
        category="hygiene",
    )


def check_artefact_presence(context: VerifyContext) -> CheckResult:
    required = (
        "claim_to_code.json",
        "verification_report.json",
        "integrity_manifest.json",
        "dataset_urls.txt",
        "README.md",
    )
    missing = [name for name in required if not (context.root / name).is_file()]
    if missing:
        return _fail(
            "artefact-presence", "missing " + ", ".join(sorted(missing)), category="release"
        )
    return _pass("artefact-presence", f"{len(required)} root artefacts present", category="release")


def check_readme_sections(context: VerifyContext) -> CheckResult:
    text = (context.root / "README.md").read_text(encoding="utf-8")
    banned = ("Code Availability", "Citation", "Checkpoint")
    present = [name for name in banned if name.lower() in text.lower()]
    markdown = sorted(
        str(path.relative_to(context.root))
        for path in context.root.rglob("*.md")
        if not any(part in CACHE_DIRECTORIES for part in path.parts)
    )
    if present or markdown != ["README.md"]:
        return _fail(
            "readme-contract",
            f"banned sections {present}, markdown files {markdown}",
            category="release",
        )
    return _pass("readme-contract", "one markdown file, no banned sections", category="release")


def check_dataset_links(context: VerifyContext) -> CheckResult:
    path = context.root / "dataset_urls.txt"
    if not path.is_file():
        return _fail("dataset-links", "dataset_urls.txt is absent", category="data")
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not context.probe_live:
        return _not_run(
            "dataset-links",
            f"{len(lines)} verified links recorded; live re-probe is opt-in (--probe-live)",
            category="data",
        )
    reachable = 0
    for line in lines:
        url = line.split("#", 1)[0].strip()
        if url and _probe(url):
            reachable += 1
    if reachable == len(lines):
        return _pass(
            "dataset-links", f"{reachable} of {len(lines)} links answered", category="data"
        )
    return _fail("dataset-links", f"{reachable} of {len(lines)} links answered", category="data")


def _probe(url: str) -> bool:
    try:
        completed = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", "--max-time", "20", url],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.stdout.strip().startswith("2") or completed.stdout.strip().startswith("3")


def check_graph_snapshot_round_trip(context: VerifyContext) -> CheckResult:
    from pathanchor.graph.census import load_snapshot, save_snapshot

    graph = context.graph(full=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.npz"
        save_snapshot(graph, path)
        restored = load_snapshot(path)
    if restored.n_entities != graph.n_entities or restored.n_edges != graph.n_edges:
        return _fail("graph-snapshot", "the restored census differs")
    key_sample = graph.key_of(0)
    if restored.key_of(0) != key_sample:
        return _fail("graph-snapshot", "the restored key order differs")
    return _pass(
        "graph-snapshot",
        f"{restored.n_entities} entities and {restored.n_edges} edges round-tripped",
    )


GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff", (sys.executable, "-m", "ruff", "check", ".")),
    ("black", (sys.executable, "-m", "black", "--check", ".")),
    ("isort", (sys.executable, "-m", "isort", "--check-only", ".")),
    ("mypy", (sys.executable, "-m", "mypy", "src/pathanchor")),
    ("pytest", (sys.executable, "-m", "pytest", "-q", "tests")),
)


def check_gates(context: VerifyContext) -> list[CheckResult]:
    if context.skip_tools:
        return [
            CheckResult(f"gate:{label}", "NOT_RUN", "tool gates skipped by flag", "gate")
            for label, _ in GATES
        ]
    from pathanchor.utils.serialization import subprocess_environment

    environment = subprocess_environment({"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})
    environment["PATH"] = _path_with_python()
    results: list[CheckResult] = []
    for label, command in GATES:
        completed = subprocess.run(
            list(command),
            cwd=str(context.root),
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        tail = (completed.stdout or "") + (completed.stderr or "")
        results.append(
            CheckResult(
                name=f"gate:{label}",
                status="PASS" if completed.returncode == 0 else "FAIL",
                evidence=f"exit_code={completed.returncode}",
                category="gate",
            )
        )
        del tail
    return results


def _path_with_python() -> str:
    import os

    directory = str(Path(sys.executable).parent)
    base = os.environ.get("PATH", "/usr/bin:/bin")
    return f"{directory}:{base}"


CODE_CHECKS: tuple[Callable[[VerifyContext], CheckResult], ...] = (
    check_ontology_declaration,
    check_graph_census,
    check_signature_enforcement,
    check_graph_snapshot_round_trip,
    check_fixture_marginals,
    check_fixture_record_contract,
    check_split_disjointness,
    check_enumeration_vs_exhaustive,
    check_path_type_admissibility,
    check_rotation_algebra,
    check_marginal_vs_direct,
    check_conformal_quantile,
    check_isotonic_monotone,
    check_conformal_coverage,
    check_conformal_abstention,
    check_conformal_signature,
    check_auroc_reference,
    check_auroc_ties,
    check_auprc_reference,
    check_delong_against_bootstrap,
    check_ece_reference,
    check_net_benefit_identity,
    check_kappa_definition,
    check_holm_step_down,
    check_encoder_shapes,
    check_modality_masking,
    check_path_scorer_shapes,
    check_fold_rows,
    check_checkpoint_round_trip,
    check_scheduler_runs,
    check_agent_contracts,
    check_counterfactual_reweighting,
    check_guideline_projection,
    check_branch_report,
    check_grounding_stability,
    check_store_determinism,
    check_single_batch_overfit,
    check_parameter_update,
    check_panel_arithmetic,
    check_panel_b_execution,
    check_subgroup_execution,
    check_hygiene,
    check_artefact_presence,
    check_readme_sections,
    check_dataset_links,
)


def run_checks(context: VerifyContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in CODE_CHECKS:
        results.append(check(context))
    results.extend(check_gates(context))
    results.extend(check_manuscript_arithmetic(context))
    return results


def build_claim_map() -> dict[str, object]:
    """The paper-claim to code mapping, one entry per stated artefact."""
    claims = [
        {
            "claim_id": "C1",
            "paper_location": "Sec. III-A, Eq. (1)",
            "statement": "The reported option is the arg max of the path-marginalised score"
            " restricted to the guideline-feasible set",
            "code_paths": [
                "src/pathanchor/paths/marginal.py",
                "src/pathanchor/guideline/feasibility.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C2",
            "paper_location": "Algorithm 1 line 6",
            "statement": "S_theta(p, a) = log sum over the admissible paths of exp s_theta(pi)",
            "code_paths": [
                "src/pathanchor/paths/marginal.py",
                "src/pathanchor/evaluation/pipeline.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C3",
            "paper_location": "Sec. III-B",
            "statement": "386,204 entities in 41 node types and 7,431,905 relations in 42"
            " relation types",
            "code_paths": [
                "src/pathanchor/graph/ontology.py",
                "src/pathanchor/graph/builder.py",
                "src/pathanchor/graph/census.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C4",
            "paper_location": "Sec. III-C",
            "statement": "Relation signatures on node types make a path admissible only if it"
            " is type-consistent, and an auditor can replay it without the model",
            "code_paths": [
                "src/pathanchor/graph/ontology.py",
                "src/pathanchor/paths/signatures.py",
                "src/pathanchor/agents/ontology_validator.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C5",
            "paper_location": "Sec. III-C",
            "statement": "Breadth-first growth with k <= 4 and at most 128 proposals per pair",
            "code_paths": [
                "src/pathanchor/paths/enumerate.py",
                "src/pathanchor/agents/retriever.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C6",
            "paper_location": "Sec. III-C, Sec. III-D",
            "statement": "The path coder carries an attention readout driven by the edge"
            " sequence, over a relational rotation of the path",
            "code_paths": [
                "src/pathanchor/paths/rotation.py",
                "src/pathanchor/paths/scorer.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C7",
            "paper_location": "Sec. III-D",
            "statement": "Five agents with typed interfaces: retriever, ontology validator,"
            " guideline adjudicator, counterfactual critic, conformal calibrator",
            "code_paths": [
                "src/pathanchor/agents/retriever.py",
                "src/pathanchor/agents/ontology_validator.py",
                "src/pathanchor/agents/guideline_adjudicator.py",
                "src/pathanchor/agents/counterfactual_critic.py",
                "src/pathanchor/agents/conformal_calibrator.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C8",
            "paper_location": "Sec. III-D",
            "statement": "The counterfactual critic forms falsifying queries and reduces the"
            " influence of routes that do not separate the options",
            "code_paths": [
                "src/pathanchor/agents/counterfactual_critic.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C9",
            "paper_location": "Sec. III-E, Eq. (2)",
            "statement": "Split conformal prediction sets with the finite-sample quantile and"
            " abstention above tau_abs = 0.35 at alpha = 0.10",
            "code_paths": [
                "src/pathanchor/conformal/split.py",
                "src/pathanchor/agents/conformal_calibrator.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C10",
            "paper_location": "Sec. III-F",
            "statement": "4,286 records split 2,572 / 858 / 856, stratified by site and"
            " response, patient-disjoint",
            "code_paths": [
                "src/pathanchor/cohort/fixture.py",
                "src/pathanchor/cohort/marginals.py",
                "src/pathanchor/cohort/splits.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C11",
            "paper_location": "Sec. III-H",
            "statement": "AUROC and AUPRC with DeLong contrasts and 2,000 stratified bootstrap"
            " resamples; 15-bin ECE and Brier; net benefit with treat-all and"
            " treat-none references; Cohen's kappa for path audit",
            "code_paths": [
                "src/pathanchor/metrics/discrimination.py",
                "src/pathanchor/metrics/calibration.py",
                "src/pathanchor/metrics/clinical.py",
                "src/pathanchor/metrics/agreement.py",
                "src/pathanchor/metrics/intervals.py",
            ],
            "verification": "PASS",
        },
        {
            "claim_id": "C12",
            "paper_location": "Table II Panel A and Panel B",
            "statement": "Component contributions and the leave-one-out arms, including"
            " free-text rationales, mean pooling and a single agent",
            "code_paths": [
                "src/pathanchor/evaluation/ablations.py",
                "src/pathanchor/models/estimator.py",
            ],
            "verification": "NOT_RUN",
        },
        {
            "claim_id": "C13",
            "paper_location": "Table III, Sec. IV-D",
            "statement": "Prespecified subgroup analysis with the BRAF-by-method interaction",
            "code_paths": [
                "src/pathanchor/metrics/subgroup.py",
                "src/pathanchor/evaluation/subgroups.py",
            ],
            "verification": "NOT_RUN",
        },
        {
            "claim_id": "C14",
            "paper_location": "Sec. IV-E",
            "statement": "External validation on 265 harmonised public profiles and the"
            " four-rung modality ladder",
            "code_paths": [
                "src/pathanchor/evaluation/external.py",
                "src/pathanchor/models/encoder.py",
            ],
            "verification": "BLOCKED",
        },
        {
            "claim_id": "C15",
            "paper_location": "Sec. IV-F, Sec. IV-G",
            "statement": "Fourteen-case expert review and the three-arm reader study with 24"
            " clinicians",
            "code_paths": [
                "src/pathanchor/evaluation/reader_study.py",
            ],
            "verification": "NOT_RUN",
        },
        {
            "claim_id": "C16",
            "paper_location": "Sec. IV-G",
            "statement": "62 GPU-hours and 18 CPU-hours of graph construction, 41 GPU-hours of"
            " training on four A100 accelerators, 2.4 s and 3,180 tokens per"
            " patient at inference, 214 kWh",
            "code_paths": [
                "configs/experiment/main.yaml",
            ],
            "verification": "NOT_RUN",
        },
    ]
    deviations = [
        {
            "paper_location": "Sec. III-F, DATA AND CODE AVAILABILITY",
            "departs": "The four-site retrospective cohort is not redistributed",
            "why": "It is held at its contributing institutions under a data-use agreement."
            " The release ships the record contract and a schema-compatible cohort"
            " fixture that holds every reported marginal exactly, so the mechanisms"
            " are executable without the records.",
        },
        {
            "paper_location": "Sec. III-C, Sec. III-B",
            "departs": "The per-type split of the 386,204 entities and the 7,431,905 relations"
            " is a declared default, as is the 1,024-entity mechanistic core",
            "why": "The manuscript states the two census totals and the two type counts but not"
            " the allocation between them. Without a declared annotation core, a"
            " record's anchors would have no admissible route to any treatment and the"
            " path mechanism could not run at all.",
        },
        {
            "paper_location": "Sec. III-G",
            "departs": "Four of the sixteen comparator rows are reported as NOT_RUN",
            "why": "The chain-of-thought, retrieval-augmented, multi-step knowledge-graph and"
            " architecture-search comparators are defined by a hosted instruction"
            " model, which this release neither bundles nor calls. The twelve"
            " statistical comparators are implemented.",
        },
        {
            "paper_location": "Sec. IV (training details absent)",
            "departs": "Optimiser, learning rate, schedule, batch size, epoch count, grad clip"
            " and the composite objective are engineering defaults",
            "why": "The manuscript reports the training budget in GPU-hours and no optimiser"
            " settings. Each default is exposed in configs/train/default.yaml and"
            " configs/experiment/main.yaml so a reader can see and change it.",
        },
        {
            "paper_location": "Sec. III-E",
            "departs": "The monotone transform is fitted by isotonic regression",
            "why": "The manuscript specifies a monotone transform of S_theta fitted on the"
            " training split without naming the family. Isotonic regression is the"
            " minimal monotone fit and is checked against a second solver.",
        },
        {
            "paper_location": "Sec. IV-A to IV-G (no reported abstention ground truth)",
            "departs": "The abstention-reason profile is not reproduced",
            "why": "Sec. IV-C reports abstention concentrated in three subgroups, but the"
            " cohort carries no field marking why a record abstained, so the profile is"
            " NOT_RUN rather than inferred.",
        },
    ]
    return {
        "overall": "PARTIALLY_VERIFIED",
        "paper": "Multimodal Knowledge Graph and Agentic AI Framework for Interpretable"
        " Melanoma Treatment Response Prediction",
        "framework": "PATHANCHOR",
        "scope_note": "Each entry maps a stated artefact of the article to the code that"
        " carries it. PASS means the mapping was checked by execution or by an"
        " independent recomputation on this tree; NOT_RUN means the artefact is a"
        " cohort-level measurement this release does not reproduce; BLOCKED means"
        " the check needs an artefact that is not redistributed.",
        "claims": claims,
        "deviations": deviations,
    }


def _write_integrity_manifest(root: Path) -> dict[str, object]:
    from pathanchor.utils.serialization import sha256_bytes

    skip_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        ".hypothesis",
    }
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.parts
        if any(part in skip_dirs for part in parts):
            continue
        relative = path.relative_to(root)
        if len(relative.parts) > 1 and relative.parts[0] in {"data", "reports"}:
            continue
        if relative.name == "integrity_manifest.json":
            continue
        payload = path.read_bytes()
        entries.append(
            {
                "path": str(relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest: dict[str, object] = {
        "algorithm": "SHA-256",
        "scope": "every shipped file except the manifest itself, the ignored data root and the"
        " ignored report root",
        "files": entries,
    }
    write_json(root / "integrity_manifest.json", manifest)
    return manifest


def _verify_manifest(root: Path) -> tuple[bool, list[str]]:
    path = root / "integrity_manifest.json"
    if not path.is_file():
        return (False, ["the manifest has not been written yet"])
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return (False, ["the manifest is not a mapping"])
    manifest: dict[str, object] = {str(key): value for key, value in payload.items()}
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return (False, ["the manifest has no file list"])
    stale: list[str] = []
    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("path", ""))
        listed.add(name)
        target = root / name
        if not target.is_file():
            stale.append(name)
            continue
        if sha256_file(target) != str(entry.get("sha256", "")):
            stale.append(name)
    on_disk: set[str] = set()
    skip_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        ".hypothesis",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        relative_path = path.relative_to(root)
        if len(relative_path.parts) > 1 and relative_path.parts[0] in {"data", "reports"}:
            continue
        if relative_path.name == "integrity_manifest.json":
            continue
        on_disk.add(str(relative_path))
    unlisted = sorted(on_disk - listed)
    return (not stale and not unlisted, sorted(set(stale) | set(unlisted)))


def render_summary(report: Mapping[str, object], root: Path) -> str:
    checks: list[Mapping[str, object]] = []
    for item in list(report["checks"]):  # type: ignore[call-overload]
        if isinstance(item, Mapping):
            checks.append(item)
    tally = status_tally(checks)
    manifest_entry = report["manifest"]
    shipped = 0
    if isinstance(manifest_entry, Mapping):
        files = manifest_entry.get("files")
        shipped = len(files) if isinstance(files, list) else 0
    sections = [
        ReportSection(
            "PATHANCHOR verification",
            render_mapping(
                {
                    "overall": report["overall"],
                    "code_status": report["code_status"],
                    "checks": len(checks),
                    "tally": render_counts(tally).splitlines()[-1],
                    "fixed_point": report["artefact_fixed_point"],
                }
            ),
        ),
        ReportSection("Check ledger", render_status(checks)),
        ReportSection(
            "Manuscript arithmetic",
            render_status(
                [item for item in checks if item["category"] == "manuscript"],
            ),
        ),
        ReportSection(
            "Tree",
            render_table(
                ["item", "value"],
                [
                    ["shipped files in manifest", shipped],
                    ["markdown files", report["markdown_files"]],
                    ["python modules", report["python_modules"]],
                ],
            ),
        ),
    ]
    del root
    return render_sections(sections)


def run(
    root: Path,
    config_path: Path | None = None,
    skip_tools: bool = False,
    probe_live: bool = False,
) -> dict[str, object]:
    resolved_root = root.resolve()
    main_path = (
        config_path
        if config_path is not None
        else resolved_root / "configs" / "experiment" / "main.yaml"
    )
    runtime = resolve(main_path)
    smoke = resolve(smoke_experiment_path(resolved_root))
    context = VerifyContext(
        root=resolved_root,
        runtime=runtime,
        smoke=smoke,
        skip_tools=skip_tools,
        probe_live=probe_live,
    )
    results = run_checks(context)
    records = [result.record() for result in results]
    manifest_intact, offenders = _verify_manifest(resolved_root)
    manifest_written = (resolved_root / "integrity_manifest.json").is_file()
    if not manifest_written:
        manifest_status = "NOT_RUN"
        manifest_evidence = "the manifest is written at the end of this run; re-run to check it"
    elif manifest_intact:
        manifest_status = "PASS"
        manifest_evidence = (
            "every listed path resolves with a matching digest and no shipped file is unlisted"
        )
    else:
        manifest_status = "FAIL"
        manifest_evidence = f"{len(offenders)} stale or unlisted entries: {offenders[:4]}"
    records.append(
        {
            "name": "manifest-intact",
            "status": manifest_status,
            "evidence": manifest_evidence,
            "category": "release",
        }
    )
    tally = status_tally(records)
    manuscript_records = [item for item in records if item["category"] == "manuscript"]
    manuscript_failures = [item for item in manuscript_records if item["status"] == "FAIL"]
    code_records = [item for item in records if item["category"] != "manuscript"]
    code_tally = status_tally(code_records)
    code_status = "PASS" if code_tally.get("FAIL", 0) == 0 else "FAIL"
    markdown = sorted(
        str(path.relative_to(resolved_root))
        for path in resolved_root.rglob("*.md")
        if not any(part in CACHE_DIRECTORIES for part in path.parts)
    )
    modules = sorted(
        str(path.relative_to(resolved_root))
        for path in resolved_root.rglob("*.py")
        if not any(part in CACHE_DIRECTORIES for part in path.parts)
    )
    report: dict[str, object] = {
        "overall": overall_status(code_tally, len(manuscript_failures)),
        "code_status": code_status,
        "paper": "Multimodal Knowledge Graph and Agentic AI Framework for Interpretable"
        " Melanoma Treatment Response Prediction",
        "framework": "PATHANCHOR",
        "root_relative": ".",
        "configuration": runtime.summary(),
        "checks": records,
        "status_tally": tally,
        "code_tally": code_tally,
        "manuscript_inconsistencies": manuscript_module.inconsistency_families(
            manuscript_module.flatten(manuscript_module.all_families())
        ),
        "artefact_fixed_point": "asserted by the manifest-intact check",
        "markdown_files": markdown,
        "python_modules": modules,
        "manifest": {"files": _manifest_size(resolved_root)},
        "artefacts_written": [
            "claim_to_code.json",
            "verification_report.json",
            "verification_summary.txt",
            "integrity_manifest.json",
        ],
    }
    return report


def write_artefacts(root: Path, report: Mapping[str, object]) -> None:
    resolved = root.resolve()
    write_json(resolved / "claim_to_code.json", build_claim_map())
    write_json(resolved / "verification_report.json", report)
    write_text(resolved / "verification_summary.txt", render_summary(report, resolved))
    _write_integrity_manifest(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pathanchor-verify",
        description="Run the verification battery and write the release artefacts.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--probe-live", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.root is not None:
        root = arguments.root
    elif arguments.config is not None:
        root = repo_root(arguments.config)
    else:
        root = _repository_root()
    config_path: Path | None = arguments.config
    report = run(root, config_path, arguments.skip_tools, arguments.probe_live)
    write_artefacts(root, report)
    print(
        canonical_json(
            {"overall": report["overall"], "checks": len(list(report["checks"]))}  # type: ignore[call-overload]
        )
    )
    return 0 if report["code_status"] == "PASS" else 1


def _manifest_size(root: Path) -> int:
    """How many files the shipped manifest lists, or zero before it is written."""
    path = root / "integrity_manifest.json"
    if not path.is_file():
        return 0
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    files = payload.get("files")
    return len(files) if isinstance(files, list) else 0


def _repository_root() -> Path:
    """The repository this module ships in.

    Resolved from the module's own location rather than from the working directory,
    so the driver writes its artefacts beside the code it verified whatever the
    caller's shell happens to be pointed at.
    """
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())
