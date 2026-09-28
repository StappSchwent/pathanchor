"""Configuration to runtime objects.

Ref: Sec. III (the configuration surface), Sec. IV-G (the reported budget).

One module owns every config-to-object conversion, so a value that reaches a
component can be traced back to the key it came from. No result value the
manuscript reports appears in a default: the defaults here are the method's
constants and the fixture's declared composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pathanchor.cohort.fixture import CohortSpec, scale_spec
from pathanchor.graph.builder import CensusSpec, spec_from_mapping
from pathanchor.models.encoder import EncoderConfig
from pathanchor.models.estimator import EstimatorConfig
from pathanchor.paths.enumerate import ProposalBudget
from pathanchor.paths.rotation import ScorerConfig
from pathanchor.training.engine import TrainConfig
from pathanchor.utils import config as config_utils

GRAPH_SECTION = "graph"
COHORT_SECTION = "cohort"
MODEL_SECTION = "model"
TRAIN_SECTION = "train"
PIPELINE_SECTION = "pipeline"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Everything the entry points need, resolved from one YAML document."""

    raw: Mapping[str, object]
    census: CensusSpec
    cohort: CohortSpec
    budget: ProposalBudget
    estimator: EstimatorConfig
    train: TrainConfig
    conformal_alpha: float
    conformal_tau_abs: float
    path_epsilon: float
    subgroup_dimensions: tuple[str, ...]
    bootstrap_resamples: int
    seeds: tuple[int, ...]

    def summary(self) -> Mapping[str, object]:
        return {
            "graph_entities": self.census.entity_total,
            "graph_edges": self.census.edge_total,
            "core_size": self.census.core_size,
            "cohort_total": self.cohort.total,
            "test_total": self.cohort.test_total,
            "max_hops": self.budget.max_hops,
            "max_per_pair": self.budget.max_per_pair,
            "branching": self.budget.branching,
            "alpha": self.conformal_alpha,
            "tau_abs": self.conformal_tau_abs,
            "epsilon": self.path_epsilon,
            "epochs": self.train.epochs,
            "batch_size": self.train.batch_size,
            "max_steps": self.train.max_steps,
            "seeds": list(self.seeds),
        }


def census_spec(data: Mapping[str, object]) -> CensusSpec:
    return spec_from_mapping(config_utils.section(data, GRAPH_SECTION))


def cohort_spec(data: Mapping[str, object]) -> CohortSpec:
    """Resolve the cohort section, scaling the declared composition to its size.

    An explicit margin in the section wins; otherwise the declared composition is
    rescaled to whatever cohort size the section names, which is what lets the
    smoke configuration run the same mechanisms on a small record set without
    restating every margin.
    """
    section = config_utils.section(data, COHORT_SECTION)
    base = CohortSpec()
    total = config_utils.get_int(section, "total", base.total)
    calibration_total = config_utils.get_int(section, "calibration_total", base.calibration_total)
    test_total = config_utils.get_int(section, "test_total", base.test_total)
    positives_total = config_utils.get_int(section, "positives_total", base.positives_total)
    declared = CohortSpec(
        total=sum(_ints(section, "molecular", base.molecular)),
        test_total=sum(_ints(section, "test_molecular", base.test_molecular)),
        calibration_total=calibration_total,
        positives_total=positives_total,
        molecular=_ints(section, "molecular", base.molecular),
        treatment=_ints(section, "treatment", base.treatment),
        site=_ints(section, "site", base.site),
        sex=_ints(section, "sex", base.sex),
        age=_ints(section, "age", base.age),
        test_molecular=_ints(section, "test_molecular", base.test_molecular),
        test_treatment=_ints(section, "test_treatment", base.test_treatment),
        test_site=_ints(section, "test_site", base.test_site),
        test_sex=_ints(section, "test_sex", base.test_sex),
        test_age=_ints(section, "test_age", base.test_age),
        signal_strength=config_utils.get_float(section, "signal_strength", base.signal_strength),
        fixture_seed=config_utils.get_int(section, "seed", base.fixture_seed),
    )
    return scale_spec(declared, total, calibration_total, test_total, positives_total)


def _ints(section: Mapping[str, object], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if key not in section:
        return default
    return tuple(int(value) for value in config_utils.get_float_list(section, key))


def proposal_budget(data: Mapping[str, object]) -> ProposalBudget:
    section = config_utils.section(data, PIPELINE_SECTION)
    return ProposalBudget(
        max_hops=config_utils.get_int(section, "max_hops", 4),
        max_per_pair=config_utils.get_int(section, "max_per_pair", 128),
        branching=config_utils.get_int(section, "branching", 24),
        min_hops=config_utils.get_int(section, "min_hops", 1),
        max_frontier=config_utils.get_int(section, "max_frontier", 256),
    )


def estimator_config(data: Mapping[str, object]) -> EstimatorConfig:
    section = config_utils.section(data, MODEL_SECTION)
    encoder = EncoderConfig(
        hidden=config_utils.get_int(section, "hidden", 64),
        blocks=config_utils.get_int(section, "blocks", 2),
        dropout=config_utils.get_float(section, "dropout", 0.1),
    )
    scorer = ScorerConfig(
        dimension=config_utils.get_int(section, "dimension", 32),
        hidden=config_utils.get_int(section, "scorer_hidden", 64),
    )
    return EstimatorConfig(
        encoder=encoder,
        scorer=scorer,
        typed_paths=config_utils.get_bool(section, "typed_paths", True),
        use_graph_paths=config_utils.get_bool(section, "use_graph_paths", True),
        use_base=config_utils.get_bool(section, "use_base", True),
    )


def train_config(data: Mapping[str, object]) -> TrainConfig:
    section = config_utils.section(data, TRAIN_SECTION)
    return TrainConfig(
        epochs=config_utils.get_int(section, "epochs", 60),
        batch_size=config_utils.get_int(section, "batch_size", 8),
        grad_accumulation=config_utils.get_int(section, "grad_accumulation", 1),
        max_steps=config_utils.get_int(section, "max_steps", 2000),
        learning_rate=config_utils.get_float(section, "learning_rate", 3.0e-4),
        weight_decay=config_utils.get_float(section, "weight_decay", 1.0e-2),
        warmup_steps=config_utils.get_int(section, "warmup_steps", 100),
        schedule=config_utils.get_str(section, "schedule", "cosine"),
        grad_clip=config_utils.get_float(section, "grad_clip", 1.0),
        seed=config_utils.get_int(section, "seed", 20260101),
        precision=config_utils.get_str(section, "precision", "fp32"),
        path_cap=config_utils.get_int(section, "path_cap", 12),
        epsilon=config_utils.get_float(section, "epsilon", 1.0e-3),
        ranking_scale=config_utils.get_float(section, "ranking_scale", 0.1),
    )


def resolve(
    path: Path,
    overrides: tuple[str, ...] = (),
) -> RuntimeConfig:
    raw = config_utils.load_config(path, overrides)
    conformal = config_utils.section(raw, "conformal")
    evaluation = config_utils.section(raw, "evaluation")
    seeds = config_utils.get_float_list(evaluation, "seeds") if "seeds" in evaluation else []
    dimensions = (
        config_utils.get_str_list(evaluation, "subgroup_dimensions")
        if "subgroup_dimensions" in evaluation
        else ["Molecular", "Treatment class", "Site", "Region", "Sex", "Age band"]
    )
    return RuntimeConfig(
        raw=raw,
        census=census_spec(raw),
        cohort=cohort_spec(raw),
        budget=proposal_budget(raw),
        estimator=estimator_config(raw),
        train=train_config(raw),
        conformal_alpha=config_utils.get_float(conformal, "alpha", 0.10),
        conformal_tau_abs=config_utils.get_float(conformal, "tau_abs", 0.35),
        path_epsilon=config_utils.get_float(conformal, "epsilon", 1.0e-3),
        subgroup_dimensions=tuple(str(item) for item in dimensions),
        bootstrap_resamples=config_utils.get_int(evaluation, "bootstrap_resamples", 2000),
        seeds=tuple(int(value) for value in seeds) or (20260101,),
    )


def repo_root(config_path: Path) -> Path:
    return config_utils.repo_root(config_path)


def default_experiment_path(root: Path) -> Path:
    return root / "configs" / "experiment" / "main.yaml"


def smoke_experiment_path(root: Path) -> Path:
    return root / "configs" / "experiment" / "_smoke.yaml"


def graph_config_path(root: Path) -> Path:
    return root / "configs" / "graph" / "census.yaml"
