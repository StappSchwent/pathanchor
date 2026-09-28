"""Shared fixtures for the suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathanchor.cohort.fixture import build_cohort_fixture
from pathanchor.cohort.schema import (
    FEATURE_WIDTH_CLINICAL,
    FEATURE_WIDTH_HISTOLOGY,
    FEATURE_WIDTH_IMAGE,
    FEATURE_WIDTH_MOLECULAR,
    MolecularClass,
    PatientRecord,
    PriorLines,
    Sex,
    Site,
    Split,
    TreatmentClass,
)
from pathanchor.evaluation.registry import RuntimeConfig, resolve
from pathanchor.graph.builder import CensusSpec, build_declared_graph
from pathanchor.graph.grounding import EntityDirectory
from pathanchor.graph.index import CompactGraph
from pathanchor.training.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "experiment" / "_smoke.yaml"
MAIN = ROOT / "configs" / "experiment" / "main.yaml"


def pytest_configure() -> None:
    set_seed(20260101)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def smoke_runtime() -> RuntimeConfig:
    return resolve(SMOKE)


@pytest.fixture(scope="session")
def main_runtime() -> RuntimeConfig:
    return resolve(MAIN)


@pytest.fixture(scope="session")
def smoke_graph(smoke_runtime: RuntimeConfig) -> CompactGraph:
    return build_declared_graph(smoke_runtime.census)


@pytest.fixture(scope="session")
def smoke_directory(smoke_graph: CompactGraph, smoke_runtime: RuntimeConfig) -> EntityDirectory:
    return EntityDirectory.from_graph(smoke_graph, core_size=smoke_runtime.census.core_size)


@pytest.fixture(scope="session")
def tiny_graph() -> CompactGraph:
    return build_declared_graph(
        CensusSpec(entity_total=3000, edge_total=12000, seed=7, core_size=64)
    )


@pytest.fixture(scope="session")
def smoke_fixture(smoke_runtime: RuntimeConfig) -> object:
    return build_cohort_fixture(smoke_runtime.cohort)


@pytest.fixture(scope="session")
def full_fixture(main_runtime: RuntimeConfig) -> object:
    return build_cohort_fixture(main_runtime.cohort)


def _record(
    molecular: MolecularClass = MolecularClass.BRAF_V600,
    prior_lines: PriorLines = PriorLines.NONE,
    has_pathology: bool = True,
    has_imaging: bool = True,
    site: Site = Site.A,
    response: int = 1,
) -> PatientRecord:
    rng = np.random.default_rng(20260101)
    treatment = (
        TreatmentClass.BRAF_MEK
        if molecular is MolecularClass.BRAF_V600
        else TreatmentClass.ANTI_PD1_MONO
    )
    return PatientRecord(
        patient_id=f"T-{molecular.name}-{prior_lines.name}",
        site=site,
        sex=Sex.FEMALE,
        age=62,
        molecular=molecular,
        treatment=treatment,
        prior_lines=prior_lines,
        has_pathology=has_pathology,
        has_imaging=has_imaging,
        follow_up_months=24.0,
        response=response,
        split=Split.TEST,
        tumour_mutational_burden=float(rng.normal()),
        pdl1_fraction=0.3,
        interferon_gamma_score=0.2,
        pathway_signature=-0.1,
        expression_vector=tuple(float(value) for value in rng.normal(size=FEATURE_WIDTH_MOLECULAR)),
        histology_features=tuple(
            float(value) for value in rng.normal(size=FEATURE_WIDTH_HISTOLOGY)
        ),
        image_features=tuple(float(value) for value in rng.normal(size=FEATURE_WIDTH_IMAGE)),
        clinical_features=tuple(float(value) for value in rng.normal(size=FEATURE_WIDTH_CLINICAL)),
    )


@pytest.fixture
def records_factory():
    """Build one record with a chosen molecular class and prior-line count."""
    return _record
