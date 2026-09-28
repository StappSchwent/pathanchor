"""The cohort fixture: a schema-compatible stand-in for the private cohort.

Ref: Sec. III-F (4,286 records over four sites; 44.2% BRAF V600, 21.6% NRAS,
12.4% NF1, 21.8% triple-wild-type; 52.3% / 27.1% / 20.6% by treatment class;
45.6% female; 40.4% response, 1,731 of 4,286; 60/20/20 partition at
2,572 / 858 / 856), Table III (the test split by subgroup).

The retrospective cohort is held at its contributing sites and is not
redistributed. What this module builds instead is a record set that satisfies the
manuscript's reported margins exactly -- the joint molecular-by-treatment table,
the site, sex and age margins, and the three partition sizes -- so that every
mechanism downstream of the record contract is executable and every reported
table row can be marked with a reason rather than a number.

The labels are a function of the drawn features plus a recorded amount of
observation noise, and the noise is set so the framework's own discrimination on
the fixture stays well below its ceiling. A fixture whose labels were an exact
function of its features would make every ranking check trivial.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.cohort.marginals import (
    MarginSpec,
    build_table,
    margin_error,
    split_proportionally,
)
from pathanchor.cohort.schema import (
    FEATURE_WIDTH_CLINICAL,
    FEATURE_WIDTH_HISTOLOGY,
    FEATURE_WIDTH_IMAGE,
    FEATURE_WIDTH_MOLECULAR,
    AgeBand,
    MolecularClass,
    PatientRecord,
    PriorLines,
    Sex,
    Site,
    Split,
    TreatmentClass,
)
from pathanchor.utils.types import FloatArray, IntArray

MOLECULAR_LEVELS: tuple[MolecularClass, ...] = (
    MolecularClass.BRAF_V600,
    MolecularClass.NRAS,
    MolecularClass.NF1,
    MolecularClass.TRIPLE_WILD_TYPE,
)
TREATMENT_LEVELS: tuple[TreatmentClass, ...] = (
    TreatmentClass.ANTI_PD1_MONO,
    TreatmentClass.ANTI_PD1_ANTI_CTLA4,
    TreatmentClass.BRAF_MEK,
)
SITE_LEVELS: tuple[Site, ...] = (Site.A, Site.B, Site.C, Site.D)
SEX_LEVELS: tuple[Sex, ...] = (Sex.FEMALE, Sex.MALE)
AGE_LEVELS: tuple[AgeBand, ...] = (AgeBand.UNDER_65, AgeBand.AT_LEAST_65)

SPLIT_LEVELS: tuple[Split, ...] = (Split.TRAIN, Split.CALIBRATION, Split.TEST)

_TABLE_SHAPE: tuple[int, ...] = (
    len(MOLECULAR_LEVELS),
    len(TREATMENT_LEVELS),
    len(SITE_LEVELS),
    len(SEX_LEVELS),
    len(AGE_LEVELS),
)

_TREATMENT_BRAF_INDEX: int = TREATMENT_LEVELS.index(TreatmentClass.BRAF_MEK)
_MOLECULAR_BRAF_INDEX: int = MOLECULAR_LEVELS.index(MolecularClass.BRAF_V600)

# Age is drawn inside the reported band, including its endpoints, so the record's
# computed band always agrees with the band its table cell was allocated to.
_AGE_RANGE: Mapping[AgeBand, tuple[int, int]] = {
    AgeBand.UNDER_65: (34, 64),
    AgeBand.AT_LEAST_65: (65, 88),
}


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """Declared composition of the fixture.

    Every field here is a design quantity of the released record set, not a
    measured result. The manuscript's reported cohort statistics are carried by
    the margins below and asserted after construction.
    """

    total: int = 4286
    test_total: int = 856
    calibration_total: int = 858
    positives_total: int = 1731
    molecular: tuple[int, ...] = (1894, 926, 531, 935)
    treatment: tuple[int, ...] = (2242, 1161, 883)
    site: tuple[int, ...] = (1412, 1107, 986, 781)
    sex: tuple[int, ...] = (1954, 2332)
    age: tuple[int, ...] = (2524, 1762)
    test_molecular: tuple[int, ...] = (378, 185, 106, 187)
    test_treatment: tuple[int, ...] = (448, 232, 176)
    test_site: tuple[int, ...] = (282, 221, 197, 156)
    test_sex: tuple[int, ...] = (391, 465)
    test_age: tuple[int, ...] = (504, 352)
    molecular_treatment: tuple[tuple[int, ...], ...] = (
        (720, 291, 883),
        (520, 406, 0),
        (300, 231, 0),
        (702, 233, 0),
    )
    test_molecular_treatment: tuple[tuple[int, ...], ...] = (
        (150, 52, 176),
        (104, 81, 0),
        (60, 46, 0),
        (134, 53, 0),
    )
    imaging_missing_site: int = 3
    imaging_missing_share: float = 0.284
    pathology_missing_share: float = 0.10
    advanced_line_share: float = 0.141
    signal_strength: float = 1.15
    fixture_seed: int = 20260101

    @property
    def train_total(self) -> int:
        return self.total - self.test_total - self.calibration_total

    def split_totals(self) -> tuple[int, int, int]:
        return (self.train_total, self.calibration_total, self.test_total)

    def positive_targets(self) -> tuple[int, int, int]:
        totals = np.asarray(self.split_totals(), dtype=np.float64)
        shares = totals / float(self.total)
        exact = shares * self.positives_total
        targets = np.floor(exact).astype(np.int64)
        remainder = int(self.positives_total - int(np.sum(targets)))
        order = np.argsort(-(exact - targets), kind="stable")
        for slot in order[: max(remainder, 0)]:
            targets[int(slot)] += 1
        return (int(targets[0]), int(targets[1]), int(targets[2]))


def scale_spec(
    base: CohortSpec,
    total: int,
    calibration_total: int,
    test_total: int,
    positives_total: int,
) -> CohortSpec:
    """Rescale a declared composition to a different cohort size.

    Used by the smoke configuration, which runs the same mechanisms on a small
    record set. The joint molecular-by-treatment table, the site, sex and age
    margins and the test-split margins are all rescaled by largest remainder, so
    the scaled fixture holds its own declared margins exactly in the same way the
    full one does, and the structural zeros stay zero.
    """
    from pathanchor.utils.numerics import allocate_counts

    declared_total = sum(base.molecular)
    declared_test = sum(base.test_molecular)
    if total == declared_total and test_total == declared_test:
        return base

    def rescale(values: Sequence[int], target: int) -> tuple[int, ...]:
        weights = [float(value) if value > 0 else 0.0 for value in values]
        if sum(weights) <= 0.0:
            raise ValueError("a margin to rescale carries no positive weight")
        positive = [value if value > 0 else 1e-9 for value in weights]
        allocated = allocate_counts(target, positive, floor=0)
        for index, value in enumerate(values):
            if value == 0:
                allocated[index] = 0
        delta = target - sum(allocated)
        if delta != 0:
            order = sorted(range(len(values)), key=lambda slot: (-values[slot], slot))
            for slot in order[: abs(delta)]:
                allocated[slot] += 1 if delta > 0 else -1
        return tuple(int(value) for value in allocated)

    from pathanchor.utils.numerics import allocate_counts as _allocate

    row_targets = _allocate(
        total,
        [float(sum(row)) if sum(row) > 0 else 1e-9 for row in base.molecular_treatment],
        floor=0,
    )
    test_row_targets = _allocate(
        test_total,
        [float(sum(row)) if sum(row) > 0 else 1e-9 for row in base.test_molecular_treatment],
        floor=0,
    )
    joint = tuple(
        rescale(row, row_targets[index]) for index, row in enumerate(base.molecular_treatment)
    )
    test_joint = tuple(
        rescale(row, test_row_targets[index])
        for index, row in enumerate(base.test_molecular_treatment)
    )
    molecular_row_sums = tuple(sum(row) for row in joint)
    test_molecular_row_sums = tuple(sum(row) for row in test_joint)
    treatment_column_sums = tuple(
        sum(joint[row][column] for row in range(len(joint))) for column in range(len(joint[0]))
    )
    test_treatment_column_sums = tuple(
        sum(test_joint[row][column] for row in range(len(test_joint)))
        for column in range(len(test_joint[0]))
    )
    return CohortSpec(
        total=total,
        test_total=test_total,
        calibration_total=calibration_total,
        positives_total=positives_total,
        molecular=molecular_row_sums,
        treatment=treatment_column_sums,
        site=rescale(base.site, total),
        sex=rescale(base.sex, total),
        age=rescale(base.age, total),
        test_molecular=test_molecular_row_sums,
        test_treatment=test_treatment_column_sums,
        test_site=rescale(base.test_site, test_total),
        test_sex=rescale(base.test_sex, test_total),
        test_age=rescale(base.test_age, test_total),
        molecular_treatment=joint,
        test_molecular_treatment=test_joint,
        imaging_missing_site=base.imaging_missing_site,
        imaging_missing_share=base.imaging_missing_share,
        pathology_missing_share=base.pathology_missing_share,
        advanced_line_share=base.advanced_line_share,
        signal_strength=base.signal_strength,
        fixture_seed=base.fixture_seed,
    )


def allowed_cells() -> np.ndarray:
    """Cells the guideline makes structurally impossible.

    BRAF-directed therapy is admissible only for a BRAF V600 mutation, so the
    BRAF/MEK column is empty for every other molecular class. This is the same
    rule the feasibility layer applies per patient, expressed once here so the
    fixture cannot contain a record the projection would have to reject.
    """
    mask = np.ones(_TABLE_SHAPE, dtype=bool)
    for molecular in range(len(MOLECULAR_LEVELS)):
        for treatment in range(len(TREATMENT_LEVELS)):
            if treatment == _TREATMENT_BRAF_INDEX and molecular != _MOLECULAR_BRAF_INDEX:
                mask[molecular, treatment, :, :, :] = False
    return mask


def _full_margins(spec: CohortSpec) -> list[MarginSpec]:
    return [
        MarginSpec(
            "molecular_treatment", (0, 1), np.asarray(spec.molecular_treatment, dtype=np.int64)
        ),
        MarginSpec("site", (2,), np.asarray(spec.site, dtype=np.int64)),
        MarginSpec("sex", (3,), np.asarray(spec.sex, dtype=np.int64)),
        MarginSpec("age", (4,), np.asarray(spec.age, dtype=np.int64)),
    ]


def _test_margins(spec: CohortSpec) -> list[MarginSpec]:
    return [
        MarginSpec(
            "molecular_treatment", (0, 1), np.asarray(spec.test_molecular_treatment, dtype=np.int64)
        ),
        MarginSpec("site", (2,), np.asarray(spec.test_site, dtype=np.int64)),
        MarginSpec("sex", (3,), np.asarray(spec.test_sex, dtype=np.int64)),
        MarginSpec("age", (4,), np.asarray(spec.test_age, dtype=np.int64)),
    ]


def _residual_margins(spec: CohortSpec) -> list[MarginSpec]:
    full = _full_margins(spec)
    test = _test_margins(spec)
    return [
        MarginSpec(left.name, left.axes, left.target - right.target)
        for left, right in zip(full, test, strict=True)
    ]


def _validate_spec(spec: CohortSpec) -> None:
    if sum(spec.molecular) != spec.total:
        raise ValueError("the molecular margin does not sum to the cohort size")
    if sum(spec.treatment) != spec.total:
        raise ValueError("the treatment margin does not sum to the cohort size")
    if sum(spec.site) != spec.total:
        raise ValueError("the site margin does not sum to the cohort size")
    if sum(spec.sex) != spec.total or sum(spec.age) != spec.total:
        raise ValueError("a demographic margin does not sum to the cohort size")
    if spec.test_total + spec.calibration_total >= spec.total:
        raise ValueError("the test and calibration partitions leave no training records")
    test_sums = (
        sum(spec.test_molecular),
        sum(spec.test_treatment),
        sum(spec.test_site),
        sum(spec.test_sex),
        sum(spec.test_age),
    )
    if any(value != spec.test_total for value in test_sums):
        raise ValueError("a test-split margin does not sum to the test partition size")
    if len(spec.molecular_treatment) != len(MOLECULAR_LEVELS):
        raise ValueError("the joint molecular-by-treatment margin has the wrong shape")
    forbidden = [
        (molecular, treatment, spec.molecular_treatment[molecular][treatment])
        for molecular in range(len(MOLECULAR_LEVELS))
        for treatment in range(len(TREATMENT_LEVELS))
        if treatment == _TREATMENT_BRAF_INDEX and molecular != _MOLECULAR_BRAF_INDEX
    ]
    if any(count != 0 for _, _, count in forbidden):
        raise ValueError("the joint margin places BRAF-directed therapy outside BRAF V600")


def build_partition_tables(spec: CohortSpec) -> dict[Split, IntArray]:
    """Fit the three partitions so their union holds every reported margin."""
    _validate_spec(spec)
    allowed = allowed_cells()
    test_table = build_table(_TABLE_SHAPE, _test_margins(spec), spec.test_total, allowed=allowed)
    rest = build_table(
        _TABLE_SHAPE,
        _residual_margins(spec),
        spec.train_total + spec.calibration_total,
        allowed=allowed,
    )
    train_table, calibration_table = split_proportionally(rest, spec.train_total)
    return {
        Split.TRAIN: train_table,
        Split.CALIBRATION: calibration_table,
        Split.TEST: test_table,
    }


def combined_table(tables: Mapping[Split, IntArray]) -> IntArray:
    total = np.zeros(_TABLE_SHAPE, dtype=np.int64)
    for table in tables.values():
        total = total + table
    return total


def _record_order(tables: Mapping[Split, IntArray]) -> list[tuple[Split, tuple[int, ...]]]:
    order: list[tuple[Split, tuple[int, ...]]] = []
    for split in SPLIT_LEVELS:
        table = tables[split]
        for index in np.ndindex(*table.shape):
            if int(table[index]) > 0:
                order.append((split, tuple(int(value) for value in index)))
    return order


def _draw_features(
    rng: np.random.Generator,
    molecular: MolecularClass,
    has_pathology: bool,
    has_imaging: bool,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Class-conditional feature draws, independent of the label.

    The three expression blocks stand for the interferon-gamma programme, the
    MAPK programme and antigen presentation, which are the mechanisms the
    treatment-conditional analysis names.
    """
    clinical = tuple(float(value) for value in rng.normal(0.0, 1.0, FEATURE_WIDTH_CLINICAL))
    expression = rng.normal(0.0, 1.0, FEATURE_WIDTH_MOLECULAR)
    if molecular is MolecularClass.BRAF_V600:
        expression[4:8] += 0.45
    else:
        expression[0:4] += 0.45
    histology = (
        tuple(float(value) for value in rng.normal(0.0, 1.0, FEATURE_WIDTH_HISTOLOGY))
        if has_pathology
        else tuple(0.0 for _ in range(FEATURE_WIDTH_HISTOLOGY))
    )
    image = (
        tuple(float(value) for value in rng.normal(0.0, 1.0, FEATURE_WIDTH_IMAGE))
        if has_imaging
        else tuple(0.0 for _ in range(FEATURE_WIDTH_IMAGE))
    )
    return (
        clinical,
        tuple(float(value) for value in expression),
        histology,
        image,
    )


def _programme_means(molecular: MolecularClass) -> dict[str, float]:
    """The coefficient vector of the latent response model, by branch.

    In BRAF V600 records the MAPK programme and antigen presentation carry the
    signal; in the wild-type branch the interferon-gamma programme does. That
    asymmetry is the manuscript's treatment-conditional finding, and it is
    reproduced here so the framework has a structure to find.
    """
    if molecular is MolecularClass.BRAF_V600:
        return {"interferon": 0.10, "mapk": -0.95, "antigen": 0.90, "clinical": 0.35}
    return {"interferon": 1.10, "mapk": -0.10, "antigen": 0.30, "clinical": 0.35}


def latent_risk(record: PatientRecord, coefficients: Mapping[str, float] | None = None) -> float:
    """The fixture's true log-odds of response for one record."""
    weights = _programme_means(record.molecular) if coefficients is None else coefficients
    expression = np.asarray(record.expression_vector, dtype=np.float64)
    clinical = np.asarray(record.clinical_features, dtype=np.float64)
    value = (
        weights["interferon"] * float(np.mean(expression[0:4]))
        + weights["mapk"] * float(np.mean(expression[4:8]))
        + weights["antigen"] * float(np.mean(expression[8:12]))
        + weights["clinical"] * float(np.mean(clinical))
    )
    if record.has_pathology:
        value += 0.05 * float(np.mean(np.asarray(record.histology_features, dtype=np.float64)))
    if record.has_imaging:
        value += 0.05 * float(np.mean(np.asarray(record.image_features, dtype=np.float64)))
    return value


@dataclass(frozen=True, slots=True)
class CohortFixture:
    """The fixture's records plus the audit of its margins."""

    spec: CohortSpec
    records: tuple[PatientRecord, ...]
    margin_deviations: Mapping[str, int]
    positive_counts: Mapping[str, int]

    def by_split(self, split: Split) -> tuple[PatientRecord, ...]:
        return tuple(record for record in self.records if record.split is split)

    def labels(self, split: Split | None = None) -> IntArray:
        selected = self.records if split is None else self.by_split(split)
        return np.asarray([record.response for record in selected], dtype=np.int64)

    def molecular_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.molecular.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def treatment_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.treatment.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def site_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.site.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def sex_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.sex.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def age_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.age_band.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def molecular_treatment_table(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = f"{record.molecular.value} | {record.treatment.value}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def region_counts(self, split: Split | None = None) -> dict[str, int]:
        selected = self.records if split is None else self.by_split(split)
        counts: dict[str, int] = {}
        for record in selected:
            key = record.region.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def response_rate(self, split: Split | None = None) -> float:
        labels = self.labels(split)
        if labels.shape[0] == 0:
            return 0.0
        return float(np.mean(labels))

    def subgroup_census(self, split: Split) -> dict[str, dict[str, int]]:
        return {
            "Molecular": self.molecular_counts(split),
            "Treatment class": self.treatment_counts(split),
            "Site": self.site_counts(split),
            "Sex": self.sex_counts(split),
            "Age band": self.age_counts(split),
        }

    def size(self, split: Split | None = None) -> int:
        return len(self.records) if split is None else len(self.by_split(split))


def _prior_lines(rng: np.random.Generator, share: float) -> PriorLines:
    draw = float(rng.random())
    if draw < share:
        return PriorLines.TWO_OR_MORE
    if draw < share + 0.42:
        return PriorLines.ONE
    return PriorLines.NONE


def _availability(rng: np.random.Generator, site: Site, spec: CohortSpec) -> tuple[bool, bool]:
    pathology = float(rng.random()) >= spec.pathology_missing_share
    imaging_missing = site is SITE_LEVELS[spec.imaging_missing_site]
    imaging = float(rng.random()) >= spec.imaging_missing_share if imaging_missing else True
    return (pathology, imaging)


def build_cohort_fixture(spec: CohortSpec | None = None) -> CohortFixture:
    """Construct the record set and audit the margins it was built to hold."""
    resolved = spec if spec is not None else CohortSpec()
    tables = build_partition_tables(resolved)
    rng = np.random.default_rng(resolved.fixture_seed)
    records: list[PatientRecord] = []
    counter = 0
    for split, index in _record_order(tables):
        molecular = MOLECULAR_LEVELS[index[0]]
        treatment = TREATMENT_LEVELS[index[1]]
        site = SITE_LEVELS[index[2]]
        sex = SEX_LEVELS[index[3]]
        age_band = AGE_LEVELS[index[4]]
        for _ in range(int(tables[split][index])):
            pathology, imaging = _availability(rng, site, resolved)
            clinical, expression, histology, image = _draw_features(
                rng, molecular, pathology, imaging
            )
            low, high = _AGE_RANGE[age_band]
            age = int(rng.integers(low, high + 1))
            records.append(
                PatientRecord(
                    patient_id=f"R{counter:06d}",
                    site=site,
                    sex=sex,
                    age=age,
                    molecular=molecular,
                    treatment=treatment,
                    prior_lines=_prior_lines(rng, resolved.advanced_line_share),
                    has_pathology=pathology,
                    has_imaging=imaging,
                    follow_up_months=float(rng.gamma(3.0, 10.5)),
                    response=0,
                    split=split,
                    tumour_mutational_burden=float(rng.normal(0.0, 1.0)),
                    pdl1_fraction=float(np.clip(rng.beta(2.0, 5.0), 0.0, 1.0)),
                    interferon_gamma_score=float(np.mean(np.asarray(expression[0:4]))),
                    pathway_signature=float(np.mean(np.asarray(expression[4:8]))),
                    expression_vector=expression,
                    histology_features=histology,
                    image_features=image,
                    clinical_features=clinical,
                )
            )
            counter += 1
    labelled = _assign_labels(records, resolved, rng)
    combined = combined_table(tables)
    deviations = margin_error(combined, _full_margins(resolved))
    positives: dict[str, int] = {}
    for split in SPLIT_LEVELS:
        positives[split.value] = int(
            sum(1 for record in labelled if record.split is split and record.response == 1)
        )
    return CohortFixture(
        spec=resolved,
        records=tuple(labelled),
        margin_deviations=deviations,
        positive_counts=positives,
    )


def _assign_labels(
    records: Sequence[PatientRecord],
    spec: CohortSpec,
    rng: np.random.Generator,
) -> list[PatientRecord]:
    """Label the records so each partition holds its declared number of positives.

    The label is the rank of a noisy latent risk, not a threshold on the risk
    itself, so the Bayes-optimal AUROC on the fixture is bounded away from one and
    a ranking check has something to be wrong about.
    """
    targets = dict(zip(SPLIT_LEVELS, spec.positive_targets(), strict=True))
    risk = np.asarray([latent_risk(record) for record in records], dtype=np.float64)
    standardised = (risk - float(np.mean(risk))) / max(float(np.std(risk)), 1e-9)
    latent = spec.signal_strength * standardised + rng.normal(0.0, 1.0, len(records))
    chosen: set[int] = set()
    for split in SPLIT_LEVELS:
        members = [index for index, record in enumerate(records) if record.split is split]
        ranked = sorted(
            members, key=lambda index: (-float(latent[index]), records[index].patient_id)
        )
        chosen.update(ranked[: targets[split]])
    return [
        _with_response(record, 1 if index in chosen else 0) for index, record in enumerate(records)
    ]


def _with_response(record: PatientRecord, response: int) -> PatientRecord:
    from dataclasses import replace

    return replace(record, response=response)


def fixture_marginals(fixture: CohortFixture) -> dict[str, int]:
    return dict(sorted(fixture.margin_deviations.items()))


def bayes_scores(fixture: CohortFixture, split: Split | None = None) -> FloatArray:
    """The fixture's true log-odds, which bounds what any model can reach."""
    selected = fixture.records if split is None else fixture.by_split(split)
    return np.asarray([latent_risk(record) for record in selected], dtype=np.float64)


def record_matrix(fixture: CohortFixture, split: Split | None = None) -> FloatArray:
    selected = fixture.records if split is None else fixture.by_split(split)
    return np.asarray([record.feature_vector() for record in selected], dtype=np.float64)


def modality_matrix(fixture: CohortFixture, split: Split | None = None) -> FloatArray:
    """Per-record modality presence in MODALITY_ORDER, for the encoder mask."""
    from pathanchor.models.encoder import MODALITY_ORDER

    selected = fixture.records if split is None else fixture.by_split(split)
    out = np.zeros((len(selected), len(MODALITY_ORDER)), dtype=np.float64)
    for row, record in enumerate(selected):
        present = set(record.observed_modalities())
        for column, name in enumerate(MODALITY_ORDER):
            out[row, column] = 1.0 if name in present else 0.0
    return out
