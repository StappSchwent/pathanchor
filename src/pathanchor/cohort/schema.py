"""Cohort record schema.

Ref: Sec. III-F (Cohorts), Table III (subgroup definitions).

The manuscript's four-site retrospective cohort is held under a data-use
agreement and is not redistributed. This module defines the record contract the
analysis consumes, so any site-supplied table with the same columns can be read
by the same code path as the fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class Site(StrEnum):
    A = "Site A"
    B = "Site B"
    C = "Site C"
    D = "Site D"


class Region(StrEnum):
    ONE = "Region I"
    TWO = "Region II"


class Sex(StrEnum):
    FEMALE = "Female"
    MALE = "Male"


class MolecularClass(StrEnum):
    BRAF_V600 = "BRAF V600 mutant"
    NRAS = "NRAS mutant"
    NF1 = "NF1 altered"
    TRIPLE_WILD_TYPE = "Triple wild-type"


class TreatmentClass(StrEnum):
    ANTI_PD1_MONO = "Anti-PD-1 monotherapy"
    ANTI_PD1_ANTI_CTLA4 = "Anti-PD-1 + anti-CTLA-4"
    BRAF_MEK = "BRAF/MEK inhibition"


class AgeBand(StrEnum):
    UNDER_65 = "Age < 65 years"
    AT_LEAST_65 = "Age >= 65 years"


class PriorLines(StrEnum):
    NONE = "no prior line"
    ONE = "one prior line"
    TWO_OR_MORE = "two or more prior lines"


class Split(StrEnum):
    TRAIN = "training"
    CALIBRATION = "calibration"
    TEST = "testing"


REGION_OF_SITE: Mapping[Site, Region] = {
    Site.A: Region.ONE,
    Site.B: Region.ONE,
    Site.C: Region.TWO,
    Site.D: Region.TWO,
}

WILD_TYPE_CLASSES: tuple[MolecularClass, ...] = (
    MolecularClass.NRAS,
    MolecularClass.NF1,
    MolecularClass.TRIPLE_WILD_TYPE,
)

FEATURE_WIDTH_MOLECULAR = 12
FEATURE_WIDTH_HISTOLOGY = 8
FEATURE_WIDTH_IMAGE = 16
FEATURE_WIDTH_CLINICAL = 6


def region_of(site: Site) -> Region:
    return REGION_OF_SITE[site]


@dataclass(frozen=True, slots=True)
class PatientRecord:
    """One de-identified record.

    Direct and quasi identifiers are absent by construction: the record carries
    an interval-valued follow-up, a coarse age band, and site codes that name no
    institution. ``molecular_flags`` and the feature vectors are the only
    quantities the estimator reads.
    """

    patient_id: str
    site: Site
    sex: Sex
    age: int
    molecular: MolecularClass
    treatment: TreatmentClass
    prior_lines: PriorLines
    has_pathology: bool
    has_imaging: bool
    follow_up_months: float
    response: int
    split: Split
    tumour_mutational_burden: float
    pdl1_fraction: float
    interferon_gamma_score: float
    pathway_signature: float
    expression_vector: tuple[float, ...]
    histology_features: tuple[float, ...]
    image_features: tuple[float, ...]
    clinical_features: tuple[float, ...]

    @property
    def region(self) -> Region:
        return region_of(self.site)

    @property
    def age_band(self) -> AgeBand:
        return AgeBand.UNDER_65 if self.age < 65 else AgeBand.AT_LEAST_65

    @property
    def is_braf_v600(self) -> bool:
        return self.molecular is MolecularClass.BRAF_V600

    def molecular_flags(self) -> dict[str, bool]:
        """Named molecular calls, used both for grounding and for features."""
        return {
            "braf_v600": self.molecular is MolecularClass.BRAF_V600,
            "nras": self.molecular is MolecularClass.NRAS,
            "nf1": self.molecular is MolecularClass.NF1,
            "tmb": True,
            "pdl1": True,
        }

    def has_molecular_profiling(self) -> bool:
        return True

    def feature_vector(self) -> tuple[float, ...]:
        values: list[float] = list(self.clinical_features)
        values.extend(self.expression_vector)
        values.extend(
            self.histology_features if self.has_pathology else _zeros(FEATURE_WIDTH_HISTOLOGY)
        )
        values.extend(self.image_features if self.has_imaging else _zeros(FEATURE_WIDTH_IMAGE))
        return tuple(values)

    def observed_modalities(self) -> tuple[str, ...]:
        present = ["clinical", "molecular"]
        if self.has_pathology:
            present.append("histology")
        if self.has_imaging:
            present.append("image")
        return tuple(present)


def _zeros(width: int) -> tuple[float, ...]:
    return tuple(0.0 for _ in range(width))


FEATURE_COLUMNS: tuple[str, ...] = (
    "clinical_features",
    "expression_vector",
    "histology_features",
    "image_features",
)


def feature_width() -> int:
    return (
        FEATURE_WIDTH_CLINICAL
        + FEATURE_WIDTH_MOLECULAR
        + FEATURE_WIDTH_HISTOLOGY
        + FEATURE_WIDTH_IMAGE
    )


def validate_record(record: PatientRecord) -> None:
    if record.response not in (0, 1):
        raise ValueError(f"record {record.patient_id}: response must be 0 or 1")
    if record.age < 18 or record.age > 120:
        raise ValueError(f"record {record.patient_id}: age outside the recorded range")
    if len(record.clinical_features) != FEATURE_WIDTH_CLINICAL:
        raise ValueError(f"record {record.patient_id}: clinical feature width is wrong")
    if len(record.expression_vector) != FEATURE_WIDTH_MOLECULAR:
        raise ValueError(f"record {record.patient_id}: molecular feature width is wrong")
    if len(record.histology_features) != FEATURE_WIDTH_HISTOLOGY:
        raise ValueError(f"record {record.patient_id}: histology feature width is wrong")
    if len(record.image_features) != FEATURE_WIDTH_IMAGE:
        raise ValueError(f"record {record.patient_id}: image feature width is wrong")


def subgroup_labels(record: PatientRecord) -> Mapping[str, Sequence[str]]:
    """The prespecified subgroup keys of Table III."""
    return {
        "Molecular": (record.molecular.value,),
        "Treatment class": (record.treatment.value,),
        "Site": (record.site.value,),
        "Region": (record.region.value,),
        "Demographic / clinical": (record.sex.value, record.age_band.value),
    }
