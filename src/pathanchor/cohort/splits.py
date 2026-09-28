"""Patient-disjoint, site-stratified partitioning.

Ref: Sec. III-F ("stratifying jointly by the site and the response status,
ensuring that each site has been included in each part. All splits are
patient-disjoint by construction").

The partition is used both to audit the fixture's own 60/20/20 composition and to
partition a caller-supplied table with the same columns. The joint stratification
is what guarantees every site appears in every part at its own response
prevalence, rather than a global stratification that could empty a site's
calibration cell.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pathanchor.cohort.schema import PatientRecord, Split
from pathanchor.utils.validation import require_disjoint


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.60
    calibration: float = 0.20
    test: float = 0.20

    def validate(self) -> None:
        total = self.train + self.calibration + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("the partition fractions must sum to one")
        if min(self.train, self.calibration, self.test) <= 0.0:
            raise ValueError("every partition fraction must be positive")


@dataclass(frozen=True, slots=True)
class Partition:
    train: tuple[PatientRecord, ...]
    calibration: tuple[PatientRecord, ...]
    test: tuple[PatientRecord, ...]

    def as_mapping(self) -> Mapping[Split, tuple[PatientRecord, ...]]:
        return {
            Split.TRAIN: self.train,
            Split.CALIBRATION: self.calibration,
            Split.TEST: self.test,
        }

    def all_ids(self) -> tuple[str, ...]:
        return tuple(
            record.patient_id
            for group in (self.train, self.calibration, self.test)
            for record in group
        )

    def sizes(self) -> tuple[int, int, int]:
        return (len(self.train), len(self.calibration), len(self.test))

    def check_disjoint(self) -> None:
        require_disjoint(
            {
                Split.TRAIN.value: [record.patient_id for record in self.train],
                Split.CALIBRATION.value: [record.patient_id for record in self.calibration],
                Split.TEST.value: [record.patient_id for record in self.test],
            },
            "the patient partition",
        )


def stratified_partition(
    records: Sequence[PatientRecord],
    fractions: SplitFractions,
    seed: int = 20260101,
) -> Partition:
    """Partition records so every (site, response) cell is split three ways."""
    fractions.validate()
    rng = np.random.default_rng(seed)
    cells: dict[tuple[str, int], list[PatientRecord]] = defaultdict(list)
    for record in records:
        cells[(record.site.value, int(record.response))].append(record)
    train: list[PatientRecord] = []
    calibration: list[PatientRecord] = []
    test: list[PatientRecord] = []
    for key in sorted(cells):
        members = sorted(cells[key], key=lambda item: item.patient_id)
        order = rng.permutation(len(members))
        shuffled = [members[int(index)] for index in order]
        n_test = round(len(shuffled) * fractions.test)
        n_calibration = round(len(shuffled) * fractions.calibration)
        n_test = min(n_test, max(len(shuffled) - 2, 0))
        n_calibration = min(n_calibration, max(len(shuffled) - n_test - 1, 0))
        test.extend(shuffled[:n_test])
        calibration.extend(shuffled[n_test : n_test + n_calibration])
        train.extend(shuffled[n_test + n_calibration :])
    partition = Partition(
        train=tuple(sorted(train, key=lambda item: item.patient_id)),
        calibration=tuple(sorted(calibration, key=lambda item: item.patient_id)),
        test=tuple(sorted(test, key=lambda item: item.patient_id)),
    )
    partition.check_disjoint()
    return partition


def reindex_splits(
    records: Sequence[PatientRecord],
    partition: Partition,
) -> tuple[PatientRecord, ...]:
    """Return ``records`` with the split field set from an existing partition."""
    from dataclasses import replace

    assignment = {
        record.patient_id: split
        for split, group in partition.as_mapping().items()
        for record in group
    }
    missing = [record.patient_id for record in records if record.patient_id not in assignment]
    if missing:
        raise ValueError(f"records absent from the partition: {missing[:4]}")
    return tuple(replace(record, split=assignment[record.patient_id]) for record in records)


def site_prevalence(records: Sequence[PatientRecord]) -> dict[str, float]:
    counts: dict[str, list[int]] = defaultdict(list)
    for record in records:
        counts[record.site.value].append(int(record.response))
    return {
        site: (float(np.mean(values)) if values else 0.0) for site, values in sorted(counts.items())
    }


def partition_report(partition: Partition) -> dict[str, object]:
    mapping = partition.as_mapping()
    sizes = partition.sizes()
    return {
        "sizes": {"train": sizes[0], "calibration": sizes[1], "test": sizes[2]},
        "response_rate": {
            split.value: (float(np.mean([record.response for record in group])) if group else 0.0)
            for split, group in mapping.items()
        },
        "site_counts": {split.value: _counts(group, "site") for split, group in mapping.items()},
        "sites_in_every_part": all(
            len(_counts(group, "site")) == len(_counts(partition_as_list(partition), "site"))
            for group in (partition.train, partition.calibration, partition.test)
        ),
        "patient_disjoint": len(set(partition.all_ids())) == len(partition.all_ids()),
    }


def partition_as_list(partition: Partition) -> list[PatientRecord]:
    return [*partition.train, *partition.calibration, *partition.test]


def _counts(records: Sequence[PatientRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(getattr(record, field).value)] += 1
    return dict(sorted(counts.items()))
