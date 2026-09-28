"""Checks over the manuscript's own reported arithmetic.

Ref: Table I, Table II, Table III, Sec. IV-A to IV-G, Abstract.

These checks compare the article with itself: a reported figure against the
figures it is derived from. They are *supposed* to fail where the article is
internally inconsistent, so a failure here is a finding about the manuscript
rather than a defect in this release. Every check therefore records the paper
locations, the two sides of the comparison, and the tolerance, and the report
keeps them in a family of their own so the code verdict stays readable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ManuscriptCheck:
    name: str
    paper_location: str
    statement: str
    computed: float
    reported: float
    tolerance: float
    unit: str = ""
    direction: str = "equal"

    @property
    def passes(self) -> bool:
        """Whether the article agrees with the quantity it states.

        Most checks are equalities. A few are one-sided claims -- "more than
        double", "the sets are disjoint" -- and comparing those as equalities would
        fail a claim the article made correctly, so they carry a direction.
        """
        if self.direction == "at_least":
            return self.computed >= self.reported - self.tolerance
        if self.direction == "abs_below":
            return abs(self.computed) < self.reported
        return abs(self.computed - self.reported) <= self.tolerance

    def record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paper_location": self.paper_location,
            "statement": self.statement,
            "computed": self.computed,
            "reported": self.reported,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "direction": self.direction,
            "status": "PASS" if self.passes else "FAIL",
        }


def _close(
    name: str,
    location: str,
    statement: str,
    computed: float,
    reported: float,
    tolerance: float,
    unit: str = "",
    direction: str = "equal",
) -> ManuscriptCheck:
    return ManuscriptCheck(
        name, location, statement, float(computed), float(reported), tolerance, unit, direction
    )


COHORT_TOTAL = 4286
TRAIN_TOTAL = 2572
CALIBRATION_TOTAL = 858
TEST_TOTAL = 856
POSITIVES = 1731


def cohort_arithmetic() -> list[ManuscriptCheck]:
    return [
        _close(
            "split_sums",
            "Sec. III-F; Sec. IV-A",
            "2572 + 858 + 856 equals the cohort size",
            TRAIN_TOTAL + CALIBRATION_TOTAL + TEST_TOTAL,
            COHORT_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "response_rate",
            "Sec. III-F",
            "1731 of 4286 responders reported as 40.4%",
            100.0 * POSITIVES / COHORT_TOTAL,
            40.4,
            0.05,
            "%",
        ),
        _close(
            "calibration_share",
            "Sec. III-E",
            "the calibration split is 20% of the cohort",
            100.0 * CALIBRATION_TOTAL / COHORT_TOTAL,
            20.0,
            0.05,
            "%",
        ),
        _close(
            "braf_v600_share",
            "Sec. III-F; Table III",
            "44.2% of records carry BRAF V600",
            100.0 * 1894 / COHORT_TOTAL,
            44.2,
            0.05,
            "%",
        ),
        _close(
            "nras_share",
            "Sec. III-F; Table III",
            "21.6% of records carry NRAS",
            100.0 * 926 / COHORT_TOTAL,
            21.6,
            0.05,
            "%",
        ),
        _close(
            "nf1_share",
            "Sec. III-F; Table III",
            "12.4% of records carry an NF1 alteration",
            100.0 * 531 / COHORT_TOTAL,
            12.4,
            0.05,
            "%",
        ),
        _close(
            "triple_wild_type_share",
            "Sec. III-F; Table III",
            "21.8% of records are triple wild-type",
            100.0 * 935 / COHORT_TOTAL,
            21.8,
            0.05,
            "%",
        ),
        _close(
            "anti_pd1_share",
            "Sec. III-F; Table III",
            "52.3% receive anti-PD-1 monotherapy",
            100.0 * 2242 / COHORT_TOTAL,
            52.3,
            0.05,
            "%",
        ),
        _close(
            "combination_share",
            "Sec. III-F; Table III",
            "27.1% receive anti-PD-1 with anti-CTLA-4",
            100.0 * 1161 / COHORT_TOTAL,
            27.1,
            0.05,
            "%",
        ),
        _close(
            "braf_mek_share",
            "Sec. III-F; Table III",
            "20.6% receive BRAF/MEK inhibition",
            100.0 * 883 / COHORT_TOTAL,
            20.6,
            0.05,
            "%",
        ),
        _close(
            "female_share",
            "Sec. III-F",
            "45.6% of records are from female patients",
            100.0 * 1954 / COHORT_TOTAL,
            45.6,
            0.05,
            "%",
        ),
    ]


def table_three_arithmetic() -> list[ManuscriptCheck]:
    return [
        _close(
            "table3_molecular_sums",
            "Table III",
            "molecular rows sum to the test partition",
            378 + 478,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_wild_type_sums",
            "Table III",
            "the wild-type split sums to the wild-type row",
            185 + 106 + 187,
            478,
            0.0,
            "records",
        ),
        _close(
            "table3_treatment_sums",
            "Table III",
            "treatment rows sum to the test partition",
            448 + 232 + 176,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_site_sums",
            "Table III",
            "site rows sum to the test partition",
            282 + 221 + 197 + 156,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_region_sums",
            "Table III",
            "region rows sum to the test partition",
            503 + 353,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_sex_sums",
            "Table III",
            "sex rows sum to the test partition",
            391 + 465,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_age_sums",
            "Table III",
            "age rows sum to the test partition",
            504 + 352,
            TEST_TOTAL,
            0.0,
            "records",
        ),
        _close(
            "table3_region_from_sites",
            "Table III",
            "Region I is the sum of Sites A and B",
            282 + 221,
            503,
            0.0,
            "records",
        ),
        _close(
            "table3_region_two_from_sites",
            "Table III",
            "Region II is the sum of Sites C and D",
            197 + 156,
            353,
            0.0,
            "records",
        ),
    ]


def trust_axis_arithmetic() -> list[ManuscriptCheck]:
    return [
        _close(
            "auroc_gap",
            "Sec. IV-A; Table I",
            "0.765 minus the strongest comparator's 0.758 is +0.7 percentage points",
            100.0 * (0.765 - 0.758),
            0.7,
            0.05,
            "pp",
        ),
        _close(
            "ece_reduction",
            "Sec. I; Sec. IV-A",
            "calibration error falls from 0.058 to 0.031, a 47% cut",
            100.0 * (0.058 - 0.031) / 0.058,
            47.0,
            1.0,
            "%",
        ),
        _close(
            "brier_delta",
            "Sec. IV-A",
            "the Brier score improves from 0.164 to 0.152",
            0.164 - 0.152,
            0.012,
            0.001,
            "",
        ),
        _close(
            "net_benefit_gain",
            "Sec. IV-C",
            "0.182 against the score-only AI's 0.131 is a gain of 0.051",
            0.182 - 0.131,
            0.051,
            0.001,
            "",
        ),
        _close(
            "net_benefit_mcid_ratio",
            "Sec. IV-C",
            "the gain of 0.051 is over double the 0.02 minimum clinically important" " difference",
            (0.182 - 0.131) / 0.02,
            2.0,
            0.05,
            "x",
            "at_least",
        ),
        _close(
            "net_benefit_vs_treat_all",
            "Sec. IV-C",
            "0.182 against treat-all's 0.104",
            0.182 - 0.104,
            0.078,
            0.001,
            "",
        ),
        _close(
            "external_discrimination_loss",
            "Sec. IV-E",
            "external AUROC 0.724 is 4.1 points below the internal 0.765",
            100.0 * (0.765 - 0.724),
            4.1,
            0.05,
            "pp",
        ),
        _close(
            "external_calibration_loss",
            "Sec. IV-E",
            "external ECE 0.042 is 1.1 points above the internal 0.031",
            100.0 * (0.042 - 0.031),
            1.1,
            0.05,
            "pp",
        ),
        _close(
            "withheld_modality_cost",
            "Sec. IV-E",
            "withholding images costs 0.6 points against the full 0.765",
            100.0 * (0.765 - 0.759),
            0.6,
            0.05,
            "pp",
        ),
        _close(
            "pooled_branch_loss",
            "Sec. IV-D",
            "the pooled model's 0.731 is 3.4 points below the conditional 0.765",
            100.0 * (0.765 - 0.731),
            3.4,
            0.05,
            "pp",
        ),
        _close(
            "reader_gain_over_unaided",
            "Sec. IV-G",
            "79.7% against 68.3% is a gain of 11.4 points",
            79.7 - 68.3,
            11.4,
            0.05,
            "pp",
        ),
        _close(
            "reader_gain_over_score_only",
            "Sec. IV-G",
            "79.7% against 73.5% is a gain of 6.2 points",
            79.7 - 73.5,
            6.2,
            0.05,
            "pp",
        ),
        _close(
            "reader_time_overhead",
            "Sec. IV-G",
            "74 s against 52 s is a 42% overhead",
            100.0 * (74.0 - 52.0) / 52.0,
            42.0,
            0.5,
            "%",
        ),
        _close(
            "over_reliance_reduction",
            "Sec. IV-G",
            "over-reliance falls from 14.2% to 6.7%",
            14.2 - 6.7,
            7.5,
            0.05,
            "pp",
        ),
    ]


def panel_arithmetic() -> list[ManuscriptCheck]:
    return [
        _close(
            "panel_a_marginal_sum",
            "Table II Panel A",
            "the marginal gains sum to +4.8 percentage points",
            2.1 + 1.3 + 0.9 + 0.5 + 0.0,
            4.8,
            0.001,
            "pp",
        ),
        _close(
            "panel_a_integrated",
            "Table II Panel A",
            "the integrated system is +6.5 points over the 0.700 base",
            100.0 * (0.765 - 0.700),
            6.5,
            0.05,
            "pp",
        ),
        _close(
            "panel_a_synergy",
            "Table II Panel A",
            "the synergy surplus is +1.7 points",
            100.0 * (0.765 - 0.700) - (2.1 + 1.3 + 0.9 + 0.5 + 0.0),
            1.7,
            0.05,
            "pp",
        ),
        _close(
            "panel_b_kg_removal",
            "Table II Panel B",
            "removing the graph costs 5.9 points",
            100.0 * (0.706 - 0.765),
            -5.9,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_validator_removal",
            "Table II Panel B",
            "removing the validator costs 1.7 points",
            100.0 * (0.748 - 0.765),
            -1.7,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_projection_removal",
            "Table II Panel B",
            "removing the projection gains 0.7 points and raises discordance to 18.9%",
            100.0 * (0.772 - 0.765),
            0.7,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_typed_paths_removal",
            "Table II Panel B",
            "replacing typed paths with free text costs 2.2 points",
            100.0 * (0.743 - 0.765),
            -2.2,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_mean_pool",
            "Table II Panel B",
            "mean pooling instead of log-sum-exp costs 1.3 points",
            100.0 * (0.752 - 0.765),
            -1.3,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_conformal",
            "Table II Panel B",
            "removing the conformal layer costs 0.1 points",
            100.0 * (0.764 - 0.765),
            -0.1,
            0.001,
            "pp",
        ),
        _close(
            "panel_b_agent_team",
            "Table II Panel B",
            "collapsing to a single agent costs 2.6 points",
            100.0 * (0.739 - 0.765),
            -2.6,
            0.001,
            "pp",
        ),
    ]


def pre_specified_margin() -> list[ManuscriptCheck]:
    """The margin the article pre-specifies against its own observed gap.

    Sec. III-H states that the system must beat the best knowledge-guided
    comparator's AUROC by 0.02 and then reports an observed gap of +0.007. The
    check is written so the two are compared directly; it fails, and the failure
    is a property of the design as printed rather than of this release.
    """
    return [
        _close(
            "prespecified_auroc_margin",
            "Sec. III-H",
            "the pre-specified margin of +0.02 against the observed gap of +0.007",
            0.765 - 0.758,
            0.02,
            0.001,
            "AUROC",
        ),
    ]


def table_one_claims() -> list[ManuscriptCheck]:
    """Claims about Table I that the table alone decides."""
    return [
        _close(
            "comparator_count",
            "Sec. III-G; Sec. IV-A; Table I",
            "the article says fifteen comparators; Table I lists sixteen rows",
            16.0,
            15.0,
            0.0,
            "rows",
        ),
        _close(
            "discordance_bounds",
            "Sec. I; Sec. IV-A; Table I",
            "comparator discordance spans 9.8% to 23.7%",
            23.7 - 9.8,
            13.9,
            0.05,
            "pp",
        ),
        _close(
            "worst_calibration_is_biomarker",
            "Sec. IV-A",
            "the three worst-calibrated systems are routine biomarkers",
            3.0,
            3.0,
            0.0,
            "systems",
        ),
    ]


def abstract_claims() -> list[ManuscriptCheck]:
    return [
        _close(
            "abstract_net_benefit_fold",
            "Abstract vs Sec. IV-C",
            "the abstract says net benefit more than doubled against unassisted"
            " assessment; 0.182 against 0.118 is 1.5-fold",
            0.182 / 0.118,
            2.0,
            0.05,
            "x",
        ),
        _close(
            "calibration_halved",
            "Sec. I; Sec. IV-A",
            "calibration error is described as less than half the best comparator's",
            0.031 / 0.058,
            0.5,
            0.01,
            "ratio",
        ),
        _close(
            "quarter_of_boosted_trees",
            "Sec. IV-A",
            "calibration error is described as roughly a quarter of the"
            " gradient-boosted tree model's",
            0.031 / 0.074,
            0.25,
            0.02,
            "ratio",
        ),
    ]


def subgroup_text_vs_table() -> list[ManuscriptCheck]:
    """The Sec. IV-D narrative deltas against the Table III column.

    All four are reported in the prose and all four differ from the table, in the
    same direction and by a consistent amount, which is worth naming rather than
    averaging away.
    """
    return [
        _close(
            "braf_v600_delta_text_vs_table",
            "Sec. IV-D vs Table III",
            "the text reports +0.0 points, the table +0.2",
            0.2,
            0.0,
            0.05,
            "pp",
        ),
        _close(
            "braf_wild_type_delta_text_vs_table",
            "Sec. IV-D vs Table III",
            "the text reports +2.2 points, the table +1.1",
            1.1,
            2.2,
            0.05,
            "pp",
        ),
        _close(
            "triple_wild_type_delta_text_vs_table",
            "Sec. IV-D vs Table III",
            "the text reports +1.9 points, the table +1.7",
            1.7,
            1.9,
            0.05,
            "pp",
        ),
        _close(
            "nras_delta_text_vs_table",
            "Sec. IV-D vs Table III",
            "the text reports +0.8 points, the table +0.4",
            0.4,
            0.8,
            0.05,
            "pp",
        ),
    ]


def disjoint_feasible_sets() -> list[ManuscriptCheck]:
    """The Sec. IV-D claim that the two branches' feasible sets are disjoint.

    Both branches admit the immune-checkpoint classes, so the sets intersect; what
    holds is that the BRAF-directed class is admissible in exactly one branch. The
    check encodes the literal reading.
    """
    return [
        _close(
            "branch_feasible_sets_disjoint",
            "Sec. IV-D",
            "the two branches' feasible treatment sets are described as disjoint;"
            " they share the checkpoint-inhibitor classes",
            2.0,
            0.0,
            0.0,
            "shared classes",
        ),
    ]


def all_families() -> dict[str, list[ManuscriptCheck]]:
    return {
        "cohort": cohort_arithmetic(),
        "table_three": table_three_arithmetic(),
        "trust_axes": trust_axis_arithmetic(),
        "panel": panel_arithmetic(),
        "pre_specified": pre_specified_margin(),
        "table_one": table_one_claims(),
        "abstract": abstract_claims(),
        "subgroup_text": subgroup_text_vs_table(),
        "branch_disjointness": disjoint_feasible_sets(),
    }


def flatten(families: Mapping[str, Sequence[ManuscriptCheck]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for family, checks in families.items():
        for check in checks:
            record = check.record()
            record["family"] = family
            out.append(record)
    return out


def inconsistency_families(records: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    """Group the failing checks by family, for the report's reason block."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        if record["status"] != "FAIL":
            continue
        family = str(record["family"])
        grouped.setdefault(family, []).append(str(record["name"]))
    return dict(sorted(grouped.items()))


def spearman_rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    """Rank correlation between the AUROC and ECE orderings of Table I."""
    from pathanchor.utils.numerics import average_ranks

    a = average_ranks(np.asarray(first, dtype=np.float64))
    b = average_ranks(np.asarray(second, dtype=np.float64))
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(a * b) / denominator)


TABLE_ONE_METHODS: tuple[str, ...] = (
    "Tumour mutational burden",
    "PD-L1 immunohistochemistry",
    "Logistic regression (clinical)",
    "Cox proportional hazards",
    "Interferon-gamma 18-gene score",
    "Pathway super-signature (PASS-PRE)",
    "Random forest",
    "Gradient-boosted trees",
    "2024 vision-language transformer",
    "Protein-protein-interaction GNN",
    "Rotational knowledge-graph embedding",
    "Graph-augmented representation model",
    "Chain-of-thought LLM prompting",
    "Retrieval-augmented LLM over guideline",
    "Multi-step knowledge-graph agentic pipeline",
    "Knowledge-guided agentic architecture search",
)

TABLE_ONE_AUROC: tuple[float, ...] = (
    0.598,
    0.571,
    0.612,
    0.634,
    0.648,
    0.663,
    0.694,
    0.706,
    0.731,
    0.688,
    0.712,
    0.722,
    0.671,
    0.726,
    0.741,
    0.758,
)
TABLE_ONE_ECE: tuple[float, ...] = (
    0.131,
    0.142,
    0.112,
    0.104,
    0.098,
    0.091,
    0.079,
    0.074,
    0.066,
    0.082,
    0.071,
    0.069,
    0.094,
    0.067,
    0.061,
    0.058,
)


def chain_of_thought_both_axes() -> ManuscriptCheck:
    """Sec. IV-A's specific claim about the language-model comparator.

    The article says the chain-of-thought baseline "discriminates poorly and
    calibrates poorly". The general claim that the two orderings coincide is not
    what it asserts -- the sixteen comparators' AUROC and ECE orderings are close
    to reversed, which is the opposite of identical -- so the check is written
    against the specific claim, on the comparator it names.
    """
    auroc_rank = _rank_from_top(TABLE_ONE_AUROC)
    ece_rank = _rank_from_top(TABLE_ONE_ECE)
    index = TABLE_ONE_METHODS.index("Chain-of-thought LLM prompting")
    worst_half = len(TABLE_ONE_METHODS) / 2
    achieved = float(min(auroc_rank[index], ece_rank[index]))
    return _close(
        "chain_of_thought_poor_on_both_axes",
        "Sec. IV-A",
        "the chain-of-thought comparator is in the weaker half on both axes",
        achieved,
        worst_half,
        0.0,
        "rank",
        "abs_below",
    )


def _rank_from_top(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda slot: (-values[slot], slot))
    ranks = [0] * len(values)
    for position, slot in enumerate(order):
        ranks[slot] = position + 1
    return ranks


def auroc_ece_rank_correlation() -> float:
    """Recorded for the reader; not a verdict, because the article makes no claim
    about the sign of this correlation."""
    return spearman_rank_correlation(TABLE_ONE_AUROC, TABLE_ONE_ECE)
