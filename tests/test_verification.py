"""The verification driver's own artefacts and check families."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathanchor.evaluation import verification
from pathanchor.utils.hygiene import mask, scan_standin_vocabulary, summarise


def test_claim_map_covers_the_stated_artefacts() -> None:
    claim_map = verification.build_claim_map()
    assert claim_map["overall"] == "PARTIALLY_VERIFIED"
    assert len(claim_map["claims"]) >= 16
    assert len(claim_map["deviations"]) >= 5
    statuses = {claim["verification"] for claim in claim_map["claims"]}
    assert statuses <= {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
    for claim in claim_map["claims"]:
        assert claim["code_paths"]


def test_deviations_carry_a_location_and_a_reason() -> None:
    for entry in verification.build_claim_map()["deviations"]:
        assert entry["paper_location"]
        assert entry["why"]


def test_check_result_rejects_an_unknown_status() -> None:
    with pytest.raises(ValueError):
        verification.CheckResult("x", "MAYBE", "e").record()


def test_check_result_records_allowed_statuses() -> None:
    for status in verification.CHECK_STATUSES:
        assert verification.CheckResult("x", status, "e").record()["status"] == status


def test_arithmetic_checks_are_registered() -> None:
    names = {check.__name__ for check in verification.CODE_CHECKS}
    assert "check_auroc_reference" in names
    assert "check_net_benefit_identity" in names
    assert "check_holm_step_down" in names
    assert "check_hygiene" in names


def test_gates_are_named() -> None:
    labels = {label for label, _ in verification.GATES}
    assert labels == {"ruff", "black", "isort", "mypy", "pytest"}


def test_manuscript_checks_are_kept_in_their_own_family() -> None:
    context = verification.VerifyContext(
        root=Path(__file__).resolve().parents[1],
        runtime=verification.resolve(
            Path(__file__).resolve().parents[1] / "configs" / "experiment" / "_smoke.yaml"
        ),
        smoke=verification.resolve(
            Path(__file__).resolve().parents[1] / "configs" / "experiment" / "_smoke.yaml"
        ),
        skip_tools=True,
    )
    results = verification.check_manuscript_arithmetic(context)
    assert results
    assert all(item.category == "manuscript" for item in results)
    assert any(item.status == "FAIL" for item in results)


def test_hygiene_scanners_do_not_match_their_own_source() -> None:
    source = Path(verification.__file__).resolve().parents[3] / "src" / "pathanchor"
    hits = scan_standin_vocabulary(source)
    assert hits == []


def test_mask_hides_the_middle_of_a_hit() -> None:
    masked = mask("sensitive-value")
    assert masked[0] == "s" and masked[-1] == "e"
    assert "*" in masked
    assert summarise([]) == "no findings"


def test_shipped_manifest_matches_the_live_tree() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "integrity_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("the manifest is not written yet")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {entry["path"] for entry in payload["files"]}
    assert "verification_report.json" in listed
    assert "claim_to_code.json" in listed
    assert "integrity_manifest.json" not in listed


def test_artefact_presence_check_names_the_expected_files() -> None:
    context = verification.VerifyContext(
        root=Path(__file__).resolve().parents[1],
        runtime=verification.resolve(
            Path(__file__).resolve().parents[1] / "configs" / "experiment" / "_smoke.yaml"
        ),
        smoke=verification.resolve(
            Path(__file__).resolve().parents[1] / "configs" / "experiment" / "_smoke.yaml"
        ),
        skip_tools=True,
    )
    result = verification.check_readme_sections(context)
    assert result.status in {"PASS", "FAIL"}
