from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from tools import m2_aggregate_acceptance as aggregate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_git(repo: Path) -> str:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "m2-test@example.invalid")
    _git(repo, "config", "user.name", "M2 Test")
    return repo.as_posix()


def _junit_fixture(root: Path) -> dict[str, Any]:
    by_suite: dict[str, list[tuple[str, str]]] = {}
    for identities in aggregate.CONDITION_TEST_IDS.values():
        for identity in identities:
            suite, test = identity.split(":", maxsplit=1)
            classname, name = test.split("::", maxsplit=1)
            by_suite.setdefault(suite, []).append((classname, name))
    suites: list[dict[str, Any]] = []
    for suite_id, cases in sorted(by_suite.items()):
        suite = ET.Element("testsuite", name=suite_id, tests=str(len(cases)))
        for classname, name in sorted(set(cases)):
            ET.SubElement(suite, "testcase", classname=classname, name=name)
        relative = Path("logs") / f"{suite_id}.xml"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
        suites.append({"suite_id": suite_id, "junit": {"path": relative.as_posix()}})
    return {"suites": suites}


def _capture_file(path: Path, *, kind: str | None = None) -> dict[str, Any]:
    observed = path.stat()
    entry: dict[str, Any] = {
        "path": path.name,
        "size": observed.st_size,
        "sha256": _sha256(path),
        "mtime_ns": observed.st_mtime_ns,
        "inode": observed.st_ino,
    }
    if kind is not None:
        entry["kind"] = kind
    return entry


def _external_fixture(root: Path) -> Path:
    model_root = root / "model"
    model_root.mkdir(parents=True)
    model_rows: list[dict[str, Any]] = []
    for index in range(16):
        suffix = ".safetensors" if index < 5 else ".json"
        path = model_root / f"model-{index:02d}{suffix}"
        path.write_bytes(f"model-{index}".encode())
        model_rows.append(
            _capture_file(path, kind="weight" if index < 5 else "metadata")
        )
    model_content = [
        {key: row[key] for key in ("path", "size", "kind", "sha256")}
        for row in model_rows
    ]

    runtime_root = root / "runtime"
    extension_root = runtime_root / "vllm"
    extension_root.mkdir(parents=True)
    extension_rows: list[dict[str, Any]] = []
    for index in range(6):
        path = extension_root / f"extension-{index}.so"
        path.write_bytes(f"extension-{index}".encode())
        row = _capture_file(path)
        row["path"] = path.relative_to(runtime_root).as_posix()
        extension_rows.append(row)
    python = root / "python"
    python.write_bytes(b"python-executable")
    python_row = _capture_file(python)
    python_row["path"] = str(python)
    runtime_content = [
        {key: row[key] for key in ("path", "size", "sha256")} for row in extension_rows
    ]
    python_content = {key: python_row[key] for key in ("path", "size", "sha256")}
    provenance = {
        "model": {
            "root": str(model_root),
            "full_hashes": True,
            "files": model_rows,
            "manifest_sha256": aggregate._canonical_digest(model_content),
        },
        "runtime_binaries": {
            "root": str(runtime_root),
            "full_hashes": True,
            "vllm_extensions": extension_rows,
            "python_executable": python_row,
            "manifest_sha256": aggregate._canonical_digest(
                {
                    "vllm_extensions": runtime_content,
                    "python_executable": python_content,
                }
            ),
        },
    }
    provenance_path = root / "provenance.json"
    _write_json(provenance_path, provenance)
    return provenance_path


def _excluded_payloads(repo: Path) -> None:
    attempts = [
        {"name": f"pilot-{index}", "gate_status": "FAILED"} for index in range(8)
    ]
    attempts.extend(
        [
            {
                "name": "run09",
                "gate_status": "FAILED",
                "cohort_eligible": False,
            },
            {
                "name": "run10",
                "gate_status": "CALIBRATED_NOT_ACCEPTED",
                "cohort_eligible": False,
            },
        ]
    )
    _write_json(
        repo / "evidence/m2/PILOT_ATTEMPTS.json",
        {
            "cohort_eligible": False,
            "acceptance_claimed": False,
            "attempts": attempts,
        },
    )
    _write_json(
        repo / "evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json",
        {
            "gate_status": "FAILED",
            "cohort_eligible": False,
            "acceptance_claimed": False,
            "execution": {"run_id": "excluded-run09"},
        },
    )
    _write_json(
        repo / "evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json",
        {
            "gate_status": "CALIBRATED_NOT_ACCEPTED",
            "cohort_eligible": False,
            "acceptance_claimed": False,
            "execution": {"run_id": "excluded-run10"},
        },
    )
    campaign_id = "campaign01"
    campaign_root = repo.parent / "campaign01"
    campaign_root.mkdir()
    records: list[dict[str, Any]] = []
    for sequence in range(1, 21):
        records.extend(
            [
                {
                    "campaign_id": campaign_id,
                    "event": "submitted",
                    "kind": "formal_run",
                    "sequence": sequence,
                },
                {
                    "campaign_id": campaign_id,
                    "event": "terminal",
                    "kind": "formal_run",
                    "sequence": sequence,
                    "status": "passed",
                    "validation": {"run_id": f"excluded-formal-{sequence:03d}"},
                },
            ]
        )
    records.extend(
        [
            {"campaign_id": campaign_id, "event": "submitted", "kind": "aggregate"},
            {
                "campaign_id": campaign_id,
                "event": "terminal",
                "kind": "aggregate",
                "status": "validation_failed",
            },
        ]
    )
    journal = campaign_root / "FORMAL_ATTEMPTS.jsonl"
    journal.write_text(
        "".join(
            json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for row in records
        ),
        encoding="utf-8",
    )
    _write_json(
        repo / "evidence/m2/v3_580_173_02/"
        "M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json",
        {
            "campaign_id": campaign_id,
            "campaign_root": str(campaign_root),
            "formal_cohort_eligible": False,
            "acceptance_claimed": False,
            "failure": {"stage": "post_aggregate_candidate_replay"},
            "attempt_journal": {
                "file": journal.name,
                "sha256": _sha256(journal),
                "record_count": len(records),
            },
        },
    )


def _eligible_run_ids() -> list[str]:
    return [f"eligible-{index:03d}" for index in range(79)]


def _cohort_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    calibration = {
        "run_count": 59,
        "all_passed": True,
        "runs": [{"run_id": f"calibration-{index:03d}"} for index in range(59)],
    }
    seal = {
        "run_count": 20,
        "ordered_runs": [{"run_id": f"formal-{index:03d}"} for index in range(20)],
    }
    preregistration = {"retry_policy": "none_stop_on_first_failure"}
    return calibration, seal, preregistration


def test_strict_json_rejects_duplicate_key_and_nonfinite_value() -> None:
    with pytest.raises(aggregate.AggregateAcceptanceError, match="duplicate JSON key"):
        aggregate._decode_json(b'{"key":1,"key":2}', label="duplicate")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="non-finite"):
        aggregate._decode_json(b'{"key":NaN}', label="nonfinite")


def test_required_junit_cases_are_bound_to_each_condition(tmp_path: Path) -> None:
    manifest = _junit_fixture(tmp_path)
    observed = aggregate._parse_junit_cases(tmp_path, manifest)
    required = {
        identity
        for identities in aggregate.CONDITION_TEST_IDS.values()
        for identity in identities
    }
    assert required <= observed
    assert set(aggregate.CONDITION_TEST_IDS) == set(range(1, 8))


def test_required_junit_missing_or_failed_case_fails_closed(tmp_path: Path) -> None:
    manifest = _junit_fixture(tmp_path)
    first = tmp_path / manifest["suites"][0]["junit"]["path"]
    tree = ET.parse(first)
    tree.getroot().remove(next(tree.getroot().iter("testcase")))
    tree.write(first, encoding="utf-8", xml_declaration=True)
    with pytest.raises(aggregate.AggregateAcceptanceError, match="cases are missing"):
        aggregate._parse_junit_cases(tmp_path, manifest)

    manifest = _junit_fixture(tmp_path)
    first = tmp_path / manifest["suites"][0]["junit"]["path"]
    tree = ET.parse(first)
    ET.SubElement(next(tree.getroot().iter("testcase")), "failure")
    tree.write(first, encoding="utf-8", xml_declaration=True)
    with pytest.raises(aggregate.AggregateAcceptanceError, match="non-pass case"):
        aggregate._parse_junit_cases(tmp_path, manifest)


def test_external_content_rehashes_exact_closed_sets(tmp_path: Path) -> None:
    provenance = _external_fixture(tmp_path)
    result = aggregate._rehash_external_content(provenance)
    assert result["total_file_count"] == 23
    assert result["model"]["file_count"] == 16
    assert result["runtime_binaries"]["extension_count"] == 6
    assert result["current_content_rehash_passed"] is True


def test_external_content_tamper_and_added_file_fail_closed(tmp_path: Path) -> None:
    provenance = _external_fixture(tmp_path)
    target = tmp_path / "runtime/vllm/extension-0.so"
    target.write_bytes(b"changed-extension")
    with pytest.raises(
        aggregate.AggregateAcceptanceError, match="stat identity differs"
    ):
        aggregate._rehash_external_content(provenance)

    provenance = _external_fixture(tmp_path / "fresh")
    (tmp_path / "fresh/model/extra.json").write_bytes(b"extra")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="closed set differs"):
        aggregate._rehash_external_content(provenance)


def test_excluded_attempt_authorities_are_read_from_git(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    _excluded_payloads(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "excluded attempts")
    head = _git(repo, "rev-parse", "HEAD")
    exclusions = aggregate._validate_excluded_attempts(
        repo,
        head,
        eligible_campaign_id="campaign02",
        eligible_run_ids=_eligible_run_ids(),
    )
    assert [item["category"] for item in exclusions["authorities"]] == [
        "pilot_index",
        "run09",
        "run10",
        "formal_campaign01",
    ]
    assert all(item["cohort_eligible"] is False for item in exclusions["authorities"])
    assert exclusions["excluded_run_id_count"] == 22
    assert exclusions["eligible_run_id_count"] == 79
    assert exclusions["eligible_run_id_intersection_count"] == 0
    assert exclusions["campaign01_journal"]["record_count"] == 42


def test_excluded_attempt_cannot_be_marked_eligible(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    _excluded_payloads(repo)
    run10 = repo / "evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json"
    payload = json.loads(run10.read_text(encoding="utf-8"))
    payload["cohort_eligible"] = True
    _write_json(run10, payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "invalid exclusion")
    head = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="run10 exclusion"):
        aggregate._validate_excluded_attempts(
            repo,
            head,
            eligible_campaign_id="campaign02",
            eligible_run_ids=_eligible_run_ids(),
        )


def test_excluded_run_id_cannot_enter_the_eligible_cohort(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    _excluded_payloads(repo)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "excluded attempts")
    head = _git(repo, "rev-parse", "HEAD")
    eligible = _eligible_run_ids()
    eligible[0] = "excluded-run09"
    with pytest.raises(aggregate.AggregateAcceptanceError, match="run IDs overlap"):
        aggregate._validate_excluded_attempts(
            repo,
            head,
            eligible_campaign_id="campaign02",
            eligible_run_ids=eligible,
        )


def test_acceptance_closed_set_requires_exact_name_modes_and_contents(
    tmp_path: Path,
) -> None:
    root = tmp_path / "accepted"
    root.mkdir()
    acceptance = root / aggregate.ACCEPTANCE_NAME
    acceptance.write_text("{}\n", encoding="utf-8")
    acceptance.chmod(0o444)
    root.chmod(0o555)
    aggregate._validate_closed_set(acceptance)

    root.chmod(0o755)
    (root / "extra").write_text("unexpected", encoding="utf-8")
    root.chmod(0o555)
    with pytest.raises(aggregate.AggregateAcceptanceError, match="one-file closed set"):
        aggregate._validate_closed_set(acceptance)


def test_exclusive_publication_refuses_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / aggregate.ACCEPTANCE_NAME
    aggregate._publish_exclusive(destination, {"first": True})
    with pytest.raises(FileExistsError):
        aggregate._publish_exclusive(destination, {"second": True})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"first": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_condition_decision_is_exactly_nine_and_claim_limited() -> None:
    conditions = aggregate._condition_payloads()
    assert [row["condition"] for row in conditions] == list(range(1, 10))
    assert all(row["status"] == "VERIFIED" for row in conditions)
    assert "No latency" in aggregate.CLAIM_SCOPE
    assert "C1" in aggregate.CLAIM_SCOPE


def test_cohort_identity_requires_exact_counts_and_global_uniqueness() -> None:
    calibration, seal, preregistration = _cohort_fixture()
    identity = aggregate._cohort_identity(
        calibration, seal, preregistration, preregistration
    )
    assert identity["calibration_run_count"] == 59
    assert identity["formal_run_count"] == 20
    assert identity["global_unique_run_id_count"] == 79

    seal["ordered_runs"][0]["run_id"] = calibration["runs"][0]["run_id"]
    with pytest.raises(aggregate.AggregateAcceptanceError, match="globally unique"):
        aggregate._cohort_identity(calibration, seal, preregistration, preregistration)


def test_cohort_identity_rejects_count_and_retry_drift() -> None:
    calibration, seal, preregistration = _cohort_fixture()
    calibration["runs"].pop()
    with pytest.raises(aggregate.AggregateAcceptanceError, match="59/59"):
        aggregate._cohort_identity(calibration, seal, preregistration, preregistration)

    calibration, seal, preregistration = _cohort_fixture()
    retry = {"retry_policy": "retry_once"}
    with pytest.raises(aggregate.AggregateAcceptanceError, match="retry policy"):
        aggregate._cohort_identity(calibration, seal, retry, preregistration)


def test_repository_publication_requires_clean_committed_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    authority = repo / "authority.txt"
    authority.write_text("frozen\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "authority")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(aggregate, "AUTHORITY_PATHS", ("authority.txt",))
    monkeypatch.setattr(aggregate, "VALIDATOR_CLOSURE_PATHS", ())
    monkeypatch.setattr(aggregate, "PROTOCOL_PATH", "authority.txt")
    monkeypatch.setattr(
        aggregate,
        "_canonical_evidence_binding",
        lambda _repo, _head: {"fixture": True},
    )
    binding = aggregate._repository_binding(repo, head)
    assert binding["head"] == head

    authority.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="must be clean"):
        aggregate._repository_binding(repo, head)


def test_descendant_validator_drift_cannot_replay_an_old_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    leaf = repo / "tools/freeze_m2_tolerance.py"
    leaf.parent.mkdir(parents=True)
    leaf.write_text("VERSION = 1\n", encoding="utf-8")
    protocol = repo / "protocol.md"
    protocol.write_text("protocol v1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "validator v1")
    first = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(aggregate, "REPO_ROOT", repo.resolve())
    monkeypatch.setattr(
        aggregate, "VALIDATOR_CLOSURE_PATHS", ("tools/freeze_m2_tolerance.py",)
    )
    monkeypatch.setattr(aggregate, "PROTOCOL_PATH", "protocol.md")
    monkeypatch.setattr(
        aggregate,
        "AUTHORITY_PATHS",
        ("tools/freeze_m2_tolerance.py", "protocol.md"),
    )
    monkeypatch.setattr(
        aggregate,
        "_canonical_evidence_binding",
        lambda _repo, _head: {"fixture": True},
    )
    recorded = aggregate._repository_binding(repo, first)

    leaf.write_text("VERSION = 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "validator v2")
    second = _git(repo, "rev-parse", "HEAD")
    assert aggregate._is_ancestor(repo, first, second)
    with pytest.raises(
        aggregate.AggregateAcceptanceError,
        match="current replay authority bytes differ",
    ):
        aggregate._validate_repository_record(recorded)


def test_publish_requires_a_fresh_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(aggregate.AggregateAcceptanceError, match="already exists"):
        aggregate.publish_aggregate_acceptance(
            output,
            component_evidence_dir=tmp_path / "component",
            expected_component_manifest_sha256="0" * 64,
            expected_component_sha256sums_sha256="0" * 64,
            formal_seal=tmp_path / "seal",
            expected_formal_seal_sha256="0" * 64,
            expected_formal_preregistration_sha256="0" * 64,
            expected_repository_head="0" * 40,
            expected_stage_gates_sha256="0" * 64,
            expected_protocol_sha256="0" * 64,
        )


def test_preparing_publication_lock_is_never_accepted(tmp_path: Path) -> None:
    output = tmp_path / "decision"
    handle = aggregate._create_publication_lock(output)
    handle.close()
    assert (
        aggregate._publication_lock_path(output).read_bytes()
        == aggregate.LOCK_PREPARING
    )
    with pytest.raises(
        aggregate.AggregateAcceptanceError, match="publication is incomplete"
    ):
        aggregate._acquire_published_lock(output)


def test_public_validator_requires_the_publication_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "decision"
    output.mkdir()
    acceptance = output / aggregate.ACCEPTANCE_NAME
    acceptance.write_text("{}\n", encoding="utf-8")
    acceptance.chmod(0o444)
    output.chmod(0o555)
    with pytest.raises(aggregate.AggregateAcceptanceError, match="lock is missing"):
        aggregate.validate_aggregate_acceptance(
            acceptance,
            expected_acceptance_sha256=hashlib.sha256(b"{}\n").hexdigest(),
        )


def test_public_validator_rejects_sidecar_replacement_during_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "decision"
    output.mkdir()
    acceptance = output / aggregate.ACCEPTANCE_NAME
    acceptance.write_text("{}\n", encoding="utf-8")
    acceptance.chmod(0o444)
    output.chmod(0o555)
    publisher = aggregate._create_publication_lock(output)
    aggregate._mark_publication_complete(publisher)
    publisher.close()
    lock_path = aggregate._publication_lock_path(output)

    def replace_sidecar(
        _path: Path, *, expected_acceptance_sha256: str
    ) -> tuple[dict[str, Any], str]:
        lock_path.unlink()
        lock_path.write_bytes(aggregate.LOCK_PREPARING)
        lock_path.chmod(0o444)
        return {}, expected_acceptance_sha256

    monkeypatch.setattr(
        aggregate, "_validate_aggregate_acceptance_unlocked", replace_sidecar
    )
    with pytest.raises(
        aggregate.AggregateAcceptanceError,
        match="publication lock changed during aggregate replay",
    ):
        aggregate.validate_aggregate_acceptance(
            acceptance,
            expected_acceptance_sha256=hashlib.sha256(b"{}\n").hexdigest(),
        )
    assert lock_path.read_bytes() == aggregate.LOCK_PREPARING


def test_successful_publish_supports_locked_and_fresh_process_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    stage_gates = repo / aggregate.STAGE_GATES_PATH
    protocol = repo / aggregate.PROTOCOL_PATH
    stage_gates.parent.mkdir(parents=True)
    protocol.parent.mkdir(parents=True)
    stage_gates.write_text("stage gates\n", encoding="utf-8")
    protocol.write_text("aggregate protocol\n", encoding="utf-8")
    stage_sha = _sha256(stage_gates)
    protocol_sha = _sha256(protocol)
    repository = {
        "root": str(repo),
        "head": "a" * 40,
        "clean_at_publication": True,
        "authorities": [
            {
                "repository_path": aggregate.STAGE_GATES_PATH,
                "sha256": stage_sha,
            },
            {
                "repository_path": aggregate.PROTOCOL_PATH,
                "sha256": protocol_sha,
            },
        ],
        "canonical_evidence": {"fixture": True},
    }
    replayed = {
        "repository": repository,
        "component_evidence": {"fixture": True},
        "formal_evidence": {"fixture": True},
        "compatibility_bridge": {"fixture": True},
        "external_content": {"fixture": True},
        "excluded_attempts": {"fixture": True},
        "conditions": aggregate._condition_payloads(),
    }
    monkeypatch.setattr(aggregate, "REPO_ROOT", repo)
    monkeypatch.setattr(aggregate, "_replay_payload_inputs", lambda **_kwargs: replayed)
    monkeypatch.setattr(
        aggregate, "_repository_binding", lambda _repo, _head: repository
    )
    replay_calls: list[Path] = []

    def replay_sealed(
        path: Path, *, expected_acceptance_sha256: str
    ) -> tuple[dict[str, Any], str]:
        aggregate._validate_closed_set(path)
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        assert observed == expected_acceptance_sha256
        replay_calls.append(path)
        return json.loads(raw), observed

    monkeypatch.setattr(
        aggregate, "_validate_aggregate_acceptance_unlocked", replay_sealed
    )
    output = tmp_path / "accepted"
    payload, acceptance_sha = aggregate.publish_aggregate_acceptance(
        output,
        component_evidence_dir=tmp_path / "component",
        expected_component_manifest_sha256="1" * 64,
        expected_component_sha256sums_sha256="2" * 64,
        formal_seal=tmp_path / "formal/M2_FORMAL_BUNDLE_SEAL.json",
        expected_formal_seal_sha256="3" * 64,
        expected_formal_preregistration_sha256="4" * 64,
        expected_repository_head="a" * 40,
        expected_stage_gates_sha256=stage_sha,
        expected_protocol_sha256=protocol_sha,
    )
    acceptance = output / aggregate.ACCEPTANCE_NAME
    assert len(replay_calls) == 1
    assert replay_calls[0].parent.name.endswith(".staging")
    assert payload["m2_accepted"] is True
    assert (
        aggregate._publication_lock_path(output).read_bytes()
        == aggregate.LOCK_PUBLISHED
    )

    validated, observed_sha = aggregate.validate_aggregate_acceptance(
        acceptance,
        expected_acceptance_sha256=acceptance_sha,
    )
    assert validated == payload
    assert observed_sha == acceptance_sha
    assert len(replay_calls) == 2

    script = """
import hashlib
import json
import sys
from pathlib import Path
from tools import m2_aggregate_acceptance as aggregate

def replay(path, *, expected_acceptance_sha256):
    aggregate._validate_closed_set(path)
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_acceptance_sha256:
        raise RuntimeError("acceptance SHA differs")
    return json.loads(raw), observed

aggregate._validate_aggregate_acceptance_unlocked = replay
_, observed = aggregate.validate_aggregate_acceptance(
    Path(sys.argv[1]), expected_acceptance_sha256=sys.argv[2]
)
print(observed)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(acceptance), acceptance_sha],
        cwd=Path(aggregate.__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == acceptance_sha


def test_hard_linked_external_file_fails_closed(tmp_path: Path) -> None:
    provenance = _external_fixture(tmp_path)
    python = tmp_path / "python"
    os.link(python, tmp_path / "python-link")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="one hard link"):
        aggregate._rehash_external_content(provenance)


def test_external_symlink_root_and_special_node_fail_closed(tmp_path: Path) -> None:
    provenance = _external_fixture(tmp_path)
    model = tmp_path / "model"
    real_model = tmp_path / "real-model"
    model.rename(real_model)
    model.symlink_to(real_model, target_is_directory=True)
    with pytest.raises(aggregate.AggregateAcceptanceError, match="root is invalid"):
        aggregate._rehash_external_content(provenance)

    fresh = tmp_path / "fresh"
    provenance = _external_fixture(fresh)
    os.mkfifo(fresh / "model/unexpected.fifo")
    with pytest.raises(aggregate.AggregateAcceptanceError, match="special node"):
        aggregate._rehash_external_content(provenance)


def test_staged_acceptance_commits_as_one_file_closed_set(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    source = staging / aggregate.ACCEPTANCE_NAME
    source.write_text("{}\n", encoding="utf-8")
    source.chmod(0o444)
    staging.chmod(0o555)
    output = tmp_path / "accepted"
    destination = aggregate._commit_staged_acceptance(staging, output)
    assert destination == output / aggregate.ACCEPTANCE_NAME
    assert not staging.exists()
    aggregate._validate_closed_set(destination, expected_root_mode=0o755)
    output.chmod(0o555)
    aggregate._validate_closed_set(destination)


def test_post_visibility_failure_quarantines_success_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir()
    source = staging / aggregate.ACCEPTANCE_NAME
    source.write_text('{"m2_accepted":true}\n', encoding="utf-8")
    source.chmod(0o444)
    staging.chmod(0o555)
    output = tmp_path / "failed"
    real_fsync = aggregate._fsync_directory
    calls = 0

    def injected_fsync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected post-visibility failure")
        real_fsync(path)

    monkeypatch.setattr(aggregate, "_fsync_directory", injected_fsync)
    with pytest.raises(OSError, match="injected post-visibility failure"):
        aggregate._commit_staged_acceptance(staging, output)
    assert not (output / aggregate.ACCEPTANCE_NAME).exists()
    assert (output / aggregate.INVALID_ACCEPTANCE_NAME).is_file()
    assert not staging.exists()


def test_canonical_index_rejects_cli_bundle_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = tmp_path / "component"
    formal = tmp_path / "formal"
    canonical = {
        "component": {
            "root": str(component),
            "manifest_sha256": "1" * 64,
            "sha256sums_sha256": "2" * 64,
        },
        "formal": {
            "root": str(formal),
            "seal_sha256": "3" * 64,
            "preregistration_sha256": "4" * 64,
        },
    }
    repository = {
        "root": str(tmp_path),
        "head": "a" * 40,
        "clean_at_publication": True,
        "authorities": [],
        "canonical_evidence": canonical,
    }
    monkeypatch.setattr(
        aggregate, "_repository_binding", lambda _repo, _head: repository
    )
    with pytest.raises(aggregate.AggregateAcceptanceError, match="component CLI"):
        aggregate._replay_payload_inputs(
            repo=tmp_path,
            repository_head="a" * 40,
            component_root=tmp_path / "alternate-component",
            expected_component_manifest_sha256="1" * 64,
            expected_component_sha256sums_sha256="2" * 64,
            formal_seal=formal / aggregate.ACCEPTANCE_NAME,
            expected_formal_seal_sha256="3" * 64,
            expected_formal_preregistration_sha256="4" * 64,
            require_clean_head=True,
        )


def test_repository_mutation_during_replay_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component_root = tmp_path / "component"
    formal_root = tmp_path / "formal"
    canonical = {
        "component": {
            "root": str(component_root),
            "manifest_sha256": "1" * 64,
            "sha256sums_sha256": "2" * 64,
        },
        "formal": {
            "root": str(formal_root),
            "seal_sha256": "3" * 64,
            "preregistration_sha256": "4" * 64,
        },
    }
    initial = {
        "root": str(tmp_path),
        "head": "a" * 40,
        "clean_at_publication": True,
        "authorities": [],
        "canonical_evidence": canonical,
    }
    changed = {**initial, "head": "b" * 40}
    calls = iter((initial, changed))
    monkeypatch.setattr(
        aggregate, "_repository_binding", lambda _repo, _head: next(calls)
    )
    monkeypatch.setattr(
        aggregate,
        "_component_evidence",
        lambda *_args, **_kwargs: ({}, {"root": str(component_root)}),
    )
    monkeypatch.setattr(
        aggregate,
        "_formal_evidence",
        lambda *_args, **_kwargs: (
            {
                "campaign_id": "campaign02",
                "ordered_runs": [
                    {"run_id": f"formal-{index:03d}"} for index in range(20)
                ],
            },
            None,
            {},
            {"runs": [{"run_id": f"calibration-{index:03d}"} for index in range(59)]},
            tmp_path / "provenance.json",
            {"root": str(formal_root)},
        ),
    )
    monkeypatch.setattr(aggregate, "_compatibility_bridge", lambda *_args: {})
    monkeypatch.setattr(aggregate, "_rehash_external_content", lambda *_args: {})
    monkeypatch.setattr(
        aggregate, "_validate_excluded_attempts", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        aggregate, "_reconcile_canonical_evidence", lambda *_args, **_kwargs: None
    )
    with pytest.raises(aggregate.AggregateAcceptanceError, match="changed during"):
        aggregate._replay_payload_inputs(
            repo=tmp_path,
            repository_head="a" * 40,
            component_root=component_root,
            expected_component_manifest_sha256="1" * 64,
            expected_component_sha256sums_sha256="2" * 64,
            formal_seal=formal_root / aggregate.ACCEPTANCE_NAME,
            expected_formal_seal_sha256="3" * 64,
            expected_formal_preregistration_sha256="4" * 64,
            require_clean_head=True,
        )


def test_protected_historical_git_tree_detects_blob_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git(repo)
    for index in range(6):
        path = repo / "src/dagkv" / f"runtime_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"RUNTIME = {index}\n", encoding="utf-8")
    for index in range(6):
        path = repo / "integrations/vllm_m2/dagkv_vllm_m2" / f"adapter_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"ADAPTER = {index}\n", encoding="utf-8")
    explicit = [
        "pyproject.toml",
        "uv.lock",
        "tests/test_domain.py",
        "tests/test_engine_adapter.py",
        "tests/test_ledger.py",
        "tests/test_orchestrator_failures.py",
        "tests/test_orchestrator_lifecycle.py",
        "integrations/vllm_m2/tests/test_contract.py",
    ]
    for relative in explicit:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "protected tree")
    first = _git(repo, "rev-parse", "HEAD")
    first_tree = aggregate._git_tree(repo, first)
    assert len(first_tree) == 20

    changed = repo / "src/dagkv/runtime_0.py"
    changed.write_text("RUNTIME = 'changed'\n", encoding="utf-8")
    _git(repo, "add", str(changed.relative_to(repo)))
    _git(repo, "commit", "-qm", "runtime drift")
    second = _git(repo, "rev-parse", "HEAD")
    assert aggregate._is_ancestor(repo, first, second)
    assert aggregate._git_tree(repo, second) != first_tree
