"""Production-runner tests for clean-source C1-B0 stage evidence."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from dagkv.c1_bundle import C1_B0_OPEN_GATES
from dagkv.c1_trace import TraceValidationError
from tools import run_m3_c1_b0_evidence as evidence


def _digest(name: str) -> str:
    return evidence._named_digest(name)


def test_source_closures_bind_runner_protocol_runtime_and_verifiers() -> None:
    assert {
        "src/dagkv/c1_commit.py",
        "src/dagkv/c1_bundle.py",
        "src/dagkv/orchestrator.py",
        "tools/run_m3_c1_b0_evidence.py",
    }.issubset(evidence.IMPLEMENTATION_PATHS)
    assert {
        "tools/run_m3_c1_b0_evidence.py",
        "tests/test_c1_bundle.py",
        "tests/test_m3_c1_b0_evidence.py",
    }.issubset(evidence.VERIFIER_PATHS)
    assert evidence.PROTOCOL_PATH in evidence.SOURCE_PATHS
    assert set(evidence.IMPLEMENTATION_PATHS).issubset(evidence.SOURCE_PATHS)
    assert set(evidence.VERIFIER_PATHS).issubset(evidence.SOURCE_PATHS)
    assert set(evidence.EXECUTED_MODULE_PATHS.values()).issubset(
        evidence.IMPLEMENTATION_PATHS
    )
    for paths in (
        evidence.SOURCE_PATHS,
        evidence.IMPLEMENTATION_PATHS,
        evidence.VERIFIER_PATHS,
        evidence.FOCUSED_TEST_PATHS,
    ):
        assert len(paths) == len(set(paths))


def test_runner_has_no_test_fixture_dependency() -> None:
    source = (evidence.REPO_ROOT / "tools/run_m3_c1_b0_evidence.py").read_text()
    assert "from tests" not in source
    assert "import tests" not in source


def test_binding_digest_is_ordered_and_tamper_evident() -> None:
    first = {"path": "a", "size": 1, "sha256": _digest("a")}
    second = {"path": "b", "size": 2, "sha256": _digest("b")}
    original = evidence._binding_digest((first, second))

    assert original != evidence._binding_digest((second, first))
    assert original != evidence._binding_digest(
        (first, {**second, "sha256": _digest("changed")})
    )


def test_testcase_identity_digest_is_exact_and_ordered() -> None:
    summary = {
        "tests": 2,
        "identities": ["tests.test_a::test_one", "tests.test_b::test_two"],
    }
    original = evidence._test_identity_digest(summary)

    assert original != evidence._test_identity_digest(
        {
            "tests": 2,
            "identities": ["tests.test_a::test_one", "tests.test_b::test_changed"],
        }
    )
    with pytest.raises(evidence.C1B0EvidenceError, match="identities"):
        evidence._test_identity_digest(
            {"tests": 2, "identities": list(reversed(summary["identities"]))}
        )


def test_controlled_scenario_contract_keeps_later_gates_open() -> None:
    assert evidence.SCENARIO_CONTRACT["predicted_deadline_ns"] == 15
    assert evidence.SCENARIO_CONTRACT["abstained_deadline_ns"] == 7
    assert evidence.SCENARIO_CONTRACT["scheduled_access_ns"] == 8
    assert not evidence.SCENARIO_CONTRACT["branch_grammar"]["b1_completeness_claimed"]
    assert not evidence.SCENARIO_CONTRACT["feature_contract"][
        "b1_leakage_acceptance_claimed"
    ]
    assert "C1_B1_SPLIT_LEAKAGE" in C1_B0_OPEN_GATES


def test_production_scenario_finalizes_positive_and_abstained_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    protocol = _digest("protocol")
    verifier = _digest("verifier")
    implementation = _digest("implementation")
    environment = _digest("environment")

    validated = evidence.build_and_finalize_bundle(
        root,
        protocol_digest=protocol,
        verifier_digest=verifier,
        implementation_digest=implementation,
        environment_digest=environment,
    )
    replayed = evidence.validate_inner_bundle(
        root,
        final_seal_sha256=validated.final_seal_sha256,
        protocol_digest=protocol,
        verifier_digest=verifier,
        implementation_digest=implementation,
        environment_digest=environment,
    )

    assert replayed == validated
    labels = {label.observation_id: label for label in replayed.demand_labels}
    assert labels["observation-predicted-resident"].first_demand == 1
    assert labels["observation-abstained-no-demand"].first_demand == 0
    assert replayed.open_gates == C1_B0_OPEN_GATES
    assert replayed.verified_observation_ids == tuple(sorted(labels))


def test_inner_replay_requires_every_external_anchor(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    anchors = {
        "protocol_digest": _digest("protocol"),
        "verifier_digest": _digest("verifier"),
        "implementation_digest": _digest("implementation"),
        "environment_digest": _digest("environment"),
    }
    validated = evidence.build_and_finalize_bundle(root, **anchors)

    for field in tuple(anchors):
        changed = {**anchors, field: _digest(f"wrong-{field}")}
        with pytest.raises(TraceValidationError, match="anchor"):
            evidence.validate_inner_bundle(
                root,
                final_seal_sha256=validated.final_seal_sha256,
                **changed,
            )


def test_validated_bundle_snapshot_is_json_serializable_shape(tmp_path: Path) -> None:
    validated = evidence.build_and_finalize_bundle(
        tmp_path / "bundle",
        protocol_digest=_digest("protocol"),
        verifier_digest=_digest("verifier"),
        implementation_digest=_digest("implementation"),
        environment_digest=_digest("environment"),
    )
    snapshot = asdict(validated)
    json_snapshot = evidence._json_value(validated)

    assert snapshot["final_seal_sha256"] == validated.final_seal_sha256
    assert isinstance(snapshot["demand_labels"], tuple)
    assert isinstance(json_snapshot["demand_labels"], list)


def test_environment_binding_covers_launcher_dependencies_and_child_env() -> None:
    dependency = {"path": "uv.lock", "size": 1, "sha256": _digest("lock")}
    binding = evidence._capture_environment_binding(
        evidence.REPO_ROOT / ".venv/bin/python",
        (dependency,),
    )

    assert binding["command_environment"] == evidence.common.BASE_ENVIRONMENT
    assert binding["dependencies"] == [dependency]
    assert binding["distributions"]["distributions"]
    assert binding["module_origins"] == evidence._current_module_origin_binding()
    assert binding["digest"] == evidence._mapping_digest(
        {key: value for key, value in binding.items() if key != "digest"}
    )


def test_inner_producer_and_fresh_replay_are_strict_json_processes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    anchors = {
        "protocol": _digest("protocol"),
        "verifier": _digest("verifier"),
        "implementation": _digest("implementation"),
        "environment": _digest("environment"),
    }
    common_arguments = [
        "--expected-protocol-digest",
        anchors["protocol"],
        "--expected-verifier-digest",
        anchors["verifier"],
        "--expected-implementation-digest",
        anchors["implementation"],
        "--expected-environment-digest",
        anchors["environment"],
    ]
    python = evidence.REPO_ROOT / ".venv/bin/python"
    producer = subprocess.run(
        [
            str(python),
            "-m",
            "tools.run_m3_c1_b0_evidence",
            "produce-inner",
            str(bundle),
            *common_arguments,
        ],
        cwd=evidence.REPO_ROOT,
        env=dict(evidence.common.BASE_ENVIRONMENT),
        capture_output=True,
        check=False,
    )
    assert producer.returncode == 0, producer.stderr.decode(errors="replace")
    produced = json.loads(producer.stdout)
    assert produced["schema_version"] == evidence.PRODUCER_RESULT_SCHEMA
    assert produced["status"] == "passed"
    assert (
        produced["execution_origin_digest"]
        == evidence._current_module_origin_binding()["digest"]
    )

    replay = subprocess.run(
        [
            str(python),
            "-m",
            "tools.run_m3_c1_b0_evidence",
            "validate-inner",
            str(bundle),
            "--expected-final-seal-sha256",
            produced["final_seal_sha256"],
            *common_arguments,
        ],
        cwd=evidence.REPO_ROOT,
        env=dict(evidence.common.BASE_ENVIRONMENT),
        capture_output=True,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr.decode(errors="replace")
    replayed = json.loads(replay.stdout)
    assert replayed["schema_version"] == evidence.REPLAY_RESULT_SCHEMA
    assert replayed["validated"] == produced["validated"]


def test_cleanup_removes_read_only_inner_bundle(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    inner = root / "bundle"
    inner.mkdir(parents=True)
    payload = inner / "sealed.json"
    payload.write_text("sealed\n")
    payload.chmod(0o440)
    inner.chmod(0o550)
    root.chmod(0o550)

    evidence._remove_tree(root)

    assert not root.exists()


def test_stable_sha_rejects_hard_linked_final_seal(tmp_path: Path) -> None:
    seal = tmp_path / "seal.json"
    linked = tmp_path / "linked.json"
    seal.write_text("sealed\n")
    os.link(seal, linked)

    with pytest.raises(evidence.C1B0EvidenceError, match="stable read"):
        evidence._stable_sha256(seal, label="test seal")


def test_output_location_must_stay_outside_repository(tmp_path: Path) -> None:
    inside = evidence.REPO_ROOT / "evidence" / "forbidden-c1-b0-runtime"
    with pytest.raises(evidence.C1B0EvidenceError, match="outside"):
        evidence.run_evidence(
            inside,
            python=Path(".venv/bin/python"),
            timeout_seconds=1,
        )

    assert tmp_path.resolve().is_absolute()
