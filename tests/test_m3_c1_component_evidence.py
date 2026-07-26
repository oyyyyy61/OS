"""Fail-closed tests for the M3/C1 component evidence runner."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools import run_m3_c1_component_evidence as evidence


def test_focused_identity_freeze_matches_the_c1_test_module() -> None:
    source = (evidence.REPO_ROOT / evidence.PROBABILITY_TEST_PATH).read_text()
    tree = ast.parse(source)
    observed = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    assert observed == evidence.EXPECTED_C1_TESTS


def test_source_closure_contains_runtime_protocol_tests_and_validator() -> None:
    assert {
        "src/dagkv/c1_leases.py",
        "src/dagkv/domain.py",
        "src/dagkv/ledger.py",
        "src/dagkv/orchestrator.py",
        "tests/test_c1_shared_leases.py",
        "tests/test_m3_c1_component_evidence.py",
        "tools/run_m3_c1_component_evidence.py",
        "research/protocols/M3_C1_SHARED_LEASE_PROTOCOL.md",
        "research/REFERENCES.md",
        "research/imported/RELATED_WORK_MATRIX.md",
    }.issubset(evidence.SOURCE_PATHS)
    assert len(evidence.SOURCE_PATHS) == len(set(evidence.SOURCE_PATHS))


@pytest.mark.parametrize(
    "payload, message",
    [
        (b'{"a":1,"a":2}\n', "duplicate JSON key"),
        (b'{"a":NaN}\n', "non-finite JSON value"),
        (b"[]\n", "must be a JSON object"),
    ],
)
def test_strict_json_rejects_ambiguous_values(
    tmp_path: Path,
    payload: bytes,
    message: str,
) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(payload)
    with pytest.raises(evidence.C1EvidenceError, match=message):
        evidence._read_json(path, label="test JSON")


def test_junit_parser_requires_unique_complete_identities(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.xml"
    path.write_text(
        "<testsuite>"
        '<testcase classname="tests.test_c1" name="test_one" />'
        '<testcase classname="tests.test_c1" name="test_one" />'
        "</testsuite>"
    )
    with pytest.raises(evidence.C1EvidenceError, match="duplicate JUnit"):
        evidence._parse_junit(path)

    path.write_text('<testsuite><testcase name="test_one" /></testsuite>')
    with pytest.raises(evidence.C1EvidenceError, match="identity is incomplete"):
        evidence._parse_junit(path)


def test_junit_parser_reports_every_terminal(tmp_path: Path) -> None:
    path = tmp_path / "terminals.xml"
    path.write_text(
        "<testsuite>"
        '<testcase classname="suite" name="pass" />'
        '<testcase classname="suite" name="fail"><failure /></testcase>'
        '<testcase classname="suite" name="error"><error /></testcase>'
        '<testcase classname="suite" name="skip"><skipped /></testcase>'
        "</testsuite>"
    )
    summary = evidence._parse_junit(path)
    assert summary["tests"] == 4
    assert summary["failures"] == 1
    assert summary["errors"] == 1
    assert summary["skipped"] == 1


def test_checksum_inventory_rejects_mutation_and_unindexed_files(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original\n")
    evidence._write_checksums(tmp_path)
    assert evidence._validate_checksums(tmp_path) == {
        "artifact.txt": evidence._sha256_file(artifact)
    }

    extra = tmp_path / "extra.txt"
    extra.write_text("unindexed\n")
    with pytest.raises(evidence.C1EvidenceError, match="inventory differs"):
        evidence._validate_checksums(tmp_path)
    extra.unlink()

    artifact.write_text("changed\n")
    with pytest.raises(evidence.C1EvidenceError, match="checksum mismatch"):
        evidence._validate_checksums(tmp_path)


def test_file_entry_rejects_boolean_size_and_bad_digest(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("payload\n")
    entry = evidence._file_entry(path, root=tmp_path)

    with pytest.raises(evidence.C1EvidenceError, match="size is invalid"):
        evidence._validate_file_entry(
            tmp_path,
            {**entry, "size": True},
            label="artifact",
        )
    with pytest.raises(evidence.C1EvidenceError, match="SHA-256 is invalid"):
        evidence._validate_file_entry(
            tmp_path,
            {**entry, "sha256": "0"},
            label="artifact",
        )


def test_command_validation_rejects_argv_or_environment_drift(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "stdout.txt"
    stderr = tmp_path / "stderr.txt"
    stdout.write_text("ok\n")
    stderr.write_text("")
    template = ("/python", "-m", "pytest")
    command = {
        "command_id": "test",
        "argv_template": list(template),
        "cwd": str(evidence.REPO_ROOT),
        "environment": dict(evidence.BASE_ENVIRONMENT),
        "started_at_utc": "2026-07-26T00:00:00+00:00",
        "ended_at_utc": "2026-07-26T00:00:01+00:00",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "timed_out": False,
        "stdout": evidence._file_entry(stdout, root=tmp_path),
        "stderr": evidence._file_entry(stderr, root=tmp_path),
    }
    evidence._validate_command(
        tmp_path,
        command,
        expected_id="test",
        expected_template=template,
    )

    with pytest.raises(evidence.C1EvidenceError, match="argv differs"):
        evidence._validate_command(
            tmp_path,
            {**command, "argv_template": ["other"]},
            expected_id="test",
            expected_template=template,
        )
    with pytest.raises(evidence.C1EvidenceError, match="environment differs"):
        evidence._validate_command(
            tmp_path,
            {**command, "environment": {}},
            expected_id="test",
            expected_template=template,
        )


def test_command_runner_prepares_only_in_staging_junit_parent(
    tmp_path: Path,
) -> None:
    result = evidence._run_command(
        "probe",
        ("/usr/bin/true", "--junitxml={output_root}/nested/result.xml"),
        output_root=tmp_path,
        cwd=evidence.REPO_ROOT,
        timeout_seconds=10,
    )
    assert result["exit_code"] == 0
    assert (tmp_path / "nested").is_dir()

    with pytest.raises(evidence.C1EvidenceError, match="escapes evidence staging"):
        evidence._run_command(
            "escape",
            ("/usr/bin/true", "--junitxml={output_root}/../escape.xml"),
            output_root=tmp_path,
            cwd=evidence.REPO_ROOT,
            timeout_seconds=10,
        )


def test_create_only_rename_preserves_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "source.txt").write_text("source\n")
    (target / "target.txt").write_text("target\n")

    with pytest.raises(evidence.C1EvidenceError, match="already exists"):
        evidence._rename_noreplace(source, target)
    assert (source / "source.txt").read_text() == "source\n"
    assert (target / "target.txt").read_text() == "target\n"
