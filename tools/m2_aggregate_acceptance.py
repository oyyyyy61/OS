#!/usr/bin/env python3
"""Publish or independently replay the aggregate M2 correctness decision."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.m2_formal_evidence import (  # noqa: E402
    FormalEvidenceError,
    validate_published_formal_bundle,
)
from tools.run_m2_component_evidence import (  # noqa: E402
    CHECKSUM_NAME as COMPONENT_CHECKSUM_NAME,
)
from tools.run_m2_component_evidence import (  # noqa: E402
    MANIFEST_NAME as COMPONENT_MANIFEST_NAME,
)
from tools.run_m2_component_evidence import (  # noqa: E402
    ComponentEvidenceError,
    validate_component_evidence,
)

SCHEMA_VERSION = "dagkv.m2.aggregate_acceptance.v1"
ACCEPTANCE_NAME = "M2_AGGREGATE_ACCEPTANCE.json"
INVALID_ACCEPTANCE_NAME = "INVALID_M2_AGGREGATE_ACCEPTANCE.json"
PUBLICATION_LOCK_SUFFIX = ".m2-aggregate-publication.lock"
LOCK_PREPARING = b"PREPARING\n"
LOCK_PUBLISHED = b"PUBLISHED\n"
GATE_STATUS = "M2_ACCEPTED_CORRECTNESS_ONLY"
CLAIM_SCOPE = (
    "M2 lifecycle and data-plane correctness for one process, one RTX 4090, "
    "and GPU plus primary CPU-DRAM under the frozen Qwen3-8B v3 ABBA profile. "
    "No latency, throughput, hit-rate, scheduling-policy, novelty, C1, C2, or "
    "C3 claim."
)
STATEMENT = (
    "All nine M2 conditions independently replayed across the sealed component "
    "and formal evidence. M2 correctness is accepted within the declared scope; "
    "all performance and M3 policy claims remain open."
)

TOOL_PATH = "tools/m2_aggregate_acceptance.py"
PROTOCOL_PATH = "research/protocols/M2_AGGREGATE_ACCEPTANCE_PROTOCOL.md"
STAGE_GATES_PATH = "research/STAGE_GATES.md"

VALIDATOR_CLOSURE_PATHS = (
    TOOL_PATH,
    "tools/m2_formal_evidence.py",
    "tools/run_m2_component_evidence.py",
    "tools/aggregate_m2_formal.py",
    "tools/aggregate_m2_calibration.py",
    "tools/freeze_m2_tolerance.py",
    "tools/m2_calibration_evidence.py",
    "tools/m2_raw_replay.py",
    "tools/run_m2_vllm_abba.py",
    "tools/nvidia_driver_userspace_bundle.py",
)

AUTHORITY_PATHS = (
    *VALIDATOR_CLOSURE_PATHS,
    PROTOCOL_PATH,
    STAGE_GATES_PATH,
    "research/M2_RUNTIME_CONTRACT.md",
    "research/protocols/M2_COMPONENT_EVIDENCE_PROTOCOL.md",
    "research/protocols/M2_FORMAL_CAMPAIGN_PROTOCOL.md",
    "research/protocols/M2_VLLM_REPLAY_PROTOCOL.md",
    "evidence/m2/M2_COMPONENT_EVIDENCE_INDEX.json",
    "evidence/m2/PILOT_ATTEMPTS.json",
    "evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json",
    "evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json",
    "evidence/m2/v3_580_173_02/M2_CALIBRATION_EVIDENCE_INDEX.json",
    "evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json",
    "evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN02_EVIDENCE_INDEX.json",
)

PROTECTED_GIT_PATHS = (
    "src/dagkv",
    "integrations/vllm_m2/dagkv_vllm_m2",
    "pyproject.toml",
    "uv.lock",
    "tests/test_domain.py",
    "tests/test_engine_adapter.py",
    "tests/test_ledger.py",
    "tests/test_orchestrator_failures.py",
    "tests/test_orchestrator_lifecycle.py",
    "integrations/vllm_m2/tests/test_contract.py",
)


def _test_id(suite: str, classname: str, name: str) -> str:
    return f"{suite}:{classname}::{name}"


CONDITION_TEST_IDS: dict[int, tuple[str, ...]] = {
    1: (
        _test_id(
            "dagkv_core",
            "tests.test_domain",
            "test_digest_identity_is_required_and_canonical",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_ledger",
            "test_binding_parent_freezes_execution_identity",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_execution_reference_is_single_use_across_binding_lifetimes",
        ),
        _test_id(
            "dagkv_vllm_adapter",
            "integrations.vllm_m2.tests.test_contract",
            "test_connector_captures_allocator_generation",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_offload.test_lifecycle",
            "test_lifecycle_identity_is_explicitly_enabled",
        ),
    ),
    2: (
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_lifecycle",
            "test_two_owner_offload_readmission_and_reclaim",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_lifecycle",
            "test_cross_owner_release_stays_rejected_after_real_owner_release",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_offload.cpu.test_manager",
            "test_strict_shared_load_has_independent_owner_binding",
        ),
    ),
    3: (
        _test_id(
            "dagkv_core",
            "tests.test_domain",
            "test_transfer_validates_direction_and_exact_terminal_replay",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_domain",
            "test_lease_allows_only_exact_terminal_replay",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_connector.unit.offloading_connector.test_worker_metadata",
            "test_mark_completed_identical_duplicate_is_idempotent",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_connector.unit.offloading_connector.test_scheduler",
            "test_duplicate_and_late_rank_reports_keep_one_terminal",
        ),
    ),
    4: (
        _test_id(
            "dagkv_core",
            "tests.test_engine_adapter",
            "test_generation_mismatch_terminalizes_before_raising",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_integrity_mismatch_cleans_reservation_and_rejects_stale_completion",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_stale_drop_and_reclaim_cannot_delete_new_generation",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_offload.cpu.test_manager",
            "test_failed_store_cleanup_emits_record_and_advances_generation",
        ),
        _test_id(
            "vllm_lifecycle_cpu",
            "tests.v1.kv_offload.cpu.test_manager",
            "test_strict_capacity_reuse_closes_generation_before_next_open",
        ),
    ),
    5: (
        _test_id(
            "dagkv_core",
            "tests.test_ledger",
            "test_audit_rejects_tampered_event_envelope_and_unknown_parent",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_ledger",
            "test_cross_family_references_block_early_free_and_failed_publish",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_lifecycle",
            "test_two_owner_offload_readmission_and_reclaim",
        ),
    ),
    6: (
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_invalid_waiter_cannot_leave_half_scheduled_h2d",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_released_h2d_waiter_is_not_published_after_completion",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_h2d_completion_does_not_publish_a_terminal_node_waiter",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_concurrent_h2d_waiters_share_one_physical_transfer",
        ),
    ),
    7: (
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_early_workflow_failure_does_not_partially_mutate_state",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_dag_failure_cancels_parallel_node_and_skips_descendants",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_required_binding_respects_the_dag_running_gate",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_workflow_failure_closes_bindings_leases_and_execution_maps",
        ),
        _test_id(
            "dagkv_core",
            "tests.test_orchestrator_failures",
            "test_workflow_failure_racing_h2d_completion_is_serializable",
        ),
    ),
}

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "accepted_at_utc",
        "gate_status",
        "verification_status",
        "repository",
        "component_evidence",
        "formal_evidence",
        "compatibility_bridge",
        "external_content",
        "excluded_attempts",
        "conditions",
        "m2_item8_accepted",
        "m2_accepted",
        "performance_claims_supported",
        "policy_claims_supported",
        "claim_scope",
        "statement",
    }
)


class AggregateAcceptanceError(RuntimeError):
    """Raised when the aggregate M2 decision cannot be reproduced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateAcceptanceError(message)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_object_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise AggregateAcceptanceError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(payload: Any) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _git_head(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-1 Git object ID",
    )
    return value


def _timestamp(value: Any, *, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} must be non-empty")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AggregateAcceptanceError(f"invalid {label}: {value}") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AggregateAcceptanceError(f"non-finite JSON constant: {value}")


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except AggregateAcceptanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateAcceptanceError(f"invalid {label}: {exc}") from exc


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode), f"{label} must be regular: {path}")
        raw = path.read_bytes()
        after = path.lstat()
    except AggregateAcceptanceError:
        raise
    except OSError as exc:
        raise AggregateAcceptanceError(f"cannot read {label}: {exc}") from exc
    require(
        _stat_identity(before) == _stat_identity(after), f"{label} changed while read"
    )
    require(len(raw) == before.st_size, f"short read from {label}: {path}")
    return raw


def _read_json(path: Path, *, label: str) -> tuple[Any, bytes]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AggregateAcceptanceError(f"cannot read {label}: {exc}") from exc
    return _decode_json(raw, label=label), raw


def _exact_fields(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    require(set(payload) == expected, f"{label} fields differ")


def _safe_relative(value: Any, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} is missing")
    path = Path(value)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"unsafe {label}: {value}",
    )
    return path


def _file_binding(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    observed = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(observed.st_mode), f"file is not regular: {path}")
    return {
        "path": str(path.resolve()),
        "size": observed.st_size,
        "sha256": _sha256_file(path),
    }


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AggregateAcceptanceError(
            f"Git command failed ({' '.join(arguments)}): {exc}"
        ) from exc


def _git_text(repo: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repo, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise AggregateAcceptanceError("Git returned non-UTF-8 text") from exc


def _git_blob_binding(repo: Path, head: str, repository_path: str) -> dict[str, Any]:
    raw = _git_bytes(repo, "ls-tree", head, "--", repository_path)
    lines = raw.decode("utf-8").splitlines()
    require(len(lines) == 1, f"Git authority path is missing: {repository_path}")
    metadata, observed_path = lines[0].split("\t", maxsplit=1)
    mode, kind, object_id = metadata.split()
    require(
        observed_path == repository_path,
        f"Git authority path differs: {repository_path}",
    )
    require(kind == "blob", f"Git authority is not a blob: {repository_path}")
    content = _git_bytes(repo, "cat-file", "blob", object_id)
    return {
        "repository_path": repository_path,
        "mode": mode,
        "git_blob": object_id,
        "size": len(content),
        "sha256": _sha256_bytes(content),
    }


def _authority_bindings(repo: Path, head: str) -> list[dict[str, Any]]:
    return [_git_blob_binding(repo, head, path) for path in AUTHORITY_PATHS]


def _validate_current_validator_closure(
    root: Path, authority_bindings: Sequence[Mapping[str, Any]]
) -> None:
    by_path: dict[str, Mapping[str, Any]] = {}
    for binding in authority_bindings:
        repository_path = binding.get("repository_path")
        require(
            isinstance(repository_path, str) and repository_path not in by_path,
            "repository authority path is invalid or duplicated",
        )
        by_path[repository_path] = binding
    for repository_path in (*VALIDATOR_CLOSURE_PATHS, PROTOCOL_PATH):
        require(
            repository_path in by_path,
            f"current replay authority is missing: {repository_path}",
        )
        binding = by_path[repository_path]
        path = root / repository_path
        try:
            require(
                path.resolve(strict=True) == path.absolute(),
                f"current replay authority has a symlink: {repository_path}",
            )
            observed = path.lstat()
        except AggregateAcceptanceError:
            raise
        except OSError as exc:
            raise AggregateAcceptanceError(
                f"cannot resolve current replay authority {repository_path}: {exc}"
            ) from exc
        require(
            stat.S_ISREG(observed.st_mode),
            f"current replay authority is not regular: {repository_path}",
        )
        raw = _read_stable_bytes(
            path, label=f"current replay authority {repository_path}"
        )
        expected_mode = binding.get("mode")
        require(
            isinstance(expected_mode, str)
            and expected_mode in {"100644", "100755"}
            and stat.S_IMODE(observed.st_mode) & 0o111
            == (0o111 if expected_mode == "100755" else 0),
            f"current replay authority mode differs: {repository_path}",
        )
        require(
            len(raw) == binding.get("size")
            and _sha256_bytes(raw) == binding.get("sha256"),
            f"current replay authority bytes differ: {repository_path}",
        )


def _git_json(
    repo: Path, head: str, repository_path: str, *, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _git_blob_binding(repo, head, repository_path)
    raw = _git_bytes(repo, "cat-file", "blob", binding["git_blob"])
    payload = _decode_json(raw, label=label)
    require(isinstance(payload, dict), f"{label} must be an object")
    return payload, binding


def _canonical_evidence_binding(repo: Path, head: str) -> dict[str, Any]:
    component_path = "evidence/m2/M2_COMPONENT_EVIDENCE_INDEX.json"
    calibration_path = "evidence/m2/v3_580_173_02/M2_CALIBRATION_EVIDENCE_INDEX.json"
    formal_path = "evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN02_EVIDENCE_INDEX.json"
    component, component_blob = _git_json(
        repo, head, component_path, label="canonical component index"
    )
    calibration, calibration_blob = _git_json(
        repo, head, calibration_path, label="canonical calibration index"
    )
    formal, formal_blob = _git_json(
        repo, head, formal_path, label="canonical formal index"
    )

    component_bundle = component.get("successful_bundle")
    require(
        component.get("schema_version") == "dagkv.m2.component_evidence_index.v1"
        and component.get("status") == "ITEMS_1_TO_7_COMPONENT_EVIDENCE_VERIFIED"
        and isinstance(component_bundle, dict)
        and component_bundle.get("all_passed") is True
        and component_bundle.get("total_tests") == 627
        and component_bundle.get("external_replay_passed") is True
        and component_bundle.get("raw_only_replay_passed") is True,
        "canonical component index is not eligible",
    )
    calibration_manifest = calibration.get("calibration_manifest")
    calibration_tolerance = calibration.get("frozen_tolerance")
    calibration_source = calibration.get("source_and_runtime")
    require(
        calibration.get("schema_version") == "dagkv.m2.v3_calibration_evidence_index.v1"
        and calibration.get("calibration_bundle_eligible") is True
        and calibration.get("acceptance_claimed") is False
        and isinstance(calibration_manifest, dict)
        and isinstance(calibration_tolerance, dict)
        and isinstance(calibration_source, dict)
        and calibration.get("observations", {}).get("run_count") == 59
        and calibration.get("observations", {}).get("all_passed") is True
        and calibration.get("attempt_journal", {}).get("retry_count") == 0
        and calibration.get("attempt_journal", {}).get("nonpassing_terminal_count")
        == 0,
        "canonical calibration index is not eligible",
    )
    formal_prereg = formal.get("campaign_preregistration")
    formal_seal = formal.get("formal_bundle_seal")
    formal_parent = formal.get("parent_binding")
    formal_source = formal.get("source_and_runtime")
    require(
        formal.get("schema_version") == "dagkv.m2.formal_campaign_evidence_index.v1"
        and formal.get("formal_cohort_eligible") is True
        and formal.get("m2_item8_accepted") is True
        and formal.get("m2_accepted") is False
        and formal.get("performance_claims_supported") is False
        and isinstance(formal_prereg, dict)
        and isinstance(formal_seal, dict)
        and isinstance(formal_parent, dict)
        and isinstance(formal_source, dict)
        and formal.get("observations", {}).get("run_count") == 20
        and formal.get("observations", {}).get("all_formal_runs_passed") is True
        and formal.get("attempt_journal", {}).get("retry_count") == 0
        and formal.get("attempt_journal", {}).get("nonpassing_terminal_count") == 0,
        "canonical formal index is not eligible",
    )
    require(
        calibration_manifest.get("sha256")
        == formal_parent.get("calibration_manifest_sha256")
        and calibration_tolerance.get("sha256")
        == formal_parent.get("frozen_tolerance_sha256"),
        "canonical formal parent differs from calibration index",
    )
    shared_fields = (
        "implementation_manifest_sha256",
        "reproducibility_fingerprint",
        "vllm_snapshot_sha256",
        "nvidia_driver_version",
        "nvidia_userspace_bundle_manifest_sha256",
        "nvidia_userspace_bundle_content_digest",
        "base_dependency_count",
        "base_dependency_manifest_sha256",
        "effective_dependency_count",
        "effective_dependency_manifest_sha256",
        "added_dependency_count",
        "added_dependency_manifest_sha256",
        "removed_dependency_count",
    )
    require(
        all(
            calibration_source.get(key) == formal_source.get(key)
            for key in shared_fields
        ),
        "canonical calibration and formal runtime identities differ",
    )
    return {
        "component": {
            "index": component_blob,
            "root": component_bundle.get("runtime_path"),
            "manifest_sha256": component_bundle.get("manifest_sha256"),
            "sha256sums_sha256": component_bundle.get("sha256sums_sha256"),
            "dagkv_head": component_bundle.get("dagkv_head"),
            "dagkv_snapshot_sha256": component_bundle.get("dagkv_snapshot_sha256"),
            "vllm_head": component_bundle.get("vllm_head"),
            "vllm_snapshot_sha256": component_bundle.get("vllm_snapshot_sha256"),
        },
        "calibration": {
            "index": calibration_blob,
            "root": calibration.get("campaign_root"),
            "manifest_sha256": calibration_manifest.get("sha256"),
            "frozen_tolerance_sha256": calibration_tolerance.get("sha256"),
            "source_and_runtime": calibration_source,
        },
        "formal": {
            "index": formal_blob,
            "campaign_id": formal.get("campaign_id"),
            "root": formal.get("campaign_root"),
            "preregistration_sha256": formal_prereg.get("sha256"),
            "seal_sha256": formal_seal.get("sha256"),
            "source_and_runtime": formal_source,
        },
    }


def _git_tree(repo: Path, head: str) -> list[dict[str, str]]:
    raw = _git_bytes(repo, "ls-tree", "-r", head, "--", *PROTECTED_GIT_PATHS)
    entries: list[dict[str, str]] = []
    for line in raw.decode("utf-8").splitlines():
        metadata, path = line.split("\t", maxsplit=1)
        mode, kind, object_id = metadata.split()
        require(kind == "blob", f"protected Git path is not a blob: {path}")
        entries.append({"mode": mode, "path": path, "git_blob": object_id})
    require(
        entries == sorted(entries, key=lambda item: item["path"]),
        "Git tree order differs",
    )
    require(len(entries) >= 20, "protected Git tree is unexpectedly small")
    return entries


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return False
        raise AggregateAcceptanceError("cannot validate Git ancestry") from exc
    except OSError as exc:
        raise AggregateAcceptanceError("cannot run Git ancestry check") from exc
    return True


def _normal_name(value: Any, *, label: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{label} name is invalid")
    return value.strip().lower().replace("_", "-")


def _dependency_pairs(rows: Any, *, label: str) -> list[tuple[str, str]]:
    require(isinstance(rows, list) and rows, f"{label} dependency list is empty")
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        require(isinstance(row, dict), f"{label} dependency row is not an object")
        name = _normal_name(row.get("name"), label=label)
        version = row.get("version")
        require(isinstance(version, str) and version, f"{label} version is invalid")
        pairs.add((name, version))
    return sorted(pairs)


def _parse_junit_cases(component_root: Path, manifest: Mapping[str, Any]) -> set[str]:
    suites = manifest.get("suites")
    require(isinstance(suites, list), "component suites are missing")
    observed: set[str] = set()
    for suite in suites:
        require(isinstance(suite, dict), "component suite entry is invalid")
        suite_id = suite.get("suite_id")
        require(isinstance(suite_id, str) and suite_id, "component suite ID is invalid")
        junit = suite.get("junit")
        require(isinstance(junit, dict), f"{suite_id} JUnit entry is missing")
        relative = _safe_relative(junit.get("path"), label=f"{suite_id} JUnit path")
        path = component_root / relative
        require(path.is_file() and not path.is_symlink(), f"missing JUnit: {path}")
        try:
            root = ET.fromstring(path.read_bytes())
        except (OSError, ET.ParseError) as exc:
            raise AggregateAcceptanceError(f"invalid {suite_id} JUnit: {exc}") from exc
        for case in root.iter("testcase"):
            classname = case.get("classname")
            name = case.get("name")
            require(
                classname is not None and name is not None, "JUnit case lacks identity"
            )
            require(
                not any(child.tag in {"failure", "error", "skipped"} for child in case),
                f"required JUnit suite contains a non-pass case: {classname}::{name}",
            )
            identity = _test_id(suite_id, classname, name)
            require(identity not in observed, f"duplicate JUnit case: {identity}")
            observed.add(identity)
    required = {
        identity for values in CONDITION_TEST_IDS.values() for identity in values
    }
    missing = sorted(required - observed)
    require(not missing, f"required M2 JUnit cases are missing: {missing}")
    return observed


def _condition_payloads() -> list[dict[str, Any]]:
    descriptions = {
        1: "canonical schemas and lifetime identities",
        2: "shared-owner isolation and reclaim safety",
        3: "idempotent release and terminal replay",
        4: "stale generation and completion rejection",
        5: "cross-family conservation and independent audit",
        6: "atomic compatible-waiter H2D publication",
        7: "workflow terminal cleanup and DAG running gate",
        8: "frozen real-vLLM calibration and formal holdouts",
        9: "content-addressed aggregate provenance",
    }
    rows: list[dict[str, Any]] = []
    for number in range(1, 10):
        sources: list[str]
        if number <= 7:
            sources = list(CONDITION_TEST_IDS[number])
        elif number == 8:
            sources = [
                "59 independently replayed calibration processes",
                "20 independently replayed formal holdouts",
                "M2 item-8 acceptance and formal bundle seal",
            ]
        else:
            sources = [
                "component external and raw-only replay",
                "formal full-bundle replay",
                "historical Git-object compatibility bridge",
                "current full hashes for 16 model files, 6 vLLM extensions, and Python",
                "create-only aggregate acceptance",
            ]
        rows.append(
            {
                "condition": number,
                "description": descriptions[number],
                "status": "VERIFIED",
                "sources": sources,
            }
        )
    return rows


def _strict_external_file(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    before = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(before.st_mode), f"{label} is not regular: {path}")
    require(before.st_nlink == 1, f"{label} must have one hard link: {path}")
    require(
        before.st_size == expected.get("size")
        and before.st_mtime_ns == expected.get("mtime_ns")
        and before.st_ino == expected.get("inode"),
        f"{label} stat identity differs: {path}",
    )
    expected_sha = _lower_sha256(expected.get("sha256"), label=f"{label} recorded SHA")
    observed_sha = _sha256_file(path)
    after = path.stat(follow_symlinks=False)
    require(
        (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{label} changed while hashing: {path}",
    )
    require(observed_sha == expected_sha, f"{label} content SHA differs: {path}")
    return {"path": str(path), "size": before.st_size, "sha256": observed_sha}


def _external_identity(path: Path) -> tuple[int, ...]:
    observed = path.stat(follow_symlinks=False)
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _validate_external_tree(root: Path, *, label: str) -> None:
    require(root.is_dir() and not root.is_symlink(), f"{label} root is invalid")
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        observed = path.stat(follow_symlinks=False)
        require(not path.is_symlink(), f"{label} tree contains a symlink: {path}")
        require(
            stat.S_ISREG(observed.st_mode) or stat.S_ISDIR(observed.st_mode),
            f"{label} tree contains a special node: {path}",
        )


def _rehash_external_content(provenance_path: Path) -> dict[str, Any]:
    provenance, raw = _read_json(provenance_path, label="formal run provenance")
    require(isinstance(provenance, dict), "formal provenance must be an object")
    model = provenance.get("model")
    runtime = provenance.get("runtime_binaries")
    require(
        isinstance(model, dict) and model.get("full_hashes") is True,
        "model full hashes missing",
    )
    require(
        isinstance(runtime, dict) and runtime.get("full_hashes") is True,
        "runtime binary full hashes missing",
    )

    model_root = Path(model.get("root", ""))
    require(model_root.is_absolute(), "model root is not absolute")
    _validate_external_tree(model_root, label="model")
    model_rows = model.get("files")
    require(
        isinstance(model_rows, list) and len(model_rows) == 16,
        "model closed set must contain 16 files",
    )
    model_paths: list[Path] = []
    for row in model_rows:
        require(isinstance(row, dict), "model file entry is invalid")
        relative = _safe_relative(row.get("path"), label="model path")
        model_paths.append(model_root / relative)

    runtime_root = Path(runtime.get("root", ""))
    require(runtime_root.is_absolute(), "runtime root is not absolute")
    _validate_external_tree(runtime_root / "vllm", label="runtime vLLM")
    extension_rows = runtime.get("vllm_extensions")
    require(
        isinstance(extension_rows, list) and len(extension_rows) == 6,
        "runtime extension closed set must contain 6 files",
    )
    extension_paths: list[Path] = []
    for row in extension_rows:
        require(isinstance(row, dict), "runtime extension entry is invalid")
        relative = _safe_relative(row.get("path"), label="runtime extension path")
        extension_paths.append(runtime_root / relative)
    python_row = runtime.get("python_executable")
    require(isinstance(python_row, dict), "runtime Python entry is invalid")
    python_path = Path(python_row.get("path", ""))
    require(python_path.is_absolute(), "runtime Python path is invalid")
    all_external_paths = [*model_paths, *extension_paths, python_path]
    require(
        len({str(path) for path in all_external_paths}) == 23,
        "external paths overlap",
    )
    identity_before = {
        str(path): _external_identity(path) for path in all_external_paths
    }

    recorded_model_paths: set[str] = set()
    model_entries: list[dict[str, Any]] = []
    model_content: list[dict[str, Any]] = []
    for row in model_rows:
        require(isinstance(row, dict), "model file entry is invalid")
        relative = _safe_relative(row.get("path"), label="model path")
        relative_text = relative.as_posix()
        require(relative_text not in recorded_model_paths, "duplicate model path")
        recorded_model_paths.add(relative_text)
        binding = _strict_external_file(model_root / relative, row, label="model file")
        binding["relative_path"] = relative_text
        binding["kind"] = row.get("kind")
        model_entries.append(binding)
        model_content.append(
            {
                "path": relative_text,
                "size": row.get("size"),
                "kind": row.get("kind"),
                "sha256": row.get("sha256"),
            }
        )
    live_model_paths = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(model_root).parts
    }
    require(live_model_paths == recorded_model_paths, "model file closed set differs")
    require(
        _canonical_digest(model_content) == model.get("manifest_sha256"),
        "model manifest digest differs",
    )

    extension_entries: list[dict[str, Any]] = []
    runtime_content: list[dict[str, Any]] = []
    recorded_extensions: set[str] = set()
    for row in extension_rows:
        require(isinstance(row, dict), "runtime extension entry is invalid")
        relative = _safe_relative(row.get("path"), label="runtime extension path")
        relative_text = relative.as_posix()
        require(relative_text not in recorded_extensions, "duplicate runtime extension")
        recorded_extensions.add(relative_text)
        binding = _strict_external_file(
            runtime_root / relative, row, label="runtime extension"
        )
        binding["relative_path"] = relative_text
        extension_entries.append(binding)
        runtime_content.append(
            {
                "path": relative_text,
                "size": row.get("size"),
                "sha256": row.get("sha256"),
            }
        )
    live_extensions = {
        path.relative_to(runtime_root).as_posix()
        for path in (runtime_root / "vllm").rglob("*.so")
        if path.is_file()
    }
    require(
        live_extensions == recorded_extensions, "runtime extension closed set differs"
    )
    python_entry = _strict_external_file(
        python_path, python_row, label="runtime Python"
    )
    python_content = {key: python_row.get(key) for key in ("path", "size", "sha256")}
    require(
        _canonical_digest(
            {
                "vllm_extensions": runtime_content,
                "python_executable": python_content,
            }
        )
        == runtime.get("manifest_sha256"),
        "runtime binary manifest digest differs",
    )
    combined = {
        "model_manifest_sha256": model.get("manifest_sha256"),
        "runtime_binary_manifest_sha256": runtime.get("manifest_sha256"),
        "python_sha256": python_entry["sha256"],
    }
    identity_after = {
        str(path): _external_identity(path) for path in all_external_paths
    }
    require(
        identity_after == identity_before,
        "external file set changed during aggregate rehash",
    )
    _validate_external_tree(model_root, label="model")
    _validate_external_tree(runtime_root / "vllm", label="runtime vLLM")
    final_model_paths = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(model_root).parts
    }
    final_extensions = {
        path.relative_to(runtime_root).as_posix()
        for path in (runtime_root / "vllm").rglob("*.so")
        if path.is_file()
    }
    require(
        final_model_paths == live_model_paths == recorded_model_paths,
        "model file set changed during aggregate rehash",
    )
    require(
        final_extensions == live_extensions == recorded_extensions,
        "runtime extension set changed during aggregate rehash",
    )
    return {
        "provenance": {
            "path": str(provenance_path.resolve()),
            "size": len(raw),
            "sha256": _sha256_bytes(raw),
        },
        "model": {
            "root": str(model_root),
            "file_count": len(model_entries),
            "manifest_sha256": model.get("manifest_sha256"),
            "files": model_entries,
        },
        "runtime_binaries": {
            "root": str(runtime_root),
            "extension_count": len(extension_entries),
            "manifest_sha256": runtime.get("manifest_sha256"),
            "extensions": extension_entries,
            "python": python_entry,
        },
        "total_file_count": len(model_entries) + len(extension_entries) + 1,
        "combined_manifest_sha256": _canonical_digest(combined),
        "current_content_rehash_passed": True,
    }


def _validate_excluded_attempts(
    repo: Path,
    head: str,
    *,
    eligible_campaign_id: str,
    eligible_run_ids: Sequence[str],
) -> dict[str, Any]:
    paths = {
        "pilot_index": "evidence/m2/PILOT_ATTEMPTS.json",
        "run09": "evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json",
        "run10": "evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json",
        "formal_campaign01": (
            "evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json"
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, repository_path in paths.items():
        binding = _git_blob_binding(repo, head, repository_path)
        raw = _git_bytes(repo, "cat-file", "blob", binding["git_blob"])
        payload = _decode_json(raw, label=label)
        require(isinstance(payload, dict), f"{label} index must be an object")
        payloads[label] = payload
        bindings[label] = binding

    pilot = payloads["pilot_index"]
    require(pilot.get("cohort_eligible") is False, "pilot index is cohort eligible")
    require(pilot.get("acceptance_claimed") is False, "pilot index claims acceptance")
    attempts = pilot.get("attempts")
    require(isinstance(attempts, list) and len(attempts) == 10, "pilot set differs")
    require(
        all(item.get("cohort_eligible") is False for item in attempts[-2:]),
        "v3 pilots must be explicitly ineligible",
    )
    run09 = payloads["run09"]
    require(
        run09.get("gate_status") == "FAILED"
        and run09.get("cohort_eligible") is False
        and run09.get("acceptance_claimed") is False,
        "run09 exclusion differs",
    )
    run10 = payloads["run10"]
    require(
        run10.get("gate_status") == "CALIBRATED_NOT_ACCEPTED"
        and run10.get("cohort_eligible") is False
        and run10.get("acceptance_claimed") is False,
        "run10 exclusion differs",
    )
    campaign01 = payloads["formal_campaign01"]
    require(
        campaign01.get("formal_cohort_eligible") is False
        and campaign01.get("acceptance_claimed") is False
        and campaign01.get("failure", {}).get("stage")
        == "post_aggregate_candidate_replay",
        "formal campaign01 exclusion differs",
    )
    campaign01_id = campaign01.get("campaign_id")
    require(
        isinstance(campaign01_id, str)
        and campaign01_id
        and isinstance(eligible_campaign_id, str)
        and eligible_campaign_id
        and campaign01_id != eligible_campaign_id,
        "excluded and eligible formal campaign identities overlap",
    )
    names = [
        *(item.get("name") for item in attempts),
        campaign01_id,
    ]
    require(
        all(isinstance(name, str) and name for name in names),
        "excluded name is invalid",
    )

    run09_execution = run09.get("execution")
    run10_execution = run10.get("execution")
    require(
        isinstance(run09_execution, dict) and isinstance(run10_execution, dict),
        "excluded execution binding is missing",
    )
    run09_id = run09_execution.get("run_id")
    run10_id = run10_execution.get("run_id")
    require(
        all(isinstance(run_id, str) and run_id for run_id in (run09_id, run10_id)),
        "excluded pilot run ID is invalid",
    )

    campaign_root_value = campaign01.get("campaign_root")
    journal = campaign01.get("attempt_journal")
    require(
        isinstance(campaign_root_value, str)
        and campaign_root_value
        and isinstance(journal, dict),
        "campaign01 journal binding is missing",
    )
    campaign_root = Path(campaign_root_value)
    require(campaign_root.is_absolute(), "campaign01 root must be absolute")
    require(
        campaign_root.is_dir() and not campaign_root.is_symlink(),
        "campaign01 root is invalid",
    )
    journal_name = journal.get("file")
    require(journal_name == "FORMAL_ATTEMPTS.jsonl", "campaign01 journal name differs")
    journal_path = campaign_root / journal_name
    journal_raw = _read_stable_bytes(journal_path, label="campaign01 attempt journal")
    expected_journal_sha = _lower_sha256(
        journal.get("sha256"), label="campaign01 journal"
    )
    require(
        _sha256_bytes(journal_raw) == expected_journal_sha,
        "campaign01 journal SHA differs",
    )
    require(journal_raw.endswith(b"\n"), "campaign01 journal is not newline terminated")
    journal_lines = journal_raw.splitlines()
    require(
        journal.get("record_count") == len(journal_lines) == 42,
        "campaign01 journal record count differs",
    )
    records: list[dict[str, Any]] = []
    for index, line in enumerate(journal_lines, start=1):
        require(bool(line), f"campaign01 journal row {index} is empty")
        row = _decode_json(line, label=f"campaign01 journal row {index}")
        require(isinstance(row, dict), f"campaign01 journal row {index} is invalid")
        records.append(row)
    formal_records = [row for row in records if row.get("kind") == "formal_run"]
    formal_terminals = [row for row in formal_records if row.get("event") == "terminal"]
    require(
        len(formal_records) == 40 and len(formal_terminals) == 20,
        "campaign01 formal journal boundary differs",
    )
    require(
        all(type(row.get("sequence")) is int for row in formal_terminals),
        "campaign01 terminal sequence is invalid",
    )
    formal_terminals.sort(key=lambda row: row.get("sequence", -1))
    require(
        [row.get("sequence") for row in formal_terminals] == list(range(1, 21))
        and all(
            row.get("status") == "passed" and row.get("campaign_id") == campaign01_id
            for row in formal_terminals
        ),
        "campaign01 terminal cohort differs",
    )
    terminal_validations = [row.get("validation") for row in formal_terminals]
    require(
        all(isinstance(validation, dict) for validation in terminal_validations),
        "campaign01 terminal validation is missing",
    )
    campaign01_run_ids = [
        validation.get("run_id") for validation in terminal_validations
    ]
    require(
        all(isinstance(run_id, str) and run_id for run_id in campaign01_run_ids)
        and len(set(campaign01_run_ids)) == 20,
        "campaign01 run IDs are invalid or duplicated",
    )

    eligible_ids = list(eligible_run_ids)
    require(
        len(eligible_ids) == 79
        and all(isinstance(run_id, str) and run_id for run_id in eligible_ids)
        and len(set(eligible_ids)) == 79,
        "eligible run ID boundary differs during exclusion audit",
    )
    excluded_ids = [run09_id, run10_id, *campaign01_run_ids]
    require(len(set(excluded_ids)) == 22, "excluded run IDs are not globally unique")
    intersection = sorted(set(excluded_ids) & set(eligible_ids))
    require(not intersection, "excluded and eligible run IDs overlap")

    authorities = [
        {
            "category": label,
            "repository_path": binding["repository_path"],
            "git_blob": binding["git_blob"],
            "sha256": binding["sha256"],
            "cohort_eligible": False,
        }
        for label, binding in bindings.items()
    ]
    return {
        "authorities": authorities,
        "excluded_campaign_ids": [campaign01_id],
        "excluded_run_ids": excluded_ids,
        "excluded_run_id_count": len(excluded_ids),
        "eligible_run_id_count": len(eligible_ids),
        "eligible_run_id_intersection_count": len(intersection),
        "campaign01_journal": {
            "path": str(journal_path),
            "size": len(journal_raw),
            "sha256": expected_journal_sha,
            "record_count": len(records),
        },
    }


def _component_evidence(
    component_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    component_root = component_root.expanduser().resolve()
    external = validate_component_evidence(component_root, verify_external=True)
    raw_only = validate_component_evidence(component_root, verify_external=False)
    require(external == raw_only, "component replay modes returned different manifests")
    manifest_binding = _file_binding(component_root / COMPONENT_MANIFEST_NAME)
    sums_binding = _file_binding(component_root / COMPONENT_CHECKSUM_NAME)
    require(
        manifest_binding["sha256"]
        == _lower_sha256(expected_manifest_sha256, label="expected component manifest"),
        "component manifest SHA differs",
    )
    require(
        sums_binding["sha256"]
        == _lower_sha256(
            expected_sha256sums_sha256, label="expected component checksums"
        ),
        "component SHA256SUMS SHA differs",
    )
    junit_cases = _parse_junit_cases(component_root, external)
    suites = [
        {
            "suite_id": suite["suite_id"],
            "tests": suite["tests"],
            "passed": suite["passed"],
        }
        for suite in external["suites"]
    ]
    return external, {
        "root": str(component_root),
        "manifest": manifest_binding,
        "sha256sums": sums_binding,
        "total_tests": external["total_tests"],
        "suites": suites,
        "required_junit_case_count": sum(map(len, CONDITION_TEST_IDS.values())),
        "observed_junit_case_count": len(junit_cases),
        "external_replay_passed": True,
        "raw_only_replay_passed": True,
    }


def _cohort_identity(
    calibration: Mapping[str, Any],
    seal: Mapping[str, Any],
    calibration_preregistration: Mapping[str, Any],
    formal_preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_runs = calibration.get("runs")
    require(
        calibration.get("run_count") == 59
        and isinstance(calibration_runs, list)
        and len(calibration_runs) == 59
        and all(isinstance(row, dict) for row in calibration_runs)
        and calibration.get("all_passed") is True,
        "calibration cohort differs from 59/59",
    )
    formal_runs = seal.get("ordered_runs")
    require(
        isinstance(formal_runs, list)
        and len(formal_runs) == seal.get("run_count") == 20,
        "formal cohort differs from 20/20",
    )
    require(
        all(isinstance(row, dict) for row in formal_runs),
        "formal cohort contains an invalid row",
    )
    require(
        calibration_preregistration.get("retry_policy") == "none_stop_on_first_failure"
        and formal_preregistration.get("retry_policy") == "none_stop_on_first_failure",
        "cohort retry policy differs",
    )
    calibration_ids = [row.get("run_id") for row in calibration_runs]
    formal_ids = [row.get("run_id") for row in formal_runs]
    require(
        all(
            isinstance(run_id, str) and run_id
            for run_id in calibration_ids + formal_ids
        ),
        "cohort run ID is invalid",
    )
    require(
        len(set(calibration_ids + formal_ids)) == 79,
        "calibration and formal run IDs are not globally unique",
    )
    ordered = {
        "calibration_run_ids": calibration_ids,
        "formal_run_ids": formal_ids,
    }
    return {
        "calibration_run_count": len(calibration_ids),
        "formal_run_count": len(formal_ids),
        "global_unique_run_id_count": len(set(calibration_ids + formal_ids)),
        "cohort_identity_sha256": _canonical_digest(ordered),
        "retry_policy": "none_stop_on_first_failure",
    }


def _formal_evidence(
    seal_path: Path,
    *,
    expected_seal_sha256: str,
    expected_preregistration_sha256: str,
) -> tuple[
    dict[str, Any],
    Any,
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    seal_path = seal_path.expanduser().resolve()
    seal, observed_sha, validation = validate_published_formal_bundle(
        seal_path,
        expected_seal_sha256=_lower_sha256(
            expected_seal_sha256, label="expected formal seal"
        ),
        expected_preregistration_sha256=_lower_sha256(
            expected_preregistration_sha256,
            label="expected formal preregistration",
        ),
    )
    require(observed_sha == expected_seal_sha256, "formal seal SHA differs")
    seal_binding = _file_binding(seal_path)
    prereg_path = seal_path.parent / seal["formal_campaign_preregistration_file"]
    prereg, prereg_raw = _read_json(prereg_path, label="formal preregistration")
    require(isinstance(prereg, dict), "formal preregistration must be an object")
    require(
        _sha256_bytes(prereg_raw) == expected_preregistration_sha256,
        "formal preregistration SHA differs",
    )
    calibration_entry = prereg.get("frozen_files", {}).get("calibration_manifest")
    tolerance_entry = prereg.get("frozen_files", {}).get("frozen_tolerance")
    require(isinstance(calibration_entry, dict), "calibration binding is missing")
    require(isinstance(tolerance_entry, dict), "tolerance binding is missing")
    calibration_path = Path(calibration_entry.get("path", ""))
    tolerance_path = Path(tolerance_entry.get("path", ""))
    calibration, calibration_raw = _read_json(
        calibration_path, label="calibration manifest"
    )
    require(isinstance(calibration, dict), "calibration manifest must be an object")
    require(
        _sha256_bytes(calibration_raw)
        == calibration_entry.get("sha256")
        == seal["calibration_manifest_sha256"],
        "calibration manifest binding differs",
    )
    tolerance_binding = _file_binding(tolerance_path)
    require(
        tolerance_binding["sha256"]
        == tolerance_entry.get("sha256")
        == seal["frozen_tolerance_sha256"],
        "frozen tolerance binding differs",
    )
    calibration_prereg_path = (
        calibration_path.parent / calibration["campaign_preregistration_file"]
    )
    calibration_prereg, _ = _read_json(
        calibration_prereg_path, label="calibration preregistration"
    )
    require(
        isinstance(calibration_prereg, dict), "calibration preregistration is invalid"
    )
    cohort = _cohort_identity(calibration, seal, calibration_prereg, prereg)
    first = seal["ordered_runs"][0]
    provenance_path = seal_path.parent / first["run_name"] / "provenance.json"
    require(
        _sha256_file(provenance_path) == first["provenance_sha256"],
        "representative formal provenance SHA differs",
    )
    summary = {
        "root": str(seal_path.parent),
        "campaign_id": seal["campaign_id"],
        "seal": seal_binding,
        "preregistration": {
            "path": str(prereg_path),
            "size": len(prereg_raw),
            "sha256": _sha256_bytes(prereg_raw),
        },
        "calibration_manifest": {
            "path": str(calibration_path),
            "size": len(calibration_raw),
            "sha256": _sha256_bytes(calibration_raw),
        },
        "frozen_tolerance": tolerance_binding,
        **cohort,
        "item8_accepted": True,
        "full_bundle_replay_passed": True,
    }
    return seal, validation, prereg, calibration, provenance_path, summary


def _compatibility_bridge(
    repo: Path,
    component: Mapping[str, Any],
    seal: Mapping[str, Any],
    provenance_path: Path,
    component_root: Path,
) -> dict[str, Any]:
    require(
        Path(component["dagkv_git"]["root"]).resolve() == repo.resolve(),
        "component DAGKV repository root differs",
    )
    component_head = _git_head(component["dagkv_git"]["head"], label="component HEAD")
    preparation_head = _git_head(
        seal["execution_binding"]["preparation_git_head"], label="preparation HEAD"
    )
    execution_head = _git_head(
        seal["execution_binding"]["execution_git_head"], label="execution HEAD"
    )
    require(
        _is_ancestor(repo, component_head, preparation_head),
        "component HEAD is not an ancestor of formal preparation",
    )
    require(
        _is_ancestor(repo, preparation_head, execution_head),
        "formal preparation is not an ancestor of execution",
    )
    parents = _git_text(
        repo, "rev-list", "--parents", "-n", "1", execution_head
    ).split()
    require(
        parents == [execution_head, preparation_head],
        "formal execution is not the direct child of preparation",
    )
    trees = {
        "component": _git_tree(repo, component_head),
        "preparation": _git_tree(repo, preparation_head),
        "execution": _git_tree(repo, execution_head),
    }
    require(
        trees["component"] == trees["preparation"] == trees["execution"],
        "protected runtime or test Git blobs differ across evidence",
    )
    provenance, _ = _read_json(provenance_path, label="formal provenance bridge")
    require(isinstance(provenance, dict), "formal provenance bridge is invalid")
    require(
        component["vllm_git"]["head"] == provenance["vllm_git"]["head"]
        and component["vllm_git"]["snapshot_sha256"]
        == provenance["vllm_git"]["snapshot_sha256"],
        "vLLM source snapshot differs across evidence",
    )
    component_python_sha = component["environment"]["vllm_python"]["sha256"]
    formal_python_sha = provenance["runtime_binaries"]["python_executable"]["sha256"]
    require(
        component_python_sha == formal_python_sha,
        "Python executable differs across evidence",
    )

    inventory_relative = _safe_relative(
        component["environment"]["vllm_distributions"]["inventory"]["path"],
        label="component distribution inventory",
    )
    component_distributions, _ = _read_json(
        component_root / inventory_relative,
        label="component vLLM distributions",
    )
    require(
        isinstance(component_distributions, dict)
        and set(component_distributions) == {"schema_version", "distributions"},
        "component distribution inventory fields differ",
    )
    formal_base = provenance["system"]["runtime_import_boundary"]["dependencies"][
        "base"
    ]
    component_pairs = _dependency_pairs(
        component_distributions["distributions"], label="component"
    )
    formal_pairs = _dependency_pairs(formal_base["packages"], label="formal base")
    require(component_pairs == formal_pairs, "normalized base dependencies differ")
    require(len(component_pairs) == 239, "normalized base dependency count differs")
    dependency_rows = [
        {"name": name, "version": version} for name, version in component_pairs
    ]
    return {
        "component_git_head": component_head,
        "formal_preparation_git_head": preparation_head,
        "formal_execution_git_head": execution_head,
        "component_is_ancestor": True,
        "execution_is_direct_child": True,
        "protected_paths": list(PROTECTED_GIT_PATHS),
        "protected_git_entry_count": len(trees["component"]),
        "protected_git_tree_sha256": _canonical_digest(trees["component"]),
        "vllm_git_head": component["vllm_git"]["head"],
        "vllm_snapshot_sha256": component["vllm_git"]["snapshot_sha256"],
        "python_sha256": component_python_sha,
        "normalized_base_dependency_count": len(dependency_rows),
        "normalized_base_dependency_sha256": _canonical_digest(dependency_rows),
        "compatible": True,
    }


def _repository_binding(repo: Path, head: str) -> dict[str, Any]:
    head = _git_head(head, label="expected repository HEAD")
    observed = _git_text(repo, "rev-parse", "HEAD")
    require(observed == head, "repository HEAD differs during publication")
    require(
        not _git_text(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "repository must be clean during publication",
    )
    bindings = _authority_bindings(repo, head)
    _validate_current_validator_closure(repo.resolve(), bindings)
    return {
        "root": str(repo.resolve()),
        "head": head,
        "clean_at_publication": True,
        "authorities": bindings,
        "canonical_evidence": _canonical_evidence_binding(repo, head),
    }


def _reconcile_canonical_evidence(
    canonical: Mapping[str, Any],
    *,
    component: Mapping[str, Any],
    component_summary: Mapping[str, Any],
    seal: Mapping[str, Any],
    formal_summary: Mapping[str, Any],
    bridge: Mapping[str, Any],
    external: Mapping[str, Any],
    provenance_path: Path,
) -> None:
    component_authority = canonical["component"]
    require(
        component_summary["root"] == component_authority["root"]
        and component_summary["manifest"]["sha256"]
        == component_authority["manifest_sha256"]
        and component_summary["sha256sums"]["sha256"]
        == component_authority["sha256sums_sha256"]
        and component["dagkv_git"]["head"] == component_authority["dagkv_head"]
        and component["dagkv_git"]["snapshot_sha256"]
        == component_authority["dagkv_snapshot_sha256"]
        and component["vllm_git"]["head"] == component_authority["vllm_head"]
        and component["vllm_git"]["snapshot_sha256"]
        == component_authority["vllm_snapshot_sha256"],
        "component evidence differs from its canonical success index",
    )
    calibration_authority = canonical["calibration"]
    formal_authority = canonical["formal"]
    require(
        Path(formal_summary["root"]).resolve()
        == Path(formal_authority["root"]).resolve()
        and formal_summary["campaign_id"] == formal_authority["campaign_id"]
        and formal_summary["seal"]["sha256"] == formal_authority["seal_sha256"]
        and formal_summary["preregistration"]["sha256"]
        == formal_authority["preregistration_sha256"],
        "formal evidence differs from its canonical success index",
    )
    require(
        Path(formal_summary["calibration_manifest"]["path"]).parent.resolve()
        == Path(calibration_authority["root"]).resolve()
        and formal_summary["calibration_manifest"]["sha256"]
        == calibration_authority["manifest_sha256"]
        and formal_summary["frozen_tolerance"]["sha256"]
        == calibration_authority["frozen_tolerance_sha256"],
        "formal parent differs from the canonical calibration index",
    )
    provenance, _ = _read_json(provenance_path, label="canonical provenance")
    require(isinstance(provenance, dict), "canonical provenance must be an object")
    source = formal_authority["source_and_runtime"]
    boundary = provenance["system"]["runtime_import_boundary"]["dependencies"]
    nvidia = provenance["nvidia_driver_userspace"]
    observed_source = {
        "implementation_manifest_sha256": provenance["implementation"][
            "manifest_sha256"
        ],
        "reproducibility_fingerprint": provenance["reproducibility_fingerprint"],
        "vllm_git_head": provenance["vllm_git"]["head"],
        "vllm_snapshot_sha256": provenance["vllm_git"]["snapshot_sha256"],
        "model_manifest_sha256": provenance["model"]["manifest_sha256"],
        "runtime_binary_manifest_sha256": provenance["runtime_binaries"][
            "manifest_sha256"
        ],
        "nvidia_driver_version": nvidia["expected_driver_version"],
        "nvidia_userspace_bundle_manifest_sha256": nvidia["expected_manifest_sha256"],
        "nvidia_userspace_bundle_content_digest": nvidia["expected_content_digest"],
        "mapped_libcuda_sha256": nvidia["libcuda_mapping"]["sha256"],
        "base_dependency_count": len(boundary["base"]["packages"]),
        "base_dependency_manifest_sha256": boundary["base"]["manifest_sha256"],
        "effective_dependency_count": boundary["effective_count"],
        "effective_dependency_manifest_sha256": boundary["effective_manifest_sha256"],
        "added_dependency_count": len(boundary["added"]),
        "added_dependency_manifest_sha256": boundary["added_manifest_sha256"],
        "removed_dependency_count": len(boundary["removed"]),
    }
    require(
        all(observed_source.get(key) == value for key, value in source.items()),
        "formal source/runtime differs from its canonical success index",
    )
    require(
        seal["implementation_manifest_sha256"]
        == source["implementation_manifest_sha256"]
        and seal["reproducibility_fingerprint"] == source["reproducibility_fingerprint"]
        and seal["nvidia_driver_version"] == source["nvidia_driver_version"]
        and seal["nvidia_userspace_bundle_manifest_sha256"]
        == source["nvidia_userspace_bundle_manifest_sha256"]
        and seal["nvidia_userspace_bundle_content_digest"]
        == source["nvidia_userspace_bundle_content_digest"],
        "formal seal differs from the canonical runtime identity",
    )
    require(
        external["model"]["manifest_sha256"] == source["model_manifest_sha256"]
        and external["runtime_binaries"]["manifest_sha256"]
        == source["runtime_binary_manifest_sha256"]
        and bridge["vllm_git_head"] == source["vllm_git_head"]
        and bridge["vllm_snapshot_sha256"] == source["vllm_snapshot_sha256"]
        and bridge["normalized_base_dependency_count"]
        == source["base_dependency_count"],
        "aggregate bridge differs from the canonical runtime identity",
    )


def _replay_payload_inputs(
    *,
    repo: Path,
    repository_head: str,
    component_root: Path,
    expected_component_manifest_sha256: str,
    expected_component_sha256sums_sha256: str,
    formal_seal: Path,
    expected_formal_seal_sha256: str,
    expected_formal_preregistration_sha256: str,
    require_clean_head: bool,
) -> dict[str, Any]:
    if require_clean_head:
        repository = _repository_binding(repo, repository_head)
    else:
        _git_head(repository_head, label="recorded repository HEAD")
        _git_bytes(repo, "cat-file", "-e", f"{repository_head}^{{commit}}")
        repository = {
            "root": str(repo.resolve()),
            "head": repository_head,
            "clean_at_publication": True,
            "authorities": _authority_bindings(repo, repository_head),
            "canonical_evidence": _canonical_evidence_binding(repo, repository_head),
        }
    canonical = repository["canonical_evidence"]
    component_authority = canonical["component"]
    formal_authority = canonical["formal"]
    require(
        component_root.expanduser().resolve()
        == Path(component_authority["root"]).resolve()
        and expected_component_manifest_sha256 == component_authority["manifest_sha256"]
        and expected_component_sha256sums_sha256
        == component_authority["sha256sums_sha256"],
        "component CLI identity differs from the canonical success index",
    )
    require(
        formal_seal.expanduser().resolve().parent
        == Path(formal_authority["root"]).resolve()
        and expected_formal_seal_sha256 == formal_authority["seal_sha256"]
        and expected_formal_preregistration_sha256
        == formal_authority["preregistration_sha256"],
        "formal CLI identity differs from the canonical success index",
    )
    component, component_summary = _component_evidence(
        component_root,
        expected_manifest_sha256=expected_component_manifest_sha256,
        expected_sha256sums_sha256=expected_component_sha256sums_sha256,
    )
    (
        seal,
        _validation,
        _prereg,
        calibration,
        provenance_path,
        formal_summary,
    ) = _formal_evidence(
        formal_seal,
        expected_seal_sha256=expected_formal_seal_sha256,
        expected_preregistration_sha256=expected_formal_preregistration_sha256,
    )
    bridge = _compatibility_bridge(
        repo,
        component,
        seal,
        provenance_path,
        component_root.expanduser().resolve(),
    )
    external = _rehash_external_content(provenance_path)
    calibration_runs = calibration.get("runs")
    formal_runs = seal.get("ordered_runs")
    require(
        isinstance(calibration_runs, list) and isinstance(formal_runs, list),
        "eligible cohort rows are missing during exclusion audit",
    )
    exclusions = _validate_excluded_attempts(
        repo,
        repository_head,
        eligible_campaign_id=seal.get("campaign_id"),
        eligible_run_ids=[
            *(row.get("run_id") for row in calibration_runs),
            *(row.get("run_id") for row in formal_runs),
        ],
    )
    _reconcile_canonical_evidence(
        canonical,
        component=component,
        component_summary=component_summary,
        seal=seal,
        formal_summary=formal_summary,
        bridge=bridge,
        external=external,
        provenance_path=provenance_path,
    )
    if require_clean_head:
        require(
            _repository_binding(repo, repository_head) == repository,
            "repository changed during aggregate input replay",
        )
    return {
        "repository": repository,
        "component_evidence": component_summary,
        "formal_evidence": formal_summary,
        "compatibility_bridge": bridge,
        "external_content": external,
        "excluded_attempts": exclusions,
        "conditions": _condition_payloads(),
    }


def _validate_repository_record(repository: Mapping[str, Any]) -> tuple[Path, str]:
    require(
        set(repository)
        == {
            "root",
            "head",
            "clean_at_publication",
            "authorities",
            "canonical_evidence",
        },
        "repository binding fields differ",
    )
    root = Path(repository["root"])
    require(
        root.is_absolute() and root.resolve() == REPO_ROOT, "repository root differs"
    )
    head = _git_head(repository["head"], label="repository HEAD")
    require(repository["clean_at_publication"] is True, "publication was not clean")
    expected = _authority_bindings(root, head)
    require(repository["authorities"] == expected, "repository authorities differ")
    require(
        repository["canonical_evidence"] == _canonical_evidence_binding(root, head),
        "canonical evidence authority differs",
    )
    _validate_current_validator_closure(root, expected)
    return root, head


def _validate_closed_set(path: Path, *, expected_root_mode: int = 0o555) -> None:
    require(
        path.name == ACCEPTANCE_NAME, f"acceptance file must be named {ACCEPTANCE_NAME}"
    )
    require(not path.is_symlink(), "acceptance file cannot be a symlink")
    root = path.parent
    require(root.is_dir() and not root.is_symlink(), "acceptance root is invalid")
    entries = sorted(item.name for item in root.iterdir())
    require(
        entries == [ACCEPTANCE_NAME],
        "acceptance directory is not a one-file closed set",
    )
    file_stat = path.stat(follow_symlinks=False)
    root_stat = root.stat(follow_symlinks=False)
    require(stat.S_ISREG(file_stat.st_mode), "acceptance is not a regular file")
    require(file_stat.st_nlink == 1, "acceptance must have one hard link")
    require(stat.S_IMODE(file_stat.st_mode) == 0o444, "acceptance mode must be 0444")
    require(
        stat.S_IMODE(root_stat.st_mode) == expected_root_mode,
        f"acceptance root mode must be {expected_root_mode:04o}",
    )


def _validate_aggregate_acceptance_unlocked(
    acceptance_path: Path,
    *,
    expected_acceptance_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Recompute inputs for a sealed staging file or while its lock is held."""

    acceptance_path = acceptance_path.expanduser()
    if not acceptance_path.is_absolute():
        acceptance_path = acceptance_path.absolute()
    _validate_closed_set(acceptance_path)
    payload, raw = _read_json(acceptance_path, label="M2 aggregate acceptance")
    require(isinstance(payload, dict), "aggregate acceptance must be an object")
    _exact_fields(payload, TOP_LEVEL_FIELDS, label="aggregate acceptance")
    observed_sha = _sha256_bytes(raw)
    require(
        observed_sha
        == _lower_sha256(expected_acceptance_sha256, label="expected acceptance"),
        "aggregate acceptance SHA differs",
    )
    require(payload["schema_version"] == SCHEMA_VERSION, "aggregate schema differs")
    _timestamp(payload["accepted_at_utc"], label="acceptance timestamp")
    require(payload["gate_status"] == GATE_STATUS, "aggregate gate status differs")
    require(payload["verification_status"] == "VERIFIED", "aggregate is unverified")
    require(payload["m2_item8_accepted"] is True, "item 8 is not accepted")
    require(payload["m2_accepted"] is True, "M2 is not accepted")
    require(
        payload["performance_claims_supported"] is False,
        "performance claim leaked into M2",
    )
    require(payload["policy_claims_supported"] is False, "policy claim leaked into M2")
    require(payload["claim_scope"] == CLAIM_SCOPE, "aggregate claim scope differs")
    require(payload["statement"] == STATEMENT, "aggregate statement differs")
    repo, head = _validate_repository_record(payload["repository"])
    component = payload["component_evidence"]
    formal = payload["formal_evidence"]
    replayed = _replay_payload_inputs(
        repo=repo,
        repository_head=head,
        component_root=Path(component["root"]),
        expected_component_manifest_sha256=component["manifest"]["sha256"],
        expected_component_sha256sums_sha256=component["sha256sums"]["sha256"],
        formal_seal=Path(formal["seal"]["path"]),
        expected_formal_seal_sha256=formal["seal"]["sha256"],
        expected_formal_preregistration_sha256=formal["preregistration"]["sha256"],
        require_clean_head=False,
    )
    for key in (
        "repository",
        "component_evidence",
        "formal_evidence",
        "compatibility_bridge",
        "external_content",
        "excluded_attempts",
        "conditions",
    ):
        require(payload[key] == replayed[key], f"aggregate {key} differs from replay")
    require(
        acceptance_path.read_bytes() == raw,
        "aggregate acceptance changed during replay",
    )
    _validate_closed_set(acceptance_path)
    return payload, observed_sha


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publication_lock_path(output_dir: Path) -> Path:
    return output_dir.with_name(f".{output_dir.name}{PUBLICATION_LOCK_SUFFIX}")


def _create_publication_lock(output_dir: Path) -> Any:
    path = _publication_lock_path(output_dir)
    try:
        handle = path.open("x+b")
    except FileExistsError as exc:
        raise AggregateAcceptanceError(
            f"publication lock already exists: {path}"
        ) from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(LOCK_PREPARING)
        handle.flush()
        os.fsync(handle.fileno())
        path.chmod(0o444)
        os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        handle.close()
        if os.path.lexists(path):
            path.unlink()
            _fsync_directory(path.parent)
        raise
    return handle


def _mark_publication_complete(handle: Any) -> None:
    handle.seek(0)
    handle.write(LOCK_PUBLISHED)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


def _acquire_published_lock(output_dir: Path) -> tuple[Any, tuple[int, ...]]:
    path = _publication_lock_path(output_dir)
    require(path.is_file() and not path.is_symlink(), "publication lock is missing")
    observed = path.lstat()
    require(stat.S_ISREG(observed.st_mode), "publication lock is not regular")
    require(observed.st_nlink == 1, "publication lock must have one hard link")
    require(stat.S_IMODE(observed.st_mode) == 0o444, "publication lock mode differs")
    try:
        handle = path.open("rb")
        opened = os.fstat(handle.fileno())
        require(
            _file_object_identity(opened) == _file_object_identity(observed),
            "publication lock changed while opened",
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        locked = os.fstat(handle.fileno())
        path_locked = path.lstat()
        require(
            _file_object_identity(locked) == _file_object_identity(path_locked),
            "publication lock path changed while waiting",
        )
        state = handle.read()
        after = os.fstat(handle.fileno())
        path_after = path.lstat()
        require(
            _stat_identity(locked)
            == _stat_identity(after)
            == _stat_identity(path_after),
            "publication lock changed while read",
        )
        require(state == LOCK_PUBLISHED, "publication is incomplete")
    except BaseException:
        if "handle" in locals():
            handle.close()
        raise
    return handle, _stat_identity(after)


def _validate_published_lock_unchanged(
    output_dir: Path, handle: Any, expected_identity: tuple[int, ...]
) -> None:
    path = _publication_lock_path(output_dir)
    try:
        before = os.fstat(handle.fileno())
        handle.seek(0)
        state = handle.read()
        after = os.fstat(handle.fileno())
        path_after = path.lstat()
    except OSError as exc:
        raise AggregateAcceptanceError(
            f"publication lock unavailable after aggregate replay: {exc}"
        ) from exc
    require(
        expected_identity
        == _stat_identity(before)
        == _stat_identity(after)
        == _stat_identity(path_after),
        "publication lock changed during aggregate replay",
    )
    require(state == LOCK_PUBLISHED, "publication became incomplete during replay")


def validate_aggregate_acceptance(
    acceptance_path: Path,
    *,
    expected_acceptance_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Recompute all aggregate inputs under the durable publication lock."""

    acceptance_path = acceptance_path.expanduser()
    if not acceptance_path.is_absolute():
        acceptance_path = acceptance_path.absolute()
    lock, lock_identity = _acquire_published_lock(acceptance_path.parent)
    try:
        result = _validate_aggregate_acceptance_unlocked(
            acceptance_path,
            expected_acceptance_sha256=expected_acceptance_sha256,
        )
        _validate_published_lock_unchanged(acceptance_path.parent, lock, lock_identity)
        return result
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()


def _publish_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_bytes(payload, pretty=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _remove_staging_directory(staging: Path) -> None:
    if not os.path.lexists(staging):
        return
    require(staging.is_dir() and not staging.is_symlink(), "staging root is invalid")
    staging.chmod(0o755)
    entries = list(staging.iterdir())
    require(
        all(item.name == ACCEPTANCE_NAME and item.is_file() for item in entries),
        "staging root contains an unexpected entry",
    )
    for item in entries:
        item.unlink()
    staging.rmdir()


def _quarantine_failed_output(output_dir: Path) -> None:
    if not os.path.lexists(output_dir):
        return
    require(
        output_dir.is_dir() and not output_dir.is_symlink(),
        "failed output root is invalid",
    )
    output_dir.chmod(0o755)
    destination = output_dir / ACCEPTANCE_NAME
    candidates = [
        item
        for item in output_dir.iterdir()
        if item.name.startswith(".M2_AGGREGATE_ACCEPTANCE.")
        and item.name.endswith(".candidate")
    ]
    require(len(candidates) <= 1, "failed output has multiple candidates")
    visible = destination if os.path.lexists(destination) else None
    source = visible or (candidates[0] if candidates else None)
    if source is not None:
        invalid = output_dir / INVALID_ACCEPTANCE_NAME
        require(not os.path.lexists(invalid), "invalid acceptance already exists")
        os.rename(source, invalid)
        _fsync_directory(output_dir)
        invalid.chmod(0o444)
        _fsync_file(invalid)
    output_dir.chmod(0o555)
    _fsync_directory(output_dir)
    _fsync_directory(output_dir.parent)


def _commit_staged_acceptance(staging: Path, output_dir: Path) -> Path:
    source = staging / ACCEPTANCE_NAME
    require(
        source.is_file() and not source.is_symlink(), "staged acceptance is missing"
    )
    output_dir.mkdir(mode=0o755)
    candidate = output_dir / f".M2_AGGREGATE_ACCEPTANCE.{uuid.uuid4().hex}.candidate"
    destination = output_dir / ACCEPTANCE_NAME
    try:
        os.link(source, candidate)
        _fsync_directory(output_dir)
        staging.chmod(0o755)
        source.unlink()
        _fsync_directory(staging)
        _fsync_file(candidate)
        staging.rmdir()
        _fsync_directory(output_dir.parent)
        require(
            candidate.stat().st_nlink == 1, "final candidate hard-link count differs"
        )
        os.rename(candidate, destination)
        _fsync_directory(output_dir)
        _fsync_directory(output_dir.parent)
    except BaseException:
        _quarantine_failed_output(output_dir)
        if os.path.lexists(staging):
            _remove_staging_directory(staging)
        raise
    return destination


def publish_aggregate_acceptance(
    output_dir: Path,
    *,
    component_evidence_dir: Path,
    expected_component_manifest_sha256: str,
    expected_component_sha256sums_sha256: str,
    formal_seal: Path,
    expected_formal_seal_sha256: str,
    expected_formal_preregistration_sha256: str,
    expected_repository_head: str,
    expected_stage_gates_sha256: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Replay every gate and publish one create-only aggregate decision."""

    output_dir = output_dir.expanduser()
    require(output_dir.is_absolute(), "output directory must be absolute")
    output_dir = output_dir.absolute()
    require(
        not os.path.lexists(output_dir),
        f"output directory already exists: {output_dir}",
    )
    require(output_dir.parent.is_dir(), "output parent directory is missing")
    require(not output_dir.parent.is_symlink(), "output parent cannot be a symlink")
    publication_lock = _create_publication_lock(output_dir)
    publication_succeeded = False
    staging: Path | None = None
    try:
        stage_sha = _sha256_file(REPO_ROOT / STAGE_GATES_PATH)
        protocol_sha = _sha256_file(REPO_ROOT / PROTOCOL_PATH)
        require(
            stage_sha
            == _lower_sha256(expected_stage_gates_sha256, label="expected stage gates"),
            "stage-gate SHA differs",
        )
        require(
            protocol_sha
            == _lower_sha256(
                expected_protocol_sha256, label="expected aggregate protocol"
            ),
            "aggregate protocol SHA differs",
        )
        replayed = _replay_payload_inputs(
            repo=REPO_ROOT,
            repository_head=expected_repository_head,
            component_root=component_evidence_dir,
            expected_component_manifest_sha256=expected_component_manifest_sha256,
            expected_component_sha256sums_sha256=(expected_component_sha256sums_sha256),
            formal_seal=formal_seal,
            expected_formal_seal_sha256=expected_formal_seal_sha256,
            expected_formal_preregistration_sha256=(
                expected_formal_preregistration_sha256
            ),
            require_clean_head=True,
        )
        authorities = {
            item["repository_path"]: item
            for item in replayed["repository"]["authorities"]
        }
        require(
            authorities[STAGE_GATES_PATH]["sha256"] == stage_sha,
            "committed stage gates differ",
        )
        require(
            authorities[PROTOCOL_PATH]["sha256"] == protocol_sha,
            "committed aggregate protocol differs",
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "accepted_at_utc": datetime.now(UTC).isoformat(),
            "gate_status": GATE_STATUS,
            "verification_status": "VERIFIED",
            **replayed,
            "m2_item8_accepted": True,
            "m2_accepted": True,
            "performance_claims_supported": False,
            "policy_claims_supported": False,
            "claim_scope": CLAIM_SCOPE,
            "statement": STATEMENT,
        }
        staging = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
        staging.mkdir(mode=0o755)
        staged_acceptance = staging / ACCEPTANCE_NAME
        _publish_exclusive(staged_acceptance, payload)
        staged_acceptance.chmod(0o444)
        _fsync_file(staged_acceptance)
        staging.chmod(0o555)
        _fsync_directory(staging)
        acceptance_sha = _sha256_file(staged_acceptance)
        published, observed_sha = _validate_aggregate_acceptance_unlocked(
            staged_acceptance,
            expected_acceptance_sha256=acceptance_sha,
        )
        require(
            published == payload and observed_sha == acceptance_sha,
            "staged acceptance did not replay exactly",
        )
        require(
            _repository_binding(REPO_ROOT, expected_repository_head)
            == replayed["repository"],
            "repository changed before aggregate publication",
        )
        destination = _commit_staged_acceptance(staging, output_dir)
        require(
            _repository_binding(REPO_ROOT, expected_repository_head)
            == replayed["repository"],
            "repository changed across aggregate publication",
        )
        require(
            _sha256_file(destination) == acceptance_sha,
            "final acceptance SHA differs",
        )
        _validate_closed_set(destination, expected_root_mode=0o755)
        output_dir.chmod(0o555)
        _fsync_directory(output_dir)
        _fsync_directory(output_dir.parent)
        _mark_publication_complete(publication_lock)
        publication_succeeded = True
        return payload, acceptance_sha
    except BaseException:
        if os.path.lexists(output_dir):
            with suppress(AggregateAcceptanceError, OSError):
                _quarantine_failed_output(output_dir)
        if staging is not None and os.path.lexists(staging):
            with suppress(AggregateAcceptanceError, OSError):
                _remove_staging_directory(staging)
        raise
    finally:
        with suppress(OSError):
            fcntl.flock(publication_lock.fileno(), fcntl.LOCK_UN)
        with suppress(OSError):
            publication_lock.close()
        if not publication_succeeded and not os.path.lexists(output_dir):
            lock_path = _publication_lock_path(output_dir)
            with suppress(OSError):
                lock_path.unlink()
                _fsync_directory(lock_path.parent)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish", help="publish a create-only M2 decision")
    publish.add_argument("--output-dir", required=True, type=Path)
    publish.add_argument("--component-evidence-dir", required=True, type=Path)
    publish.add_argument("--expected-component-manifest-sha256", required=True)
    publish.add_argument("--expected-component-sha256sums-sha256", required=True)
    publish.add_argument("--formal-seal", required=True, type=Path)
    publish.add_argument("--expected-formal-seal-sha256", required=True)
    publish.add_argument("--expected-formal-preregistration-sha256", required=True)
    publish.add_argument("--expected-repository-head", required=True)
    publish.add_argument("--expected-stage-gates-sha256", required=True)
    publish.add_argument("--expected-protocol-sha256", required=True)
    validate = subparsers.add_parser("validate", help="replay a published M2 decision")
    validate.add_argument("acceptance", type=Path)
    validate.add_argument("--expected-acceptance-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "publish":
            payload, acceptance_sha = publish_aggregate_acceptance(
                args.output_dir,
                component_evidence_dir=args.component_evidence_dir,
                expected_component_manifest_sha256=(
                    args.expected_component_manifest_sha256
                ),
                expected_component_sha256sums_sha256=(
                    args.expected_component_sha256sums_sha256
                ),
                formal_seal=args.formal_seal,
                expected_formal_seal_sha256=args.expected_formal_seal_sha256,
                expected_formal_preregistration_sha256=(
                    args.expected_formal_preregistration_sha256
                ),
                expected_repository_head=args.expected_repository_head,
                expected_stage_gates_sha256=args.expected_stage_gates_sha256,
                expected_protocol_sha256=args.expected_protocol_sha256,
            )
            destination = args.output_dir / ACCEPTANCE_NAME
            print(
                f"M2 aggregate acceptance published: {destination} "
                f"sha256={acceptance_sha} conditions={len(payload['conditions'])}"
            )
        else:
            payload, acceptance_sha = validate_aggregate_acceptance(
                args.acceptance,
                expected_acceptance_sha256=args.expected_acceptance_sha256,
            )
            print(
                f"M2 aggregate replay passed: {payload['gate_status']} "
                f"sha256={acceptance_sha}"
            )
    except (
        AggregateAcceptanceError,
        ComponentEvidenceError,
        FormalEvidenceError,
        OSError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"M2 aggregate acceptance failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
