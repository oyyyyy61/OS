#!/usr/bin/env python3
"""Create and independently replay a clean-source C1-B0 evidence bundle."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from dagkv.c1_bundle import (
    C1_B0_OPEN_GATES,
    FINAL_INVENTORY,
    FINAL_SEAL,
    LIFECYCLE_PAYLOAD,
    MANIFEST,
    SCHEDULE_PAYLOAD,
    TRACE_PAYLOAD,
    ValidatedC1B0Bundle,
    finalize_c1_b0_bundle,
    validate_c1_b0_bundle,
)
from dagkv.c1_commit import (
    AttemptPayload,
    CanonicalTraceCommitter,
    DemandCommitRequest,
    ObservationCloseRequest,
    ObservationTerminalSpec,
    TraceEnvelope,
    TracePreambleRequest,
)
from dagkv.c1_leases import (
    DependenceGroup,
    ForecastSource,
    JointOutcome,
    ReuseClaim,
    SharedLeaseForecast,
)
from dagkv.c1_lifecycle import (
    LIFECYCLE_CLOCK_DOMAIN,
    LIFECYCLE_SIDECAR_SCHEMA_VERSION,
    ClosedLifecycleArtifact,
    make_lifecycle_closure,
    write_lifecycle_artifact,
)
from dagkv.c1_schedule import (
    SCHEDULE_CLOCK_DOMAIN,
    SCHEDULE_EVENT_ORDER_RULE,
    SCHEDULE_SIDECAR_SCHEMA_VERSION,
    ClosedScheduleArtifact,
    ReplayScheduleClosure,
    ScheduleDemandEvent,
    ScheduleEpoch,
    make_schedule_checkpoint,
    schedule_stream_digest,
    write_schedule_artifact,
)
from dagkv.c1_trace import (
    AbstainedAttemptPayload,
    AbstentionReason,
    AtomicCutoffView,
    ForecastAttemptContext,
    ForecastAttemptStatus,
    PredictedAttemptPayload,
    ReplayScheduleWatermarkPayload,
    ResidentExecMapService,
    ScheduleProducerKind,
    ServiceDisposition,
    TerminalReason,
    TerminalStatus,
    TraceHeaderPayload,
    TraceValidationError,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
)
from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    LedgerAction,
    ReplicaId,
    Tier,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)
from dagkv.orchestrator import LifecycleOrchestrator
from tools import run_m3_c1_component_evidence as common

SCHEMA_VERSION = "dagkv.m3.c1_b0_stage_evidence.v1"
PRODUCER_RESULT_SCHEMA = "dagkv.m3.c1_b0_inner_producer_result.v1"
REPLAY_RESULT_SCHEMA = "dagkv.m3.c1_b0_inner_replay_result.v1"
MANIFEST_NAME = "M3_C1_B0_STAGE_EVIDENCE.json"
CHECKSUM_NAME = common.CHECKSUM_NAME
PUBLICATION_LOCK_SUFFIX = ".m3-c1-b0-publication.lock"
LOCK_PREPARING = b"PREPARING\n"
LOCK_PUBLISHED = b"PUBLISHED\n"
REPO_ROOT = common.REPO_ROOT
DEFAULT_TIMEOUT_SECONDS = 900
EXPECTED_FOCUSED_TESTS = 192
EXPECTED_FOCUSED_IDENTITIES_SHA256 = (
    "2f224769a23cdec169de0a903cc08011e2c841758dab5dd3478cf388ee3a1f49"
)
EXPECTED_FULL_TESTS = 575
EXPECTED_FULL_IDENTITIES_SHA256 = (
    "3375e800db068c510a777cbc1f4d0de247eaa5a9a40fba56e243ac0ae212a528"
)

STAGE_STATUS = "C1_B0_STAGE_EVIDENCE_VERIFIED"
ACCEPTED_GATE_STATUS = "C1_B0_STAGE_ACCEPTED"
CLAIM_SCOPE = (
    "C1-B0 schema, durable operation ordering, lifecycle/schedule closure, and "
    "demand-label reconstruction only. No branch-grammar completeness, leakage, "
    "calibration, policy benefit, latency, throughput, GPU, or novelty claim."
)

PROTOCOL_PATH = "research/protocols/M3_C1_TRACE_CALIBRATION_PROTOCOL.md"
DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")
IMPLEMENTATION_PATHS = (
    "src/dagkv/__init__.py",
    "src/dagkv/domain.py",
    "src/dagkv/ledger.py",
    "src/dagkv/c1_leases.py",
    "src/dagkv/c1_schedule.py",
    "src/dagkv/c1_lifecycle.py",
    "src/dagkv/c1_trace.py",
    "src/dagkv/c1_commit.py",
    "src/dagkv/c1_bundle.py",
    "src/dagkv/orchestrator.py",
    "tools/run_m3_c1_component_evidence.py",
    "tools/run_m3_c1_b0_evidence.py",
)
VERIFIER_PATHS = (
    "src/dagkv/domain.py",
    "src/dagkv/ledger.py",
    "src/dagkv/c1_schedule.py",
    "src/dagkv/c1_lifecycle.py",
    "src/dagkv/c1_trace.py",
    "src/dagkv/c1_commit.py",
    "src/dagkv/c1_bundle.py",
    "tools/run_m3_c1_component_evidence.py",
    "tools/run_m3_c1_b0_evidence.py",
    "tests/test_c1_trace.py",
    "tests/test_c1_trace_adversarial.py",
    "tests/test_c1_schedule.py",
    "tests/test_c1_lifecycle.py",
    "tests/test_c1_commit.py",
    "tests/test_c1_formal_runtime.py",
    "tests/test_c1_bundle.py",
    "tests/test_m3_c1_b0_evidence.py",
)
SOURCE_PATHS = tuple(
    sorted(
        {
            PROTOCOL_PATH,
            "README.md",
            "research/ARCHITECTURE.md",
            "research/REFERENCES.md",
            "research/STAGE_GATES.md",
            "research/imported/RELATED_WORK_MATRIX.md",
            "evidence/m3/c1/M3_C1_COMPONENT_EVIDENCE_INDEX.json",
            *DEPENDENCY_PATHS,
            *IMPLEMENTATION_PATHS,
            *VERIFIER_PATHS,
        }
    )
)
FOCUSED_TEST_PATHS = (
    "tests/test_c1_trace.py",
    "tests/test_c1_trace_adversarial.py",
    "tests/test_c1_schedule.py",
    "tests/test_c1_lifecycle.py",
    "tests/test_c1_commit.py",
    "tests/test_c1_formal_runtime.py",
    "tests/test_c1_bundle.py",
    "tests/test_m3_c1_b0_evidence.py",
)
EXECUTED_MODULE_PATHS = {
    "dagkv": "src/dagkv/__init__.py",
    "dagkv.c1_bundle": "src/dagkv/c1_bundle.py",
    "dagkv.c1_commit": "src/dagkv/c1_commit.py",
    "dagkv.c1_leases": "src/dagkv/c1_leases.py",
    "dagkv.c1_lifecycle": "src/dagkv/c1_lifecycle.py",
    "dagkv.c1_schedule": "src/dagkv/c1_schedule.py",
    "dagkv.c1_trace": "src/dagkv/c1_trace.py",
    "dagkv.domain": "src/dagkv/domain.py",
    "dagkv.ledger": "src/dagkv/ledger.py",
    "dagkv.orchestrator": "src/dagkv/orchestrator.py",
    "tools.run_m3_c1_b0_evidence": "tools/run_m3_c1_b0_evidence.py",
    "tools.run_m3_c1_component_evidence": "tools/run_m3_c1_component_evidence.py",
}

SCENARIO_CONTRACT: dict[str, Any] = {
    "schema_version": "dagkv.m3.c1_b0_controlled_scenario.v1",
    "scenario_id": "c1-b0-resident-and-abstention",
    "clock_domain": SCHEDULE_CLOCK_DOMAIN,
    "cutoff_ns": 5,
    "predicted_deadline_ns": 15,
    "abstained_deadline_ns": 7,
    "scheduled_access_ns": 8,
    "schedule_closed_through_ns": 16,
    "branch_grammar": {
        "scope": "single declared-independent group",
        "outcomes": ["demand", "no-demand"],
        "b1_completeness_claimed": False,
    },
    "feature_contract": {
        "scope": "deterministic component fixture",
        "future_fields_allowed": False,
        "b1_leakage_acceptance_claimed": False,
    },
    "split_contract": {
        "component_id": "c1-b0-controlled-component",
        "role": "component-only",
        "b1_role_assignment_claimed": False,
    },
}


class C1B0EvidenceError(RuntimeError):
    """Raised when C1-B0 stage evidence cannot be created or replayed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise C1B0EvidenceError(message)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _named_digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


def _mapping_digest(value: Mapping[str, Any]) -> str:
    return sha256(common._canonical_json(value)).hexdigest()


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=common._strict_object,
            parse_constant=common._reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, common.C1EvidenceError) as exc:
        raise C1B0EvidenceError(f"invalid {label}: {exc}") from exc


def _json_value(value: Any) -> Any:
    return _strict_json_bytes(canonical_json(value), label="canonical snapshot")


def _capture_distribution_inventory(python: Path) -> dict[str, Any]:
    probe_source = """
import importlib.metadata as metadata
import json
import re
import sys

rows = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not isinstance(name, str) or not name.strip():
        raise SystemExit("installed distribution lacks a Name")
    if not isinstance(version, str) or not version.strip():
        raise SystemExit(f"installed distribution lacks a version: {name}")
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    rows.append({"name": name, "normalized_name": normalized, "version": version})
rows.sort(key=lambda row: (row["normalized_name"], row["name"], row["version"]))
normalized_names = [row["normalized_name"] for row in rows]
if len(normalized_names) != len(set(normalized_names)):
    raise SystemExit("duplicate installed distribution identity")
payload = {
    "schema_version": "dagkv.python_distribution_inventory.v1",
    "distributions": rows,
    "sys_path": list(sys.path),
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
""".strip()
    result = subprocess.run(
        [str(python), "-I", "-c", probe_source],
        env=dict(common.BASE_ENVIRONMENT),
        capture_output=True,
        timeout=30,
        check=False,
    )
    require(
        result.returncode == 0,
        "installed-distribution probe failed: "
        f"{result.stderr.decode(errors='replace')[-1000:].strip()}",
    )
    value = _strict_json_bytes(result.stdout, label="distribution inventory")
    require(
        isinstance(value, dict)
        and set(value) == {"schema_version", "distributions", "sys_path"}
        and value["schema_version"] == "dagkv.python_distribution_inventory.v1",
        "distribution inventory fields differ",
    )
    distributions = value["distributions"]
    require(
        isinstance(distributions, list) and bool(distributions),
        "distribution inventory is empty",
    )
    previous: tuple[str, str, str] | None = None
    normalized_names: set[str] = set()
    for row in distributions:
        require(
            isinstance(row, dict)
            and set(row) == {"name", "normalized_name", "version"}
            and all(isinstance(item, str) and item for item in row.values()),
            "distribution inventory row differs",
        )
        identity = (row["normalized_name"], row["name"], row["version"])
        require(previous is None or previous < identity, "distribution order differs")
        require(
            row["normalized_name"] not in normalized_names,
            "duplicate normalized distribution identity",
        )
        previous = identity
        normalized_names.add(row["normalized_name"])
    require(
        isinstance(value["sys_path"], list)
        and all(isinstance(item, str) for item in value["sys_path"]),
        "distribution sys.path differs",
    )
    return value


def _module_origin_binding(entries: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    payload = {"entries": list(entries)}
    return {**payload, "digest": _mapping_digest(payload)}


def _validate_module_origin_entries(value: Any) -> list[dict[str, str]]:
    require(isinstance(value, list), "module origin entries must be an array")
    require(len(value) == len(EXECUTED_MODULE_PATHS), "module origin count differs")
    entries: list[dict[str, str]] = []
    for index, (expected_module, expected_relative) in enumerate(
        sorted(EXECUTED_MODULE_PATHS.items())
    ):
        entry = value[index]
        require(
            isinstance(entry, dict)
            and set(entry) == {"module", "path", "sha256"}
            and entry["module"] == expected_module
            and entry["path"] == str((REPO_ROOT / expected_relative).resolve())
            and isinstance(entry["sha256"], str)
            and len(entry["sha256"]) == 64
            and all(character in "0123456789abcdef" for character in entry["sha256"]),
            f"module origin differs: {expected_module}",
        )
        entries.append(entry)
    return entries


def _current_module_origin_binding() -> dict[str, Any]:
    entries: list[dict[str, str]] = []
    for module_name, expected_relative in sorted(EXECUTED_MODULE_PATHS.items()):
        module = importlib.import_module(module_name)
        origin = getattr(module, "__file__", None)
        require(
            isinstance(origin, str) and origin, f"module lacks origin: {module_name}"
        )
        resolved = Path(origin).resolve(strict=True)
        require(
            resolved == (REPO_ROOT / expected_relative).resolve(strict=True)
            and resolved.is_file()
            and not resolved.is_symlink(),
            f"module loaded outside bound source: {module_name}",
        )
        entries.append(
            {
                "module": module_name,
                "path": str(resolved),
                "sha256": common._sha256_file(resolved),
            }
        )
    return _module_origin_binding(entries)


def _capture_module_origin_binding(python: Path) -> dict[str, Any]:
    expected_json = json.dumps(
        EXECUTED_MODULE_PATHS,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    probe_source = f"""
import importlib
import hashlib
import json
from pathlib import Path

repo_root = Path({str(REPO_ROOT)!r})
expected = json.loads({expected_json!r})
entries = []
for module_name, relative in sorted(expected.items()):
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise SystemExit(f"module lacks origin: {{module_name}}")
    resolved = Path(origin).resolve(strict=True)
    expected_path = (repo_root / relative).resolve(strict=True)
    if resolved != expected_path or not resolved.is_file() or resolved.is_symlink():
        raise SystemExit(f"module loaded outside bound source: {{module_name}}")
    entries.append({{
        "module": module_name,
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }})
print(json.dumps({{"entries": entries}}, sort_keys=True, separators=(",", ":")))
""".strip()
    result = subprocess.run(
        [str(python), "-c", probe_source],
        cwd=REPO_ROOT,
        env=dict(common.BASE_ENVIRONMENT),
        capture_output=True,
        timeout=30,
        check=False,
    )
    require(
        result.returncode == 0,
        "module-origin probe failed: "
        f"{result.stderr.decode(errors='replace')[-1000:].strip()}",
    )
    value = _strict_json_bytes(result.stdout, label="module origin probe")
    require(
        isinstance(value, dict) and set(value) == {"entries"},
        "module origin probe fields differ",
    )
    entries = _validate_module_origin_entries(value["entries"])
    return _module_origin_binding(entries)


def _capture_environment_binding(
    python: Path,
    dependency_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "python": common._capture_python_binding(python),
        "command_environment": dict(common.BASE_ENVIRONMENT),
        "distributions": _capture_distribution_inventory(python),
        "module_origins": _capture_module_origin_binding(python),
        "dependencies": list(dependency_entries),
    }
    return {**payload, "digest": _mapping_digest(payload)}


def _repository_state() -> tuple[str, str, str, str]:
    return (
        common._git_text("branch", "--show-current"),
        common._git_text("rev-parse", "HEAD"),
        common._git_text("show", "-s", "--format=%T", "HEAD"),
        common._git_text("status", "--porcelain=v1", "--untracked-files=all"),
    )


def _assert_repository_binding(*, head: str, tree: str) -> None:
    branch, observed_head, observed_tree, status = _repository_state()
    require(branch == "main", "C1-B0 evidence requires branch main")
    require(observed_head == head, "repository HEAD changed during evidence run")
    require(observed_tree == tree, "repository tree changed during evidence run")
    require(not status, "repository changed during evidence run")


def _stable_sha256(path: Path, *, label: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    before = path.stat()
    first = path.read_bytes()
    middle = path.stat()
    second = path.read_bytes()
    after = path.stat()

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            stat.S_IMODE(item.st_mode),
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    require(
        identity(before) == identity(middle) == identity(after)
        and first == second
        and before.st_nlink == 1,
        f"{label} changed during stable read",
    )
    return common._sha256_bytes(first)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    require(root.is_dir() and not root.is_symlink(), "cleanup root is invalid")
    for directory, child_directories, filenames in os.walk(
        root, topdown=False, followlinks=False
    ):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if not path.is_symlink():
                os.chmod(path, 0o600)
        for child in child_directories:
            path = directory_path / child
            if not path.is_symlink():
                os.chmod(path, 0o700)
        os.chmod(directory_path, 0o700)
    shutil.rmtree(root)
    require(not root.exists(), f"cleanup left residual tree: {root}")


def _source_entry(head: str, relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
    blob = common._git("show", f"{head}:{relative}").stdout
    require(blob == path.read_bytes(), f"working source differs from HEAD: {relative}")
    return {
        "path": relative,
        "size": len(blob),
        "sha256": common._sha256_bytes(blob),
    }


def _binding_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return _mapping_digest({"entries": list(entries)})


def _test_identity_digest(summary: Mapping[str, Any]) -> str:
    identities = summary.get("identities")
    require(
        isinstance(identities, list)
        and summary.get("tests") == len(identities)
        and identities == sorted(identities)
        and len(identities) == len(set(identities))
        and all(isinstance(identity, str) and identity for identity in identities),
        "JUnit testcase identities are invalid",
    )
    return _mapping_digest({"identities": identities})


def _select_entries(
    entries_by_path: Mapping[str, dict[str, Any]],
    paths: Sequence[str],
) -> list[dict[str, Any]]:
    return [entries_by_path[path] for path in paths]


def _capture_anchors(
    head: str,
    python: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_entries = [_source_entry(head, path) for path in SOURCE_PATHS]
    entries_by_path = {entry["path"]: entry for entry in source_entries}
    implementation_entries = _select_entries(entries_by_path, IMPLEMENTATION_PATHS)
    verifier_entries = _select_entries(entries_by_path, VERIFIER_PATHS)
    dependency_entries = _select_entries(entries_by_path, DEPENDENCY_PATHS)
    environment = _capture_environment_binding(python, dependency_entries)
    require(
        _current_module_origin_binding() == environment["module_origins"],
        "launcher modules differ from the bound execution source",
    )
    anchors = {
        "protocol": entries_by_path[PROTOCOL_PATH],
        "implementation": {
            "paths": list(IMPLEMENTATION_PATHS),
            "entries": implementation_entries,
            "digest": _binding_digest(implementation_entries),
        },
        "verifier": {
            "paths": list(VERIFIER_PATHS),
            "entries": verifier_entries,
            "digest": _binding_digest(verifier_entries),
        },
        "environment": environment,
    }
    return anchors, source_entries


def _capture_source_archive(
    root: Path,
    *,
    head: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    archive = common._git(
        "archive",
        "--format=tar",
        head,
        "--",
        *SOURCE_PATHS,
    ).stdout
    archive_path = root / "source" / "c1-b0-source.tar"
    common._write_new(archive_path, archive)
    return {
        "paths": list(SOURCE_PATHS),
        "entries": entries,
        "archive": common._file_entry(archive_path, root=root),
    }


def _contract_digest(section: str) -> str:
    value = SCENARIO_CONTRACT[section]
    require(isinstance(value, dict), f"scenario contract section is invalid: {section}")
    return _mapping_digest({section: value})


@dataclass(frozen=True, slots=True)
class _Scenario:
    runtime: LifecycleOrchestrator
    committer: CanonicalTraceCommitter
    schedule_digest: str
    schedule_event: ScheduleDemandEvent
    request_handle: BindingHandle
    target_replica: ReplicaId
    predicted_observation_id: str
    abstained_observation_id: str


def _schedule(
    *,
    block_key: BlockKey,
    workflow: WorkflowSpec,
    retention: BindingHandle,
    request: BindingHandle,
    execution: ExecutionRef,
) -> ClosedScheduleArtifact:
    event = ScheduleDemandEvent(
        event_ordinal=0,
        schedule_event_id="c1-b0-schedule-event-1",
        scheduled_access_ns=SCENARIO_CONTRACT["scheduled_access_ns"],
        claim_id="c1-b0-claim-1",
        retention_binding_id=retention.binding_id,
        request_binding_id=request.binding_id,
        workflow=workflow.key,
        node_id="agent",
        execution_ref=execution,
        block_key=block_key,
        reuse_epoch_id="c1-b0-epoch-1",
        source_record_id="c1-b0-plan-record-1",
        source_record_digest=_named_digest("c1-b0-plan-record-1"),
    )
    epoch = ScheduleEpoch(
        reuse_epoch_id=event.reuse_epoch_id,
        access_ns=event.scheduled_access_ns,
        block_key=event.block_key,
        schedule_event_ids=(event.schedule_event_id,),
    )
    events = (event,)
    epochs = (epoch,)
    checkpoint = make_schedule_checkpoint(
        "c1-b0-checkpoint-1",
        SCENARIO_CONTRACT["schedule_closed_through_ns"],
        events,
        epochs,
    )
    event_digest = schedule_stream_digest(events)
    return ClosedScheduleArtifact(
        schema_version=SCHEDULE_SIDECAR_SCHEMA_VERSION,
        artifact_id="c1-b0-controlled-schedule-artifact",
        trace_pair_id="c1-b0-controlled-pair",
        run_id="c1-b0-controlled-run",
        schedule_id="c1-b0-controlled-schedule",
        schedule_case_id="c1-b0-controlled-case",
        producer_kind=ScheduleProducerKind.REPLAY,
        producer_id="c1-b0-controlled-producer",
        source_artifact_digest=_named_digest("c1-b0-controlled-source"),
        source_schema_digest=_named_digest("c1-b0-controlled-source-schema"),
        source_case_digest=_mapping_digest({"scenario": SCENARIO_CONTRACT}),
        clock_domain=SCHEDULE_CLOCK_DOMAIN,
        event_order_rule=SCHEDULE_EVENT_ORDER_RULE,
        events=events,
        epochs=epochs,
        checkpoints=(checkpoint,),
        closure=ReplayScheduleClosure(
            declared_plan_event_count=1,
            plan_event_digest=event_digest,
            final_consumed_plan_event_count=1,
        ),
        final_event_digest=event_digest,
        final_epoch_digest=schedule_stream_digest(epochs),
        final_checkpoint_id=checkpoint.checkpoint_id,
    )


def _attempt_context(view: AtomicCutoffView, *, name: str) -> ForecastAttemptContext:
    return ForecastAttemptContext(
        feature_view_digest=_named_digest(f"{name}-feature-view"),
        information_cutoff_digest=view.view_digest,
        model_artifact_digest=_named_digest(f"{name}-model"),
        predictor_digest=_named_digest(f"{name}-predictor"),
        dependence_digest=_named_digest(f"{name}-dependence"),
        outcome_catalog_digest=_named_digest(f"{name}-outcome-catalog"),
        grouping_rules_digest=_named_digest(f"{name}-grouping"),
        model_inputs_digest=_named_digest(f"{name}-model-inputs"),
    )


def _prepare_scenario(
    root: Path,
    *,
    implementation_digest: str,
    environment_digest: str,
) -> _Scenario:
    block_key = BlockKey(
        content_digest=_named_digest("c1-b0-content"),
        parent_digest=_named_digest("c1-b0-parent"),
        model_fingerprint="qwen3-8b-c1-b0-component",
        tokenizer_fingerprint="qwen3-tokenizer-c1-b0-component",
        adapter_fingerprint=None,
        block_size_tokens=16,
        kv_dtype="bfloat16",
        cache_salt="c1-b0-controlled-replay",
    )
    workflow = WorkflowSpec(
        key=WorkflowKey("c1-b0-workflow", 0),
        nodes=(WorkflowNode("agent"),),
    )
    retention = BindingHandle(workflow.key, "retention", "c1-b0-retention")
    request_handle = BindingHandle(workflow.key, "request", "c1-b0-request")
    execution = ExecutionRef(workflow.key, "request", "sequence", 0)
    schedule = _schedule(
        block_key=block_key,
        workflow=workflow,
        retention=retention,
        request=request_handle,
        execution=execution,
    )
    schedule_digest = write_schedule_artifact(root / SCHEDULE_PAYLOAD, schedule)
    grammar_digest = _contract_digest("branch_grammar")
    committer = CanonicalTraceCommitter(
        root / TRACE_PAYLOAD,
        envelope=TraceEnvelope(
            trace_id="c1-b0-controlled-trace",
            run_id=schedule.run_id,
            schedule_id=schedule.schedule_id,
            schedule_case_id=schedule.schedule_case_id,
        ),
        schedule=schedule,
        schedule_artifact_digest=schedule_digest,
    )
    committer.commit_preamble(
        TracePreambleRequest(
            operation_id="c1-b0-preamble",
            header=TraceHeaderPayload(
                trace_pair_id=schedule.trace_pair_id,
                source_digest=schedule.source_artifact_digest,
                schedule_digest=schedule_digest,
                split_manifest_digest=_contract_digest("split_contract"),
                branch_grammar_digest=grammar_digest,
                feature_contract_digest=_contract_digest("feature_contract"),
                implementation_digest=implementation_digest,
                environment_digest=environment_digest,
            ),
            topologies=(
                WorkflowTopologyPayload(
                    workflow_spec=workflow,
                    workflow_template_digest=_named_digest("c1-b0-workflow-template"),
                    source_case_digest=schedule.source_case_digest,
                    split_component_id="c1-b0-controlled-component",
                    branch_grammar_digest=grammar_digest,
                ),
            ),
        )
    )
    runtime = LifecycleOrchestrator(
        run_id=schedule.run_id,
        phase="m3_c1b",
        source="dagkv.orchestrator",
        trace_required=True,
        formal_trace_committer=committer,
    )
    runtime.register_workflow(workflow)
    target_replica = ReplicaId(Tier.GPU, "logical:0", "c1-b0-slot", 1)
    runtime.register_gpu_block(
        block_key,
        target_replica,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=_named_digest("c1-b0-payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "agent", timestamp_ns=2)
    runtime.bind_owner(
        retention,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=3,
    )
    runtime.bind_owner(
        request_handle,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=execution,
        timestamp_ns=4,
    )

    def predicted_attempt(view: AtomicCutoffView) -> AttemptPayload:
        context = _attempt_context(view, name="c1-b0-predicted")
        claim = ReuseClaim(
            claim_id=schedule.events[0].claim_id,
            binding_id=retention.binding_id,
            workflow=workflow.key,
            node_id="agent",
            reuse_epoch_id=schedule.epochs[0].reuse_epoch_id,
            access_ns=schedule.events[0].scheduled_access_ns,
        )
        return PredictedAttemptPayload(
            status=ForecastAttemptStatus.PREDICTED,
            context=context,
            forecast=SharedLeaseForecast(
                forecast_id="c1-b0-forecast",
                block_key=block_key,
                runtime_event_count=len(view.lifecycle_prefix),
                generated_ns=view.cutoff_ns,
                horizon_ns=view.deadline_ns,
                source=ForecastSource.PREDICTED,
                predictor_digest=context.predictor_digest,
                dependence_digest=context.dependence_digest,
                independence_basis="one controlled component group",
                groups=(
                    DependenceGroup(
                        group_id="c1-b0-group",
                        outcomes=(
                            JointOutcome("demand", 500_000, (claim,)),
                            JointOutcome("no-demand", 500_000),
                        ),
                    ),
                ),
            ),
        )

    def abstained_attempt(view: AtomicCutoffView) -> AttemptPayload:
        return AbstainedAttemptPayload(
            status=ForecastAttemptStatus.ABSTAINED,
            context=_attempt_context(view, name="c1-b0-abstained"),
            reason=AbstentionReason.INSUFFICIENT_DATA,
        )

    predicted_observation_id = "observation-predicted-resident"
    abstained_observation_id = "observation-abstained-no-demand"
    runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=SCENARIO_CONTRACT["cutoff_ns"],
        horizon_duration_ns=(
            SCENARIO_CONTRACT["predicted_deadline_ns"] - SCENARIO_CONTRACT["cutoff_ns"]
        ),
        operation_id="c1-b0-cutoff-predicted",
        observation_id=predicted_observation_id,
        attempt_factory=predicted_attempt,
    )
    runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=SCENARIO_CONTRACT["cutoff_ns"],
        horizon_duration_ns=(
            SCENARIO_CONTRACT["abstained_deadline_ns"] - SCENARIO_CONTRACT["cutoff_ns"]
        ),
        operation_id="c1-b0-cutoff-abstained",
        observation_id=abstained_observation_id,
        attempt_factory=abstained_attempt,
    )
    demand = runtime.ensure_h2d_traced(
        block_key,
        target_replica,
        (request_handle,),
        transfer_id="c1-b0-resident-demand",
        timestamp_ns=SCENARIO_CONTRACT["scheduled_access_ns"],
        request=DemandCommitRequest(
            operation_id="c1-b0-demand",
            observation_id=predicted_observation_id,
            schedule_event_ids=(schedule.events[0].schedule_event_id,),
        ),
    )
    require(demand.command is None, "controlled resident demand scheduled a transfer")
    exec_map_event = next(
        event
        for event in reversed(runtime.events)
        if event.action == LedgerAction.EXEC_MAP
        and event.binding_id == request_handle.binding_id
    )
    seal_event = runtime.seal_lifecycle()
    events = runtime.events
    write_lifecycle_artifact(
        root / LIFECYCLE_PAYLOAD,
        ClosedLifecycleArtifact(
            schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            artifact_id="c1-b0-controlled-lifecycle",
            trace_pair_id=schedule.trace_pair_id,
            run_id=schedule.run_id,
            phase="m3_c1b",
            source="dagkv.orchestrator",
            clock_domain=LIFECYCLE_CLOCK_DOMAIN,
            implementation_digest=implementation_digest,
            environment_digest=environment_digest,
            events=events,
            closure=make_lifecycle_closure(events),
        ),
    )
    checkpoint = schedule.checkpoints[-1]
    schedule_closure = schedule.closure
    require(
        isinstance(schedule_closure, ReplayScheduleClosure),
        "controlled schedule lost replay closure",
    )
    watermark = ReplayScheduleWatermarkPayload(
        producer_kind=schedule.producer_kind,
        producer_id=schedule.producer_id,
        producer_artifact_digest=schedule_closure.plan_event_digest,
        schedule_digest=schedule_digest,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_digest=canonical_digest(checkpoint),
        consumed_event_count=checkpoint.consumed_event_count,
        last_schedule_event_id=checkpoint.last_schedule_event_id,
        max_closed_timestamp_ns=checkpoint.closed_through_ns,
        event_prefix_digest=checkpoint.event_prefix_digest,
        closed_epoch_count=checkpoint.closed_epoch_count,
        epoch_prefix_digest=checkpoint.epoch_prefix_digest,
    )
    terminal = ObservationTerminalSpec(
        status=TerminalStatus.COMPLETE,
        reason=TerminalReason.WINDOW_COMPLETE,
        label_available_ns=max(checkpoint.closed_through_ns, seal_event.timestamp_ns),
        last_verified_event_count=len(events),
        last_verified_event_id=seal_event.event_id,
        last_verified_event_timestamp_ns=seal_event.timestamp_ns,
    )
    committer.close_observation(
        ObservationCloseRequest(
            operation_id="c1-b0-close-predicted",
            observation_id=predicted_observation_id,
            services=(
                ResidentExecMapService(
                    intent_record_id=demand.receipt.commit.record_ids[0],
                    disposition=ServiceDisposition.RESIDENT_EXEC_MAP,
                    exec_map_event_id=exec_map_event.event_id,
                ),
            ),
            watermark=watermark,
            terminal=terminal,
        )
    )
    committer.close_observation(
        ObservationCloseRequest(
            operation_id="c1-b0-close-abstained",
            observation_id=abstained_observation_id,
            services=(),
            watermark=watermark,
            terminal=terminal,
        )
    )
    return _Scenario(
        runtime=runtime,
        committer=committer,
        schedule_digest=schedule_digest,
        schedule_event=schedule.events[0],
        request_handle=request_handle,
        target_replica=target_replica,
        predicted_observation_id=predicted_observation_id,
        abstained_observation_id=abstained_observation_id,
    )


def build_and_finalize_bundle(
    root: Path,
    *,
    protocol_digest: str,
    verifier_digest: str,
    implementation_digest: str,
    environment_digest: str,
) -> ValidatedC1B0Bundle:
    """Generate the controlled runtime trace and finalize its inner bundle."""

    root = root.expanduser().resolve()
    require(root.is_absolute(), "inner bundle root must be absolute")
    root.mkdir(mode=0o750)
    scenario = _prepare_scenario(
        root,
        implementation_digest=implementation_digest,
        environment_digest=environment_digest,
    )
    validated = finalize_c1_b0_bundle(
        root,
        bundle_id="c1-b0-controlled-stage-bundle",
        trace_committer=scenario.committer,
        protocol_digest=protocol_digest,
        verifier_digest=verifier_digest,
        expected_implementation_digest=implementation_digest,
        expected_environment_digest=environment_digest,
    )
    labels = {label.observation_id: label for label in validated.demand_labels}
    predicted = labels.get(scenario.predicted_observation_id)
    abstained = labels.get(scenario.abstained_observation_id)
    require(
        predicted is not None
        and predicted.first_demand == 1
        and predicted.epoch_count == 1
        and predicted.repeat_count == 0,
        "predicted resident label differs",
    )
    require(
        abstained is not None
        and abstained.first_demand == 0
        and abstained.epoch_count == 0
        and abstained.repeat_count == 0,
        "abstained no-demand label differs",
    )
    return validated


def validate_inner_bundle(
    root: Path,
    *,
    final_seal_sha256: str,
    protocol_digest: str,
    verifier_digest: str,
    implementation_digest: str,
    environment_digest: str,
) -> ValidatedC1B0Bundle:
    return validate_c1_b0_bundle(
        root.expanduser().resolve(),
        expected_final_seal_sha256=final_seal_sha256,
        expected_protocol_digest=protocol_digest,
        expected_verifier_digest=verifier_digest,
        expected_implementation_digest=implementation_digest,
        expected_environment_digest=environment_digest,
    )


def _validate_source_record(
    record: object,
    *,
    head: str,
    expected_paths: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    require(isinstance(record, dict), f"{label} binding must be an object")
    require(
        set(record) == {"paths", "entries", "digest"},
        f"{label} binding fields differ",
    )
    require(record["paths"] == list(expected_paths), f"{label} paths differ")
    entries = record["entries"]
    require(
        isinstance(entries, list) and len(entries) == len(expected_paths),
        f"{label} entries differ",
    )
    for expected_path, entry in zip(expected_paths, entries, strict=True):
        require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size", "sha256"}
            and entry["path"] == expected_path,
            f"{label} source entry fields differ",
        )
        blob = common._git("show", f"{head}:{expected_path}").stdout
        require(
            len(blob) == entry["size"]
            and common._sha256_bytes(blob) == entry["sha256"],
            f"{label} Git blob differs: {expected_path}",
        )
    require(record["digest"] == _binding_digest(entries), f"{label} digest differs")
    return entries


def _validate_anchors(anchors: object, *, head: str) -> dict[str, Any]:
    require(isinstance(anchors, dict), "external anchors must be an object")
    require(
        set(anchors) == {"protocol", "implementation", "verifier", "environment"},
        "external anchor fields differ",
    )
    protocol = anchors["protocol"]
    require(
        isinstance(protocol, dict)
        and set(protocol) == {"path", "size", "sha256"}
        and protocol["path"] == PROTOCOL_PATH,
        "protocol anchor fields differ",
    )
    protocol_blob = common._git("show", f"{head}:{PROTOCOL_PATH}").stdout
    require(
        len(protocol_blob) == protocol["size"]
        and common._sha256_bytes(protocol_blob) == protocol["sha256"],
        "protocol anchor differs from Git",
    )
    _validate_source_record(
        anchors["implementation"],
        head=head,
        expected_paths=IMPLEMENTATION_PATHS,
        label="implementation",
    )
    _validate_source_record(
        anchors["verifier"],
        head=head,
        expected_paths=VERIFIER_PATHS,
        label="verifier",
    )
    environment = anchors["environment"]
    require(
        isinstance(environment, dict)
        and set(environment)
        == {
            "python",
            "command_environment",
            "distributions",
            "module_origins",
            "dependencies",
            "digest",
        },
        "environment anchor fields differ",
    )
    python = common._validate_python_binding(environment["python"])
    require(
        environment["command_environment"] == common.BASE_ENVIRONMENT,
        "bound command environment differs",
    )
    dependencies = environment["dependencies"]
    require(
        isinstance(dependencies, list) and len(dependencies) == len(DEPENDENCY_PATHS),
        "environment dependencies differ",
    )
    for expected_path, entry in zip(DEPENDENCY_PATHS, dependencies, strict=True):
        require(
            isinstance(entry, dict)
            and entry.get("path") == expected_path
            and set(entry) == {"path", "size", "sha256"},
            "environment dependency entry differs",
        )
        blob = common._git("show", f"{head}:{expected_path}").stdout
        require(
            len(blob) == entry["size"]
            and common._sha256_bytes(blob) == entry["sha256"],
            f"environment dependency differs: {expected_path}",
        )
    module_origins = environment["module_origins"]
    require(
        isinstance(module_origins, dict)
        and set(module_origins) == {"entries", "digest"},
        "module origin binding fields differ",
    )
    origin_entries = _validate_module_origin_entries(module_origins["entries"])
    require(
        module_origins == _module_origin_binding(origin_entries),
        "module origin binding digest differs",
    )
    for entry, (module_name, relative) in zip(
        origin_entries,
        sorted(EXECUTED_MODULE_PATHS.items()),
        strict=True,
    ):
        require(entry["module"] == module_name, "module origin order differs")
        blob = common._git("show", f"{head}:{relative}").stdout
        require(
            common._sha256_bytes(blob) == entry["sha256"],
            f"module origin differs from Git: {module_name}",
        )
    environment_payload = {
        key: environment[key] for key in environment if key != "digest"
    }
    require(
        environment["digest"] == _mapping_digest(environment_payload),
        "environment digest differs",
    )
    require(
        _capture_environment_binding(python, dependencies) == environment,
        "bound execution environment differs",
    )
    require(
        _current_module_origin_binding() == environment["module_origins"],
        "validator modules differ from the bound execution source",
    )
    return anchors


def _validate_source_capture(root: Path, source: object, *, head: str) -> None:
    require(isinstance(source, dict), "source capture must be an object")
    require(set(source) == {"paths", "entries", "archive"}, "source fields differ")
    require(source["paths"] == list(SOURCE_PATHS), "source paths differ")
    entries = source["entries"]
    require(
        isinstance(entries, list) and len(entries) == len(SOURCE_PATHS),
        "source entries differ",
    )
    for expected_path, entry in zip(SOURCE_PATHS, entries, strict=True):
        require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size", "sha256"}
            and entry["path"] == expected_path,
            "source entry fields differ",
        )
        blob = common._git("show", f"{head}:{expected_path}").stdout
        require(
            len(blob) == entry["size"]
            and common._sha256_bytes(blob) == entry["sha256"],
            f"source Git blob differs: {expected_path}",
        )
    archive_path = common._validate_file_entry(root, source["archive"], label="source")
    expected_archive = common._git(
        "archive", "--format=tar", head, "--", *SOURCE_PATHS
    ).stdout
    require(archive_path.read_bytes() == expected_archive, "source archive differs")


def _command_specs(
    python: Path,
    anchors: Mapping[str, Any],
    final_seal_sha256: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "c1_b0_focused",
            (
                str(python),
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml={output_root}/logs/c1-b0-focused.junit.xml",
                *FOCUSED_TEST_PATHS,
            ),
        ),
        (
            "repository_full",
            (
                str(python),
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml={output_root}/logs/repository-full.junit.xml",
            ),
        ),
        ("ruff_check", (str(python), "-m", "ruff", "check", ".")),
        (
            "ruff_format_check",
            (str(python), "-m", "ruff", "format", "--check", "."),
        ),
        (
            "inner_bundle_producer",
            (
                str(python),
                "-m",
                "tools.run_m3_c1_b0_evidence",
                "produce-inner",
                "{output_root}/bundle",
                "--expected-protocol-digest",
                anchors["protocol"]["sha256"],
                "--expected-verifier-digest",
                anchors["verifier"]["digest"],
                "--expected-implementation-digest",
                anchors["implementation"]["digest"],
                "--expected-environment-digest",
                anchors["environment"]["digest"],
            ),
        ),
        (
            "fresh_bundle_replay",
            (
                str(python),
                "-m",
                "tools.run_m3_c1_b0_evidence",
                "validate-inner",
                "{output_root}/bundle",
                "--expected-final-seal-sha256",
                final_seal_sha256,
                "--expected-protocol-digest",
                anchors["protocol"]["sha256"],
                "--expected-verifier-digest",
                anchors["verifier"]["digest"],
                "--expected-implementation-digest",
                anchors["implementation"]["digest"],
                "--expected-environment-digest",
                anchors["environment"]["digest"],
            ),
        ),
    )


def _validate_inner_result(
    raw: bytes,
    *,
    label: str,
    expected_schema: str,
    expected_final_seal_sha256: str,
    expected_validated: Mapping[str, Any],
    expected_execution_origin_digest: str,
) -> None:
    result = _strict_json_bytes(raw, label=label)
    require(isinstance(result, dict), f"{label} must be an object")
    require(
        set(result)
        == {
            "schema_version",
            "status",
            "final_seal_sha256",
            "execution_origin_digest",
            "validated",
        },
        f"{label} fields differ",
    )
    require(result["schema_version"] == expected_schema, f"{label} schema differs")
    require(result["status"] == "passed", f"{label} status differs")
    require(
        result["final_seal_sha256"] == expected_final_seal_sha256,
        f"{label} final seal differs",
    )
    require(
        result["execution_origin_digest"] == expected_execution_origin_digest,
        f"{label} execution origin differs",
    )
    require(result["validated"] == expected_validated, f"{label} snapshot differs")


def validate_evidence(
    root: Path,
    *,
    expected_manifest_sha256: str | None,
    expected_checksums_sha256: str | None,
    require_sealed: bool = True,
) -> tuple[dict[str, Any], str, str]:
    root = root.expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), "evidence root is invalid")
    manifest, raw = common._read_json(root / MANIFEST_NAME, label="C1-B0 manifest")
    manifest_sha = common._sha256_bytes(raw)
    checksums_sha = common._sha256_file(root / CHECKSUM_NAME)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "manifest SHA-256 differs")
    if expected_checksums_sha256 is not None:
        require(checksums_sha == expected_checksums_sha256, "checksums SHA-256 differs")
    common._validate_checksums(root)
    require(
        set(manifest)
        == {
            "schema_version",
            "created_at_utc",
            "status",
            "accepted_gate_status",
            "claim_scope",
            "repository",
            "anchors",
            "source",
            "scenario_contract",
            "bundle",
            "commands",
            "junit",
            "open_gates",
            "gpu_used",
            "performance_claims_supported",
            "novelty_claims_supported",
            "all_passed",
        },
        "manifest fields differ",
    )
    require(manifest["schema_version"] == SCHEMA_VERSION, "schema version differs")
    require(manifest["status"] == STAGE_STATUS, "stage evidence status differs")
    require(
        manifest["accepted_gate_status"] == ACCEPTED_GATE_STATUS,
        "accepted gate status differs",
    )
    require(manifest["claim_scope"] == CLAIM_SCOPE, "claim scope differs")
    require(manifest["scenario_contract"] == SCENARIO_CONTRACT, "scenario differs")
    require(manifest["open_gates"] == list(C1_B0_OPEN_GATES), "open gates differ")
    require(manifest["gpu_used"] is False, "evidence claims GPU use")
    require(
        manifest["performance_claims_supported"] is False,
        "evidence claims performance support",
    )
    require(
        manifest["novelty_claims_supported"] is False,
        "evidence claims novelty support",
    )
    require(manifest["all_passed"] is True, "stage evidence did not pass")
    try:
        created = datetime.fromisoformat(manifest["created_at_utc"])
    except (TypeError, ValueError) as exc:
        raise C1B0EvidenceError("manifest timestamp is invalid") from exc
    require(created.tzinfo is not None, "manifest timestamp lacks timezone")

    repository = manifest["repository"]
    require(isinstance(repository, dict), "repository binding is invalid")
    require(
        set(repository) == {"root", "head", "branch", "tree", "clean"},
        "repository fields differ",
    )
    require(Path(repository["root"]).resolve() == REPO_ROOT, "repository root differs")
    head = repository["head"]
    require(
        isinstance(head, str)
        and repository["branch"] == "main"
        and repository["clean"] is True,
        "repository state differs",
    )
    require(
        common._git("cat-file", "-e", f"{head}^{{commit}}", check=False).returncode
        == 0,
        "repository commit is absent",
    )
    require(
        common._git_text("show", "-s", "--format=%T", head) == repository["tree"],
        "repository tree differs",
    )
    anchors = _validate_anchors(manifest["anchors"], head=head)
    python = Path(anchors["environment"]["python"]["launcher"]["path"])
    _validate_source_capture(root, manifest["source"], head=head)

    bundle = manifest["bundle"]
    require(isinstance(bundle, dict), "bundle binding must be an object")
    require(
        set(bundle) == {"root", "inventory", "final_seal", "manifest", "validated"},
        "bundle binding fields differ",
    )
    require(bundle["root"] == "bundle", "bundle root differs")
    bundle_root = root / bundle["root"]
    require(bundle["inventory"] == list(FINAL_INVENTORY), "bundle inventory differs")
    final_path = common._validate_file_entry(
        root, bundle["final_seal"], label="inner final seal"
    )
    inner_manifest_path = common._validate_file_entry(
        root, bundle["manifest"], label="inner manifest"
    )
    require(final_path == bundle_root / FINAL_SEAL, "inner final seal path differs")
    require(
        inner_manifest_path == bundle_root / MANIFEST, "inner manifest path differs"
    )
    validated = validate_inner_bundle(
        bundle_root,
        final_seal_sha256=bundle["final_seal"]["sha256"],
        protocol_digest=anchors["protocol"]["sha256"],
        verifier_digest=anchors["verifier"]["digest"],
        implementation_digest=anchors["implementation"]["digest"],
        environment_digest=anchors["environment"]["digest"],
    )
    validated_snapshot = _json_value(validated)
    require(validated_snapshot == bundle["validated"], "validated bundle differs")

    commands = manifest["commands"]
    require(isinstance(commands, list) and len(commands) == 6, "commands differ")
    specs = _command_specs(python, anchors, validated.final_seal_sha256)
    for command, (command_id, template) in zip(commands, specs, strict=True):
        common._validate_command(
            root,
            command,
            expected_id=command_id,
            expected_template=template,
        )
    producer_stdout = common._validate_file_entry(
        root, commands[-2]["stdout"], label="inner producer stdout"
    )
    _validate_inner_result(
        producer_stdout.read_bytes(),
        label="inner producer result",
        expected_schema=PRODUCER_RESULT_SCHEMA,
        expected_final_seal_sha256=validated.final_seal_sha256,
        expected_validated=validated_snapshot,
        expected_execution_origin_digest=anchors["environment"]["module_origins"][
            "digest"
        ],
    )
    fresh_stdout = common._validate_file_entry(
        root, commands[-1]["stdout"], label="fresh replay stdout"
    )
    _validate_inner_result(
        fresh_stdout.read_bytes(),
        label="fresh replay result",
        expected_schema=REPLAY_RESULT_SCHEMA,
        expected_final_seal_sha256=validated.final_seal_sha256,
        expected_validated=validated_snapshot,
        expected_execution_origin_digest=anchors["environment"]["module_origins"][
            "digest"
        ],
    )

    junit = manifest["junit"]
    require(
        isinstance(junit, dict) and set(junit) == {"focused", "full"},
        "JUnit fields differ",
    )
    summaries: dict[str, dict[str, Any]] = {}
    for label in ("focused", "full"):
        record = junit[label]
        require(
            isinstance(record, dict) and set(record) == {"file", "summary"},
            f"{label} JUnit record differs",
        )
        path = common._validate_file_entry(root, record["file"], label=f"{label} JUnit")
        summary = common._parse_junit(path)
        require(summary == record["summary"], f"{label} JUnit summary differs")
        require(
            all(summary[key] == 0 for key in ("failures", "errors", "skipped")),
            f"{label} JUnit has non-pass terminals",
        )
        summaries[label] = summary
    require(
        summaries["focused"]["tests"] == EXPECTED_FOCUSED_TESTS,
        "focused test count differs from the frozen C1-B0 set",
    )
    require(
        _test_identity_digest(summaries["focused"])
        == EXPECTED_FOCUSED_IDENTITIES_SHA256,
        "focused testcase identities differ from the frozen C1-B0 set",
    )
    require(
        summaries["full"]["tests"] == EXPECTED_FULL_TESTS,
        "full test count differs from the frozen repository set",
    )
    require(
        _test_identity_digest(summaries["full"]) == EXPECTED_FULL_IDENTITIES_SHA256,
        "full testcase identities differ from the frozen repository set",
    )
    require(
        set(summaries["focused"]["identities"]).issubset(
            summaries["full"]["identities"]
        ),
        "full regression omits a focused C1-B0 testcase",
    )

    if require_sealed:
        require(
            stat.S_IMODE(root.stat().st_mode) == 0o555,
            "evidence root is not read-only",
        )
        for path in root.rglob("*"):
            expected_mode = 0o555 if path.is_dir() else 0o444
            require(
                stat.S_IMODE(path.stat().st_mode) == expected_mode,
                f"sealed mode differs: {path}",
            )
        lock_path = root.parent / f".{root.name}{PUBLICATION_LOCK_SUFFIX}"
        require(
            lock_path.is_file()
            and not lock_path.is_symlink()
            and lock_path.read_bytes() == LOCK_PUBLISHED
            and stat.S_IMODE(lock_path.stat().st_mode) == 0o444,
            "publication sidecar differs",
        )
    return manifest, manifest_sha, checksums_sha


def run_evidence(
    output_root: Path,
    *,
    python: Path,
    timeout_seconds: int,
) -> tuple[str, str, int, int]:
    output_root = output_root.expanduser().resolve()
    python = Path(os.path.abspath(python.expanduser()))
    require(output_root.is_absolute(), "output root must be absolute")
    require(
        not output_root.is_relative_to(REPO_ROOT),
        "evidence output must be outside the repository",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}{PUBLICATION_LOCK_SUFFIX}"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    published = False
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(lock_descriptor, LOCK_PREPARING)
        os.fsync(lock_descriptor)
        require(not output_root.exists(), "evidence output already exists")
        branch, head, tree, status = _repository_state()
        require(branch == "main", "C1-B0 evidence requires branch main")
        require(not status, "repository must be clean")
        anchors, source_entries = _capture_anchors(head, python)
        _assert_repository_binding(head=head, tree=tree)
        specs_before_bundle = _command_specs(
            python,
            anchors,
            "0" * 64,
        )[:4]
        commands = [
            common._run_command(
                command_id,
                template,
                output_root=staging,
                cwd=REPO_ROOT,
                timeout_seconds=timeout_seconds,
            )
            for command_id, template in specs_before_bundle
        ]
        failed = [
            command
            for command in commands
            if command["exit_code"] != 0 or command["timed_out"]
        ]
        if failed:
            failed_command = failed[0]
            terminal = (staging / failed_command["stderr"]["path"]).read_text(
                errors="replace"
            )[-2000:]
            raise C1B0EvidenceError(
                f"{failed_command['command_id']} failed with exit "
                f"{failed_command['exit_code']}: {terminal.strip()}"
            )
        focused_junit = staging / "logs" / "c1-b0-focused.junit.xml"
        full_junit = staging / "logs" / "repository-full.junit.xml"
        focused_summary = common._parse_junit(focused_junit)
        full_summary = common._parse_junit(full_junit)
        require(
            focused_summary["tests"] == EXPECTED_FOCUSED_TESTS,
            "focused test count differs from the frozen C1-B0 set",
        )
        require(
            _test_identity_digest(focused_summary)
            == EXPECTED_FOCUSED_IDENTITIES_SHA256,
            "focused testcase identities differ from the frozen C1-B0 set",
        )
        require(
            full_summary["tests"] == EXPECTED_FULL_TESTS,
            "full test count differs from the frozen repository set",
        )
        require(
            _test_identity_digest(full_summary) == EXPECTED_FULL_IDENTITIES_SHA256,
            "full testcase identities differ from the frozen repository set",
        )
        require(
            all(
                summary[key] == 0
                for summary in (focused_summary, full_summary)
                for key in ("failures", "errors", "skipped")
            ),
            "test evidence contains a non-pass terminal",
        )
        _assert_repository_binding(head=head, tree=tree)

        bundle_root = staging / "bundle"
        producer_spec = _command_specs(python, anchors, "0" * 64)[-2]
        producer_command = common._run_command(
            producer_spec[0],
            producer_spec[1],
            output_root=staging,
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        )
        require(
            producer_command["exit_code"] == 0 and not producer_command["timed_out"],
            "inner-bundle producer failed",
        )
        commands.append(producer_command)
        independently_read_final_digest = _stable_sha256(
            bundle_root / FINAL_SEAL,
            label="inner final seal",
        )
        validated = validate_inner_bundle(
            bundle_root,
            final_seal_sha256=independently_read_final_digest,
            protocol_digest=anchors["protocol"]["sha256"],
            verifier_digest=anchors["verifier"]["digest"],
            implementation_digest=anchors["implementation"]["digest"],
            environment_digest=anchors["environment"]["digest"],
        )
        validated_snapshot = _json_value(validated)
        producer_stdout = staging / producer_command["stdout"]["path"]
        _validate_inner_result(
            producer_stdout.read_bytes(),
            label="inner producer result",
            expected_schema=PRODUCER_RESULT_SCHEMA,
            expected_final_seal_sha256=independently_read_final_digest,
            expected_validated=validated_snapshot,
            expected_execution_origin_digest=anchors["environment"]["module_origins"][
                "digest"
            ],
        )
        require(
            independently_read_final_digest == validated.final_seal_sha256,
            "final seal differs from independent launcher readback",
        )
        _assert_repository_binding(head=head, tree=tree)
        fresh_spec = _command_specs(
            python,
            anchors,
            independently_read_final_digest,
        )[-1]
        fresh_command = common._run_command(
            fresh_spec[0],
            fresh_spec[1],
            output_root=staging,
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        )
        require(
            fresh_command["exit_code"] == 0 and not fresh_command["timed_out"],
            "fresh-process bundle replay failed",
        )
        commands.append(fresh_command)
        fresh_stdout = staging / fresh_command["stdout"]["path"]
        _validate_inner_result(
            fresh_stdout.read_bytes(),
            label="fresh replay result",
            expected_schema=REPLAY_RESULT_SCHEMA,
            expected_final_seal_sha256=independently_read_final_digest,
            expected_validated=validated_snapshot,
            expected_execution_origin_digest=anchors["environment"]["module_origins"][
                "digest"
            ],
        )
        _assert_repository_binding(head=head, tree=tree)
        source = _capture_source_archive(
            staging,
            head=head,
            entries=source_entries,
        )
        bundle_record = {
            "root": "bundle",
            "inventory": list(FINAL_INVENTORY),
            "final_seal": common._file_entry(
                bundle_root / FINAL_SEAL,
                root=staging,
            ),
            "manifest": common._file_entry(
                bundle_root / MANIFEST,
                root=staging,
            ),
            "validated": validated_snapshot,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "status": STAGE_STATUS,
            "accepted_gate_status": ACCEPTED_GATE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "repository": {
                "root": str(REPO_ROOT),
                "head": head,
                "branch": branch,
                "tree": tree,
                "clean": True,
            },
            "anchors": anchors,
            "source": source,
            "scenario_contract": SCENARIO_CONTRACT,
            "bundle": bundle_record,
            "commands": commands,
            "junit": {
                "focused": {
                    "file": common._file_entry(focused_junit, root=staging),
                    "summary": focused_summary,
                },
                "full": {
                    "file": common._file_entry(full_junit, root=staging),
                    "summary": full_summary,
                },
            },
            "open_gates": list(C1_B0_OPEN_GATES),
            "gpu_used": False,
            "performance_claims_supported": False,
            "novelty_claims_supported": False,
            "all_passed": True,
        }
        common._write_new(staging / MANIFEST_NAME, common._canonical_json(manifest))
        common._write_checksums(staging)
        _assert_repository_binding(head=head, tree=tree)
        validate_evidence(
            staging,
            expected_manifest_sha256=None,
            expected_checksums_sha256=None,
            require_sealed=False,
        )
        common._seal(staging)
        common._rename_noreplace(staging, output_root)
        published = True
        common._fsync_directory(output_root.parent)
        manifest_sha = common._sha256_file(output_root / MANIFEST_NAME)
        checksums_sha = common._sha256_file(output_root / CHECKSUM_NAME)
        os.lseek(lock_descriptor, 0, os.SEEK_SET)
        os.ftruncate(lock_descriptor, 0)
        os.write(lock_descriptor, LOCK_PUBLISHED)
        os.fsync(lock_descriptor)
        os.fchmod(lock_descriptor, 0o444)
        common._fsync_directory(output_root.parent)
        validate_evidence(
            output_root,
            expected_manifest_sha256=manifest_sha,
            expected_checksums_sha256=checksums_sha,
            require_sealed=True,
        )
        return (
            manifest_sha,
            checksums_sha,
            focused_summary["tests"],
            full_summary["tests"],
        )
    except BaseException:
        if published and output_root.exists():
            _remove_tree(output_root)
            published = False
            with suppress(OSError):
                os.fchmod(lock_descriptor, 0o600)
        if staging.exists():
            _remove_tree(staging)
        raise
    finally:
        os.close(lock_descriptor)
        if not published:
            with suppress(FileNotFoundError):
                lock_path.unlink()


def _add_bundle_anchor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-protocol-digest", required=True)
    parser.add_argument("--expected-verifier-digest", required=True)
    parser.add_argument("--expected-implementation-digest", required=True)
    parser.add_argument("--expected-environment-digest", required=True)


def _add_inner_anchor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-final-seal-sha256", required=True)
    _add_bundle_anchor_arguments(parser)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run and publish C1-B0 stage evidence")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    validate = subparsers.add_parser("validate", help="replay outer stage evidence")
    validate.add_argument("evidence", type=Path)
    validate.add_argument("--expected-manifest-sha256", required=True)
    validate.add_argument("--expected-checksums-sha256", required=True)
    producer = subparsers.add_parser(
        "produce-inner", help="produce one controlled inner bundle"
    )
    producer.add_argument("bundle", type=Path)
    _add_bundle_anchor_arguments(producer)
    inner = subparsers.add_parser("validate-inner", help="replay the sealed bundle")
    inner.add_argument("bundle", type=Path)
    _add_inner_anchor_arguments(inner)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            require(arguments.timeout_seconds > 0, "timeout must be positive")
            manifest_sha, checksums_sha, focused, full = run_evidence(
                arguments.output_dir,
                python=arguments.python,
                timeout_seconds=arguments.timeout_seconds,
            )
            print(
                "C1-B0 stage evidence passed: "
                f"focused={focused} full={full} "
                f"manifest_sha256={manifest_sha} "
                f"checksums_sha256={checksums_sha}"
            )
        elif arguments.command == "validate":
            manifest, manifest_sha, checksums_sha = validate_evidence(
                arguments.evidence,
                expected_manifest_sha256=arguments.expected_manifest_sha256,
                expected_checksums_sha256=arguments.expected_checksums_sha256,
            )
            print(
                f"C1-B0 stage replay passed: {manifest['accepted_gate_status']} "
                f"manifest_sha256={manifest_sha} "
                f"checksums_sha256={checksums_sha}"
            )
        elif arguments.command == "produce-inner":
            execution_origin_digest = _current_module_origin_binding()["digest"]
            validated = build_and_finalize_bundle(
                arguments.bundle,
                protocol_digest=arguments.expected_protocol_digest,
                verifier_digest=arguments.expected_verifier_digest,
                implementation_digest=arguments.expected_implementation_digest,
                environment_digest=arguments.expected_environment_digest,
            )
            independently_read_final_digest = _stable_sha256(
                arguments.bundle.expanduser().resolve() / FINAL_SEAL,
                label="produced inner final seal",
            )
            require(
                independently_read_final_digest == validated.final_seal_sha256,
                "produced final seal differs from stable readback",
            )
            sys.stdout.buffer.write(
                common._canonical_json(
                    {
                        "schema_version": PRODUCER_RESULT_SCHEMA,
                        "status": "passed",
                        "final_seal_sha256": independently_read_final_digest,
                        "execution_origin_digest": execution_origin_digest,
                        "validated": _json_value(validated),
                    }
                )
            )
        else:
            execution_origin_digest = _current_module_origin_binding()["digest"]
            validated = validate_inner_bundle(
                arguments.bundle,
                final_seal_sha256=arguments.expected_final_seal_sha256,
                protocol_digest=arguments.expected_protocol_digest,
                verifier_digest=arguments.expected_verifier_digest,
                implementation_digest=arguments.expected_implementation_digest,
                environment_digest=arguments.expected_environment_digest,
            )
            sys.stdout.buffer.write(
                common._canonical_json(
                    {
                        "schema_version": REPLAY_RESULT_SCHEMA,
                        "status": "passed",
                        "final_seal_sha256": validated.final_seal_sha256,
                        "execution_origin_digest": execution_origin_digest,
                        "validated": _json_value(validated),
                    }
                )
            )
        return 0
    except (
        C1B0EvidenceError,
        common.C1EvidenceError,
        TraceValidationError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"C1-B0 stage evidence failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
