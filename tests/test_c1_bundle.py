"""Fresh replay and adversarial tests for the C1-B0 evidence bundle."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import dagkv.c1_bundle as bundle_module
from dagkv.c1_bundle import (
    ATTEMPT,
    C1_B0_CLAIM_SCOPE,
    C1_B0_OPEN_GATES,
    C1_B0_STATUS,
    COMMIT_NAMES,
    FINAL_INVENTORY,
    FINAL_SEAL,
    LIFECYCLE_PAYLOAD,
    MANIFEST,
    SCHEDULE_PAYLOAD,
    TRACE_COMMIT,
    TRACE_PAYLOAD,
    C1B0BundleManifest,
    C1B0FinalSeal,
    C1BundleSegmentCommit,
    ValidatedC1B0Bundle,
    finalize_c1_b0_bundle,
    validate_c1_b0_bundle,
)
from dagkv.c1_commit import (
    CanonicalTraceCommitter,
    ObservationCloseRequest,
    ObservationTerminalSpec,
    SealedTraceReceipt,
)
from dagkv.c1_lifecycle import (
    LIFECYCLE_CLOCK_DOMAIN,
    LIFECYCLE_SIDECAR_SCHEMA_VERSION,
    ClosedLifecycleArtifact,
    make_lifecycle_closure,
    write_lifecycle_artifact,
)
from dagkv.c1_schedule import (
    NaturalScheduleClosure,
    ReplayScheduleClosure,
    ScheduleProducerKind,
    write_schedule_artifact,
)
from dagkv.c1_trace import (
    ReplayScheduleWatermarkPayload,
    ResidentExecMapService,
    ServiceDisposition,
    TerminalReason,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceValidationError,
    canonical_digest,
    parse_canonical_dataclass,
)
from dagkv.domain import BlockKey, LedgerAction
from tests.test_c1_formal_runtime import _formal_scenario, _FormalScenario
from tests.test_c1_trace import _digest

PROTOCOL_DIGEST = _digest("c1-b0-protocol")
VERIFIER_DIGEST = _digest("c1-b0-verifier")
IMPLEMENTATION_DIGEST = _digest("formal-implementation")
ENVIRONMENT_DIGEST = _digest("formal-environment")


@dataclass(frozen=True, slots=True)
class _PreparedBundle:
    root: Path
    scenario: _FormalScenario


def _prepare_bundle(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> _PreparedBundle:
    root = tmp_path / "c1-b0-bundle"
    root.mkdir(mode=0o750)
    scenario = _formal_scenario(
        root,
        block_key,
        digest,  # type: ignore[arg-type]
        trace_basename=TRACE_PAYLOAD,
    )
    observed_schedule_digest = write_schedule_artifact(
        root / SCHEDULE_PAYLOAD,
        scenario.committer.schedule,
    )
    assert observed_schedule_digest == scenario.schedule_digest
    scenario.runtime.commit_shared_lease_cutoff_traced(
        block_key,
        cutoff_ns=5,
        horizon_duration_ns=10,
        operation_id="formal-cutoff",
        observation_id="formal-observation",
        attempt_factory=scenario.attempt_factory,
    )
    demand = scenario.runtime.ensure_h2d_traced(
        block_key,
        scenario.target_replica,
        (scenario.request_handle,),
        transfer_id="resident-demand",
        timestamp_ns=8,
        request=scenario.demand_request,
    )
    exec_map_event = next(
        event
        for event in reversed(scenario.runtime.events)
        if event.action == LedgerAction.EXEC_MAP
        and event.binding_id == scenario.request_handle.binding_id
    )
    seal_event = scenario.runtime.seal_lifecycle()
    header = scenario.committer.records[0].payload
    assert isinstance(header, TraceHeaderPayload)
    events = scenario.runtime.events
    lifecycle = ClosedLifecycleArtifact(
        schema_version=LIFECYCLE_SIDECAR_SCHEMA_VERSION,
        artifact_id="formal-lifecycle-artifact",
        trace_pair_id=scenario.committer.schedule.trace_pair_id,
        run_id=scenario.committer.run_id,
        phase="m3_c1b",
        source="dagkv.orchestrator",
        clock_domain=LIFECYCLE_CLOCK_DOMAIN,
        implementation_digest=header.implementation_digest,
        environment_digest=header.environment_digest,
        events=events,
        closure=make_lifecycle_closure(events),
    )
    write_lifecycle_artifact(root / LIFECYCLE_PAYLOAD, lifecycle)
    checkpoint = scenario.committer.schedule.checkpoints[-1]
    closure = scenario.committer.schedule.closure
    assert isinstance(closure, ReplayScheduleClosure)
    scenario.committer.close_observation(
        ObservationCloseRequest(
            operation_id="formal-close",
            observation_id="formal-observation",
            services=(
                ResidentExecMapService(
                    intent_record_id=demand.receipt.commit.record_ids[0],
                    disposition=ServiceDisposition.RESIDENT_EXEC_MAP,
                    exec_map_event_id=exec_map_event.event_id,
                ),
            ),
            watermark=ReplayScheduleWatermarkPayload(
                producer_kind=scenario.committer.schedule.producer_kind,
                producer_id=scenario.committer.schedule.producer_id,
                producer_artifact_digest=closure.plan_event_digest,
                schedule_digest=scenario.schedule_digest,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_digest=canonical_digest(checkpoint),
                consumed_event_count=checkpoint.consumed_event_count,
                last_schedule_event_id=checkpoint.last_schedule_event_id,
                max_closed_timestamp_ns=checkpoint.closed_through_ns,
                event_prefix_digest=checkpoint.event_prefix_digest,
                closed_epoch_count=checkpoint.closed_epoch_count,
                epoch_prefix_digest=checkpoint.epoch_prefix_digest,
            ),
            terminal=ObservationTerminalSpec(
                status=TerminalStatus.COMPLETE,
                reason=TerminalReason.WINDOW_COMPLETE,
                label_available_ns=max(
                    checkpoint.closed_through_ns,
                    seal_event.timestamp_ns,
                ),
                last_verified_event_count=len(events),
                last_verified_event_id=seal_event.event_id,
                last_verified_event_timestamp_ns=seal_event.timestamp_ns,
            ),
        )
    )
    return _PreparedBundle(root=root, scenario=scenario)


def _publish(prepared: _PreparedBundle) -> ValidatedC1B0Bundle:
    return finalize_c1_b0_bundle(
        prepared.root,
        bundle_id="formal-c1-b0-bundle",
        trace_committer=prepared.scenario.committer,
        protocol_digest=PROTOCOL_DIGEST,
        verifier_digest=VERIFIER_DIGEST,
        expected_implementation_digest=IMPLEMENTATION_DIGEST,
        expected_environment_digest=ENVIRONMENT_DIGEST,
    )


def _fresh_validate(root: Path, final_digest: str) -> ValidatedC1B0Bundle:
    return validate_c1_b0_bundle(
        root,
        expected_final_seal_sha256=final_digest,
        expected_protocol_digest=PROTOCOL_DIGEST,
        expected_verifier_digest=VERIFIER_DIGEST,
        expected_implementation_digest=IMPLEMENTATION_DIGEST,
        expected_environment_digest=ENVIRONMENT_DIGEST,
    )


def test_finalize_and_fresh_validate_exact_three_role_bundle(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)

    published = _publish(prepared)
    replayed = _fresh_validate(prepared.root, published.final_seal_sha256)

    assert replayed == published
    assert tuple(sorted(path.name for path in prepared.root.iterdir())) == tuple(
        sorted(FINAL_INVENTORY)
    )
    assert published.verified_observation_ids == ("formal-observation",)
    assert published.status == C1_B0_STATUS
    assert published.claim_scope == C1_B0_CLAIM_SCOPE
    assert published.open_gates == C1_B0_OPEN_GATES
    assert published.demand_labels[0].first_demand == 1
    assert published.demand_labels[0].epoch_count == 1
    assert prepared.root.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0 for path in prepared.root.iterdir())

    manifest = parse_canonical_dataclass(
        (prepared.root / MANIFEST).read_bytes(),
        C1B0BundleManifest,
        artifact_name="test C1-B0 manifest",
        max_bytes=bundle_module.MAX_BUNDLE_JSON_BYTES,
    )
    assert manifest.claim_scope == C1_B0_CLAIM_SCOPE
    assert tuple(reference.commit_basename for reference in manifest.segments) == (
        COMMIT_NAMES
    )


def test_external_final_protocol_and_verifier_anchors_are_mandatory(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)

    attempts = (
        (
            _digest("wrong-seal"),
            PROTOCOL_DIGEST,
            VERIFIER_DIGEST,
            IMPLEMENTATION_DIGEST,
            ENVIRONMENT_DIGEST,
        ),
        (
            published.final_seal_sha256,
            _digest("wrong-protocol"),
            VERIFIER_DIGEST,
            IMPLEMENTATION_DIGEST,
            ENVIRONMENT_DIGEST,
        ),
        (
            published.final_seal_sha256,
            PROTOCOL_DIGEST,
            _digest("wrong-verifier"),
            IMPLEMENTATION_DIGEST,
            ENVIRONMENT_DIGEST,
        ),
        (
            published.final_seal_sha256,
            PROTOCOL_DIGEST,
            VERIFIER_DIGEST,
            _digest("wrong-implementation"),
            ENVIRONMENT_DIGEST,
        ),
        (
            published.final_seal_sha256,
            PROTOCOL_DIGEST,
            VERIFIER_DIGEST,
            IMPLEMENTATION_DIGEST,
            _digest("wrong-environment"),
        ),
    )
    for (
        final_digest,
        protocol_digest,
        verifier_digest,
        implementation_digest,
        environment_digest,
    ) in attempts:
        with pytest.raises(TraceValidationError, match="anchor"):
            validate_c1_b0_bundle(
                prepared.root,
                expected_final_seal_sha256=final_digest,
                expected_protocol_digest=protocol_digest,
                expected_verifier_digest=verifier_digest,
                expected_implementation_digest=implementation_digest,
                expected_environment_digest=environment_digest,
            )


def test_trace_descriptor_persists_replayable_operation_boundaries(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    descriptor = parse_canonical_dataclass(
        (prepared.root / TRACE_COMMIT).read_bytes(),
        C1BundleSegmentCommit,
        artifact_name="test trace segment",
        max_bytes=bundle_module.MAX_BUNDLE_JSON_BYTES,
    )
    assert descriptor.sealed_trace is not None

    records = bundle_module._verify_typed_trace(
        (prepared.root / TRACE_PAYLOAD).read_bytes(),
        descriptor.sealed_trace,
    )

    assert len(descriptor.sealed_trace.operations) == 4
    assert descriptor.sealed_trace.closure.record_count == len(records)
    assert published.trace_sha256 == descriptor.payload_sha256


def test_forged_operation_kind_is_rejected_against_raw_trace(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    _publish(prepared)
    descriptor = parse_canonical_dataclass(
        (prepared.root / TRACE_COMMIT).read_bytes(),
        C1BundleSegmentCommit,
        artifact_name="test trace segment",
        max_bytes=bundle_module.MAX_BUNDLE_JSON_BYTES,
    )
    sealed = descriptor.sealed_trace
    assert sealed is not None
    forged_operation = replace(
        sealed.operations[1],
        kind=bundle_module.TraceOperationKind.DEMAND_INTENT,
    )
    forged = SealedTraceReceipt(
        trace_pair_id=sealed.trace_pair_id,
        trace_basename=sealed.trace_basename,
        closure=sealed.closure,
        operations=(sealed.operations[0], forged_operation, *sealed.operations[2:]),
    )

    with pytest.raises(TraceValidationError, match="operation chain"):
        bundle_module._verify_typed_trace(
            (prepared.root / TRACE_PAYLOAD).read_bytes(),
            forged,
        )


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("operation_id", "forged-cutoff-operation"),
        ("request_digest", _digest("forged-cutoff-request")),
        ("runtime_event_count", 987654),
        ("runtime_view_digest", _digest("forged-cutoff-view")),
    ),
)
def test_forged_reconstructible_operation_metadata_is_rejected(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
    field_name: str,
    forged_value: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    _publish(prepared)
    descriptor = parse_canonical_dataclass(
        (prepared.root / TRACE_COMMIT).read_bytes(),
        C1BundleSegmentCommit,
        artifact_name="test trace segment",
        max_bytes=bundle_module.MAX_BUNDLE_JSON_BYTES,
    )
    sealed = descriptor.sealed_trace
    assert sealed is not None
    forged_operation = replace(
        sealed.operations[1],
        **{field_name: forged_value},
    )
    forged = SealedTraceReceipt(
        trace_pair_id=sealed.trace_pair_id,
        trace_basename=sealed.trace_basename,
        closure=sealed.closure,
        operations=(sealed.operations[0], forged_operation, *sealed.operations[2:]),
    )

    with pytest.raises(TraceValidationError, match="typed operation"):
        bundle_module._verify_typed_trace(
            (prepared.root / TRACE_PAYLOAD).read_bytes(),
            forged,
        )


def test_payload_tamper_extra_file_and_hardlink_fail_closed(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    alias = tmp_path / "trace-hardlink"
    os.link(prepared.root / TRACE_PAYLOAD, alias)

    with pytest.raises(TraceValidationError, match="singly linked"):
        _fresh_validate(prepared.root, published.final_seal_sha256)

    alias.unlink()
    os.chmod(prepared.root, 0o750)
    extra = prepared.root / "unexpected.txt"
    extra.write_text("unexpected", encoding="ascii")
    os.chmod(extra, 0o440)
    os.chmod(prepared.root, 0o550)
    with pytest.raises(TraceValidationError, match="inventory"):
        _fresh_validate(prepared.root, published.final_seal_sha256)


def test_bundle_rejects_natural_schedule_component(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    source_digest = _digest("sealed-natural-source")
    natural_schedule = replace(
        prepared.scenario.committer.schedule,
        producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
        source_artifact_digest=source_digest,
        closure=NaturalScheduleClosure(
            source_eof_record_count=1,
            source_eof_digest=source_digest,
            capture_start_ns=0,
            capture_end_ns=20,
            dropped_record_count=0,
            clean_eof=True,
        ),
    )
    schedule_path = prepared.root / SCHEDULE_PAYLOAD
    os.chmod(schedule_path, 0o640)
    schedule_path.write_bytes(bundle_module.canonical_json(natural_schedule))
    os.chmod(schedule_path, 0o440)

    with pytest.raises(TraceValidationError, match="controlled replay"):
        _publish(prepared)

    assert (prepared.root / ATTEMPT).exists()


def test_sealed_bundle_rejects_writable_root_and_child(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    os.chmod(prepared.root, 0o750)
    with pytest.raises(TraceValidationError, match="root must be read-only"):
        _fresh_validate(prepared.root, published.final_seal_sha256)

    os.chmod(prepared.root, 0o550)
    os.chmod(prepared.root / TRACE_PAYLOAD, 0o640)
    with pytest.raises(TraceValidationError, match="read-only"):
        _fresh_validate(prepared.root, published.final_seal_sha256)


def test_sealed_bundle_rejects_root_and_child_symlinks(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    trace_path = prepared.root / TRACE_PAYLOAD
    trace_backup = tmp_path / "trace-backup"
    os.chmod(prepared.root, 0o750)
    trace_path.rename(trace_backup)
    trace_path.symlink_to(trace_backup)
    os.chmod(prepared.root, 0o550)
    with pytest.raises(TraceValidationError, match="regular|read-only"):
        _fresh_validate(prepared.root, published.final_seal_sha256)

    os.chmod(prepared.root, 0o750)
    trace_path.unlink()
    trace_backup.rename(trace_path)
    os.chmod(prepared.root, 0o550)
    real_root = tmp_path / "sealed-real-root"
    prepared.root.rename(real_root)
    prepared.root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(TraceValidationError, match="non-symlink"):
        _fresh_validate(prepared.root, published.final_seal_sha256)


def test_concurrent_child_tamper_during_replay_fails_closed(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    trace_path = prepared.root / TRACE_PAYLOAD
    original_replay = bundle_module._evidence_replay

    def tamper_after_replay(*args: object, **kwargs: object) -> object:
        result = original_replay(*args, **kwargs)  # type: ignore[arg-type]
        raw = trace_path.read_bytes()
        replacement = b"0" if raw[-2:-1] != b"0" else b"1"
        os.chmod(trace_path, 0o640)
        trace_path.write_bytes(raw[:-2] + replacement + raw[-1:])
        os.chmod(trace_path, 0o440)
        return result

    monkeypatch.setattr(bundle_module, "_evidence_replay", tamper_after_replay)

    with pytest.raises(TraceValidationError, match="changed during"):
        _fresh_validate(prepared.root, published.final_seal_sha256)


def test_second_finalization_cannot_overwrite_sealed_bundle(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    before = {path.name: path.read_bytes() for path in prepared.root.iterdir()}

    with pytest.raises(TraceValidationError, match="inventory"):
        _publish(prepared)

    assert {path.name: path.read_bytes() for path in prepared.root.iterdir()} == before
    assert _fresh_validate(prepared.root, published.final_seal_sha256) == published


def test_first_descriptor_write_failure_makes_root_nonresumable(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    original_write = bundle_module.os.write
    write_count = 0

    def fail_descriptor_write(descriptor: int, raw: bytes) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            return original_write(descriptor, raw)
        raise OSError("injected segment write failure")

    with monkeypatch.context() as context:
        context.setattr(bundle_module.os, "write", fail_descriptor_write)
        with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
            _publish(prepared)

    assert (prepared.root / ATTEMPT).exists()
    assert (prepared.root / COMMIT_NAMES[0]).exists()
    assert not (prepared.root / FINAL_SEAL).exists()
    with pytest.raises(TraceValidationError, match="inventory"):
        _publish(prepared)


def test_semantic_failure_consumes_bundle_attempt(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    lifecycle_path = prepared.root / LIFECYCLE_PAYLOAD
    original = lifecycle_path.read_bytes()
    os.chmod(lifecycle_path, 0o640)
    lifecycle_path.write_bytes(b"{")
    os.chmod(lifecycle_path, 0o440)

    with pytest.raises(TraceValidationError):
        _publish(prepared)

    assert (prepared.root / ATTEMPT).exists()
    assert not (prepared.root / FINAL_SEAL).exists()
    os.chmod(lifecycle_path, 0o640)
    lifecycle_path.write_bytes(original)
    os.chmod(lifecycle_path, 0o440)

    with pytest.raises(TraceValidationError, match="inventory"):
        _publish(prepared)


def test_finalizer_rejects_committer_subclasses_before_attempt(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)

    class DerivedCommitter(CanonicalTraceCommitter):
        pass

    derived = object.__new__(DerivedCommitter)
    with pytest.raises(TraceValidationError, match="concrete"):
        finalize_c1_b0_bundle(
            prepared.root,
            bundle_id="formal-c1-b0-bundle",
            trace_committer=derived,
            protocol_digest=PROTOCOL_DIGEST,
            verifier_digest=VERIFIER_DIGEST,
            expected_implementation_digest=IMPLEMENTATION_DIGEST,
            expected_environment_digest=ENVIRONMENT_DIGEST,
        )

    assert not (prepared.root / ATTEMPT).exists()


def test_fresh_subprocess_replays_only_from_bundle_and_external_anchors(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)
    script = (
        "from pathlib import Path; "
        "from dagkv.c1_bundle import validate_c1_b0_bundle; "
        f"v=validate_c1_b0_bundle(Path({str(prepared.root)!r}), "
        f"expected_final_seal_sha256={published.final_seal_sha256!r}, "
        f"expected_protocol_digest={PROTOCOL_DIGEST!r}, "
        f"expected_verifier_digest={VERIFIER_DIGEST!r}, "
        f"expected_implementation_digest={IMPLEMENTATION_DIGEST!r}, "
        f"expected_environment_digest={ENVIRONMENT_DIGEST!r}); "
        "print(v.final_seal_sha256)"
    )

    completed = subprocess.run(
        [str(Path(".venv/bin/python").absolute()), "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == published.final_seal_sha256


def test_manifest_and_final_seal_are_canonical_closed_dataclasses(
    tmp_path: Path,
    block_key: BlockKey,
    digest: object,
) -> None:
    prepared = _prepare_bundle(tmp_path, block_key, digest)
    published = _publish(prepared)

    final_seal = parse_canonical_dataclass(
        (prepared.root / FINAL_SEAL).read_bytes(),
        C1B0FinalSeal,
        artifact_name="test final seal",
        max_bytes=bundle_module.MAX_BUNDLE_JSON_BYTES,
    )

    assert final_seal.manifest_sha256 == published.manifest_sha256
    assert tuple(item.basename for item in final_seal.preseal_files) == (
        bundle_module.PRESEAL_NAMES
    )
