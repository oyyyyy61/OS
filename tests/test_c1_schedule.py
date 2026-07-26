"""Focused and adversarial tests for canonical C1 schedule evidence."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv.c1_schedule as schedule_module
from dagkv.c1_schedule import (
    SCHEDULE_SIDECAR_SCHEMA_VERSION,
    CanonicalScheduleEvidenceGate,
    ClosedScheduleArtifact,
    NaturalScheduleClosure,
    ReplayScheduleClosure,
    ScheduleCheckpoint,
    ScheduleDemandEvent,
    ScheduleEpoch,
    load_schedule_artifact,
    make_schedule_checkpoint,
    write_schedule_artifact,
)
from dagkv.c1_trace import (
    DemandIntentPayload,
    EvidenceRole,
    NaturalTraceWatermarkPayload,
    ObservationTerminalPayload,
    ReplayScheduleWatermarkPayload,
    ReuseEpochPayload,
    ScheduleProducerKind,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceRecord,
    TraceRecordType,
    TraceValidationError,
    canonical_digest,
    canonical_json,
    trace_stream_digest,
    validate_trace,
)
from dagkv.domain import BlockKey, WorkflowKey
from tests.test_c1_trace import _digest, _trace


def _jsonl_digest(values: tuple[object, ...]) -> str:
    return sha256(
        b"".join(canonical_json(value) + b"\n" for value in values)
    ).hexdigest()


def _record_of_type(
    records: tuple[TraceRecord, ...], record_type: TraceRecordType
) -> TraceRecord:
    return next(record for record in records if record.record_type == record_type)


def _schedule_event(records: tuple[TraceRecord, ...]) -> ScheduleDemandEvent:
    intent = _record_of_type(records, TraceRecordType.DEMAND_INTENT)
    payload = intent.payload
    assert isinstance(payload, DemandIntentPayload)
    return ScheduleDemandEvent(
        event_ordinal=0,
        schedule_event_id=payload.schedule_event_id,
        scheduled_access_ns=payload.scheduled_access_ns,
        claim_id=payload.claim_id,
        retention_binding_id=payload.retention_binding_id,
        request_binding_id=payload.request_binding_id,
        workflow=payload.workflow,
        node_id=payload.node_id,
        execution_ref=payload.execution_ref,
        block_key=payload.block_key,
        reuse_epoch_id=payload.reuse_epoch_id,
        source_record_id="plan-record-1",
        source_record_digest=_digest("plan-record-1"),
    )


def _schedule_epoch(event: ScheduleDemandEvent) -> ScheduleEpoch:
    return ScheduleEpoch(
        reuse_epoch_id=event.reuse_epoch_id,
        access_ns=event.scheduled_access_ns,
        block_key=event.block_key,
        schedule_event_ids=(event.schedule_event_id,),
    )


def _artifact(
    events: tuple[ScheduleDemandEvent, ...],
    epochs: tuple[ScheduleEpoch, ...],
    *,
    closed_through_ns: int = 31,
    checkpoints: tuple[ScheduleCheckpoint, ...] | None = None,
    producer_kind: ScheduleProducerKind = ScheduleProducerKind.REPLAY,
    closure: ReplayScheduleClosure | NaturalScheduleClosure | None = None,
    trace_pair_id: str = "pair-1",
    run_id: str = "run-1",
    schedule_id: str = "schedule-1",
    schedule_case_id: str = "case-1",
    source_artifact_digest: str | None = None,
    source_case_digest: str | None = None,
) -> ClosedScheduleArtifact:
    if checkpoints is None:
        checkpoints = (
            make_schedule_checkpoint(
                checkpoint_id="checkpoint-1",
                closed_through_ns=closed_through_ns,
                events=events,
                epochs=epochs,
            ),
        )
    event_digest = _jsonl_digest(events)
    epoch_digest = _jsonl_digest(epochs)
    if closure is None:
        closure = ReplayScheduleClosure(
            declared_plan_event_count=len(events),
            plan_event_digest=event_digest,
            final_consumed_plan_event_count=len(events),
        )
    return ClosedScheduleArtifact(
        schema_version=SCHEDULE_SIDECAR_SCHEMA_VERSION,
        artifact_id="artifact-1",
        trace_pair_id=trace_pair_id,
        run_id=run_id,
        schedule_id=schedule_id,
        schedule_case_id=schedule_case_id,
        producer_kind=producer_kind,
        producer_id="scheduler-1",
        source_artifact_digest=(
            source_artifact_digest
            or (
                closure.source_eof_digest
                if isinstance(closure, NaturalScheduleClosure)
                else _digest("source")
            )
        ),
        source_schema_digest=_digest("schedule-source-schema"),
        source_case_digest=source_case_digest or _digest("case"),
        clock_domain="campaign_monotonic_ns",
        event_order_rule="timestamp_then_ordinal_v1",
        events=events,
        epochs=epochs,
        checkpoints=checkpoints,
        closure=closure,
        final_event_digest=event_digest,
        final_epoch_digest=epoch_digest,
        final_checkpoint_id=checkpoints[-1].checkpoint_id,
    )


def _bind_trace_to_artifact(
    records: tuple[TraceRecord, ...],
    artifact: ClosedScheduleArtifact,
    artifact_digest: str,
) -> tuple[TraceRecord, ...]:
    checkpoint = artifact.checkpoints[-1]
    bound = list(records)
    header_index = next(
        index
        for index, record in enumerate(bound)
        if record.record_type == TraceRecordType.TRACE_HEADER
    )
    header = bound[header_index]
    header_payload = header.payload
    assert isinstance(header_payload, TraceHeaderPayload)
    bound[header_index] = replace(
        header,
        payload=replace(
            header_payload,
            source_digest=(
                artifact.source_artifact_digest
                if isinstance(artifact.closure, NaturalScheduleClosure)
                else header_payload.source_digest
            ),
            schedule_digest=artifact_digest,
        ),
    )

    watermark_index = next(
        index
        for index, record in enumerate(bound)
        if record.record_type == TraceRecordType.SCHEDULE_WATERMARK
    )
    watermark = bound[watermark_index]
    if isinstance(artifact.closure, ReplayScheduleClosure):
        producer_artifact_digest = artifact.closure.plan_event_digest
        watermark_payload = ReplayScheduleWatermarkPayload(
            producer_kind=ScheduleProducerKind.REPLAY,
            producer_id=artifact.producer_id,
            producer_artifact_digest=producer_artifact_digest,
            schedule_digest=artifact_digest,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_digest=canonical_digest(checkpoint),
            consumed_event_count=checkpoint.consumed_event_count,
            last_schedule_event_id=checkpoint.last_schedule_event_id,
            max_closed_timestamp_ns=checkpoint.closed_through_ns,
            event_prefix_digest=checkpoint.event_prefix_digest,
            closed_epoch_count=checkpoint.closed_epoch_count,
            epoch_prefix_digest=checkpoint.epoch_prefix_digest,
        )
    else:
        closure = artifact.closure
        producer_artifact_digest = closure.source_eof_digest
        watermark_payload = NaturalTraceWatermarkPayload(
            producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
            producer_id=artifact.producer_id,
            producer_artifact_digest=producer_artifact_digest,
            schedule_digest=artifact_digest,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_digest=canonical_digest(checkpoint),
            consumed_event_count=checkpoint.consumed_event_count,
            last_schedule_event_id=checkpoint.last_schedule_event_id,
            max_closed_timestamp_ns=checkpoint.closed_through_ns,
            event_prefix_digest=checkpoint.event_prefix_digest,
            closed_epoch_count=checkpoint.closed_epoch_count,
            epoch_prefix_digest=checkpoint.epoch_prefix_digest,
            source_eof_record_count=closure.source_eof_record_count,
            source_eof_digest=closure.source_eof_digest,
            capture_start_ns=closure.capture_start_ns,
            capture_end_ns=closure.capture_end_ns,
            dropped_record_count=closure.dropped_record_count,
            clean_eof=closure.clean_eof,
        )
    bound[watermark_index] = replace(watermark, payload=watermark_payload)

    terminal_index = next(
        index
        for index, record in enumerate(bound)
        if record.record_type == TraceRecordType.OBSERVATION_TERMINAL
    )
    terminal = bound[terminal_index]
    terminal_payload = terminal.payload
    assert isinstance(terminal_payload, ObservationTerminalPayload)
    bound[terminal_index] = replace(
        terminal,
        payload=replace(
            terminal_payload,
            label_available_ns=max(
                terminal_payload.label_available_ns or 0,
                checkpoint.closed_through_ns,
            ),
        ),
    )
    return tuple(bound)


def _write_and_bind(
    tmp_path: Path,
    artifact: ClosedScheduleArtifact,
    records: tuple[TraceRecord, ...],
    *,
    name: str = "schedule.json",
) -> tuple[Path, tuple[TraceRecord, ...], str]:
    path = tmp_path / name
    artifact_digest = write_schedule_artifact(path, artifact)
    return (
        path,
        _bind_trace_to_artifact(records, artifact, artifact_digest),
        artifact_digest,
    )


def _gate(
    path: Path,
    *,
    producer_source_path: Path | None = None,
) -> CanonicalScheduleEvidenceGate:
    return CanonicalScheduleEvidenceGate(
        path,
        _digest("schedule-verifier"),
        producer_source_path=producer_source_path,
    )


def _write_natural_source(path: Path, *, record_count: int = 7) -> tuple[str, int]:
    records = [b"plan-record-1"] + [
        f"source-record-{index}".encode() for index in range(2, record_count + 1)
    ]
    raw = b"\n".join(records) + b"\n"
    path.write_bytes(raw)
    return sha256(raw).hexdigest(), len(records)


def _second_event(
    event: ScheduleDemandEvent,
    *,
    event_ordinal: int = 1,
    scheduled_access_ns: int | None = None,
    reuse_epoch_id: str = "epoch-2",
) -> ScheduleDemandEvent:
    execution = replace(
        event.execution_ref,
        request_id=f"request-{event_ordinal + 1}",
        sequence_id=f"sequence-{event_ordinal + 1}",
    )
    return replace(
        event,
        event_ordinal=event_ordinal,
        schedule_event_id=f"schedule-event-{event_ordinal + 1}",
        scheduled_access_ns=(
            event.scheduled_access_ns
            if scheduled_access_ns is None
            else scheduled_access_ns
        ),
        claim_id=f"claim-{event_ordinal + 1}",
        request_binding_id=f"request-binding-{event_ordinal + 1}",
        execution_ref=execution,
        reuse_epoch_id=reuse_epoch_id,
        source_record_id=f"plan-record-{event_ordinal + 1}",
        source_record_digest=_digest(f"plan-record-{event_ordinal + 1}"),
    )


def test_schedule_artifact_round_trip_and_gate_receipt(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path, bound_records, artifact_digest = _write_and_bind(tmp_path, artifact, records)

    loaded = load_schedule_artifact(path)
    trace = validate_trace(bound_records)
    receipt = _gate(path).verify_schedule(trace)

    assert loaded.artifact == artifact
    assert loaded.digest == artifact_digest
    assert loaded.size_bytes == path.stat().st_size
    assert receipt.role == EvidenceRole.SCHEDULE
    assert receipt.trace_pair_id == "pair-1"
    assert receipt.trace_digest == trace_stream_digest(bound_records)
    assert receipt.artifact_digest == artifact_digest
    assert receipt.verifier_digest == _digest("schedule-verifier")
    assert receipt.verified_observation_ids == ("observation-1",)


def test_schedule_artifact_write_is_create_only(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path = tmp_path / "schedule.json"
    first_digest = write_schedule_artifact(path, artifact)
    first_bytes = path.read_bytes()

    with pytest.raises((OSError, TraceValidationError), match="create-only|exist"):
        write_schedule_artifact(path, artifact)

    assert path.read_bytes() == first_bytes
    assert sha256(first_bytes).hexdigest() == first_digest


def test_schedule_writer_retries_short_writes(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path = tmp_path / "short-write.json"
    real_write = schedule_module.os.write

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(schedule_module.os, "write", short_write)

    digest = write_schedule_artifact(path, artifact)

    assert load_schedule_artifact(path).digest == digest


@pytest.mark.parametrize("failure", ("write", "fsync", "readback"))
def test_schedule_writer_never_returns_a_digest_after_io_failure(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path = tmp_path / f"failed-{failure}.json"

    if failure == "write":
        monkeypatch.setattr(schedule_module.os, "write", lambda *_: 0)
    elif failure == "fsync":

        def fail_fsync(_: int) -> None:
            raise OSError("injected fsync failure")

        monkeypatch.setattr(schedule_module.os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(schedule_module.os, "pread", lambda *_: b"")

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_schedule_artifact(path, artifact)


def test_schedule_artifact_paths_reject_symlinks(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    target = tmp_path / "target.json"
    target.write_bytes(b"sentinel")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    with pytest.raises((OSError, TraceValidationError), match="symlink|regular|create"):
        write_schedule_artifact(alias, artifact)
    with pytest.raises(TraceValidationError, match="symlink|regular"):
        load_schedule_artifact(alias)

    assert target.read_bytes() == b"sentinel"


def test_schedule_artifact_rejects_unknown_clock_domain(block_key: BlockKey) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))

    with pytest.raises(TraceValidationError, match="clock domain"):
        replace(artifact, clock_domain="unix_epoch_ns")


def test_schedule_load_rechecks_single_link_after_open(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path = tmp_path / "schedule.json"
    alias = tmp_path / "late-hardlink.json"
    write_schedule_artifact(path, artifact)
    real_fstat = schedule_module.os.fstat
    linked = False

    def add_link_before_first_fstat(descriptor: int) -> object:
        nonlocal linked
        if not linked:
            alias.hardlink_to(path)
            linked = True
        return real_fstat(descriptor)

    monkeypatch.setattr(schedule_module.os, "fstat", add_link_before_first_fstat)

    with pytest.raises(TraceValidationError, match="changed|link"):
        load_schedule_artifact(path)


def test_load_rejects_noncanonical_artifact_bytes(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path = tmp_path / "schedule.json"
    write_schedule_artifact(path, artifact)
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(TraceValidationError, match="canonical|trailing|decode"):
        load_schedule_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    (
        "file_bytes",
        "trace_pair_id",
        "run_id",
        "schedule_id",
        "schedule_case_id",
        "source_artifact_digest",
        "source_case_digest",
    ),
)
def test_gate_rejects_artifact_tampering_or_identity_mismatch(
    tmp_path: Path, block_key: BlockKey, mutation: str
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    identity_changes: dict[str, object] = {}
    if mutation != "file_bytes":
        identity_changes[mutation] = (
            _digest(f"different-{mutation}")
            if mutation in {"source_artifact_digest", "source_case_digest"}
            else f"different-{mutation}"
        )
    artifact = _artifact(
        (event,),
        (_schedule_epoch(event),),
        **identity_changes,
    )
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)
    if mutation == "file_bytes":
        path.write_bytes(path.read_bytes().replace(b"scheduler-1", b"scheduler-x", 1))

    with pytest.raises(
        TraceValidationError,
        match="artifact|trace|digest|case|identity|header",
    ):
        _gate(path).verify_schedule(validate_trace(bound_records))


@pytest.mark.parametrize(
    "mutation",
    ("missing_member", "unknown_member", "duplicate_event", "overlapping_epoch"),
)
def test_artifact_requires_an_exact_event_epoch_partition(
    block_key: BlockKey, mutation: str
) -> None:
    records = _trace(block_key)
    first = _schedule_event(records)
    second = _second_event(first)
    events = (first, second)
    epochs = (_schedule_epoch(first), _schedule_epoch(second))
    if mutation == "missing_member":
        epochs = (_schedule_epoch(first),)
    elif mutation == "unknown_member":
        epochs = (
            replace(
                _schedule_epoch(first),
                schedule_event_ids=("unknown-schedule-event",),
            ),
            _schedule_epoch(second),
        )
    elif mutation == "duplicate_event":
        events = (first, replace(first, event_ordinal=1))
        epochs = (_schedule_epoch(first),)
    else:
        epochs = (
            _schedule_epoch(first),
            replace(
                _schedule_epoch(second),
                reuse_epoch_id="overlap-epoch",
                access_ns=first.scheduled_access_ns,
                schedule_event_ids=(first.schedule_event_id,),
            ),
        )

    with pytest.raises(TraceValidationError, match="event|epoch|partition|duplicate"):
        _artifact(events, epochs)


def test_make_checkpoint_recomputes_exact_closed_prefixes(
    block_key: BlockKey,
) -> None:
    records = _trace(block_key)
    first = _schedule_event(records)
    second = _second_event(first, scheduled_access_ns=40)
    epochs = (_schedule_epoch(first), _schedule_epoch(second))

    empty = make_schedule_checkpoint(
        checkpoint_id="empty",
        closed_through_ns=10,
        events=(first, second),
        epochs=epochs,
    )
    prefix = make_schedule_checkpoint(
        checkpoint_id="prefix",
        closed_through_ns=31,
        events=(first, second),
        epochs=epochs,
    )

    assert empty.consumed_event_count == 0
    assert empty.last_schedule_event_id is None
    assert empty.event_prefix_digest == sha256(b"").hexdigest()
    assert empty.closed_epoch_count == 0
    assert empty.epoch_prefix_digest == sha256(b"").hexdigest()
    assert prefix.consumed_event_count == 1
    assert prefix.last_schedule_event_id == first.schedule_event_id
    assert prefix.event_prefix_digest == _jsonl_digest((first,))
    assert prefix.closed_epoch_count == 1
    assert prefix.epoch_prefix_digest == _jsonl_digest((epochs[0],))


@pytest.mark.parametrize(
    "changes",
    (
        {"consumed_event_count": 0, "last_schedule_event_id": None},
        {"last_schedule_event_id": "wrong-event"},
        {"event_prefix_digest": _digest("wrong-event-prefix")},
        {"closed_epoch_count": 0},
        {"epoch_prefix_digest": _digest("wrong-epoch-prefix")},
    ),
)
def test_artifact_rejects_forged_checkpoint_prefix(
    block_key: BlockKey, changes: dict[str, object]
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    epoch = _schedule_epoch(event)
    checkpoint = make_schedule_checkpoint(
        checkpoint_id="checkpoint-1",
        closed_through_ns=31,
        events=(event,),
        epochs=(epoch,),
    )
    forged = replace(checkpoint, **changes)

    with pytest.raises(TraceValidationError, match="checkpoint|prefix|event|epoch"):
        _artifact((event,), (epoch,), checkpoints=(forged,))


@pytest.mark.parametrize(
    "changes",
    (
        {"checkpoint_id": "another-checkpoint"},
        {"checkpoint_digest": _digest("another-checkpoint")},
        {"consumed_event_count": 2},
        {"last_schedule_event_id": "another-event"},
        {"event_prefix_digest": _digest("another-event-prefix")},
        {"closed_epoch_count": 2},
        {"epoch_prefix_digest": _digest("another-epoch-prefix")},
        {"producer_id": "another-producer"},
        {"producer_artifact_digest": _digest("another-plan")},
    ),
)
def test_complete_watermark_must_bind_the_exact_checkpoint(
    tmp_path: Path, block_key: BlockKey, changes: dict[str, object]
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)
    bound = list(bound_records)
    index = next(
        index
        for index, record in enumerate(bound)
        if record.record_type == TraceRecordType.SCHEDULE_WATERMARK
    )
    payload = bound[index].payload
    assert isinstance(payload, ReplayScheduleWatermarkPayload)
    bound[index] = replace(bound[index], payload=replace(payload, **changes))

    with pytest.raises(
        TraceValidationError, match="checkpoint|prefix|producer|schedule"
    ):
        _gate(path).verify_schedule(validate_trace(tuple(bound)))


@pytest.mark.parametrize(
    "field",
    (
        "schedule_event_id",
        "scheduled_access_ns",
        "claim_id",
        "retention_binding_id",
        "request_binding_id",
        "workflow",
        "node_id",
        "execution_ref",
        "block_key",
        "reuse_epoch_id",
    ),
)
def test_demand_intent_identity_mismatch_fails_closed(
    tmp_path: Path, block_key: BlockKey, field: str
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    if field == "schedule_event_id":
        changed = replace(event, schedule_event_id="different-event")
    elif field == "scheduled_access_ns":
        changed = replace(event, scheduled_access_ns=21)
    elif field == "claim_id":
        changed = replace(event, claim_id="different-claim")
    elif field == "retention_binding_id":
        changed = replace(event, retention_binding_id="different-retention")
    elif field == "request_binding_id":
        changed = replace(event, request_binding_id="different-request-binding")
    elif field == "workflow":
        workflow = WorkflowKey("different-workflow", 0)
        changed = replace(
            event,
            workflow=workflow,
            execution_ref=replace(event.execution_ref, workflow=workflow),
        )
    elif field == "node_id":
        changed = replace(event, node_id="different-node")
    elif field == "execution_ref":
        changed = replace(
            event,
            execution_ref=replace(event.execution_ref, request_id="different-request"),
        )
    elif field == "block_key":
        changed = replace(event, block_key=replace(block_key, cache_salt="different"))
    else:
        changed = replace(event, reuse_epoch_id="different-epoch")
    epoch = _schedule_epoch(changed)
    artifact = _artifact((changed,), (epoch,))
    path, bound_records, _ = _write_and_bind(
        tmp_path, artifact, records, name=f"{field}.json"
    )

    with pytest.raises(
        TraceValidationError, match="demand|intent|event|identity|epoch"
    ):
        _gate(path).verify_schedule(validate_trace(bound_records))


def test_deleted_schedule_event_cannot_forge_a_zero_demand_label(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    artifact = _artifact((), ())
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    with pytest.raises(TraceValidationError, match="demand|intent|missing|bijection"):
        _gate(path).verify_schedule(validate_trace(bound_records))


def test_extra_eligible_schedule_event_fails_closed(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    first = _schedule_event(records)
    second = _second_event(first)
    artifact = _artifact(
        (first, second),
        (_schedule_epoch(first), _schedule_epoch(second)),
    )
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    with pytest.raises(TraceValidationError, match="demand|intent|extra|bijection"):
        _gate(path).verify_schedule(validate_trace(bound_records))


def test_events_at_cutoff_and_after_deadline_are_outside_the_label_window(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    inside = _schedule_event(records)
    at_cutoff = replace(
        inside,
        event_ordinal=0,
        schedule_event_id="schedule-event-cutoff",
        scheduled_access_ns=10,
        claim_id="claim-cutoff",
        request_binding_id="request-binding-cutoff",
        execution_ref=replace(
            inside.execution_ref,
            request_id="request-cutoff",
            sequence_id="sequence-cutoff",
        ),
        reuse_epoch_id="epoch-cutoff",
        source_record_id="plan-record-cutoff",
        source_record_digest=_digest("plan-record-cutoff"),
    )
    inside = replace(inside, event_ordinal=1)
    after_deadline = _second_event(
        inside,
        event_ordinal=2,
        scheduled_access_ns=31,
        reuse_epoch_id="epoch-after",
    )
    events = (at_cutoff, inside, after_deadline)
    epochs = tuple(_schedule_epoch(event) for event in events)
    artifact = _artifact(events, epochs, closed_through_ns=32)
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    receipt = _gate(path).verify_schedule(validate_trace(bound_records))

    assert receipt.verified_observation_ids == ("observation-1",)


def test_event_at_deadline_is_included_in_the_label_window(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = list(_trace(block_key))
    intent_index = next(
        index
        for index, record in enumerate(records)
        if record.record_type == TraceRecordType.DEMAND_INTENT
    )
    intent = records[intent_index]
    intent_payload = intent.payload
    assert isinstance(intent_payload, DemandIntentPayload)
    records[intent_index] = replace(
        intent, payload=replace(intent_payload, scheduled_access_ns=30)
    )
    epoch_index = next(
        index
        for index, record in enumerate(records)
        if record.record_type == TraceRecordType.REUSE_EPOCH
    )
    epoch_record = records[epoch_index]
    epoch_payload = epoch_record.payload
    assert isinstance(epoch_payload, ReuseEpochPayload)
    records[epoch_index] = replace(
        epoch_record, payload=replace(epoch_payload, access_ns=30)
    )
    changed_records = tuple(records)
    event = _schedule_event(changed_records)
    artifact = _artifact((event,), (_schedule_epoch(event),))
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, changed_records)

    receipt = _gate(path).verify_schedule(validate_trace(bound_records))

    assert receipt.verified_observation_ids == ("observation-1",)


def test_zero_demand_is_authorized_by_an_empty_prefix_beyond_deadline(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key, predicted=False, service="none")
    artifact = _artifact((), (), closed_through_ns=31)
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    receipt = _gate(path).verify_schedule(validate_trace(bound_records))

    assert receipt.verified_observation_ids == ("observation-1",)


def test_zero_demand_checkpoint_must_exceed_the_deadline(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key, predicted=False, service="none")
    artifact = _artifact((), (), closed_through_ns=30)
    _, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    with pytest.raises(TraceValidationError, match="exceed.*deadline"):
        validate_trace(bound_records)


def test_natural_source_count_is_separate_but_total_normalization_is_required(
    tmp_path: Path, block_key: BlockKey
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    source_path = tmp_path / "natural-source.jsonl"
    source_digest, source_count = _write_natural_source(source_path)
    closure = NaturalScheduleClosure(
        source_eof_record_count=source_count,
        source_eof_digest=source_digest,
        capture_start_ns=0,
        capture_end_ns=40,
        dropped_record_count=0,
        clean_eof=True,
    )
    artifact = _artifact(
        (event,),
        (_schedule_epoch(event),),
        producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
        closure=closure,
    )
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    assert closure.source_eof_record_count != len(artifact.events)
    with pytest.raises(TraceValidationError, match="total normalization"):
        _gate(path, producer_source_path=source_path).verify_schedule(
            validate_trace(bound_records)
        )


def test_natural_zero_demand_cannot_use_vacuous_source_membership(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    records = _trace(block_key, predicted=False, service="none")
    source_path = tmp_path / "natural-source.jsonl"
    source_digest, source_count = _write_natural_source(source_path)
    closure = NaturalScheduleClosure(
        source_eof_record_count=source_count,
        source_eof_digest=source_digest,
        capture_start_ns=0,
        capture_end_ns=40,
        dropped_record_count=0,
        clean_eof=True,
    )
    artifact = _artifact(
        (),
        (),
        producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
        closure=closure,
    )
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)

    with pytest.raises(TraceValidationError, match="total normalization"):
        _gate(path, producer_source_path=source_path).verify_schedule(
            validate_trace(bound_records)
        )


@pytest.mark.parametrize("mutation", ("missing", "digest", "count", "record"))
def test_natural_source_artifact_is_verified_independently(
    tmp_path: Path,
    block_key: BlockKey,
    mutation: str,
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    source_path = tmp_path / "natural-source.jsonl"
    source_digest, source_count = _write_natural_source(source_path)
    closure = NaturalScheduleClosure(
        source_eof_record_count=(
            source_count + 1 if mutation == "count" else source_count
        ),
        source_eof_digest=(
            _digest("wrong-natural-source") if mutation == "digest" else source_digest
        ),
        capture_start_ns=0,
        capture_end_ns=40,
        dropped_record_count=0,
        clean_eof=True,
    )
    changed_event = (
        replace(event, source_record_digest=_digest("missing-source-record"))
        if mutation == "record"
        else event
    )
    artifact = _artifact(
        (changed_event,),
        (_schedule_epoch(changed_event),),
        producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
        closure=closure,
    )
    path, bound_records, _ = _write_and_bind(tmp_path, artifact, records)
    gate_source = None if mutation == "missing" else source_path

    with pytest.raises(TraceValidationError, match="natural|source|EOF|record"):
        _gate(path, producer_source_path=gate_source).verify_schedule(
            validate_trace(bound_records)
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"capture_end_ns": 30},
        {"dropped_record_count": 1},
        {"clean_eof": False},
    ),
)
def test_incomplete_natural_closure_never_authorizes_complete_observation(
    tmp_path: Path, block_key: BlockKey, changes: dict[str, object]
) -> None:
    records = _trace(block_key)
    event = _schedule_event(records)
    source_path = tmp_path / "natural-source.jsonl"
    source_digest, source_count = _write_natural_source(source_path)
    closure = NaturalScheduleClosure(
        source_eof_record_count=source_count,
        source_eof_digest=source_digest,
        capture_start_ns=0,
        capture_end_ns=40,
        dropped_record_count=0,
        clean_eof=True,
    )

    with pytest.raises(TraceValidationError, match="natural|capture|drop|EOF|complete"):
        changed_closure = replace(closure, **changes)
        artifact = _artifact(
            (event,),
            (_schedule_epoch(event),),
            producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
            closure=changed_closure,
        )
        path, bound_records, _ = _write_and_bind(
            tmp_path,
            artifact,
            records,
            name=f"natural-{next(iter(changes))}.json",
        )
        _gate(path, producer_source_path=source_path).verify_schedule(
            validate_trace(bound_records)
        )
