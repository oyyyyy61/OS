"""Adversarial C1-B trace authorization and persistence regressions."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock

import pytest

import dagkv.c1_trace as trace_module
from dagkv.c1_trace import (
    AbstainedAttemptPayload,
    CutoffPayload,
    DurableTraceWriter,
    EvidenceRole,
    H2DFailedService,
    NaturalTraceWatermarkPayload,
    ObservationTerminalPayload,
    PredictedAttemptPayload,
    ReplayScheduleWatermarkPayload,
    ReuseEpochPayload,
    ScheduleProducerKind,
    TerminalReason,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceDurabilityError,
    TraceRecord,
    TraceRecordType,
    TraceValidationError,
    TraceValidationReceipt,
    ValidatedTrace,
    WorkflowTopologyPayload,
    encode_trace_record,
    reconstruct_demand_label,
    trace_stream_digest,
    validate_trace,
    validate_trace_for_labels,
)
from dagkv.domain import BlockKey, WorkflowKey, WorkflowSpec
from tests.test_c1_trace import _digest, _trace


def _valid_trace(
    block_key: BlockKey,
    *,
    predicted: bool = True,
    service: str = "resident",
) -> tuple[TraceRecord, ...]:
    """Return the shared fixture with the frozen atomic cutoff identity."""

    return _trace(block_key, predicted=predicted, service=service)


def _complete_observation_ids(trace: ValidatedTrace) -> tuple[str, ...]:
    return tuple(
        sorted(
            observation.observation_id
            for observation in trace.observations
            if isinstance(observation.terminal.payload, ObservationTerminalPayload)
            and observation.terminal.payload.status == TerminalStatus.COMPLETE
        )
    )


def _receipt(
    trace: ValidatedTrace,
    *,
    role: EvidenceRole,
    artifact_digest: str,
    verifier: str,
    trace_digest: str | None = None,
    verified_observation_ids: tuple[str, ...] | None = None,
) -> TraceValidationReceipt:
    header = trace.header.payload
    assert isinstance(header, trace_module.TraceHeaderPayload)
    return TraceValidationReceipt(
        role=role,
        trace_pair_id=header.trace_pair_id,
        trace_digest=trace_digest or trace_stream_digest(trace.records),
        artifact_digest=artifact_digest,
        verifier_digest=_digest(verifier),
        verified_observation_ids=(
            _complete_observation_ids(trace)
            if verified_observation_ids is None
            else verified_observation_ids
        ),
    )


class _LifecycleGate:
    def __init__(
        self,
        *,
        trace_digest: str | None = None,
        artifact_digest: str | None = None,
        verified_observation_ids: tuple[str, ...] | None = None,
        role: EvidenceRole = EvidenceRole.LIFECYCLE,
    ) -> None:
        self.trace_digest = trace_digest
        self.artifact_digest = artifact_digest
        self.verified_observation_ids = verified_observation_ids
        self.role = role

    def verify_lifecycle(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        header = trace.header.payload
        assert isinstance(header, trace_module.TraceHeaderPayload)
        return _receipt(
            trace,
            role=self.role,
            artifact_digest=self.artifact_digest or header.source_digest,
            verifier="lifecycle-verifier",
            trace_digest=self.trace_digest,
            verified_observation_ids=self.verified_observation_ids,
        )


class _ScheduleGate:
    def __init__(
        self,
        *,
        trace_digest: str | None = None,
        artifact_digest: str | None = None,
        verified_observation_ids: tuple[str, ...] | None = None,
        role: EvidenceRole = EvidenceRole.SCHEDULE,
    ) -> None:
        self.trace_digest = trace_digest
        self.artifact_digest = artifact_digest
        self.verified_observation_ids = verified_observation_ids
        self.role = role

    def verify_schedule(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        header = trace.header.payload
        assert isinstance(header, trace_module.TraceHeaderPayload)
        return _receipt(
            trace,
            role=self.role,
            artifact_digest=self.artifact_digest or header.schedule_digest,
            verifier="schedule-verifier",
            trace_digest=self.trace_digest,
            verified_observation_ids=self.verified_observation_ids,
        )


def _rechain(records: list[TraceRecord]) -> tuple[TraceRecord, ...]:
    header_id = records[0].record_id
    latest_by_observation: dict[str, str] = {}
    result: list[TraceRecord] = []
    for sequence, record in enumerate(records):
        if record.record_type == TraceRecordType.TRACE_HEADER:
            parent = None
        elif record.record_type in {
            TraceRecordType.WORKFLOW_TOPOLOGY,
            TraceRecordType.CUTOFF,
        }:
            parent = header_id
        else:
            assert record.observation_id is not None
            parent = latest_by_observation[record.observation_id]
        updated = replace(
            record,
            sequence=sequence,
            parent_record_id=parent,
        )
        result.append(updated)
        if updated.observation_id is not None:
            latest_by_observation[updated.observation_id] = updated.record_id
    return tuple(result)


def _noncomplete_terminal(
    terminal: TraceRecord,
    *,
    watermark_record_id: str | None,
    unresolved: tuple[str, ...],
) -> TraceRecord:
    return replace(
        terminal,
        payload=ObservationTerminalPayload(
            status=TerminalStatus.CENSORED,
            reason=TerminalReason.TRACE_TRUNCATED,
            label_available_ns=None,
            schedule_watermark_record_id=watermark_record_id,
            last_verified_event_count=0,
            last_verified_event_id=None,
            last_verified_event_timestamp_ns=None,
            unresolved_demand_intent_record_ids=unresolved,
        ),
    )


def test_structural_validation_does_not_authorize_labels(block_key: BlockKey) -> None:
    structural = validate_trace(_valid_trace(block_key))

    with pytest.raises(
        TraceValidationError,
        match="authoriz|evidence gate|schedule verification",
    ):
        reconstruct_demand_label(structural, "observation-1")


def test_typed_verifier_receipts_authorize_labels(block_key: BlockKey) -> None:
    records = _valid_trace(block_key)
    authorized = validate_trace_for_labels(
        records,
        lifecycle_gate=_LifecycleGate(),
        schedule_gate=_ScheduleGate(),
    )

    label = reconstruct_demand_label(authorized, "observation-1")
    assert label.first_demand == 1
    assert label.epoch_count == 1


def test_label_authorization_does_not_survive_dataclass_replace(
    block_key: BlockKey,
) -> None:
    records = _valid_trace(block_key)
    authorized = validate_trace_for_labels(
        records,
        lifecycle_gate=_LifecycleGate(),
        schedule_gate=_ScheduleGate(),
    )

    copied = replace(authorized)

    with pytest.raises(TraceValidationError, match="opaque.*authorization"):
        reconstruct_demand_label(copied, "observation-1")


def test_label_authorization_token_cannot_be_transplanted(
    block_key: BlockKey,
) -> None:
    records = _valid_trace(block_key)
    authorized = validate_trace_for_labels(
        records,
        lifecycle_gate=_LifecycleGate(),
        schedule_gate=_ScheduleGate(),
    )
    forged = validate_trace(records)
    token = authorized._authorization_token
    assert token is not None
    with pytest.raises(AttributeError):
        object.__setattr__(token, "trace_identity", id(forged))
    object.__setattr__(
        forged,
        "lifecycle_validation",
        authorized.lifecycle_validation,
    )
    object.__setattr__(
        forged,
        "schedule_validation",
        authorized.schedule_validation,
    )
    object.__setattr__(
        forged,
        "_authorization_token",
        token,
    )

    with pytest.raises(TraceValidationError, match="opaque.*authorization"):
        reconstruct_demand_label(forged, "observation-1")


def test_label_reconstruction_ignores_tampered_cached_observations(
    block_key: BlockKey,
) -> None:
    records = _valid_trace(block_key)
    authorized = validate_trace_for_labels(
        records,
        lifecycle_gate=_LifecycleGate(),
        schedule_gate=_ScheduleGate(),
    )
    object.__setattr__(authorized.observations[0], "epochs", ())

    label = reconstruct_demand_label(authorized, "observation-1")

    assert label.first_demand == 1
    assert label.epoch_count == 1
    assert label.repeat_count == 0


@pytest.mark.parametrize(
    ("lifecycle_gate", "schedule_gate", "message"),
    [
        (
            _LifecycleGate(trace_digest=_digest("wrong-trace")),
            _ScheduleGate(),
            "trace (digest|stream)",
        ),
        (
            _LifecycleGate(),
            _ScheduleGate(artifact_digest=_digest("wrong-schedule")),
            "schedule.*(digest|frozen)|frozen schedule",
        ),
        (
            _LifecycleGate(role=EvidenceRole.SCHEDULE),
            _ScheduleGate(),
            "lifecycle.*wrong role",
        ),
        (
            _LifecycleGate(),
            _ScheduleGate(role=EvidenceRole.LIFECYCLE),
            "schedule.*wrong role",
        ),
    ],
)
def test_label_gate_rejects_wrong_trace_or_schedule_digest(
    block_key: BlockKey,
    lifecycle_gate: _LifecycleGate,
    schedule_gate: _ScheduleGate,
    message: str,
) -> None:
    with pytest.raises(TraceValidationError, match=message):
        validate_trace_for_labels(
            _valid_trace(block_key),
            lifecycle_gate=lifecycle_gate,
            schedule_gate=schedule_gate,
        )


@pytest.mark.parametrize("predicted", [True, False], ids=["predicted", "abstained"])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("feature_view_digest", "feature view"),
        ("information_cutoff_digest", "information cutoff"),
    ],
)
def test_attempt_context_must_match_exact_cutoff(
    block_key: BlockKey,
    predicted: bool,
    field: str,
    message: str,
) -> None:
    records = list(_valid_trace(block_key, predicted=predicted, service="none"))
    attempt = records[3]
    assert isinstance(
        attempt.payload,
        (PredictedAttemptPayload, AbstainedAttemptPayload),
    )
    context = replace(
        attempt.payload.context,
        **{field: _digest(f"wrong-{field}")},
    )
    records[3] = replace(
        attempt,
        payload=replace(attempt.payload, context=context),
    )

    with pytest.raises(TraceValidationError, match=message):
        validate_trace(tuple(records))


def test_duplicate_semantic_demand_intent_is_rejected(block_key: BlockKey) -> None:
    records = list(_valid_trace(block_key))
    intent = records[4]
    duplicate = replace(intent, record_id="semantic-duplicate-intent")
    records.insert(5, duplicate)

    with pytest.raises(TraceValidationError, match="semantic.*intent|intent.*semantic"):
        validate_trace(_rechain(records))


def test_duplicate_statistical_observation_unit_is_rejected(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key, service="none"))
    observation_rows = records[2:]
    duplicate_ids = {
        record.record_id: f"duplicate-{record.record_id}" for record in observation_rows
    }
    for record in observation_rows:
        payload = record.payload
        if isinstance(payload, ObservationTerminalPayload):
            assert payload.schedule_watermark_record_id is not None
            payload = replace(
                payload,
                schedule_watermark_record_id=duplicate_ids[
                    payload.schedule_watermark_record_id
                ],
            )
        records.append(
            replace(
                record,
                record_id=duplicate_ids[record.record_id],
                observation_id="observation-2",
                payload=payload,
            )
        )

    with pytest.raises(
        TraceValidationError,
        match="statistical observation unit.*duplicat|observation unit.*duplicat",
    ):
        validate_trace(_rechain(records))


def test_duplicate_schedule_event_claim_pair_is_rejected(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key, predicted=False))
    first_intent = records[4]
    assert isinstance(first_intent.payload, trace_module.DemandIntentPayload)
    duplicate = replace(
        first_intent,
        record_id="duplicate-schedule-claim-intent",
        payload=replace(
            first_intent.payload,
            request_binding_id="request-binding-2",
            execution_ref=replace(
                first_intent.payload.execution_ref,
                request_id="request-2",
            ),
            reuse_epoch_id="reuse-epoch-2",
        ),
    )
    records.insert(5, duplicate)

    with pytest.raises(
        TraceValidationError,
        match="schedule-event.*claim.*duplicat|claim.*schedule-event.*duplicat",
    ):
        validate_trace(_rechain(records))


def test_duplicate_reuse_epoch_identity_is_rejected(block_key: BlockKey) -> None:
    records = list(_valid_trace(block_key))
    first_intent = records[4]
    first_epoch = records[5]
    second_intent = replace(
        first_intent,
        record_id="second-intent",
        payload=replace(
            first_intent.payload,
            schedule_event_id="schedule-event-2",
            claim_id="claim-2",
            request_binding_id="request-binding-2",
            execution_ref=replace(
                first_intent.payload.execution_ref,
                request_id="request-2",
            ),
        ),
    )
    assert isinstance(first_epoch.payload, ReuseEpochPayload)
    first_service = first_epoch.payload.service_terminals[0]
    second_epoch = replace(
        first_epoch,
        record_id="second-epoch",
        payload=replace(
            first_epoch.payload,
            demand_intent_record_ids=(second_intent.record_id,),
            service_terminals=(
                replace(first_service, intent_record_id=second_intent.record_id),
            ),
        ),
    )
    mutated = records[:5] + [second_intent, first_epoch, second_epoch] + records[6:]

    with pytest.raises(
        TraceValidationError, match="reuse epoch.*duplicat|duplicate.*epoch"
    ):
        validate_trace(_rechain(mutated))


def test_cutoff_topology_workflow_must_match_owner_exactly(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key, service="none"))
    topology = records[1]
    assert isinstance(topology.payload, WorkflowTopologyPayload)
    records[1] = replace(
        topology,
        payload=replace(
            topology.payload,
            workflow_spec=WorkflowSpec(
                key=WorkflowKey("another-workflow", 0),
                nodes=topology.payload.workflow_spec.nodes,
            ),
        ),
    )

    with pytest.raises(
        TraceValidationError, match="topology.*workflow|workflow.*topology"
    ):
        validate_trace(tuple(records))


def test_cutoff_eligible_nodes_must_exist_in_owner_topology(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key, service="none"))
    cutoff = records[2]
    assert isinstance(cutoff.payload, CutoffPayload)
    owner = cutoff.payload.snapshot.owners[0]
    snapshot = replace(
        cutoff.payload.snapshot,
        owners=(replace(owner, eligible_node_ids=("missing-node",)),),
    )
    records[2] = replace(
        cutoff,
        payload=replace(cutoff.payload, snapshot=snapshot),
    )

    with pytest.raises(
        TraceValidationError,
        match="eligible.*(topology|node)|unknown eligible",
    ):
        validate_trace(tuple(records))


@pytest.mark.parametrize("closed_intent", [True, False], ids=["closed", "open"])
def test_noncomplete_unresolved_intents_are_exact(
    block_key: BlockKey,
    closed_intent: bool,
) -> None:
    records = list(_valid_trace(block_key))
    intent_id = records[4].record_id
    terminal = records[-1]
    records.pop(-2)  # A non-complete terminal has no schedule watermark here.
    if not closed_intent:
        records.pop(5)  # Leave the demand intent open.
    records[-1] = _noncomplete_terminal(
        terminal,
        watermark_record_id=None,
        unresolved=(intent_id,) if closed_intent else (),
    )

    with pytest.raises(
        TraceValidationError,
        match="unresolved.*(exact|differ)|open intent",
    ):
        validate_trace(_rechain(records))


def test_noncomplete_terminal_references_exact_observed_watermark(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key))
    terminal = records[-1]
    records[-1] = _noncomplete_terminal(
        terminal,
        watermark_record_id="another-watermark",
        unresolved=(),
    )

    with pytest.raises(TraceValidationError, match="watermark"):
        validate_trace(tuple(records))


def test_h2d_waiter_set_contains_the_request_binding(block_key: BlockKey) -> None:
    records = list(_valid_trace(block_key, service="h2d_failed"))
    epoch = records[5]
    assert isinstance(epoch.payload, ReuseEpochPayload)
    service = epoch.payload.service_terminals[0]
    assert isinstance(service, H2DFailedService)
    records[5] = replace(
        epoch,
        payload=replace(
            epoch.payload,
            service_terminals=(
                replace(service, waiter_binding_ids=("another-request-binding",)),
            ),
        ),
    )

    with pytest.raises(TraceValidationError, match="request.*waiter|waiter.*request"):
        validate_trace(tuple(records))


def test_natural_watermark_separates_schedule_count_from_source_eof_count(
    block_key: BlockKey,
) -> None:
    records = list(_valid_trace(block_key, service="none"))
    watermark = records[-2]
    assert isinstance(watermark.payload, ReplayScheduleWatermarkPayload)
    records[-2] = replace(
        watermark,
        payload=NaturalTraceWatermarkPayload(
            producer_kind=ScheduleProducerKind.SEALED_NATURAL_TRACE,
            producer_id=watermark.payload.producer_id,
            producer_artifact_digest=watermark.payload.producer_artifact_digest,
            schedule_digest=watermark.payload.schedule_digest,
            checkpoint_id=watermark.payload.checkpoint_id,
            checkpoint_digest=watermark.payload.checkpoint_digest,
            consumed_event_count=watermark.payload.consumed_event_count,
            last_schedule_event_id=watermark.payload.last_schedule_event_id,
            max_closed_timestamp_ns=watermark.payload.max_closed_timestamp_ns,
            event_prefix_digest=watermark.payload.event_prefix_digest,
            closed_epoch_count=watermark.payload.closed_epoch_count,
            epoch_prefix_digest=watermark.payload.epoch_prefix_digest,
            source_eof_record_count=watermark.payload.consumed_event_count + 7,
            source_eof_digest=_digest("natural-eof"),
            capture_start_ns=0,
            capture_end_ns=31,
            dropped_record_count=0,
            clean_eof=True,
        ),
    )

    validated = validate_trace(tuple(records))
    natural = validated.observations[0].watermark
    assert natural is not None
    assert isinstance(natural.payload, NaturalTraceWatermarkPayload)
    assert natural.payload.source_eof_record_count == 8


def test_writer_rejects_concurrent_append(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "concurrent.jsonl").resolve()
    writer = DurableTraceWriter(path)
    real_write = os.write
    first_write_entered = Event()
    release_first_write = Event()
    selection_lock = Lock()
    first_selected = False

    def blocking_first_write(descriptor: int, data: bytes) -> int:
        nonlocal first_selected
        with selection_lock:
            is_first = not first_selected
            first_selected = True
        if is_first:
            first_write_entered.set()
            assert release_first_write.wait(timeout=5)
        return real_write(descriptor, data)

    monkeypatch.setattr(trace_module.os, "write", blocking_first_write)
    header = (_valid_trace(block_key)[0],)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            writer.append_durable,
            header,
            event_count=0,
            view_digest=_digest("first-view"),
        )
        assert first_write_entered.wait(timeout=5)
        second = pool.submit(
            writer.append_durable,
            header,
            event_count=0,
            view_digest=_digest("second-view"),
        )
        try:
            with pytest.raises(
                (TraceDurabilityError, TraceValidationError),
                match="concurrent",
            ):
                second.result(timeout=5)
        finally:
            release_first_write.set()
        first.result(timeout=5)
    writer.close(finalize=False)


def test_writer_detects_external_append_before_next_batch(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    path = (tmp_path / "external-append.jsonl").resolve()
    records = _valid_trace(block_key)
    writer = DurableTraceWriter(path)
    writer.append_durable(
        records[:4],
        event_count=0,
        view_digest=_digest("cutoff-view"),
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, b"{}\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        with pytest.raises(
            (TraceDurabilityError, TraceValidationError),
            match="external|changed|size",
        ):
            writer.append_durable(
                records[4:],
                event_count=0,
                view_digest=_digest("terminal-view"),
            )
    finally:
        writer.close(finalize=False)


def test_writer_creation_base_exception_closes_both_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "creation-base-exception.jsonl").resolve()
    real_close = trace_module.os.close
    closed_descriptors: list[int] = []

    class InjectedAbort(BaseException):
        pass

    def fail_fsync(_descriptor: int) -> None:
        raise InjectedAbort("injected creation abort")

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    with monkeypatch.context() as context:
        context.setattr(trace_module.os, "fsync", fail_fsync)
        context.setattr(trace_module.os, "close", record_close)
        with pytest.raises(InjectedAbort, match="creation abort"):
            DurableTraceWriter(path)

    assert path.exists()
    assert len(closed_descriptors) == 2


def test_writer_receipt_preserves_trace_sequence_order(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    path = (tmp_path / "sequence-order.jsonl").resolve()
    records = list(_valid_trace(block_key)[:2])
    records[0] = replace(records[0], record_id="z-header")
    records[1] = replace(
        records[1],
        record_id="a-topology",
        parent_record_id="z-header",
    )
    writer = DurableTraceWriter(path)
    try:
        receipt = writer.append_durable(
            tuple(records),
            event_count=0,
            view_digest=_digest("sequence-order-view"),
        )
    finally:
        writer.close(finalize=False)

    assert receipt.record_ids == ("z-header", "a-topology")


def test_writer_validates_receipt_before_writing(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "receipt-prevalidation.jsonl").resolve()
    writer = DurableTraceWriter(path)

    def reject_receipt(**_: object) -> None:
        raise TraceValidationError("injected receipt validation failure")

    monkeypatch.setattr(trace_module, "DurableCommitReceipt", reject_receipt)
    try:
        with pytest.raises(TraceValidationError, match="receipt validation"):
            writer.append_durable(
                (_valid_trace(block_key)[0],),
                event_count=0,
                view_digest=_digest("receipt-prevalidation-view"),
            )
        assert path.stat().st_size == 0
    finally:
        writer.close(finalize=False)


def test_writer_rejects_same_size_overwrite_during_append(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "same-size-overwrite.jsonl").resolve()
    records = _valid_trace(block_key)
    writer = DurableTraceWriter(path)
    writer.append_durable(
        (records[0],),
        event_count=0,
        view_digest=_digest("initial-prefix-view"),
    )
    prefix = path.read_bytes()
    trace_id = b'"trace_id":"trace-1"'
    mutation_offset = prefix.index(trace_id) + len(b'"trace_id":"trace-')
    real_write = os.write
    overwrite_injected = False

    def overwrite_then_append(descriptor: int, data: bytes) -> int:
        nonlocal overwrite_injected
        if not overwrite_injected:
            overwrite_injected = True
            external = os.open(path, os.O_WRONLY)
            try:
                assert os.pwrite(external, b"X", mutation_offset) == 1
                os.fsync(external)
            finally:
                os.close(external)
        return real_write(descriptor, data)

    monkeypatch.setattr(trace_module.os, "write", overwrite_then_append)
    try:
        with pytest.raises(
            TraceCommitIndeterminateError,
            match="indeterminate",
        ):
            writer.append_durable(
                (records[1],),
                event_count=0,
                view_digest=_digest("raced-prefix-view"),
            )
    finally:
        writer.close(finalize=False)


def test_writer_rejects_stream_larger_than_frozen_limit(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (tmp_path / "oversize.jsonl").resolve()
    writer = DurableTraceWriter(path)
    monkeypatch.setattr(trace_module, "MAX_TRACE_BYTES", 64)
    try:
        with pytest.raises(TraceValidationError, match="size|large"):
            writer.append_durable(
                (_valid_trace(block_key)[0],),
                event_count=0,
                view_digest=_digest("oversize-view"),
            )
        assert path.stat().st_size == 0
    finally:
        writer.close(finalize=False)


def test_writer_round_trip_rejects_constructor_bypassed_value(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    path = (tmp_path / "invalid-object.jsonl").resolve()
    writer = DurableTraceWriter(path)
    header = deepcopy(_valid_trace(block_key)[0])
    object.__setattr__(header, "sequence", False)

    try:
        with pytest.raises(TraceValidationError, match="round.trip|canonical|integer"):
            writer.append_durable(
                (header,),
                event_count=0,
                view_digest=_digest("invalid-view"),
            )
        assert path.stat().st_size == 0
    finally:
        writer.close(finalize=False)


def test_trace_stream_digest_is_canonical_jsonl_digest(block_key: BlockKey) -> None:
    records = _valid_trace(block_key)
    expected = _digest(
        b"".join(encode_trace_record(record) + b"\n" for record in records).decode()
    )

    assert trace_stream_digest(records) == expected
