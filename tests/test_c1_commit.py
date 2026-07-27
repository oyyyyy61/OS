"""Operation-boundary tests for the canonical C1 trace committer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv.c1_commit as commit_module
import dagkv.c1_trace as trace_module
from dagkv.c1_commit import (
    CanonicalTraceCommitter,
    CutoffCommitRequest,
    DemandCommitRequest,
    ObservationCloseRequest,
    ObservationTerminalSpec,
    TraceEnvelope,
    TraceOperationKind,
    TracePreambleRequest,
    WriterIssuedCommitReceipt,
)
from dagkv.c1_schedule import ClosedScheduleArtifact, ReplayScheduleClosure
from dagkv.c1_trace import (
    AtomicCutoffView,
    DemandServiceMode,
    ForecastAttemptContext,
    PredictedAttemptPayload,
    ReplayScheduleWatermarkPayload,
    ResidentExecMapService,
    ServiceDisposition,
    TerminalReason,
    TerminalStatus,
    TraceHeaderPayload,
    TraceRecordType,
    TraceValidationError,
    WaiterIdentity,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
    load_trace_jsonl,
    parse_trace_record,
    validate_trace,
)
from dagkv.domain import BindingState, BlockKey, LedgerAction, ReplicaId, Tier
from tests.test_c1_schedule import _artifact, _schedule_epoch, _schedule_event
from tests.test_c1_trace import _digest, _trace


@dataclass(frozen=True, slots=True)
class _CommitScenario:
    path: Path
    committer: CanonicalTraceCommitter
    schedule: ClosedScheduleArtifact
    schedule_digest: str
    preamble: TracePreambleRequest
    cutoff_request: CutoffCommitRequest
    cutoff_view: AtomicCutoffView
    demand_request: DemandCommitRequest
    demand_view: trace_module.PreServiceDemandView


def _scenario(
    tmp_path: Path, block_key: BlockKey, *, name: str = "trace"
) -> _CommitScenario:
    template = _trace(block_key)
    schedule_event = _schedule_event(template)
    schedule = _artifact(
        (schedule_event,),
        (_schedule_epoch(schedule_event),),
    )
    schedule_digest = sha256(canonical_json(schedule)).hexdigest()

    header = template[0].payload
    topology = template[1].payload
    cutoff = template[2].payload
    attempt = template[3].payload
    assert isinstance(header, TraceHeaderPayload)
    assert isinstance(topology, WorkflowTopologyPayload)
    assert isinstance(attempt, PredictedAttemptPayload)
    preamble = TracePreambleRequest(
        operation_id="preamble-1",
        header=replace(
            header,
            source_digest=schedule.source_artifact_digest,
            schedule_digest=schedule_digest,
        ),
        topologies=(topology,),
    )
    cutoff_view = AtomicCutoffView(
        snapshot=cutoff.snapshot,
        owner_specs=(topology.workflow_spec,),
        lifecycle_prefix=(),
        cutoff_ns=cutoff.cutoff_ns,
        horizon_duration_ns=cutoff.horizon_duration_ns,
        deadline_ns=cutoff.deadline_ns,
    )
    context = replace(
        attempt.context,
        information_cutoff_digest=cutoff_view.view_digest,
    )
    assert isinstance(context, ForecastAttemptContext)
    cutoff_request = CutoffCommitRequest(
        operation_id="cutoff-1",
        observation_id="observation-1",
        attempt=replace(attempt, context=context),
    )
    waiter = WaiterIdentity(
        binding_id=schedule_event.request_binding_id,
        workflow=schedule_event.workflow,
        request_id=schedule_event.execution_ref.request_id,
        node_id=schedule_event.node_id,
        execution_ref=schedule_event.execution_ref,
        state=BindingState.RETAINED,
    )
    demand_request = DemandCommitRequest(
        operation_id="demand-1",
        observation_id="observation-1",
        schedule_event_ids=(schedule_event.schedule_event_id,),
    )
    demand_view = trace_module.PreServiceDemandView(
        block_key=block_key,
        demand_commit_id=demand_request.operation_id,
        target_replica=ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1),
        action=LedgerAction.LOAD,
        service_mode=DemandServiceMode.RESIDENT,
        effective_transfer_id=None,
        effective_transfer_action=None,
        timestamp_ns=schedule_event.scheduled_access_ns,
        lifecycle_prefix=(),
        runtime_event_count=0,
        last_event_id=None,
        last_event_timestamp_ns=None,
        location_version=cutoff.snapshot.location_version,
        residency=cutoff.snapshot.residency,
        waiters=(waiter,),
    )
    path = tmp_path / f"{name}.jsonl"
    committer = CanonicalTraceCommitter(
        path,
        envelope=TraceEnvelope(
            trace_id=template[0].trace_id,
            run_id=schedule.run_id,
            schedule_id=schedule.schedule_id,
            schedule_case_id=schedule.schedule_case_id,
        ),
        schedule=schedule,
        schedule_artifact_digest=schedule_digest,
    )
    return _CommitScenario(
        path=path,
        committer=committer,
        schedule=schedule,
        schedule_digest=schedule_digest,
        preamble=preamble,
        cutoff_request=cutoff_request,
        cutoff_view=cutoff_view,
        demand_request=demand_request,
        demand_view=demand_view,
    )


def _commit_through_demand(
    scenario: _CommitScenario,
) -> WriterIssuedCommitReceipt:
    scenario.committer.commit_preamble(scenario.preamble)
    scenario.committer.commit_cutoff(
        scenario.cutoff_request,
        scenario.cutoff_view,
    )
    return scenario.committer.commit_demands(
        scenario.demand_request,
        scenario.demand_view,
    )


def _parse_incomplete_prefix(path: Path) -> tuple[trace_module.TraceRecord, ...]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    return tuple(parse_trace_record(line) for line in raw[:-1].split(b"\n"))


def _complete_request(
    scenario: _CommitScenario,
    demand_receipt: WriterIssuedCommitReceipt,
) -> ObservationCloseRequest:
    checkpoint = scenario.schedule.checkpoints[-1]
    closure = scenario.schedule.closure
    assert isinstance(closure, ReplayScheduleClosure)
    return ObservationCloseRequest(
        operation_id="close-1",
        observation_id=scenario.cutoff_request.observation_id,
        services=(
            ResidentExecMapService(
                intent_record_id=demand_receipt.commit.record_ids[0],
                disposition=ServiceDisposition.RESIDENT_EXEC_MAP,
                exec_map_event_id="event-exec-map-1",
            ),
        ),
        watermark=ReplayScheduleWatermarkPayload(
            producer_kind=scenario.schedule.producer_kind,
            producer_id=scenario.schedule.producer_id,
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
            label_available_ns=checkpoint.closed_through_ns,
            last_verified_event_count=0,
            last_verified_event_id=None,
            last_verified_event_timestamp_ns=None,
        ),
    )


def test_typed_committer_builds_and_seals_a_replayable_trace(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    close_receipt = scenario.committer.close_observation(
        _complete_request(scenario, demand_receipt)
    )

    sealed = scenario.committer.seal_trace()
    replayed_seal = scenario.committer.seal_trace()
    loaded = load_trace_jsonl(scenario.path)
    validated = validate_trace(loaded)

    assert loaded == scenario.committer.records
    assert replayed_seal is sealed
    assert sealed.closure.record_count == len(loaded) == 8
    assert sealed.closure.stream_digest == close_receipt.commit.committed_stream_digest
    assert tuple(record.record_type for record in loaded) == (
        TraceRecordType.TRACE_HEADER,
        TraceRecordType.WORKFLOW_TOPOLOGY,
        TraceRecordType.CUTOFF,
        TraceRecordType.FORECAST_ATTEMPT,
        TraceRecordType.DEMAND_INTENT,
        TraceRecordType.REUSE_EPOCH,
        TraceRecordType.SCHEDULE_WATERMARK,
        TraceRecordType.OBSERVATION_TERMINAL,
    )
    assert validated.observations[0].observation_id == "observation-1"


def test_exact_operation_replay_returns_same_receipt_without_appending(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    first = scenario.committer.commit_preamble(scenario.preamble)
    size = scenario.path.stat().st_size

    replay = scenario.committer.commit_preamble(scenario.preamble)

    assert replay is first
    assert scenario.path.stat().st_size == size
    with pytest.raises(TraceValidationError, match="reused"):
        scenario.committer.commit_preamble(
            replace(
                scenario.preamble,
                header=replace(
                    scenario.preamble.header,
                    implementation_digest=_digest("changed-implementation"),
                ),
            )
        )
    scenario.committer.abort()


def test_concurrent_exact_replay_appends_once_and_returns_one_capability(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(scenario.committer.commit_preamble, scenario.preamble)
            for _ in range(2)
        )
        receipts = tuple(future.result(timeout=5) for future in futures)

    assert receipts[0] is receipts[1]
    assert len(scenario.committer.commits) == 1
    assert _parse_incomplete_prefix(scenario.path) == scenario.committer.records
    scenario.committer.abort()


def test_cutoff_and_attempt_are_one_idempotent_operation(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer.commit_preamble(scenario.preamble)

    first = scenario.committer.commit_cutoff(
        scenario.cutoff_request,
        scenario.cutoff_view,
    )
    size = scenario.path.stat().st_size
    replay = scenario.committer.commit_cutoff(
        scenario.cutoff_request,
        scenario.cutoff_view,
    )

    assert replay is first
    assert first.commit.sequence_end - first.commit.sequence_start == 2
    assert first.commit.kind == TraceOperationKind.CUTOFF_ATTEMPT
    assert scenario.path.stat().st_size == size
    scenario.committer.abort()


def test_schedule_waiter_mismatch_is_rejected_before_any_append(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer.commit_preamble(scenario.preamble)
    scenario.committer.commit_cutoff(
        scenario.cutoff_request,
        scenario.cutoff_view,
    )
    size = scenario.path.stat().st_size
    mismatched_waiter = replace(scenario.demand_view.waiters[0], node_id="other")

    with pytest.raises(TraceValidationError, match="runtime waiter"):
        scenario.committer.commit_demands(
            scenario.demand_request,
            replace(scenario.demand_view, waiters=(mismatched_waiter,)),
        )

    assert scenario.path.stat().st_size == size
    assert scenario.committer.poisoned_reason is None
    scenario.committer.commit_demands(
        scenario.demand_request,
        scenario.demand_view,
    )
    scenario.committer.abort()


def test_receipt_must_be_the_exact_capability_issued_by_this_writer(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    first = _scenario(tmp_path, block_key, name="first")
    second = _scenario(tmp_path, block_key, name="second")
    first_receipt = first.committer.commit_preamble(first.preamble)
    second_receipt = second.committer.commit_preamble(second.preamble)
    forged = WriterIssuedCommitReceipt(
        commit=first_receipt.commit,
        receipt_id=first_receipt.receipt_id,
    )

    for receipt in (forged, second_receipt):
        with pytest.raises(TraceValidationError, match="not issued"):
            first.committer.verify_receipt(
                receipt,
                kind=TraceOperationKind.PREAMBLE,
                operation_id=first.preamble.operation_id,
                runtime_event_count=0,
                runtime_view_digest=canonical_digest(first.preamble),
            )
    first.committer.verify_receipt(
        first_receipt,
        kind=TraceOperationKind.PREAMBLE,
        operation_id=first.preamble.operation_id,
        runtime_event_count=0,
        runtime_view_digest=canonical_digest(first.preamble),
    )
    first.committer.abort()
    second.committer.abort()


def test_in_place_mutation_of_genuine_receipt_poisons_writer(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    receipt = scenario.committer.commit_preamble(scenario.preamble)
    changed_commit = replace(
        receipt.commit,
        batch_digest=_digest("forged-batch"),
    )
    object.__setattr__(receipt, "commit", changed_commit)
    object.__setattr__(receipt, "receipt_id", canonical_digest(changed_commit))

    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="receipt changed",
    ):
        scenario.committer.verify_receipt(
            receipt,
            kind=TraceOperationKind.PREAMBLE,
            operation_id=scenario.preamble.operation_id,
            runtime_event_count=0,
            runtime_view_digest=canonical_digest(scenario.preamble),
        )

    assert scenario.committer.poisoned_reason is not None
    scenario.committer.abort()


def test_exact_replay_checks_receipt_integrity_before_operation_kind(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    receipt = scenario.committer.commit_preamble(scenario.preamble)
    changed_commit = replace(
        receipt.commit,
        kind=TraceOperationKind.DEMAND_INTENT,
    )
    object.__setattr__(receipt, "commit", changed_commit)
    object.__setattr__(receipt, "receipt_id", canonical_digest(changed_commit))

    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="receipt changed",
    ):
        scenario.committer.commit_preamble(scenario.preamble)

    assert scenario.committer.poisoned_reason is not None
    scenario.committer.abort()


def test_seal_revalidates_detached_receipts_against_internal_commits(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    preamble_receipt = scenario.committer.commit_preamble(scenario.preamble)
    scenario.committer.commit_cutoff(
        scenario.cutoff_request,
        scenario.cutoff_view,
    )
    demand_receipt = scenario.committer.commit_demands(
        scenario.demand_request,
        scenario.demand_view,
    )
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    original_internal_digest = scenario.committer.commits[0].batch_digest
    changed_commit = replace(
        preamble_receipt.commit,
        batch_digest=_digest("mutated-external-receipt"),
    )
    object.__setattr__(preamble_receipt, "commit", changed_commit)
    object.__setattr__(
        preamble_receipt,
        "receipt_id",
        canonical_digest(changed_commit),
    )

    assert scenario.committer.commits[0].batch_digest == original_internal_digest
    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="receipt changed",
    ):
        scenario.committer.seal_trace()

    assert scenario.committer.poisoned_reason is not None
    scenario.committer.abort()


def test_watermark_must_equal_a_frozen_schedule_checkpoint(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    request = _complete_request(scenario, demand_receipt)
    assert isinstance(request.watermark, ReplayScheduleWatermarkPayload)
    changed = replace(
        request,
        watermark=replace(
            request.watermark,
            producer_artifact_digest=_digest("other-producer"),
        ),
    )
    size = scenario.path.stat().st_size

    with pytest.raises(TraceValidationError, match="exact frozen"):
        scenario.committer.close_observation(changed)

    assert scenario.path.stat().st_size == size
    scenario.committer.close_observation(request)
    scenario.committer.seal_trace()


def test_trace_cannot_seal_with_an_open_observation(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)

    with pytest.raises(TraceValidationError, match="open observations"):
        scenario.committer.seal_trace()

    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    scenario.committer.seal_trace()


def test_sealed_receipt_rejects_cross_trace_duplicate_and_nonzero_chain(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    sealed = scenario.committer.seal_trace()

    with pytest.raises(TraceValidationError, match="another trace pair"):
        replace(sealed, trace_pair_id="different-pair")
    duplicated = replace(
        sealed.operations[1],
        operation_id=sealed.operations[0].operation_id,
    )
    with pytest.raises(TraceValidationError, match="duplicated"):
        replace(
            sealed,
            operations=(sealed.operations[0], duplicated, *sealed.operations[2:]),
        )
    shifted = replace(sealed.operations[0], byte_start=1)
    with pytest.raises(TraceValidationError, match="start at zero"):
        replace(sealed, operations=(shifted, *sealed.operations[1:]))

    scenario.committer._poison(
        "post-seal response",
        _InjectedAbort("injected response loss"),
    )
    with pytest.raises(trace_module.TraceCommitIndeterminateError, match="poisoned"):
        scenario.committer.seal_trace()


def test_tamper_between_writer_seal_and_close_invalidates_attempt(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    receipt_type = commit_module.SealedTraceReceipt

    def tampering_receipt(*args: object, **kwargs: object) -> object:
        with scenario.path.open("r+b", buffering=0) as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(b"[" if first != b"[" else b"{")
        return receipt_type(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(commit_module, "SealedTraceReceipt", tampering_receipt)
        with pytest.raises(
            trace_module.TraceCommitIndeterminateError,
            match="sealed trace identity",
        ):
            scenario.committer.seal_trace()

    assert scenario.committer.poisoned_reason is not None


def test_tamper_after_writer_close_is_rejected_before_first_seal_return(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    original_close = scenario.committer._writer.close

    def close_then_tamper(*args: object, **kwargs: object) -> None:
        original_close(*args, **kwargs)  # type: ignore[arg-type]
        with scenario.path.open("r+b", buffering=0) as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(b"[" if first != b"[" else b"{")

    monkeypatch.setattr(scenario.committer._writer, "close", close_then_tamper)
    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="sealed trace file changed",
    ):
        scenario.committer.seal_trace()

    assert scenario.committer.poisoned_reason is not None


def test_mutated_final_receipt_cannot_be_replayed_or_snapshotted(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    sealed = scenario.committer.seal_trace()
    changed = replace(
        sealed.operations[1],
        runtime_view_digest=_digest("forged-runtime-view"),
    )
    object.__setattr__(
        sealed,
        "operations",
        (sealed.operations[0], changed, *sealed.operations[2:]),
    )

    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="sealed trace receipt changed",
    ):
        scenario.committer.seal_trace()

    assert scenario.committer.poisoned_reason is not None


def test_sealed_fast_path_revalidates_trace_file(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    demand_receipt = _commit_through_demand(scenario)
    scenario.committer.close_observation(_complete_request(scenario, demand_receipt))
    scenario.committer.seal_trace()
    with scenario.path.open("r+b", buffering=0) as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(b"[" if first != b"[" else b"{")

    with pytest.raises(
        trace_module.TraceCommitIndeterminateError,
        match="sealed trace file changed",
    ):
        scenario.committer.seal_trace()

    assert scenario.committer.poisoned_reason is not None


class _InjectedAbort(BaseException):
    pass


class _UnprintableAbort(BaseException):
    def __str__(self) -> str:
        raise RuntimeError("injected exception formatting failure")


class _FailingCounter(int):
    def __iadd__(self, _value: int) -> int:
        raise _InjectedAbort("post-fsync state publication failed")


def test_writer_response_loss_permanently_poisons_committer(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    original = scenario.committer._writer.append_durable

    def durable_then_lost(*args: object, **kwargs: object) -> object:
        original(*args, **kwargs)  # type: ignore[arg-type]
        raise _UnprintableAbort()

    monkeypatch.setattr(
        scenario.committer._writer,
        "append_durable",
        durable_then_lost,
    )
    with pytest.raises(_UnprintableAbort):
        scenario.committer.commit_preamble(scenario.preamble)

    assert scenario.path.stat().st_size > 0
    assert "unprintable failure" in (scenario.committer.poisoned_reason or "")
    scenario.committer.abort()


def test_post_fsync_writer_state_failure_poisons_both_layers(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer._writer._record_count = _FailingCounter(0)

    with pytest.raises(_InjectedAbort, match="state publication"):
        scenario.committer.commit_preamble(scenario.preamble)

    assert scenario.path.stat().st_size > 0
    assert scenario.committer._writer.poisoned
    assert scenario.committer.poisoned_reason is not None
    scenario.committer.abort()


def test_base_exception_during_write_permanently_poisons_committer(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer.commit_preamble(scenario.preamble)

    def fail_write(*_args: object, **_kwargs: object) -> int:
        raise _InjectedAbort("injected asynchronous abort")

    with monkeypatch.context() as context:
        context.setattr(trace_module.os, "write", fail_write)
        with pytest.raises(_InjectedAbort, match="asynchronous"):
            scenario.committer.commit_cutoff(
                scenario.cutoff_request,
                scenario.cutoff_view,
            )

    assert scenario.committer.poisoned_reason is not None
    with pytest.raises(trace_module.TraceCommitIndeterminateError, match="poisoned"):
        scenario.committer.commit_cutoff(
            scenario.cutoff_request,
            scenario.cutoff_view,
        )
    scenario.committer.abort()


def test_request_snapshot_blocks_digest_record_toctou(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    original_digest = scenario.preamble.header.implementation_digest
    original = CanonicalTraceCommitter._request_digest

    def mutate_external_request_after_digest(
        committer: CanonicalTraceCommitter,
        request: object,
        view_digest: str | None = None,
    ) -> str:
        result = original(committer, request, view_digest)
        object.__setattr__(
            scenario.preamble.header,
            "implementation_digest",
            _digest("mutated-after-digest"),
        )
        return result

    with monkeypatch.context() as context:
        context.setattr(
            CanonicalTraceCommitter,
            "_request_digest",
            mutate_external_request_after_digest,
        )
        scenario.committer.commit_preamble(scenario.preamble)

    header = _parse_incomplete_prefix(scenario.path)[0].payload
    assert isinstance(header, TraceHeaderPayload)
    assert header.implementation_digest == original_digest
    with pytest.raises(TraceValidationError, match="reused"):
        scenario.committer.commit_preamble(scenario.preamble)
    scenario.committer.abort()


def test_public_schedule_and_path_cannot_rebind_internal_identity(
    tmp_path: Path,
    block_key: BlockKey,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    exposed_schedule = scenario.committer.schedule
    object.__setattr__(exposed_schedule.events[0], "claim_id", "forged-claim")

    assert scenario.committer.schedule.events[0].claim_id != "forged-claim"
    with pytest.raises(AttributeError):
        scenario.committer.path = tmp_path / "other.jsonl"  # type: ignore[misc]
    _commit_through_demand(scenario)
    demand = next(
        record
        for record in _parse_incomplete_prefix(scenario.path)
        if record.record_type == TraceRecordType.DEMAND_INTENT
    )
    assert isinstance(demand.payload, trace_module.DemandIntentPayload)
    assert demand.payload.claim_id == scenario.schedule.events[0].claim_id
    scenario.committer.abort()


def test_post_durable_state_apply_failure_permanently_poisons_committer(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer.commit_preamble(scenario.preamble)

    def fail_apply(*_args: object, **_kwargs: object) -> None:
        raise _InjectedAbort("injected post-durable state failure")

    with monkeypatch.context() as context:
        context.setattr(CanonicalTraceCommitter, "_apply_cutoff", fail_apply)
        with pytest.raises(_InjectedAbort, match="post-durable"):
            scenario.committer.commit_cutoff(
                scenario.cutoff_request,
                scenario.cutoff_view,
            )

    assert len(_parse_incomplete_prefix(scenario.path)) == 4
    assert scenario.committer.poisoned_reason is not None
    with pytest.raises(trace_module.TraceCommitIndeterminateError, match="poisoned"):
        scenario.committer.commit_cutoff(
            scenario.cutoff_request,
            scenario.cutoff_view,
        )
    scenario.committer.abort()


def test_abort_close_failure_still_permanently_poisons_committer(
    tmp_path: Path,
    block_key: BlockKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario(tmp_path, block_key)
    scenario.committer.commit_preamble(scenario.preamble)

    def fail_integrity_check(_writer: object) -> None:
        raise OSError("injected abort integrity failure")

    with monkeypatch.context() as context:
        context.setattr(
            trace_module.DurableTraceWriter,
            "_assert_unchanged",
            fail_integrity_check,
        )
        with pytest.raises(OSError, match="abort integrity"):
            scenario.committer.abort()

    assert "abort close failed" in (scenario.committer.poisoned_reason or "")
    with pytest.raises(trace_module.TraceCommitIndeterminateError, match="poisoned"):
        scenario.committer.commit_preamble(scenario.preamble)
