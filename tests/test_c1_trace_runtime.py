"""Atomic runtime boundaries for C1-B cutoff and pre-service tracing."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from threading import Event, get_ident

import pytest

from dagkv.c1_trace import (
    AtomicCutoffView,
    DurableCommitReceipt,
    PreServiceDemandView,
)
from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    IdentityError,
    LedgerAction,
    ReplicaId,
    StateTransitionError,
    Tier,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)
from dagkv.orchestrator import LifecycleOrchestrator

_BATCH_DIGEST = sha256(b"c1-runtime-test-batch").hexdigest()


def _receipt(
    *,
    record_id: str,
    event_count: int,
    view_digest: str,
) -> DurableCommitReceipt:
    return DurableCommitReceipt(
        record_ids=(record_id,),
        event_count=event_count,
        view_digest=view_digest,
        batch_digest=_BATCH_DIGEST,
    )


class _CutoffCommitter:
    def __init__(self) -> None:
        self.views: list[AtomicCutoffView] = []

    def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
        self.views.append(view)
        return _receipt(
            record_id="cutoff-record",
            event_count=len(view.lifecycle_prefix),
            view_digest=view.view_digest,
        )


class _DemandCommitter:
    def __init__(self, runtime: LifecycleOrchestrator) -> None:
        self.runtime = runtime
        self.views: list[PreServiceDemandView] = []
        self.event_prefixes: list[tuple[object, ...]] = []

    def commit_demands(
        self,
        view: PreServiceDemandView,
    ) -> DurableCommitReceipt:
        self.views.append(view)
        self.event_prefixes.append(self.runtime.events)
        return _receipt(
            record_id="demand-record",
            event_count=view.runtime_event_count,
            view_digest=view.view_digest,
        )


def _workflow() -> WorkflowSpec:
    return WorkflowSpec(
        key=WorkflowKey("c1-runtime-workflow", 0),
        nodes=(WorkflowNode("agent"),),
    )


def _runtime_with_retained_waiter(
    block_key: BlockKey,
    digest: Callable[[str], str],
    *,
    trace_required: bool,
) -> tuple[
    LifecycleOrchestrator,
    WorkflowSpec,
    BindingHandle,
    BindingHandle,
    ReplicaId,
]:
    runtime = LifecycleOrchestrator(
        run_id="c1-runtime",
        phase="m3_c1b",
        trace_required=trace_required,
    )
    workflow = _workflow()
    runtime.register_workflow(workflow)
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1)
    runtime.register_gpu_block(
        block_key,
        gpu,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "agent", timestamp_ns=2)

    retention = BindingHandle(workflow.key, "retention", "retention-binding")
    runtime.bind_owner(
        retention,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=3,
    )
    request = BindingHandle(workflow.key, "request", "request-binding")
    runtime.bind_owner(
        request,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(workflow.key, "request", "sequence", 0),
        timestamp_ns=4,
    )
    return runtime, workflow, retention, request, gpu


def _drop_to_cpu(
    runtime: LifecycleOrchestrator,
    block_key: BlockKey,
    gpu: ReplicaId,
) -> ReplicaId:
    cpu = ReplicaId(Tier.CPU, "numa:0", "slot-0", 1)
    save = runtime.begin_d2h(
        block_key,
        cpu,
        transfer_id="save-for-c1",
        timestamp_ns=5,
    )
    assert save is not None
    runtime.complete_transfer(
        save.transfer_id,
        timestamp_ns=6,
        observed_bytes=save.byte_count,
        observed_digest=save.payload_digest,
    )
    snapshot = runtime.block_snapshot(block_key)
    assert runtime.drop_gpu(
        block_key,
        expected_gpu=gpu,
        expected_location_version=snapshot.location_version,
        timestamp_ns=7,
    )
    return cpu


def test_cutoff_holds_runtime_lock_and_binds_exact_prefix(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, retention, _, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    entered = Event()
    release = Event()

    class BlockingCommitter:
        def commit_cutoff(
            self,
            view: AtomicCutoffView,
        ) -> DurableCommitReceipt:
            entered.set()
            assert release.wait(timeout=5)
            return _receipt(
                record_id="cutoff-record",
                event_count=len(view.lifecycle_prefix),
                view_digest=view.view_digest,
            )

    mutation_started = Event()
    read_started = Event()

    def mutate() -> str:
        mutation_started.set()
        return runtime.open_lease(
            retention,
            "post-cutoff-lease",
            registered_ns=5,
            deadline_ns=9,
            reason="post_cutoff",
        )

    def read_events() -> tuple[object, ...]:
        read_started.set()
        return runtime.events

    with ThreadPoolExecutor(max_workers=3) as pool:
        cutoff_future = pool.submit(
            runtime.commit_shared_lease_cutoff,
            block_key,
            cutoff_ns=4,
            horizon_duration_ns=10,
            committer=BlockingCommitter(),
        )
        assert entered.wait(timeout=5)
        mutation_future = pool.submit(mutate)
        read_future = pool.submit(read_events)
        assert mutation_started.wait(timeout=5)
        assert read_started.wait(timeout=5)
        with pytest.raises(StateTransitionError, match="trace committer callback"):
            mutation_future.result(timeout=5)
        assert not read_future.done()
        release.set()
        view, receipt = cutoff_future.result(timeout=5)
        post_commit_events = read_future.result(timeout=5)

    assert view.owner_specs == (workflow,)
    assert view.snapshot.runtime_event_count == len(view.lifecycle_prefix)
    assert view.lifecycle_prefix[-1].event_id == "evt-000000000005"
    assert receipt.event_count == len(view.lifecycle_prefix)
    assert post_commit_events[: len(view.lifecycle_prefix)] == view.lifecycle_prefix
    final_events = runtime.events
    assert final_events == view.lifecycle_prefix


def test_lifecycle_seal_wins_cutoff_lock_race_before_durable_commit(
    monkeypatch: pytest.MonkeyPatch,
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, _, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _CutoffCommitter()
    outer_guard_passed = Event()
    main_thread_id = get_ident()
    original_guard = runtime._guard_runtime_mutation

    def observed_guard() -> None:
        original_guard()
        if get_ident() != main_thread_id and not outer_guard_passed.is_set():
            outer_guard_passed.set()

    monkeypatch.setattr(runtime, "_guard_runtime_mutation", observed_guard)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with runtime._lock:
            future = pool.submit(
                runtime.commit_shared_lease_cutoff,
                block_key,
                cutoff_ns=10,
                horizon_duration_ns=20,
                committer=committer,
            )
            assert outer_guard_passed.wait(timeout=2)
            runtime.seal_lifecycle()

        with pytest.raises(StateTransitionError, match="lifecycle stream is sealed"):
            future.result()
    assert committer.views == []


def test_lifecycle_seal_wins_workflow_registration_lock_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = LifecycleOrchestrator(run_id="seal-registration-race")
    workflow = WorkflowSpec(
        WorkflowKey("late-workflow", 0),
        (WorkflowNode("agent"),),
    )
    outer_guard_passed = Event()
    main_thread_id = get_ident()
    original_guard = runtime._guard_runtime_mutation

    def observed_guard() -> None:
        original_guard()
        if get_ident() != main_thread_id and not outer_guard_passed.is_set():
            outer_guard_passed.set()

    monkeypatch.setattr(runtime, "_guard_runtime_mutation", observed_guard)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with runtime._lock:
            future = pool.submit(runtime.register_workflow, workflow)
            assert outer_guard_passed.wait(timeout=2)
            runtime.seal_lifecycle()

        with pytest.raises(StateTransitionError, match="lifecycle stream is sealed"):
            future.result()
    with pytest.raises(IdentityError, match="unknown workflow"):
        runtime.workflow_snapshot(workflow.key)


def test_cutoff_callback_failure_preserves_runtime(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    before_events = runtime.events
    before_block = runtime.block_snapshot(block_key)
    before_binding = runtime.binding_snapshot(request)

    class FailingCommitter:
        def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
            raise OSError(f"injected fsync failure for {view.view_digest}")

    with pytest.raises(OSError, match="injected fsync failure"):
        runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=4,
            horizon_duration_ns=10,
            committer=FailingCommitter(),
        )

    assert runtime.events == before_events
    assert runtime.block_snapshot(block_key) == before_block
    assert runtime.binding_snapshot(request) == before_binding


def test_cutoff_committer_cannot_reenter_a_runtime_mutation(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, retention, _, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    before_events = runtime.events
    before_block = runtime.block_snapshot(block_key)

    class ReentrantCommitter:
        def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
            runtime.open_lease(
                retention,
                "reentrant-lease",
                registered_ns=5,
                deadline_ns=9,
                reason="forbidden_reentry",
            )
            raise AssertionError("mutation guard did not reject reentry")

    with pytest.raises(StateTransitionError, match="trace committer callback"):
        runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=4,
            horizon_duration_ns=10,
            committer=ReentrantCommitter(),
        )

    assert runtime.events == before_events
    assert runtime.block_snapshot(block_key) == before_block


def test_cutoff_committer_cannot_register_a_workflow(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, _, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    nested_workflow = WorkflowSpec(
        key=WorkflowKey("reentrant-workflow", 0),
        nodes=(WorkflowNode("agent"),),
    )
    before_events = runtime.events

    class ReentrantCommitter:
        def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
            runtime.register_workflow(nested_workflow)
            raise AssertionError("mutation guard did not reject workflow registration")

    with pytest.raises(StateTransitionError, match="trace committer callback"):
        runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=4,
            horizon_duration_ns=10,
            committer=ReentrantCommitter(),
        )

    assert runtime.events == before_events
    with pytest.raises(IdentityError, match="unknown workflow"):
        runtime.workflow_snapshot(nested_workflow.key)


def test_cutoff_committer_cannot_reenter_an_idempotent_single_flight_join(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, _, replay_waiter, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (replay_waiter,),
        transfer_id="resident-before-replay",
        timestamp_ns=5,
        demand_commit_id="replay-demand",
        demand_committer=committer,
    )
    runtime.set_binding_state(
        replay_waiter,
        BindingState.RETAINED,
        timestamp_ns=5,
    )
    _drop_to_cpu(runtime, block_key, gpu)

    transfer_waiter = BindingHandle(
        workflow.key,
        "transfer-request",
        "transfer-request-binding",
    )
    runtime.bind_owner(
        transfer_waiter,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(
            workflow.key,
            "transfer-request",
            "sequence",
            0,
        ),
        timestamp_ns=8,
    )
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    command = runtime.ensure_h2d(
        block_key,
        target,
        (transfer_waiter,),
        transfer_id="active-load",
        timestamp_ns=9,
        demand_commit_id="active-load-demand",
        demand_committer=committer,
    )
    assert command is not None
    before_events = runtime.events
    before_transfer = runtime.transfer_snapshot(command.transfer_id)
    before_binding = runtime.binding_snapshot(replay_waiter)

    class ReentrantCommitter:
        def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
            runtime.ensure_h2d(
                block_key,
                target,
                (replay_waiter,),
                transfer_id="ignored-replay-transfer",
                timestamp_ns=9,
                demand_commit_id="replay-demand",
            )
            raise AssertionError("mutation guard did not reject single-flight replay")

    with pytest.raises(StateTransitionError, match="trace committer callback"):
        runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=9,
            horizon_duration_ns=10,
            committer=ReentrantCommitter(),
        )

    assert runtime.events == before_events
    assert runtime.transfer_snapshot(command.transfer_id) == before_transfer
    assert runtime.binding_snapshot(replay_waiter) == before_binding


def test_cutoff_rejects_receipt_for_another_runtime_view(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, _, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )

    class WrongReceiptCommitter:
        def commit_cutoff(
            self,
            view: AtomicCutoffView,
        ) -> DurableCommitReceipt:
            return _receipt(
                record_id="cutoff-record",
                event_count=len(view.lifecycle_prefix) + 1,
                view_digest=view.view_digest,
            )

    before_events = runtime.events
    with pytest.raises(StateTransitionError, match="event count differs"):
        runtime.commit_shared_lease_cutoff(
            block_key,
            cutoff_ns=4,
            horizon_duration_ns=10,
            committer=WrongReceiptCommitter(),
        )
    assert runtime.events == before_events


def test_resident_demand_is_durable_before_exec_map(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    before_count = len(runtime.events)
    committer = _DemandCommitter(runtime)

    command = runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="resident-demand",
        timestamp_ns=5,
        demand_commit_id="resident-demand-1",
        demand_committer=committer,
    )

    assert command is None
    assert len(committer.views) == 1
    assert committer.views[0].runtime_event_count == before_count
    assert len(committer.event_prefixes[0]) == before_count
    assert runtime.events[before_count].action == LedgerAction.EXEC_MAP
    assert runtime.binding_snapshot(request).state == BindingState.REQUIRED
    assert runtime.is_ready(request)


def test_h2d_demand_is_durable_before_transfer_schedule(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    before_count = len(runtime.events)
    committer = _DemandCommitter(runtime)

    command = runtime.ensure_h2d(
        block_key,
        target,
        (request,),
        transfer_id="load-for-c1",
        timestamp_ns=8,
        demand_commit_id="h2d-demand-1",
        demand_committer=committer,
    )

    assert command is not None
    assert len(committer.event_prefixes[0]) == before_count
    assert committer.views[0].runtime_event_count == before_count
    assert committer.views[0].target_replica == target
    assert [event.action for event in runtime.events[before_count:]] == [
        LedgerAction.ALLOCATE,
        LedgerAction.LOAD,
        LedgerAction.BIND_STATE,
        LedgerAction.WAITER_JOIN,
        LedgerAction.BLOCK_STATE,
    ]
    assert runtime.binding_snapshot(request).state == BindingState.WAITING


def test_resident_mapping_replay_does_not_create_a_second_demand(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="resident-demand",
        timestamp_ns=5,
        demand_commit_id="resident-demand-1",
        demand_committer=committer,
    )
    committed_events = runtime.events

    assert (
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="resident-demand-replay",
            timestamp_ns=5,
            demand_commit_id="resident-demand-1",
        )
        is None
    )
    assert len(committer.views) == 1
    assert runtime.events == committed_events


def test_dispatched_resident_demand_cannot_publish_a_second_service(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="resident-demand",
        timestamp_ns=5,
        demand_commit_id="resident-demand-1",
        demand_committer=committer,
    )
    runtime.set_binding_state(request, BindingState.RETAINED, timestamp_ns=5)
    before_events = runtime.events

    with pytest.raises(StateTransitionError, match="second resident service"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="resident-demand-replay",
            timestamp_ns=5,
            demand_commit_id="resident-demand-1",
        )
    assert runtime.events == before_events
    assert len(committer.views) == 1


def test_same_demand_id_replay_is_independent_of_waiter_order(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, _, first, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    second = BindingHandle(
        workflow.key,
        "second-request",
        "second-request-binding",
    )
    runtime.bind_owner(
        second,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(
            workflow.key,
            "second-request",
            "sequence",
            0,
        ),
        timestamp_ns=5,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (first, second),
        transfer_id="ordered-resident-demand",
        timestamp_ns=6,
        demand_commit_id="two-waiter-demand",
        demand_committer=committer,
    )
    committed_events = runtime.events

    assert (
        runtime.ensure_h2d(
            block_key,
            gpu,
            (second, first),
            transfer_id="reversed-resident-replay",
            timestamp_ns=6,
            demand_commit_id="two-waiter-demand",
        )
        is None
    )
    assert len(committer.views) == 1
    assert runtime.events == committed_events


def test_same_execution_cannot_claim_a_second_resident_demand_id(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="resident-demand",
        timestamp_ns=5,
        demand_commit_id="resident-demand-1",
        demand_committer=committer,
    )
    with pytest.raises(IdentityError, match="already has another demand commit ID"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="resident-demand-2",
            timestamp_ns=5,
            demand_commit_id="resident-demand-2",
            demand_committer=committer,
        )

    assert len(committer.views) == 1


def test_new_request_binding_commits_a_new_resident_demand(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, workflow, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="resident-demand",
        timestamp_ns=5,
        demand_commit_id="resident-demand-1",
        demand_committer=committer,
    )
    second = BindingHandle(workflow.key, "request-2", "request-binding-2")
    runtime.bind_owner(
        second,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(workflow.key, "request-2", "sequence", 0),
        timestamp_ns=6,
    )
    before_count = len(runtime.events)

    assert (
        runtime.ensure_h2d(
            block_key,
            gpu,
            (second,),
            transfer_id="resident-demand-2",
            timestamp_ns=7,
            demand_commit_id="resident-demand-2",
            demand_committer=committer,
        )
        is None
    )
    assert [view.demand_commit_id for view in committer.views] == [
        "resident-demand-1",
        "resident-demand-2",
    ]
    assert runtime.events[before_count].action == LedgerAction.EXEC_MAP


def test_existing_single_flight_waiter_replay_does_not_create_a_second_demand(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    committer = _DemandCommitter(runtime)
    command = runtime.ensure_h2d(
        block_key,
        target,
        (request,),
        transfer_id="load-for-c1",
        timestamp_ns=8,
        demand_commit_id="h2d-demand-1",
        demand_committer=committer,
    )
    assert command is not None
    committed_events = runtime.events

    replay = runtime.ensure_h2d(
        block_key,
        target,
        (request,),
        transfer_id="ignored-single-flight-replay",
        timestamp_ns=8,
        demand_commit_id="h2d-demand-1",
    )

    assert replay == command
    assert len(committer.views) == 1
    assert runtime.events == committed_events


def test_failed_h2d_physical_retry_requires_an_attempt_chain_schema(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    first_target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    committer = _DemandCommitter(runtime)
    first = runtime.ensure_h2d(
        block_key,
        first_target,
        (request,),
        transfer_id="load-for-c1",
        timestamp_ns=8,
        demand_commit_id="h2d-demand-1",
        demand_committer=committer,
    )
    assert first is not None
    runtime.set_binding_state(
        request,
        BindingState.RETAINED,
        timestamp_ns=8,
    )
    assert (
        request.binding_id
        not in runtime.transfer_snapshot(first.transfer_id).waiter_binding_ids
    )
    runtime.fail_transfer(
        first.transfer_id,
        timestamp_ns=9,
        observed_bytes=0,
        observed_digest=None,
        error="injected worker failure",
    )

    with pytest.raises(IdentityError, match="another logical demand"):
        runtime.ensure_h2d(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot-0", 3),
            (request,),
            transfer_id="load-for-c1-retry",
            timestamp_ns=10,
            demand_commit_id="h2d-demand-1",
        )
    assert len(committer.views) == 1


def test_trace_committer_requires_an_explicit_demand_commit_id(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    before_events = runtime.events

    with pytest.raises(StateTransitionError, match="demand commit ID"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="missing-demand-id",
            timestamp_ns=5,
            demand_committer=committer,
        )

    assert committer.views == []
    assert runtime.events == before_events


def test_demand_callback_failure_preserves_runtime(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    before_events = runtime.events
    before_block = runtime.block_snapshot(block_key)
    before_binding = runtime.binding_snapshot(request)

    class FailingCommitter:
        def commit_demands(
            self,
            view: PreServiceDemandView,
        ) -> DurableCommitReceipt:
            raise OSError(f"injected demand fsync failure for {view.view_digest}")

    with pytest.raises(OSError, match="injected demand fsync failure"):
        runtime.ensure_h2d(
            block_key,
            target,
            (request,),
            transfer_id="load-for-c1",
            timestamp_ns=8,
            demand_commit_id="failing-demand-1",
            demand_committer=FailingCommitter(),
        )

    assert runtime.events == before_events
    assert runtime.block_snapshot(block_key) == before_block
    assert runtime.binding_snapshot(request) == before_binding


def test_demand_committer_cannot_reenter_service(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    before_events = runtime.events
    before_binding = runtime.binding_snapshot(request)
    nested = _DemandCommitter(runtime)

    class ReentrantCommitter:
        def commit_demands(
            self,
            view: PreServiceDemandView,
        ) -> DurableCommitReceipt:
            runtime.ensure_h2d(
                block_key,
                gpu,
                (request,),
                transfer_id="nested-resident-demand",
                timestamp_ns=5,
                demand_commit_id="nested-demand",
                demand_committer=nested,
            )
            raise AssertionError("service guard did not reject reentry")

    with pytest.raises(StateTransitionError, match="trace committer callback"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="outer-resident-demand",
            timestamp_ns=5,
            demand_commit_id="outer-demand",
            demand_committer=ReentrantCommitter(),
        )

    assert nested.views == []
    assert runtime.events == before_events
    assert runtime.binding_snapshot(request) == before_binding


def test_default_runtime_rejects_late_resident_demand_instrumentation(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="untraced-resident-demand",
        timestamp_ns=5,
    )
    committer = _DemandCommitter(runtime)
    before_events = runtime.events
    before_binding = runtime.binding_snapshot(request)

    with pytest.raises(StateTransitionError, match="fresh demand waiter"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="late-resident-demand",
            timestamp_ns=5,
            demand_commit_id="late-resident-demand",
            demand_committer=committer,
        )

    assert committer.views == []
    assert runtime.events == before_events
    assert runtime.binding_snapshot(request) == before_binding


def test_default_runtime_rejects_late_single_flight_demand_instrumentation(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)
    command = runtime.ensure_h2d(
        block_key,
        target,
        (request,),
        transfer_id="untraced-active-load",
        timestamp_ns=8,
    )
    assert command is not None
    committer = _DemandCommitter(runtime)
    before_events = runtime.events
    before_binding = runtime.binding_snapshot(request)
    before_transfer = runtime.transfer_snapshot(command.transfer_id)

    with pytest.raises(StateTransitionError, match="fresh demand waiter"):
        runtime.ensure_h2d(
            block_key,
            target,
            (request,),
            transfer_id="late-active-load",
            timestamp_ns=8,
            demand_commit_id="late-active-load-demand",
            demand_committer=committer,
        )

    assert committer.views == []
    assert runtime.events == before_events
    assert runtime.binding_snapshot(request) == before_binding
    assert runtime.transfer_snapshot(command.transfer_id) == before_transfer


def test_default_runtime_rejects_fresh_demand_after_execution_unmap(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    runtime.ensure_h2d(
        block_key,
        gpu,
        (request,),
        transfer_id="untraced-resident-service",
        timestamp_ns=5,
    )
    runtime.set_binding_state(
        request,
        BindingState.RETAINED,
        timestamp_ns=6,
    )
    committer = _DemandCommitter(runtime)
    before_events = runtime.events
    before_binding = runtime.binding_snapshot(request)

    with pytest.raises(StateTransitionError, match="execution-map history"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="late-fresh-resident-service",
            timestamp_ns=7,
            demand_commit_id="late-fresh-resident-demand",
            demand_committer=committer,
        )

    assert committer.views == []
    assert runtime.events == before_events
    assert runtime.binding_snapshot(request) == before_binding


def test_default_runtime_rejects_fresh_demand_after_untraced_h2d_failure(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, gpu = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=False,
    )
    _drop_to_cpu(runtime, block_key, gpu)
    first = runtime.ensure_h2d(
        block_key,
        ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2),
        (request,),
        transfer_id="untraced-failed-load",
        timestamp_ns=8,
    )
    assert first is not None
    runtime.set_binding_state(
        request,
        BindingState.RETAINED,
        timestamp_ns=8,
    )
    assert (
        request.binding_id
        not in runtime.transfer_snapshot(first.transfer_id).waiter_binding_ids
    )
    runtime.fail_transfer(
        first.transfer_id,
        timestamp_ns=9,
        observed_bytes=0,
        observed_digest=None,
        error="injected untraced failure",
    )
    committer = _DemandCommitter(runtime)
    before_events = runtime.events
    before_binding = runtime.binding_snapshot(request)
    before_block = runtime.block_snapshot(block_key)

    with pytest.raises(StateTransitionError, match="transfer-service history"):
        runtime.ensure_h2d(
            block_key,
            ReplicaId(Tier.GPU, "cuda:0", "slot-0", 3),
            (request,),
            transfer_id="late-fresh-load-retry",
            timestamp_ns=10,
            demand_commit_id="late-fresh-load-demand",
            demand_committer=committer,
        )

    assert committer.views == []
    assert runtime.events == before_events
    assert runtime.binding_snapshot(request) == before_binding
    assert runtime.block_snapshot(block_key) == before_block


def test_trace_required_rejects_all_exec_map_bypasses(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime = LifecycleOrchestrator(run_id="trace-required", trace_required=True)
    workflow = _workflow()
    runtime.register_workflow(workflow)
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1)
    runtime.register_gpu_block(
        block_key,
        gpu,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "agent", timestamp_ns=2)
    request = BindingHandle(workflow.key, "request", "request-binding")
    execution_ref = ExecutionRef(workflow.key, "request", "sequence", 0)

    before_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="bind as retained"):
        runtime.bind_owner(
            request,
            node_id="agent",
            block_key=block_key,
            kind=BindingKind.REQUEST,
            state=BindingState.REQUIRED,
            execution_ref=execution_ref,
            timestamp_ns=3,
        )
    assert len(runtime.events) == before_count

    runtime.bind_owner(
        request,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=execution_ref,
        timestamp_ns=3,
    )
    before_count = len(runtime.events)
    with pytest.raises(StateTransitionError, match="demand gate"):
        runtime.set_binding_state(
            request,
            BindingState.REQUIRED,
            timestamp_ns=4,
        )
    assert len(runtime.events) == before_count

    with pytest.raises(StateTransitionError, match="durable demand commit"):
        runtime.ensure_h2d(
            block_key,
            gpu,
            (request,),
            transfer_id="naked-resident-demand",
            timestamp_ns=4,
            demand_commit_id="naked-resident-demand-1",
        )
    assert len(runtime.events) == before_count
    assert runtime.binding_snapshot(request).state == BindingState.RETAINED


def test_service_preflight_rejects_before_demand_callback(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime, _, _, request, _ = _runtime_with_retained_waiter(
        block_key,
        digest,
        trace_required=True,
    )
    committer = _DemandCommitter(runtime)
    wrong_target = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 2)

    with pytest.raises(StateTransitionError, match="target disagrees"):
        runtime.ensure_h2d(
            block_key,
            wrong_target,
            (request,),
            transfer_id="invalid-target",
            timestamp_ns=5,
            demand_committer=committer,
        )

    assert committer.views == []


def test_default_runtime_keeps_untraced_mapping_compatibility(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    runtime = LifecycleOrchestrator(run_id="default-runtime")
    workflow = _workflow()
    runtime.register_workflow(workflow)
    gpu = ReplicaId(Tier.GPU, "cuda:0", "slot-0", 1)
    runtime.register_gpu_block(
        block_key,
        gpu,
        byte_capacity=4096,
        payload_size=3072,
        payload_digest=digest("payload"),
        timestamp_ns=1,
    )
    runtime.start_node(workflow.key, "agent", timestamp_ns=2)
    request = BindingHandle(workflow.key, "request", "request-binding")
    runtime.bind_owner(
        request,
        node_id="agent",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.REQUIRED,
        execution_ref=ExecutionRef(workflow.key, "request", "sequence", 0),
        timestamp_ns=3,
    )

    assert runtime.is_ready(request)
