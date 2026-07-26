"""Sole-writer lifecycle orchestrator for DAG-aware KV-cache state."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import monotonic_ns

from dagkv.c1_leases import LeaseOwnerSnapshot, SharedLeasePolicySnapshot
from dagkv.c1_trace import (
    AtomicCutoffView,
    CutoffCommitter,
    DemandCommitter,
    DurableCommitReceipt,
    PreServiceDemandView,
    WaiterIdentity,
)
from dagkv.domain import (
    AuditReport,
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    BlockRecord,
    BlockStateSnapshot,
    ContentMapping,
    ExecutionMapping,
    ExecutionRef,
    IdentityError,
    Lease,
    LeaseState,
    LedgerAction,
    LedgerStatus,
    LifecycleEvent,
    NodeRecord,
    NodeStatus,
    OwnerBinding,
    ReplicaId,
    ReplicaRecord,
    ReplicaReservation,
    ResidencyState,
    StateTransitionError,
    Tier,
    Transfer,
    TransferCommand,
    TransferDirection,
    TransferIntegrityError,
    TransferState,
    WorkflowKey,
    WorkflowRecord,
    WorkflowSpec,
    WorkflowStatus,
    require_sha256,
    require_text,
)
from dagkv.ledger import EventDraft, EventLedger


@dataclass(frozen=True, slots=True)
class _BindingReleasePlan:
    """Ephemeral transition plan committed only after one ledger batch passes."""

    binding: OwnerBinding
    leases: tuple[Lease, ...]
    mapping: ExecutionMapping | None
    drafts: tuple[EventDraft, ...]
    timestamp_ns: int
    reason: str


class LifecycleOrchestrator:
    """Own every mutable lifecycle registry behind one atomic writer lock."""

    def __init__(
        self,
        *,
        run_id: str,
        phase: str = "m2_component",
        source: str = "dagkv.orchestrator",
        trace_required: bool = False,
    ) -> None:
        if type(trace_required) is not bool:
            raise IdentityError("trace_required must be a bool")
        self._lock = RLock()
        self._trace_callback_active = False
        self._ledger = EventLedger(
            run_id=run_id,
            phase=phase,
            source=source,
            mutation_guard=self._guard_runtime_mutation,
            require_complete_state=True,
        )
        self._trace_required = trace_required
        self._workflows: dict[WorkflowKey, WorkflowRecord] = {}
        self._blocks: dict[BlockKey, BlockRecord] = {}
        self._bindings: dict[str, OwnerBinding] = {}
        self._leases: dict[str, Lease] = {}
        self._content_mappings: dict[str, ContentMapping] = {}
        self._execution_owners: dict[ExecutionRef, str] = {}
        self._execution_history: dict[ExecutionRef, tuple[BlockKey, str]] = {}
        self._execution_mappings: dict[ExecutionRef, ExecutionMapping] = {}
        self._execution_mapping_generations: dict[str, int] = {}
        self._committed_demand_ids: dict[
            str,
            tuple[
                BlockKey,
                tuple[tuple[str, ExecutionRef], ...],
                ReplicaId,
                LedgerAction,
                int,
            ],
        ] = {}
        self._demand_id_by_execution: dict[ExecutionRef, str] = {}
        self._dispatched_demand_ids: set[str] = set()
        # Monotonic: removing an active waiter must not erase prior service.
        self._transfer_waiter_history: set[str] = set()
        self._transfers: dict[str, Transfer] = {}
        self._reservations: dict[ReplicaId, ReplicaReservation] = {}

        # These indices prevent ABA reuse even after a slot is physically free.
        self._slot_generations: dict[tuple[Tier, str, str], int] = {}
        self._slot_occupants: dict[tuple[Tier, str, str], ReplicaId] = {}
        self._hard_failures: list[str] = []
        self._runtime_poisoned: str | None = None
        self._seal_event: LifecycleEvent | None = None

    @property
    def events(self) -> tuple[LifecycleEvent, ...]:
        """Return an immutable ledger snapshot."""

        with self._lock:
            return self._ledger.events

    def workflow_snapshot(self, key: WorkflowKey) -> WorkflowRecord:
        """Return a detached workflow snapshot."""

        with self._lock:
            return deepcopy(self._workflow(key))

    def block_snapshot(self, key: BlockKey) -> BlockRecord:
        """Return a detached block snapshot."""

        with self._lock:
            return deepcopy(self._block(key))

    def binding_snapshot(self, handle: BindingHandle) -> OwnerBinding:
        """Return a detached owner binding snapshot."""

        with self._lock:
            return deepcopy(self._binding(handle))

    def transfer_snapshot(self, transfer_id: str) -> Transfer:
        """Return a detached transfer snapshot."""

        with self._lock:
            try:
                return deepcopy(self._transfers[transfer_id])
            except KeyError as exc:
                raise IdentityError(f"unknown transfer: {transfer_id}") from exc

    def seal_lifecycle(self) -> LifecycleEvent:
        """Close the lifecycle stream at a sole-writer monotonic-clock boundary."""

        with self._lock:
            if self._seal_event is not None:
                return self._seal_event
            self._guard_runtime_mutation()
            report = self.audit()
            if not report.passed:
                raise StateTransitionError(
                    f"cannot seal an inconsistent lifecycle: {report.issues[0]}"
                )
            timestamp_ns = monotonic_ns()
            if timestamp_ns < self._ledger.last_timestamp_ns:
                raise StateTransitionError(
                    "monotonic seal clock predates the lifecycle stream"
                )
            event = self._ledger.append(
                EventDraft(
                    action=LedgerAction.STREAM_SEAL,
                    status=LedgerStatus.COMPLETED,
                    reason="lifecycle_stream_closed",
                    timestamp_ns=timestamp_ns,
                    operation_id="lifecycle-stream-seal",
                )
            )
            self._seal_event = event
            return event

    def shared_lease_policy_snapshot(
        self,
        key: BlockKey,
    ) -> SharedLeasePolicySnapshot:
        """Project active retention owners into an immutable C1 input."""

        with self._lock:
            lifecycle_prefix = self._ledger.events
            return self._shared_lease_policy_snapshot_locked(
                key,
                runtime_event_count=len(lifecycle_prefix),
            )

    def commit_shared_lease_cutoff(
        self,
        key: BlockKey,
        *,
        cutoff_ns: int,
        horizon_duration_ns: int,
        committer: CutoffCommitter,
    ) -> tuple[AtomicCutoffView, DurableCommitReceipt]:
        """Durably bind one forecast attempt to an atomic runtime prefix."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            view = self._atomic_cutoff_view_locked(
                key,
                cutoff_ns=cutoff_ns,
                horizon_duration_ns=horizon_duration_ns,
            )
            if self._trace_callback_active:
                raise StateTransitionError("trace committer callback cannot reenter")
            self._trace_callback_active = True
            try:
                receipt = committer.commit_cutoff(view)
            finally:
                self._trace_callback_active = False
            current = self._atomic_cutoff_view_locked(
                key,
                cutoff_ns=cutoff_ns,
                horizon_duration_ns=horizon_duration_ns,
            )
            if current != view:
                raise StateTransitionError(
                    "cutoff committer changed the atomic runtime view"
                )
            self._validate_trace_receipt(
                receipt,
                event_count=len(view.lifecycle_prefix),
                view_digest=view.view_digest,
            )
            return view, receipt

    def register_workflow(self, spec: WorkflowSpec) -> bool:
        """Register one immutable DAG, initializing roots as ready."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            existing = self._workflows.get(spec.key)
            if existing is not None:
                if existing.spec == spec:
                    return False
                raise IdentityError(f"workflow identity conflict: {spec.key}")
            nodes = {
                node.node_id: NodeRecord(
                    workflow=spec.key,
                    node_id=node.node_id,
                    predecessors=node.predecessors,
                    successors=spec.successors(node.node_id),
                    status=(
                        NodeStatus.READY
                        if not node.predecessors
                        else NodeStatus.PENDING
                    ),
                )
                for node in spec.nodes
            }
            self._workflows[spec.key] = WorkflowRecord(spec=spec, nodes=nodes)
            return True

    def start_node(
        self,
        workflow: WorkflowKey,
        node_id: str,
        *,
        timestamp_ns: int,
        reason: str = "dependencies_satisfied",
    ) -> bool:
        """Start one ready DAG node and open its ledger lifetime."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            record = self._active_workflow(workflow)
            node = self._node(record, node_id)
            if node.status == NodeStatus.RUNNING:
                if node.started_ns == timestamp_ns:
                    return False
                raise StateTransitionError("conflicting node start replay")
            if node.status != NodeStatus.READY:
                raise StateTransitionError(
                    f"node {node_id} is not ready: {node.status.value}"
                )
            operation_id = self._node_operation(workflow, node_id)
            event = self._ledger.append(
                EventDraft(
                    action=LedgerAction.NODE,
                    status=LedgerStatus.SCHEDULED,
                    reason=reason,
                    timestamp_ns=timestamp_ns,
                    operation_id=operation_id,
                    workflow=workflow,
                    node_id=node_id,
                )
            )
            with self._committed_runtime_apply("start_node"):
                node.status = NodeStatus.RUNNING
                node.started_ns = timestamp_ns
                node.scheduled_event_id = event.event_id
            return True

    def complete_node(
        self,
        workflow: WorkflowKey,
        node_id: str,
        *,
        timestamp_ns: int,
        reason: str = "node_completed",
    ) -> bool:
        """Complete one running node and release newly ready successors."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            record = self._workflow(workflow)
            node = self._node(record, node_id)
            if node.status == NodeStatus.DONE:
                if node.terminal_ns == timestamp_ns:
                    return False
                raise StateTransitionError("conflicting node completion replay")
            if record.status != WorkflowStatus.ACTIVE:
                raise StateTransitionError(
                    f"workflow is terminal: {record.status.value}"
                )
            if node.status != NodeStatus.RUNNING or node.scheduled_event_id is None:
                raise StateTransitionError(f"node {node_id} is not running")
            mapped_bindings = sorted(
                binding.binding_id
                for binding in self._bindings.values()
                if binding.active
                and binding.workflow == workflow
                and binding.node_id == node_id
                and self._mapping_for_binding(binding) is not None
            )
            if mapped_bindings:
                raise StateTransitionError(
                    f"node still has live execution mappings: {mapped_bindings}"
                )
            self._ledger.append(
                EventDraft(
                    action=LedgerAction.NODE,
                    status=LedgerStatus.COMPLETED,
                    reason=reason,
                    timestamp_ns=timestamp_ns,
                    operation_id=self._node_operation(workflow, node_id),
                    parent_event_id=node.scheduled_event_id,
                    workflow=workflow,
                    node_id=node_id,
                )
            )
            with self._committed_runtime_apply("complete_node"):
                node.status = NodeStatus.DONE
                node.terminal_ns = timestamp_ns
                for successor_id in node.successors:
                    successor = record.nodes[successor_id]
                    if successor.status != NodeStatus.PENDING:
                        continue
                    if all(
                        record.nodes[parent].status == NodeStatus.DONE
                        for parent in successor.predecessors
                    ):
                        successor.status = NodeStatus.READY
            return True

    def fail_node(
        self,
        workflow: WorkflowKey,
        node_id: str,
        *,
        timestamp_ns: int,
        error: str,
    ) -> bool:
        """Fail one running node and close the entire workflow safely."""

        self._guard_runtime_mutation()
        require_text("node error", error)
        with self._lock:
            self._guard_runtime_mutation()
            record = self._workflow(workflow)
            node = self._node(record, node_id)
            if node.status == NodeStatus.FAILED:
                if node.terminal_ns == timestamp_ns and node.error == error:
                    return False
                raise StateTransitionError("conflicting node failure replay")
            if record.status != WorkflowStatus.ACTIVE:
                raise StateTransitionError(
                    f"workflow is terminal: {record.status.value}"
                )
            if node.status != NodeStatus.RUNNING or node.scheduled_event_id is None:
                raise StateTransitionError(f"node {node_id} is not running")
            self._fail_workflow_locked(
                record,
                timestamp_ns=timestamp_ns,
                error=error,
                failed_node=node,
            )
            return True

    def finish_workflow(
        self,
        workflow: WorkflowKey,
        *,
        timestamp_ns: int,
    ) -> bool:
        """Close all owners after every DAG node has completed."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            record = self._workflow(workflow)
            if record.status == WorkflowStatus.DONE:
                if record.terminal_ns == timestamp_ns:
                    return False
                raise StateTransitionError("conflicting workflow completion replay")
            if record.status == WorkflowStatus.FAILED:
                raise StateTransitionError("failed workflow cannot finish")
            unfinished = [
                node.node_id
                for node in record.nodes.values()
                if node.status != NodeStatus.DONE
            ]
            if unfinished:
                raise StateTransitionError(
                    f"workflow has unfinished nodes: {unfinished}"
                )
            self._validate_terminal_timestamp(timestamp_ns)
            self._release_workflow_bindings(record, timestamp_ns=timestamp_ns)
            record.status = WorkflowStatus.DONE
            record.terminal_ns = timestamp_ns
            return True

    def fail_workflow(
        self,
        workflow: WorkflowKey,
        *,
        timestamp_ns: int,
        error: str,
    ) -> bool:
        """Cancel active nodes and release every owner in one workflow."""

        self._guard_runtime_mutation()
        require_text("workflow error", error)
        with self._lock:
            self._guard_runtime_mutation()
            record = self._workflow(workflow)
            if record.status == WorkflowStatus.FAILED:
                if record.terminal_ns == timestamp_ns and record.error == error:
                    return False
                raise StateTransitionError("conflicting workflow failure replay")
            if record.status == WorkflowStatus.DONE:
                raise StateTransitionError("completed workflow cannot fail")
            self._fail_workflow_locked(record, timestamp_ns=timestamp_ns, error=error)
            return True

    def register_gpu_block(
        self,
        block_key: BlockKey,
        replica_id: ReplicaId,
        *,
        byte_capacity: int,
        payload_size: int,
        payload_digest: str,
        timestamp_ns: int,
        reason: str = "producer_allocation",
    ) -> bool:
        """Publish an initial GPU allocation and its content mapping."""

        self._guard_runtime_mutation()
        require_sha256("payload_digest", payload_digest)
        if replica_id.tier != Tier.GPU:
            raise IdentityError("initial replica must be on GPU")
        self._validate_replica_sizes(byte_capacity, payload_size)
        with self._lock:
            self._guard_runtime_mutation()
            block = self._blocks.get(block_key)
            if block is not None and Tier.GPU in block.replicas:
                existing = block.replicas[Tier.GPU]
                if (
                    existing.replica_id == replica_id
                    and existing.byte_capacity == byte_capacity
                    and existing.payload_size == payload_size
                    and existing.payload_digest == payload_digest
                ):
                    return False
                raise StateTransitionError("block already has a different GPU replica")
            if block is not None and block.inflight_transfer_id is not None:
                raise StateTransitionError(
                    "cannot publish GPU while a block transfer owns a reservation"
                )
            if block is not None and Tier.CPU in block.replicas:
                raise StateTransitionError(
                    "CPU-only block must publish GPU through H2D"
                )
            if block is not None and (
                block.payload_size != payload_size
                or block.payload_digest != payload_digest
            ):
                raise TransferIntegrityError(
                    "published GPU payload conflicts with canonical block payload"
                )
            self._validate_new_slot(replica_id)
            mapping_id = self._content_mapping_id(block_key, replica_id)
            if mapping_id in self._content_mappings:
                raise StateTransitionError(f"duplicate content map: {mapping_id}")
            block_for_plan = block or BlockRecord(
                block_key=block_key,
                payload_size=payload_size,
                payload_digest=payload_digest,
            )
            state_after = self._planned_block_state(
                block_for_plan,
                replicas=(replica_id,),
                location_version=block_for_plan.location_version + 1,
                reclaimed=False,
            )
            events = self._ledger.append_batch(
                (
                    self._allocation_draft(
                        block_key,
                        replica_id,
                        byte_capacity=byte_capacity,
                        payload_size=payload_size,
                        payload_digest=payload_digest,
                        timestamp_ns=timestamp_ns,
                        reason=reason,
                        local_id="allocate",
                    ),
                    EventDraft(
                        action=LedgerAction.MAP,
                        status=LedgerStatus.COMPLETED,
                        reason="content_published",
                        timestamp_ns=timestamp_ns,
                        operation_id=mapping_id,
                        parent_local_id="allocate",
                        block_key=block_key,
                        blocks=(replica_id,),
                        mapping_id=mapping_id,
                        payload_size=payload_size,
                        payload_digest=payload_digest,
                    ),
                    self._block_state_draft(
                        block_key,
                        state_after,
                        timestamp_ns=timestamp_ns,
                        reason="gpu_content_published",
                    ),
                )
            )
            with self._committed_runtime_apply("register_gpu_block"):
                block = block_for_plan
                block.payload_size = payload_size
                block.payload_digest = payload_digest
                block.reclaimed = False
                block.location_version += 1
                block.replicas[Tier.GPU] = ReplicaRecord(
                    replica_id=replica_id,
                    byte_capacity=byte_capacity,
                    payload_size=payload_size,
                    payload_digest=payload_digest,
                    allocate_event_id=events[0].event_id,
                    mapping_id=mapping_id,
                )
                self._blocks[block_key] = block
                self._content_mappings[mapping_id] = ContentMapping(
                    mapping_id=mapping_id,
                    block_key=block_key,
                    replica_id=replica_id,
                    map_event_id=events[1].event_id,
                )
                self._occupy_new_slot(replica_id)
            return True

    def bind_owner(
        self,
        handle: BindingHandle,
        *,
        node_id: str,
        block_key: BlockKey,
        kind: BindingKind,
        state: BindingState,
        execution_ref: ExecutionRef | None,
        timestamp_ns: int,
        reason: str = "owner_attached",
    ) -> BindingHandle:
        """Attach one owner, optionally publishing a request execution map."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            if (
                self._trace_required
                and kind == BindingKind.REQUEST
                and state == BindingState.REQUIRED
            ):
                raise StateTransitionError(
                    "trace-required requests must bind as retained before service"
                )
            existing = self._bindings.get(handle.binding_id)
            if existing is not None:
                expected = (
                    existing.handle == handle
                    and existing.node_id == node_id
                    and existing.block_key == block_key
                    and existing.kind == kind
                    and existing.execution_ref == execution_ref
                    and existing.created_ns == timestamp_ns
                    and existing.state == state
                )
                if expected:
                    return handle
                raise IdentityError(f"binding identity conflict: {handle.binding_id}")
            valid_initial_state = (
                kind == BindingKind.REQUEST
                and state in {BindingState.REQUIRED, BindingState.RETAINED}
            ) or (
                kind == BindingKind.WORKFLOW_RETENTION
                and state == BindingState.RETAINED
            )
            if not valid_initial_state:
                raise StateTransitionError(
                    "binding must start as required request or retained owner"
                )
            workflow = self._active_workflow(handle.workflow)
            node = self._node(workflow, node_id)
            if node.status in {
                NodeStatus.DONE,
                NodeStatus.FAILED,
                NodeStatus.SKIPPED,
            }:
                raise StateTransitionError("cannot bind owner to an inactive DAG node")
            if (
                kind == BindingKind.REQUEST
                and state == BindingState.REQUIRED
                and node.status != NodeStatus.RUNNING
            ):
                raise StateTransitionError(
                    "required request binding needs a running DAG node"
                )
            block = self._block(block_key)
            if block.reclaimed:
                raise StateTransitionError("cannot bind a reclaimed block")
            binding = OwnerBinding(
                handle=handle,
                node_id=node_id,
                block_key=block_key,
                kind=kind,
                execution_ref=execution_ref,
                created_ns=timestamp_ns,
                state=state,
            )
            if execution_ref is not None:
                if execution_ref in self._execution_owners:
                    raise StateTransitionError(
                        "execution reference already has an owner"
                    )
                if execution_ref in self._execution_history:
                    raise StateTransitionError("execution reference was already used")

            bind_draft = EventDraft(
                action=LedgerAction.BIND,
                status=LedgerStatus.COMPLETED,
                reason=reason,
                timestamp_ns=timestamp_ns,
                operation_id=handle.binding_id,
                local_id="bind",
                workflow=handle.workflow,
                request_id=handle.request_id,
                node_id=node_id,
                block_key=block_key,
                binding_id=handle.binding_id,
                binding_kind=kind,
                binding_state_after=state,
                execution_ref=execution_ref,
            )
            drafts = [bind_draft]
            gpu = block.replicas.get(Tier.GPU)
            mapping_id: str | None = None
            mapping_generation: int | None = None
            if kind == BindingKind.REQUEST and state == BindingState.REQUIRED:
                if gpu is None or execution_ref is None:
                    raise StateTransitionError(
                        "required request binding needs a published GPU replica"
                    )
                mapping_generation = self._next_execution_mapping_generation(
                    handle.binding_id
                )
                mapping_id = self._execution_mapping_id(
                    handle,
                    gpu.replica_id,
                    mapping_generation,
                )
                drafts.append(
                    self._execution_map_draft(
                        binding,
                        gpu.replica_id,
                        mapping_id=mapping_id,
                        timestamp_ns=timestamp_ns,
                        parent_local_id="bind",
                    )
                )
            events = self._ledger.append_batch(drafts)
            with self._committed_runtime_apply("bind_owner"):
                binding.bind_event_id = events[0].event_id
                self._bindings[handle.binding_id] = binding
                if execution_ref is not None:
                    self._execution_owners[execution_ref] = handle.binding_id
                    self._execution_history[execution_ref] = (
                        block_key,
                        handle.binding_id,
                    )
                if mapping_generation is not None:
                    self._execution_mapping_generations[handle.binding_id] = (
                        mapping_generation
                    )
                workflow.binding_ids.add(handle.binding_id)
                block.binding_ids.add(handle.binding_id)
                if (
                    mapping_id is not None
                    and execution_ref is not None
                    and gpu is not None
                ):
                    self._execution_mappings[execution_ref] = ExecutionMapping(
                        mapping_id=mapping_id,
                        execution_ref=execution_ref,
                        binding_id=handle.binding_id,
                        block_key=block_key,
                        gpu_replica=gpu.replica_id,
                        location_version=block.location_version,
                        map_event_id=events[1].event_id,
                    )
            return handle

    def set_binding_state(
        self,
        handle: BindingHandle,
        state: BindingState,
        *,
        timestamp_ns: int,
        reason: str = "owner_state_changed",
    ) -> bool:
        """Move a request binding between required and retained states."""

        self._guard_runtime_mutation()
        if state in {BindingState.WAITING, BindingState.RELEASED}:
            raise StateTransitionError(
                "waiting and released states require lifecycle-specific operations"
            )
        with self._lock:
            self._guard_runtime_mutation()
            binding = self._binding(handle)
            if not binding.active:
                raise StateTransitionError("released binding cannot transition")
            if binding.kind != BindingKind.REQUEST:
                raise StateTransitionError("retention binding state is fixed")
            if binding.state == state:
                return False
            if self._trace_required and state == BindingState.REQUIRED:
                raise StateTransitionError(
                    "trace-required request mapping must pass the demand gate"
                )
            mapping = self._mapping_for_binding(binding)
            drafts: list[EventDraft] = []
            new_mapping: tuple[str, ReplicaRecord, int] | None = None
            if binding.state == BindingState.WAITING:
                if state != BindingState.RETAINED:
                    raise StateTransitionError(
                        "waiting binding can only detach from its active transfer"
                    )
                transfer = self._active_waiter_transfer(binding.binding_id)
                waiter_ids = set(transfer.waiter_binding_ids)
                waiter_ids.remove(binding.binding_id)
                drafts.append(
                    self._waiter_draft(
                        transfer,
                        binding,
                        LedgerAction.WAITER_LEAVE,
                        waiter_ids,
                        timestamp_ns=timestamp_ns,
                        reason=reason,
                    )
                )
            if state == BindingState.RETAINED:
                if mapping is not None:
                    drafts.append(
                        self._execution_unmap_draft(
                            binding,
                            mapping,
                            timestamp_ns=timestamp_ns,
                            reason=reason,
                        )
                    )
            else:
                workflow = self._active_workflow(binding.workflow)
                node = self._node(workflow, binding.node_id)
                if node.status != NodeStatus.RUNNING:
                    raise StateTransitionError(
                        "required request binding needs a running DAG node"
                    )
                block = self._block(binding.block_key)
                gpu = block.replicas.get(Tier.GPU)
                if gpu is None or binding.execution_ref is None:
                    raise StateTransitionError("required binding has no GPU replica")
                if mapping is None:
                    mapping_generation = self._next_execution_mapping_generation(
                        binding.binding_id
                    )
                    mapping_id = self._execution_mapping_id(
                        handle,
                        gpu.replica_id,
                        mapping_generation,
                    )
                    drafts.append(
                        self._execution_map_draft(
                            binding,
                            gpu.replica_id,
                            mapping_id=mapping_id,
                            timestamp_ns=timestamp_ns,
                            parent_event_id=binding.bind_event_id,
                            reason=reason,
                        )
                    )
                    new_mapping = (mapping_id, gpu, mapping_generation)
                elif mapping.gpu_replica != gpu.replica_id:
                    raise StateTransitionError(
                        "binding points at a stale GPU generation"
                    )
            drafts.append(
                self._binding_state_draft(
                    binding,
                    state,
                    timestamp_ns=timestamp_ns,
                    reason=reason,
                )
            )
            events = self._ledger.append_batch(drafts)
            with self._committed_runtime_apply("set_binding_state"):
                if mapping is not None and state == BindingState.RETAINED:
                    self._execution_mappings.pop(mapping.execution_ref)
                if new_mapping is not None and binding.execution_ref is not None:
                    mapping_id, gpu, mapping_generation = new_mapping
                    block = self._block(binding.block_key)
                    self._execution_mappings[binding.execution_ref] = ExecutionMapping(
                        mapping_id=mapping_id,
                        execution_ref=binding.execution_ref,
                        binding_id=binding.binding_id,
                        block_key=binding.block_key,
                        gpu_replica=gpu.replica_id,
                        location_version=block.location_version,
                        map_event_id=next(
                            event.event_id
                            for event in events
                            if event.operation_id == mapping_id
                        ),
                    )
                    self._execution_mapping_generations[binding.binding_id] = (
                        mapping_generation
                    )
                binding.transition(state)
                self._remove_waiter(binding.binding_id)
            return True

    def open_lease(
        self,
        handle: BindingHandle,
        lease_id: str,
        *,
        registered_ns: int,
        deadline_ns: int,
        reason: str,
    ) -> str:
        """Open a fresh TTL generation on a retention binding."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            binding = self._binding(handle)
            if not binding.active or binding.kind != BindingKind.WORKFLOW_RETENTION:
                raise StateTransitionError("lease requires an active retention binding")
            existing = self._leases.get(lease_id)
            if existing is not None:
                if (
                    existing.active
                    and existing.binding_id == binding.binding_id
                    and existing.block_key == binding.block_key
                    and existing.registered_ns == registered_ns
                    and existing.deadline_ns == deadline_ns
                    and existing.reason == reason
                ):
                    return lease_id
                raise IdentityError(f"lease ID already used: {lease_id}")
            lease = Lease(
                lease_id=lease_id,
                binding_id=binding.binding_id,
                block_key=binding.block_key,
                registered_ns=registered_ns,
                deadline_ns=deadline_ns,
                reason=reason,
            )
            event = self._ledger.append(
                EventDraft(
                    action=LedgerAction.LEASE,
                    status=LedgerStatus.SCHEDULED,
                    reason=reason,
                    timestamp_ns=registered_ns,
                    operation_id=lease_id,
                    parent_event_id=binding.bind_event_id,
                    workflow=binding.workflow,
                    request_id=binding.request_id,
                    node_id=binding.node_id,
                    block_key=binding.block_key,
                    binding_id=binding.binding_id,
                    binding_kind=binding.kind,
                    lease_id=lease_id,
                    lease_deadline_ns=deadline_ns,
                )
            )
            with self._committed_runtime_apply("open_lease"):
                lease.scheduled_event_id = event.event_id
                self._leases[lease_id] = lease
                self._block(binding.block_key).lease_ids.add(lease_id)
            return lease_id

    def terminate_lease(
        self,
        handle: BindingHandle,
        lease_id: str,
        state: LeaseState,
        *,
        timestamp_ns: int,
        error: str | None = None,
    ) -> bool:
        """Close a lease using an exact owner-qualified terminal."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            binding = self._binding(handle)
            lease = self._lease_for_binding(lease_id, binding)
            if not lease.active:
                return lease.terminate(state, timestamp_ns, error=error)
            if state == LeaseState.ACTIVE:
                raise StateTransitionError("lease terminal must be non-active")
            if timestamp_ns < lease.registered_ns:
                raise StateTransitionError("lease terminal predates registration")
            if state == LeaseState.EXPIRED and timestamp_ns < lease.deadline_ns:
                raise StateTransitionError("lease cannot expire before its deadline")
            self._append_lease_terminal(
                binding,
                lease,
                state,
                timestamp_ns=timestamp_ns,
                error=error,
            )
            with self._committed_runtime_apply("terminate_lease"):
                lease.terminate(state, timestamp_ns, error=error)
                self._block(lease.block_key).lease_ids.discard(lease_id)
            return True

    def expire_leases(self, *, timestamp_ns: int) -> tuple[str, ...]:
        """Expire every active lease whose declared deadline has passed."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            due = sorted(
                lease.lease_id
                for lease in self._leases.values()
                if lease.active and lease.deadline_ns <= timestamp_ns
            )
            for lease_id in due:
                lease = self._leases[lease_id]
                binding = self._bindings[lease.binding_id]
                self.terminate_lease(
                    binding.handle,
                    lease_id,
                    LeaseState.EXPIRED,
                    timestamp_ns=timestamp_ns,
                )
            return tuple(due)

    def begin_d2h(
        self,
        block_key: BlockKey,
        target_replica: ReplicaId,
        *,
        transfer_id: str,
        timestamp_ns: int,
        reason: str = "policy_offload",
    ) -> TransferCommand | None:
        """Reserve CPU memory and schedule one GPU-to-CPU save."""

        self._guard_runtime_mutation()
        if target_replica.tier != Tier.CPU:
            raise IdentityError("D2H target must be CPU")
        with self._lock:
            self._guard_runtime_mutation()
            block = self._block(block_key)
            source = block.replicas.get(Tier.GPU)
            if source is None:
                raise StateTransitionError("D2H source GPU replica is absent")
            existing_cpu = block.replicas.get(Tier.CPU)
            if existing_cpu is not None:
                if (
                    existing_cpu.payload_size == source.payload_size
                    and existing_cpu.payload_digest == source.payload_digest
                ):
                    return None
                raise StateTransitionError("CPU replica has conflicting content")
            self._ensure_no_active_lease(block)
            if block.inflight_transfer_id is not None:
                transfer = self._transfers[block.inflight_transfer_id]
                if (
                    transfer.transfer_id == transfer_id
                    and transfer.direction == TransferDirection.D2H
                    and transfer.target_replica == target_replica
                ):
                    return self._transfer_command(transfer)
                raise StateTransitionError("block already has an in-flight transfer")
            return self._schedule_transfer(
                block,
                source,
                target_replica,
                transfer_id=transfer_id,
                action=LedgerAction.SAVE,
                direction=TransferDirection.D2H,
                waiter_bindings=(),
                timestamp_ns=timestamp_ns,
                reason=reason,
            )

    def ensure_h2d(
        self,
        block_key: BlockKey,
        target_replica: ReplicaId,
        waiter_handles: Iterable[BindingHandle],
        *,
        transfer_id: str,
        timestamp_ns: int,
        action: LedgerAction = LedgerAction.LOAD,
        reason: str = "owner_readmission",
        demand_commit_id: str | None = None,
        demand_committer: DemandCommitter | None = None,
    ) -> TransferCommand | None:
        """Publish residency or coalesce the logical consumers in this call.

        A traced adapter uses a fresh demand commit ID for each new scheduled
        access batch. Reuse is limited to an event-free replay of that batch.
        """

        self._guard_runtime_mutation()
        if target_replica.tier != Tier.GPU:
            raise IdentityError("H2D target must be GPU")
        if action not in {LedgerAction.LOAD, LedgerAction.PREFETCH}:
            raise IdentityError("H2D action must be load or prefetch")
        with self._lock:
            self._guard_runtime_mutation()
            waiter_handles_tuple = tuple(waiter_handles)
            block = self._block(block_key)
            waiters = self._validate_waiters(block_key, waiter_handles_tuple)
            if action == LedgerAction.LOAD and not waiters:
                raise StateTransitionError("load requires at least one waiter")
            if (
                self._trace_required
                or demand_committer is not None
                or demand_commit_id is not None
            ) and timestamp_ns < self._ledger.last_timestamp_ns:
                raise StateTransitionError("H2D timestamp predates ledger state")
            gpu = block.replicas.get(Tier.GPU)
            if gpu is not None:
                if gpu.replica_id != target_replica:
                    raise StateTransitionError(
                        "requested target disagrees with live GPU"
                    )
                self._validate_gpu_fast_path_waiters(gpu, waiters)
            elif block.inflight_transfer_id is not None:
                transfer = self._transfers[block.inflight_transfer_id]
                if transfer.direction != TransferDirection.H2D:
                    raise StateTransitionError("D2H conflicts with requested H2D")
                if transfer.target_replica != target_replica:
                    raise StateTransitionError(
                        "single-flight target generation changed"
                    )
                self._validate_h2d_waiter_preparation(
                    waiters,
                    existing_transfer=transfer,
                )
            else:
                source = block.replicas.get(Tier.CPU)
                if source is None:
                    raise StateTransitionError("H2D source CPU replica is absent")
                self._validate_h2d_waiter_preparation(waiters)
                require_text("transfer_id", transfer_id)
                if transfer_id in self._transfers:
                    raise IdentityError(f"transfer ID already used: {transfer_id}")
                self._validate_new_slot(target_replica)

            trace_gate_enabled = (
                self._trace_required
                or demand_committer is not None
                or demand_commit_id is not None
            )
            demand_signature: (
                tuple[
                    BlockKey,
                    tuple[tuple[str, ExecutionRef], ...],
                    ReplicaId,
                    LedgerAction,
                    int,
                ]
                | None
            ) = None
            demand_waiters: tuple[OwnerBinding, ...] = ()
            if waiters and trace_gate_enabled:
                if demand_commit_id is None:
                    raise StateTransitionError(
                        "traced H2D waiters require a demand commit ID"
                    )
                require_text("demand_commit_id", demand_commit_id)
                identified_waiters = tuple(
                    (binding.binding_id, binding.execution_ref)
                    for binding in sorted(waiters, key=lambda item: item.binding_id)
                )
                if any(
                    execution_ref is None for _, execution_ref in identified_waiters
                ):
                    raise StateTransitionError("request waiter lost execution identity")
                demand_signature = (
                    block_key,
                    tuple(
                        (binding_id, execution_ref)
                        for binding_id, execution_ref in identified_waiters
                        if execution_ref is not None
                    ),
                    target_replica,
                    action,
                    timestamp_ns,
                )
                prior_signature = self._committed_demand_ids.get(demand_commit_id)
                if prior_signature is None:
                    conflicting_id = next(
                        (
                            self._demand_id_by_execution[execution_ref]
                            for _, execution_ref in demand_signature[1]
                            if execution_ref in self._demand_id_by_execution
                            and self._demand_id_by_execution[execution_ref]
                            != demand_commit_id
                        ),
                        None,
                    )
                    if conflicting_id is not None:
                        raise IdentityError(
                            "request execution already has another demand commit ID"
                        )
                    self._validate_fresh_demand_waiters(block, waiters)
                    demand_waiters = waiters
                elif prior_signature != demand_signature:
                    raise IdentityError(
                        "demand commit ID was reused for another logical demand"
                    )
                elif demand_commit_id in self._dispatched_demand_ids:
                    self._validate_idempotent_demand_replay(
                        block,
                        waiters,
                        gpu=gpu,
                    )
            elif not waiters and (
                demand_committer is not None or demand_commit_id is not None
            ):
                raise StateTransitionError(
                    "demand commit requires at least one request waiter"
                )

            if demand_waiters:
                if demand_committer is None:
                    raise StateTransitionError(
                        "trace-required H2D waiters need a durable demand commit"
                    )
                assert demand_commit_id is not None
                assert demand_signature is not None
                demand_view = self._pre_service_demand_view_locked(
                    block,
                    demand_commit_id=demand_commit_id,
                    target_replica=target_replica,
                    action=action,
                    transfer_id=transfer_id,
                    timestamp_ns=timestamp_ns,
                    waiters=demand_waiters,
                )
                if self._trace_callback_active:
                    raise StateTransitionError(
                        "trace committer callback cannot reenter"
                    )
                self._trace_callback_active = True
                try:
                    receipt = demand_committer.commit_demands(demand_view)
                finally:
                    self._trace_callback_active = False
                current = self._pre_service_demand_view_locked(
                    self._block(block_key),
                    demand_commit_id=demand_commit_id,
                    target_replica=target_replica,
                    action=action,
                    transfer_id=transfer_id,
                    timestamp_ns=timestamp_ns,
                    waiters=self._validate_waiters(
                        block_key,
                        waiter_handles_tuple,
                    ),
                )
                if current != demand_view:
                    raise StateTransitionError(
                        "demand committer changed the pre-service runtime view"
                    )
                self._validate_trace_receipt(
                    receipt,
                    event_count=demand_view.runtime_event_count,
                    view_digest=demand_view.view_digest,
                )
                self._committed_demand_ids[demand_commit_id] = demand_signature
                for _, execution_ref in demand_signature[1]:
                    self._demand_id_by_execution[execution_ref] = demand_commit_id

            if gpu is not None:
                self._publish_waiter_mappings(
                    block,
                    gpu,
                    waiters,
                    timestamp_ns=timestamp_ns,
                    reason="gpu_fast_path",
                )
                if demand_commit_id is not None:
                    self._dispatched_demand_ids.add(demand_commit_id)
                return None
            if block.inflight_transfer_id is not None:
                transfer = self._transfers[block.inflight_transfer_id]
                drafts: list[EventDraft] = []
                waiter_ids = set(transfer.waiter_binding_ids)
                ordered_waiters = tuple(
                    sorted(waiters, key=lambda item: item.binding_id)
                )
                for binding in ordered_waiters:
                    if (
                        binding.state == BindingState.WAITING
                        and binding.binding_id in waiter_ids
                    ):
                        continue
                    drafts.append(
                        self._binding_state_draft(
                            binding,
                            BindingState.WAITING,
                            timestamp_ns=timestamp_ns,
                            reason="coalesced_h2d_waiter_prepared",
                        )
                    )
                    waiter_ids.add(binding.binding_id)
                    drafts.append(
                        self._waiter_draft(
                            transfer,
                            binding,
                            LedgerAction.WAITER_JOIN,
                            waiter_ids,
                            timestamp_ns=timestamp_ns,
                            reason="coalesced_h2d_waiter_joined",
                        )
                    )
                self._ledger.append_batch(drafts)
                with self._committed_runtime_apply("join_coalesced_h2d"):
                    for binding in ordered_waiters:
                        self._prepare_waiter(binding)
                        transfer.waiter_binding_ids.add(binding.binding_id)
                        self._transfer_waiter_history.add(binding.binding_id)
                if demand_commit_id is not None:
                    self._dispatched_demand_ids.add(demand_commit_id)
                return self._transfer_command(transfer)
            source = block.replicas.get(Tier.CPU)
            if source is None:  # pragma: no cover - protected by locked preflight
                raise StateTransitionError("H2D source disappeared after preflight")
            command = self._schedule_transfer(
                block,
                source,
                target_replica,
                transfer_id=transfer_id,
                action=action,
                direction=TransferDirection.H2D,
                waiter_bindings=waiters,
                timestamp_ns=timestamp_ns,
                reason=reason,
            )
            if demand_commit_id is not None:
                self._dispatched_demand_ids.add(demand_commit_id)
            return command

    def complete_transfer(
        self,
        transfer_id: str,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str,
    ) -> bool:
        """Validate and atomically publish one completed physical transfer."""

        self._guard_runtime_mutation()
        require_sha256("observed_digest", observed_digest)
        with self._lock:
            self._guard_runtime_mutation()
            transfer = self._transfer(transfer_id)
            if not transfer.active:
                return transfer.terminate(
                    TransferState.COMPLETED,
                    timestamp_ns,
                    observed_bytes=observed_bytes,
                    observed_digest=observed_digest,
                )
            mismatches: list[str] = []
            if observed_bytes != transfer.declared_bytes:
                mismatches.append(
                    f"bytes {observed_bytes} != {transfer.declared_bytes}"
                )
            if observed_digest != transfer.payload_digest:
                mismatches.append("payload digest mismatch")
            if mismatches:
                error = "; ".join(mismatches)
                self._finalize_transfer_failure(
                    transfer,
                    TransferState.FAILED,
                    timestamp_ns=timestamp_ns,
                    observed_bytes=observed_bytes,
                    observed_digest=observed_digest,
                    error=error,
                    cleanup_reason="integrity_failure_cleanup",
                )
                issue = f"transfer {transfer_id} integrity failure: {error}"
                self._hard_failures.append(issue)
                raise TransferIntegrityError(issue)
            self._finalize_transfer_success(
                transfer,
                timestamp_ns=timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
            )
            return True

    def fail_transfer(
        self,
        transfer_id: str,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str | None,
        error: str,
    ) -> bool:
        """Record a worker-reported failure and release its target reservation."""

        self._guard_runtime_mutation()
        require_text("transfer error", error)
        with self._lock:
            self._guard_runtime_mutation()
            transfer = self._transfer(transfer_id)
            if not transfer.active:
                return transfer.terminate(
                    TransferState.FAILED,
                    timestamp_ns,
                    observed_bytes=observed_bytes,
                    observed_digest=observed_digest,
                    error=error,
                )
            self._finalize_transfer_failure(
                transfer,
                TransferState.FAILED,
                timestamp_ns=timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
                error=error,
                cleanup_reason="failed_transfer_cleanup",
            )
            if observed_bytes > transfer.declared_bytes:
                issue = f"transfer {transfer_id} reported a byte overrun"
                self._hard_failures.append(issue)
                raise TransferIntegrityError(issue)
            return True

    def cancel_transfer(
        self,
        transfer_id: str,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str | None = None,
        reason: str = "transfer_cancelled",
    ) -> bool:
        """Cancel one transfer while preserving its exact observed progress."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            transfer = self._transfer(transfer_id)
            if not transfer.active:
                return transfer.terminate(
                    TransferState.CANCELLED,
                    timestamp_ns,
                    observed_bytes=observed_bytes,
                    observed_digest=observed_digest,
                    error=reason,
                )
            self._finalize_transfer_failure(
                transfer,
                TransferState.CANCELLED,
                timestamp_ns=timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
                error=reason,
                cleanup_reason="cancelled_transfer_cleanup",
            )
            if observed_bytes > transfer.declared_bytes:
                issue = f"transfer {transfer_id} reported a byte overrun"
                self._hard_failures.append(issue)
                raise TransferIntegrityError(issue)
            return True

    def drop_gpu(
        self,
        block_key: BlockKey,
        *,
        expected_gpu: ReplicaId,
        expected_location_version: int,
        timestamp_ns: int,
        reason: str = "gpu_capacity",
    ) -> bool:
        """Close an explicitly versioned GPU allocation after owners detach."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            block = self._block(block_key)
            gpu = block.replicas.get(Tier.GPU)
            if gpu is None:
                return False
            if (
                gpu.replica_id != expected_gpu
                or block.location_version != expected_location_version
            ):
                raise StateTransitionError(
                    "stale GPU drop command does not match current location"
                )
            if block.inflight_transfer_id is not None:
                raise StateTransitionError("cannot drop GPU during a transfer")
            self._ensure_no_active_lease(block)
            active_bindings = [
                self._bindings[binding_id]
                for binding_id in block.binding_ids
                if self._bindings[binding_id].active
            ]
            required = [
                binding.binding_id
                for binding in active_bindings
                if binding.state == BindingState.REQUIRED
            ]
            if required:
                raise StateTransitionError(
                    f"required owners still use the GPU replica: {required}"
                )
            if active_bindings and Tier.CPU not in block.replicas:
                raise StateTransitionError("live owners require a CPU fallback")
            mappings = [
                mapping
                for mapping in self._execution_mappings.values()
                if mapping.block_key == block_key
            ]
            if mappings:
                raise StateTransitionError("execution mappings remain on GPU drop")
            content_map = self._content_mappings[gpu.mapping_id]
            remaining_replicas = tuple(
                replica.replica_id
                for tier, replica in block.replicas.items()
                if tier != Tier.GPU
            )
            drafts = (
                EventDraft(
                    action=LedgerAction.UNMAP,
                    status=LedgerStatus.COMPLETED,
                    reason=reason,
                    timestamp_ns=timestamp_ns,
                    operation_id=content_map.mapping_id,
                    parent_event_id=content_map.map_event_id,
                    block_key=block_key,
                    blocks=(gpu.replica_id,),
                    mapping_id=content_map.mapping_id,
                    payload_size=gpu.payload_size,
                    payload_digest=gpu.payload_digest,
                ),
                self._eviction_draft(
                    block_key,
                    gpu,
                    timestamp_ns=timestamp_ns,
                    reason=reason,
                ),
                self._block_state_draft(
                    block_key,
                    self._planned_block_state(
                        block,
                        replicas=remaining_replicas,
                        location_version=block.location_version + 1,
                        reclaimed=False,
                    ),
                    timestamp_ns=timestamp_ns,
                    reason="gpu_dropped_state",
                ),
            )
            self._ledger.append_batch(drafts)
            with self._committed_runtime_apply("drop_gpu"):
                self._content_mappings.pop(gpu.mapping_id)
                block.replicas.pop(Tier.GPU)
                block.location_version += 1
                self._free_slot(gpu.replica_id)
            return True

    def release_binding(
        self,
        handle: BindingHandle,
        *,
        timestamp_ns: int,
        reason: str = "owner_detached",
    ) -> bool:
        """Release one owner after validating its full caller capability."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            binding = self._binding(handle)
            if not binding.active:
                return False
            plan = self._plan_binding_release(
                binding,
                timestamp_ns=timestamp_ns,
                reason=reason,
            )
            self._ledger.append_batch(plan.drafts)
            with self._committed_runtime_apply("release_binding"):
                self._commit_binding_release(plan)
            return True

    def reclaim(
        self,
        block_key: BlockKey,
        *,
        expected_replicas: tuple[ReplicaId, ...],
        expected_location_version: int,
        timestamp_ns: int,
        reason: str = "no_live_references",
    ) -> bool:
        """Release an explicitly versioned replica set after all edges close."""

        self._guard_runtime_mutation()
        with self._lock:
            self._guard_runtime_mutation()
            block = self._block(block_key)
            if block.reclaimed:
                return False
            if len(expected_replicas) != len(set(expected_replicas)):
                raise IdentityError("reclaim guard contains duplicate replicas")
            current_replicas = {
                replica.replica_id for replica in block.replicas.values()
            }
            if (
                current_replicas != set(expected_replicas)
                or block.location_version != expected_location_version
            ):
                raise StateTransitionError(
                    "stale reclaim command does not match current block state"
                )
            live_bindings = [
                binding_id
                for binding_id in block.binding_ids
                if self._bindings[binding_id].active
            ]
            live_leases = [
                lease_id
                for lease_id in block.lease_ids
                if self._leases[lease_id].active
            ]
            live_exec = [
                mapping.mapping_id
                for mapping in self._execution_mappings.values()
                if mapping.block_key == block_key
            ]
            if live_bindings or live_leases or live_exec:
                raise StateTransitionError(
                    "cannot reclaim block with live owner, lease, or execution map"
                )
            if block.inflight_transfer_id is not None:
                raise StateTransitionError("cannot reclaim block during a transfer")
            drafts: list[EventDraft] = []
            replicas = sorted(
                block.replicas.values(), key=lambda item: item.replica_id.tier.value
            )
            for replica in replicas:
                content_map = self._content_mappings[replica.mapping_id]
                drafts.extend(
                    (
                        EventDraft(
                            action=LedgerAction.UNMAP,
                            status=LedgerStatus.COMPLETED,
                            reason=reason,
                            timestamp_ns=timestamp_ns,
                            operation_id=content_map.mapping_id,
                            parent_event_id=content_map.map_event_id,
                            block_key=block_key,
                            blocks=(replica.replica_id,),
                            mapping_id=content_map.mapping_id,
                            payload_size=replica.payload_size,
                            payload_digest=replica.payload_digest,
                        ),
                        self._eviction_draft(
                            block_key,
                            replica,
                            timestamp_ns=timestamp_ns,
                            reason=reason,
                        ),
                    )
                )
            drafts.append(
                self._block_state_draft(
                    block_key,
                    self._planned_block_state(
                        block,
                        replicas=(),
                        location_version=block.location_version + 1,
                        reclaimed=True,
                    ),
                    timestamp_ns=timestamp_ns,
                    reason="block_reclaimed_state",
                )
            )
            self._ledger.append_batch(drafts)
            with self._committed_runtime_apply("reclaim"):
                for replica in replicas:
                    self._content_mappings.pop(replica.mapping_id)
                    self._free_slot(replica.replica_id)
                block.replicas.clear()
                block.reclaimed = True
                block.location_version += 1
            return True

    def is_ready(self, handle: BindingHandle) -> bool:
        """Return whether one request binding can safely execute on GPU."""

        with self._lock:
            binding = self._binding(handle)
            if (
                not binding.active
                or binding.kind != BindingKind.REQUEST
                or binding.state != BindingState.REQUIRED
                or binding.execution_ref is None
            ):
                return False
            workflow = self._workflows.get(binding.workflow)
            node = workflow.nodes.get(binding.node_id) if workflow is not None else None
            if node is None or node.status != NodeStatus.RUNNING:
                return False
            mapping = self._execution_mappings.get(binding.execution_ref)
            if mapping is None or mapping.binding_id != binding.binding_id:
                return False
            block = self._blocks.get(binding.block_key)
            if block is None or block.reclaimed:
                return False
            gpu = block.replicas.get(Tier.GPU)
            return bool(
                gpu is not None
                and mapping.gpu_replica == gpu.replica_id
                and mapping.location_version == block.location_version
            )

    def audit(self, *, require_quiescent: bool = False) -> AuditReport:
        """Reconcile every derived index and replay ledger conservation."""

        with self._lock:
            issues = list(self._hard_failures)
            active_bindings = {
                binding_id: binding
                for binding_id, binding in self._bindings.items()
                if binding.active
            }
            active_leases = {
                lease_id: lease
                for lease_id, lease in self._leases.items()
                if lease.active
            }
            active_transfers = {
                transfer_id: transfer
                for transfer_id, transfer in self._transfers.items()
                if transfer.active
            }

            self._audit_workflows(issues, active_bindings)
            self._audit_blocks(
                issues,
                active_bindings,
                active_leases,
                active_transfers,
            )
            self._audit_slots(issues)
            issues.extend(
                self._ledger.audit(
                    require_quiescent=require_quiescent,
                    require_complete_state=True,
                )
            )

            ledger_counts = self._ledger.live_counts()
            runtime_counts = {
                "allocations": self._live_replica_count() + len(self._reservations),
                "content_mappings": len(self._content_mappings),
                "bindings": len(active_bindings),
                "leases": len(active_leases),
                "transfers": len(active_transfers),
                "execution_mappings": len(self._execution_mappings),
            }
            for name, runtime_count in runtime_counts.items():
                ledger_count = ledger_counts[name]
                if ledger_count != runtime_count:
                    issues.append(
                        f"{name} ledger/runtime mismatch: "
                        f"{ledger_count} != {runtime_count}"
                    )

            runtime_block_states = {
                key: self._planned_block_state(
                    block,
                    inflight_transfer_id=block.inflight_transfer_id,
                    inflight_direction=block.inflight_direction,
                    reclaimed=block.reclaimed,
                )
                for key, block in self._blocks.items()
            }
            ledger_block_states = self._ledger.block_states()
            if ledger_block_states != runtime_block_states:
                issues.append("block state ledger/runtime mismatch")

            runtime_binding_states = {
                binding_id: binding.state
                for binding_id, binding in active_bindings.items()
            }
            ledger_binding_states = self._ledger.binding_states()
            if ledger_binding_states != runtime_binding_states:
                issues.append("binding state ledger/runtime mismatch")

            runtime_transfer_waiters = {
                transfer_id: frozenset(transfer.waiter_binding_ids)
                for transfer_id, transfer in active_transfers.items()
            }
            ledger_transfer_waiters = self._ledger.active_transfer_waiters()
            if ledger_transfer_waiters != runtime_transfer_waiters:
                issues.append("transfer waiter ledger/runtime mismatch")
            if self._ledger.transfer_waiter_history() != frozenset(
                self._transfer_waiter_history
            ):
                issues.append("transfer waiter history ledger/runtime mismatch")

            ledger_transfers = self._ledger.transfer_records()
            if set(ledger_transfers) != set(self._transfers):
                issues.append("transfer history ledger/runtime identity mismatch")
            terminal_status = {
                TransferState.SCHEDULED: LedgerStatus.SCHEDULED,
                TransferState.COMPLETED: LedgerStatus.COMPLETED,
                TransferState.FAILED: LedgerStatus.FAILED,
                TransferState.CANCELLED: LedgerStatus.CANCELLED,
            }
            for transfer_id in sorted(set(ledger_transfers) & set(self._transfers)):
                ledger_transfer = ledger_transfers[transfer_id]
                runtime_transfer = self._transfers[transfer_id]
                observed = (
                    runtime_transfer.transfer_id,
                    runtime_transfer.ledger_action,
                    runtime_transfer.block_key,
                    runtime_transfer.source_replica,
                    runtime_transfer.target_replica,
                    terminal_status[runtime_transfer.state],
                    runtime_transfer.started_ns,
                    runtime_transfer.terminal_ns,
                    runtime_transfer.declared_bytes,
                    runtime_transfer.observed_bytes,
                    runtime_transfer.payload_digest,
                    runtime_transfer.observed_digest,
                    runtime_transfer.error,
                    runtime_transfer.scheduled_event_id,
                    tuple(sorted(runtime_transfer.waiter_binding_ids)),
                )
                expected = (
                    ledger_transfer.transfer_id,
                    ledger_transfer.action,
                    ledger_transfer.block_key,
                    ledger_transfer.source_replica,
                    ledger_transfer.target_replica,
                    ledger_transfer.status,
                    ledger_transfer.started_ns,
                    ledger_transfer.terminal_ns,
                    ledger_transfer.declared_bytes,
                    ledger_transfer.observed_bytes,
                    ledger_transfer.payload_digest,
                    ledger_transfer.observed_digest,
                    ledger_transfer.error,
                    ledger_transfer.scheduled_event_id,
                    ledger_transfer.waiter_binding_ids,
                )
                if observed != expected:
                    issues.append(
                        f"transfer history ledger/runtime mismatch: {transfer_id}"
                    )
            if require_quiescent:
                active_workflows = [
                    str(key)
                    for key, workflow in self._workflows.items()
                    if workflow.status == WorkflowStatus.ACTIVE
                ]
                if active_workflows:
                    issues.append(f"active workflows remain: {active_workflows}")
                if any(runtime_counts.values()):
                    issues.append(f"runtime resources remain: {runtime_counts}")

            return AuditReport(
                issues=tuple(dict.fromkeys(issues)),
                active_bindings=len(active_bindings),
                active_leases=len(active_leases),
                content_mappings=len(self._content_mappings),
                execution_mappings=len(self._execution_mappings),
                live_replicas=self._live_replica_count(),
                reservations=len(self._reservations),
                inflight_transfers=len(active_transfers),
            )

    def _shared_lease_policy_snapshot_locked(
        self,
        key: BlockKey,
        *,
        runtime_event_count: int,
    ) -> SharedLeasePolicySnapshot:
        block = self._block(key)
        owners: list[LeaseOwnerSnapshot] = []
        for binding_id in sorted(block.binding_ids):
            binding = self._bindings[binding_id]
            if not binding.active or binding.kind != BindingKind.WORKFLOW_RETENTION:
                continue
            workflow = self._workflow(binding.workflow)
            eligible_node_ids = tuple(
                sorted(
                    node.node_id
                    for node in workflow.nodes.values()
                    if node.status
                    in {NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING}
                )
            )
            owners.append(
                LeaseOwnerSnapshot(
                    binding_id=binding.binding_id,
                    workflow=binding.workflow,
                    created_ns=binding.created_ns,
                    eligible_node_ids=eligible_node_ids,
                )
            )
        return SharedLeasePolicySnapshot(
            block_key=key,
            runtime_event_count=runtime_event_count,
            location_version=block.location_version,
            residency=block.residency,
            owners=tuple(owners),
        )

    def _atomic_cutoff_view_locked(
        self,
        key: BlockKey,
        *,
        cutoff_ns: int,
        horizon_duration_ns: int,
    ) -> AtomicCutoffView:
        lifecycle_prefix = self._ledger.events
        snapshot = self._shared_lease_policy_snapshot_locked(
            key,
            runtime_event_count=len(lifecycle_prefix),
        )
        owner_workflows = sorted({owner.workflow for owner in snapshot.owners})
        owner_specs = tuple(
            deepcopy(self._workflow(workflow).spec) for workflow in owner_workflows
        )
        return AtomicCutoffView(
            snapshot=snapshot,
            owner_specs=owner_specs,
            lifecycle_prefix=lifecycle_prefix,
            cutoff_ns=cutoff_ns,
            horizon_duration_ns=horizon_duration_ns,
            deadline_ns=cutoff_ns + horizon_duration_ns,
        )

    @staticmethod
    def _validate_trace_receipt(
        receipt: DurableCommitReceipt,
        *,
        event_count: int,
        view_digest: str,
    ) -> None:
        if not isinstance(receipt, DurableCommitReceipt):
            raise StateTransitionError("trace committer returned an invalid receipt")
        if receipt.event_count != event_count:
            raise StateTransitionError("trace receipt event count differs from runtime")
        if receipt.view_digest != view_digest:
            raise StateTransitionError("trace receipt view digest differs from runtime")

    def _guard_runtime_mutation(self) -> None:
        if self._runtime_poisoned is not None:
            raise StateTransitionError(
                f"runtime is poisoned after a committed apply failure: "
                f"{self._runtime_poisoned}"
            )
        if self._seal_event is not None:
            raise StateTransitionError("lifecycle stream is sealed")
        if self._trace_callback_active:
            raise StateTransitionError(
                "runtime mutation from a trace committer callback is forbidden"
            )

    @contextmanager
    def _committed_runtime_apply(self, operation: str) -> Iterator[None]:
        """Poison the writer if in-memory apply fails after ledger commit."""

        try:
            yield
        except BaseException as exc:
            issue = (
                f"post-ledger runtime apply failed for {operation}: "
                f"{type(exc).__name__}: {exc}"
            )
            self._runtime_poisoned = issue
            self._hard_failures.append(issue)
            raise

    def _workflow(self, key: WorkflowKey) -> WorkflowRecord:
        try:
            return self._workflows[key]
        except KeyError as exc:
            raise IdentityError(f"unknown workflow: {key}") from exc

    def _active_workflow(self, key: WorkflowKey) -> WorkflowRecord:
        record = self._workflow(key)
        if record.status != WorkflowStatus.ACTIVE:
            raise StateTransitionError(f"workflow is terminal: {record.status.value}")
        return record

    @staticmethod
    def _node(record: WorkflowRecord, node_id: str) -> NodeRecord:
        try:
            return record.nodes[node_id]
        except KeyError as exc:
            raise IdentityError(f"unknown workflow node: {node_id}") from exc

    def _block(self, key: BlockKey) -> BlockRecord:
        try:
            return self._blocks[key]
        except KeyError as exc:
            raise IdentityError(f"unknown block: {key}") from exc

    def _binding(self, handle: BindingHandle) -> OwnerBinding:
        try:
            binding = self._bindings[handle.binding_id]
        except KeyError as exc:
            raise IdentityError(f"unknown binding: {handle.binding_id}") from exc
        if binding.handle != handle:
            raise IdentityError("binding capability does not match its owner")
        return binding

    def _transfer(self, transfer_id: str) -> Transfer:
        try:
            return self._transfers[transfer_id]
        except KeyError as exc:
            raise IdentityError(f"unknown transfer: {transfer_id}") from exc

    @staticmethod
    def _node_operation(workflow: WorkflowKey, node_id: str) -> str:
        return f"node:{workflow.workflow_id}:{workflow.epoch}:{node_id}"

    @staticmethod
    def _allocation_operation(replica_id: ReplicaId) -> str:
        return (
            f"allocation:{replica_id.tier.value}:{replica_id.device_id}:"
            f"{replica_id.slot_id}:{replica_id.generation}"
        )

    @staticmethod
    def _content_mapping_id(block_key: BlockKey, replica_id: ReplicaId) -> str:
        return (
            f"content-map:{block_key.content_digest}:"
            f"{replica_id.tier.value}:{replica_id.device_id}:"
            f"{replica_id.slot_id}:{replica_id.generation}"
        )

    def _next_execution_mapping_generation(self, binding_id: str) -> int:
        return self._execution_mapping_generations.get(binding_id, 0) + 1

    @staticmethod
    def _execution_mapping_id(
        handle: BindingHandle,
        replica_id: ReplicaId,
        mapping_generation: int,
    ) -> str:
        return (
            f"execution-map:{handle.binding_id}:{replica_id.device_id}:"
            f"{replica_id.slot_id}:{replica_id.generation}:"
            f"activation:{mapping_generation}"
        )

    @staticmethod
    def _validate_replica_sizes(byte_capacity: int, payload_size: int) -> None:
        if byte_capacity <= 0:
            raise IdentityError("replica byte_capacity must be positive")
        if payload_size <= 0 or payload_size > byte_capacity:
            raise IdentityError("replica payload_size must fit positive capacity")

    def _validate_new_slot(self, replica_id: ReplicaId) -> None:
        slot = replica_id.physical_slot
        if slot in self._slot_occupants:
            raise StateTransitionError(f"physical slot is occupied: {slot}")
        expected = self._slot_generations.get(slot, 0) + 1
        if replica_id.generation != expected:
            raise StateTransitionError(
                "stale or skipped slot generation: "
                f"{replica_id.generation} != {expected}"
            )

    def _occupy_new_slot(self, replica_id: ReplicaId) -> None:
        slot = replica_id.physical_slot
        self._slot_generations[slot] = replica_id.generation
        self._slot_occupants[slot] = replica_id

    def _free_slot(self, replica_id: ReplicaId) -> None:
        slot = replica_id.physical_slot
        if self._slot_occupants.get(slot) != replica_id:
            raise StateTransitionError("physical slot occupant changed before release")
        self._slot_occupants.pop(slot)

    @staticmethod
    def _planned_block_state(
        block: BlockRecord,
        *,
        replicas: Iterable[ReplicaId] | None = None,
        location_version: int | None = None,
        inflight_transfer_id: str | None = None,
        inflight_direction: TransferDirection | None = None,
        reclaimed: bool | None = None,
    ) -> BlockStateSnapshot:
        replica_ids = tuple(
            sorted(
                replicas
                if replicas is not None
                else (replica.replica_id for replica in block.replicas.values())
            )
        )
        final_reclaimed = block.reclaimed if reclaimed is None else reclaimed
        if final_reclaimed:
            residency = ResidencyState.FREED
        elif inflight_direction == TransferDirection.D2H:
            residency = ResidencyState.D2H_COPYING
        elif inflight_direction == TransferDirection.H2D:
            residency = ResidencyState.H2D_COPYING
        else:
            tiers = frozenset(replica.tier for replica in replica_ids)
            residency = {
                frozenset(): ResidencyState.ABSENT,
                frozenset({Tier.GPU}): ResidencyState.GPU_ONLY,
                frozenset({Tier.CPU}): ResidencyState.CPU_ONLY,
                frozenset({Tier.GPU, Tier.CPU}): ResidencyState.GPU_AND_CPU,
            }[tiers]
        return BlockStateSnapshot(
            location_version=(
                block.location_version if location_version is None else location_version
            ),
            residency=residency,
            replicas=replica_ids,
            inflight_transfer_id=inflight_transfer_id,
            inflight_direction=inflight_direction,
            reclaimed=final_reclaimed,
        )

    @staticmethod
    def _block_state_draft(
        block_key: BlockKey,
        state: BlockStateSnapshot,
        *,
        timestamp_ns: int,
        reason: str,
    ) -> EventDraft:
        return EventDraft(
            action=LedgerAction.BLOCK_STATE,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=(
                f"block-state:{block_key.content_digest}:"
                f"{state.location_version}:{state.residency.value}"
            ),
            block_key=block_key,
            block_state_after=state,
        )

    @staticmethod
    def _binding_state_draft(
        binding: OwnerBinding,
        state: BindingState,
        *,
        timestamp_ns: int,
        reason: str,
    ) -> EventDraft:
        if binding.bind_event_id is None:
            raise StateTransitionError("binding state transition lacks its open event")
        return EventDraft(
            action=LedgerAction.BIND_STATE,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=binding.binding_id,
            parent_event_id=binding.bind_event_id,
            workflow=binding.workflow,
            request_id=binding.request_id,
            node_id=binding.node_id,
            block_key=binding.block_key,
            binding_id=binding.binding_id,
            binding_kind=binding.kind,
            binding_state_before=binding.state,
            binding_state_after=state,
            execution_ref=binding.execution_ref,
        )

    @staticmethod
    def _waiter_draft(
        transfer: Transfer,
        binding: OwnerBinding,
        action: LedgerAction,
        waiter_binding_ids_after: Iterable[str],
        *,
        timestamp_ns: int,
        reason: str,
    ) -> EventDraft:
        if action not in {LedgerAction.WAITER_JOIN, LedgerAction.WAITER_LEAVE}:
            raise StateTransitionError("invalid waiter ledger action")
        return EventDraft(
            action=action,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=f"{transfer.transfer_id}:{binding.binding_id}:{action.value}",
            parent_event_id=transfer.scheduled_event_id,
            workflow=binding.workflow,
            request_id=binding.request_id,
            node_id=binding.node_id,
            block_key=binding.block_key,
            binding_id=binding.binding_id,
            binding_kind=binding.kind,
            transfer_id=transfer.transfer_id,
            execution_ref=binding.execution_ref,
            waiter_binding_ids_after=tuple(sorted(waiter_binding_ids_after)),
        )

    def _allocation_draft(
        self,
        block_key: BlockKey,
        replica_id: ReplicaId,
        *,
        byte_capacity: int,
        payload_size: int,
        payload_digest: str,
        timestamp_ns: int,
        reason: str,
        local_id: str | None = None,
    ) -> EventDraft:
        return EventDraft(
            action=LedgerAction.ALLOCATE,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=self._allocation_operation(replica_id),
            local_id=local_id,
            block_key=block_key,
            blocks=(replica_id,),
            payload_size=payload_size,
            byte_count=byte_capacity,
            payload_digest=payload_digest,
        )

    def _eviction_draft(
        self,
        block_key: BlockKey,
        replica: ReplicaRecord | ReplicaReservation,
        *,
        timestamp_ns: int,
        reason: str,
    ) -> EventDraft:
        return EventDraft(
            action=LedgerAction.EVICT,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=self._allocation_operation(replica.replica_id),
            parent_event_id=replica.allocate_event_id,
            block_key=block_key,
            blocks=(replica.replica_id,),
            payload_size=replica.payload_size,
            byte_count=replica.byte_capacity,
            payload_digest=replica.payload_digest,
        )

    def _execution_map_draft(
        self,
        binding: OwnerBinding,
        replica_id: ReplicaId,
        *,
        mapping_id: str,
        timestamp_ns: int,
        parent_event_id: str | None = None,
        parent_local_id: str | None = None,
        reason: str = "execution_mapping_published",
    ) -> EventDraft:
        return EventDraft(
            action=LedgerAction.EXEC_MAP,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=mapping_id,
            parent_event_id=parent_event_id,
            parent_local_id=parent_local_id,
            workflow=binding.workflow,
            request_id=binding.request_id,
            node_id=binding.node_id,
            block_key=binding.block_key,
            blocks=(replica_id,),
            binding_id=binding.binding_id,
            binding_kind=binding.kind,
            mapping_id=mapping_id,
            execution_ref=binding.execution_ref,
        )

    @staticmethod
    def _execution_unmap_draft(
        binding: OwnerBinding,
        mapping: ExecutionMapping,
        *,
        timestamp_ns: int,
        reason: str,
    ) -> EventDraft:
        return EventDraft(
            action=LedgerAction.EXEC_UNMAP,
            status=LedgerStatus.COMPLETED,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=mapping.mapping_id,
            parent_event_id=mapping.map_event_id,
            workflow=binding.workflow,
            request_id=binding.request_id,
            node_id=binding.node_id,
            block_key=binding.block_key,
            blocks=(mapping.gpu_replica,),
            binding_id=binding.binding_id,
            binding_kind=binding.kind,
            mapping_id=mapping.mapping_id,
            execution_ref=mapping.execution_ref,
        )

    def _mapping_for_binding(
        self,
        binding: OwnerBinding,
    ) -> ExecutionMapping | None:
        if binding.execution_ref is None:
            return None
        mapping = self._execution_mappings.get(binding.execution_ref)
        if mapping is not None and mapping.binding_id != binding.binding_id:
            raise StateTransitionError("execution mapping owner index is corrupt")
        return mapping

    def _lease_for_binding(
        self,
        lease_id: str,
        binding: OwnerBinding,
    ) -> Lease:
        try:
            lease = self._leases[lease_id]
        except KeyError as exc:
            raise IdentityError(f"unknown lease: {lease_id}") from exc
        if (
            lease.binding_id != binding.binding_id
            or lease.block_key != binding.block_key
        ):
            raise IdentityError("lease capability does not match its owner")
        return lease

    @staticmethod
    def _lease_status(state: LeaseState) -> LedgerStatus:
        return {
            LeaseState.EXPIRED: LedgerStatus.COMPLETED,
            LeaseState.CANCELLED: LedgerStatus.CANCELLED,
            LeaseState.FAILED: LedgerStatus.FAILED,
        }[state]

    def _lease_terminal_draft(
        self,
        binding: OwnerBinding,
        lease: Lease,
        state: LeaseState,
        *,
        timestamp_ns: int,
        error: str | None,
    ) -> EventDraft:
        if lease.scheduled_event_id is None:
            raise StateTransitionError("lease is missing its scheduled event")
        reason = (
            "deadline_expired"
            if state == LeaseState.EXPIRED
            else (error or state.value.lower())
        )
        return EventDraft(
            action=LedgerAction.LEASE,
            status=self._lease_status(state),
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=lease.lease_id,
            parent_event_id=lease.scheduled_event_id,
            workflow=binding.workflow,
            request_id=binding.request_id,
            node_id=binding.node_id,
            block_key=binding.block_key,
            binding_id=binding.binding_id,
            binding_kind=binding.kind,
            lease_id=lease.lease_id,
            lease_deadline_ns=lease.deadline_ns,
            error=error,
        )

    def _append_lease_terminal(
        self,
        binding: OwnerBinding,
        lease: Lease,
        state: LeaseState,
        *,
        timestamp_ns: int,
        error: str | None,
    ) -> None:
        self._ledger.append(
            self._lease_terminal_draft(
                binding,
                lease,
                state,
                timestamp_ns=timestamp_ns,
                error=error,
            )
        )

    def _ensure_no_active_lease(self, block: BlockRecord) -> None:
        active = [
            lease_id for lease_id in block.lease_ids if self._leases[lease_id].active
        ]
        if active:
            raise StateTransitionError(f"active leases protect the block: {active}")

    def _pre_service_demand_view_locked(
        self,
        block: BlockRecord,
        *,
        demand_commit_id: str,
        target_replica: ReplicaId,
        action: LedgerAction,
        transfer_id: str,
        timestamp_ns: int,
        waiters: tuple[OwnerBinding, ...],
    ) -> PreServiceDemandView:
        lifecycle_prefix = self._ledger.events
        last_event = lifecycle_prefix[-1] if lifecycle_prefix else None
        waiter_identities: list[WaiterIdentity] = []
        for binding in sorted(waiters, key=lambda item: item.binding_id):
            if binding.execution_ref is None:
                raise StateTransitionError("request waiter lost execution identity")
            waiter_identities.append(
                WaiterIdentity(
                    binding_id=binding.binding_id,
                    workflow=binding.workflow,
                    request_id=binding.request_id,
                    node_id=binding.node_id,
                    execution_ref=binding.execution_ref,
                    state=binding.state,
                )
            )
        return PreServiceDemandView(
            block_key=block.block_key,
            demand_commit_id=demand_commit_id,
            target_replica=target_replica,
            action=action,
            transfer_id=transfer_id,
            timestamp_ns=timestamp_ns,
            lifecycle_prefix=lifecycle_prefix,
            runtime_event_count=len(lifecycle_prefix),
            last_event_id=last_event.event_id if last_event is not None else None,
            last_event_timestamp_ns=(
                last_event.timestamp_ns if last_event is not None else None
            ),
            location_version=block.location_version,
            residency=block.residency,
            waiters=tuple(waiter_identities),
        )

    def _validate_gpu_fast_path_waiters(
        self,
        gpu: ReplicaRecord,
        waiters: tuple[OwnerBinding, ...],
    ) -> None:
        for binding in waiters:
            mapping = self._mapping_for_binding(binding)
            if mapping is not None:
                if mapping.gpu_replica != gpu.replica_id:
                    raise StateTransitionError("waiter maps a stale GPU generation")
                if binding.state != BindingState.REQUIRED:
                    raise StateTransitionError(
                        "mapped GPU fast-path waiter is not required"
                    )
            elif binding.state != BindingState.RETAINED:
                raise StateTransitionError(
                    f"invalid GPU fast-path waiter state: {binding.state.value}"
                )

    def _validate_fresh_demand_waiters(
        self,
        block: BlockRecord,
        waiters: tuple[OwnerBinding, ...],
    ) -> None:
        active_transfer_waiters: set[str] = set()
        if block.inflight_transfer_id is not None:
            transfer = self._transfers[block.inflight_transfer_id]
            if transfer.active:
                active_transfer_waiters = transfer.waiter_binding_ids
        for binding in waiters:
            if binding.state != BindingState.RETAINED:
                raise StateTransitionError(
                    "fresh demand waiter must be retained before service"
                )
            if self._mapping_for_binding(binding) is not None:
                raise StateTransitionError(
                    "fresh demand waiter already has an execution map"
                )
            if binding.binding_id in active_transfer_waiters:
                raise StateTransitionError(
                    "fresh demand waiter already belongs to an active transfer"
                )
            if binding.binding_id in self._execution_mapping_generations:
                raise StateTransitionError(
                    "fresh demand waiter has prior execution-map history"
                )
            if binding.binding_id in self._transfer_waiter_history:
                raise StateTransitionError(
                    "fresh demand waiter has prior transfer-service history"
                )

    def _validate_idempotent_demand_replay(
        self,
        block: BlockRecord,
        waiters: tuple[OwnerBinding, ...],
        *,
        gpu: ReplicaRecord | None,
    ) -> None:
        if gpu is not None:
            if all(
                binding.state == BindingState.REQUIRED
                and (mapping := self._mapping_for_binding(binding)) is not None
                and mapping.gpu_replica == gpu.replica_id
                for binding in waiters
            ):
                return
            raise StateTransitionError(
                "committed demand replay would create a second resident service"
            )
        transfer = (
            self._transfers.get(block.inflight_transfer_id)
            if block.inflight_transfer_id is not None
            else None
        )
        if (
            transfer is not None
            and transfer.active
            and all(
                binding.state == BindingState.WAITING
                and binding.binding_id in transfer.waiter_binding_ids
                for binding in waiters
            )
        ):
            return
        raise StateTransitionError(
            "committed demand replay would create a second H2D service"
        )

    def _validate_waiters(
        self,
        block_key: BlockKey,
        handles: Iterable[BindingHandle],
    ) -> tuple[OwnerBinding, ...]:
        handles_tuple = tuple(handles)
        if len(handles_tuple) != len(set(handles_tuple)):
            raise IdentityError("duplicate H2D waiter capability")
        waiters: list[OwnerBinding] = []
        for handle in handles_tuple:
            binding = self._binding(handle)
            if not binding.active or binding.kind != BindingKind.REQUEST:
                raise StateTransitionError("H2D waiter must be an active request")
            if binding.block_key != block_key or binding.execution_ref is None:
                raise IdentityError("H2D waiter refers to another block")
            self._require_running_binding_node(binding)
            waiters.append(binding)
        return tuple(waiters)

    def _require_running_binding_node(self, binding: OwnerBinding) -> None:
        workflow = self._active_workflow(binding.workflow)
        node = self._node(workflow, binding.node_id)
        if node.status != NodeStatus.RUNNING:
            raise StateTransitionError("H2D waiter requires a running DAG node")

    def _binding_node_is_running(self, binding: OwnerBinding) -> bool:
        workflow = self._workflows.get(binding.workflow)
        return (
            workflow is not None
            and workflow.status == WorkflowStatus.ACTIVE
            and workflow.nodes.get(binding.node_id) is not None
            and workflow.nodes[binding.node_id].status == NodeStatus.RUNNING
        )

    def _prepare_waiter(self, binding: OwnerBinding) -> None:
        mapping = self._mapping_for_binding(binding)
        if mapping is not None:
            raise StateTransitionError("CPU-only waiter still has an execution map")
        binding.transition(BindingState.WAITING)

    def _validate_h2d_waiter_preparation(
        self,
        waiters: tuple[OwnerBinding, ...],
        *,
        existing_transfer: Transfer | None = None,
    ) -> None:
        for binding in waiters:
            if self._mapping_for_binding(binding) is not None:
                raise StateTransitionError("H2D waiter still has an execution map")
            already_waiting = (
                existing_transfer is not None
                and binding.binding_id in existing_transfer.waiter_binding_ids
            )
            valid_state = binding.state == BindingState.RETAINED or (
                binding.state == BindingState.WAITING and already_waiting
            )
            if not valid_state:
                raise StateTransitionError(
                    f"invalid H2D waiter state: {binding.state.value}"
                )

    def _publish_waiter_mappings(
        self,
        block: BlockRecord,
        gpu: ReplicaRecord,
        waiters: tuple[OwnerBinding, ...],
        *,
        timestamp_ns: int,
        reason: str,
    ) -> None:
        pending: list[tuple[OwnerBinding, str, int]] = []
        drafts: list[EventDraft] = []
        for binding in sorted(waiters, key=lambda item: item.binding_id):
            self._require_running_binding_node(binding)
            mapping = self._mapping_for_binding(binding)
            if mapping is not None:
                if mapping.gpu_replica != gpu.replica_id:
                    raise StateTransitionError("waiter maps a stale GPU generation")
                if binding.state != BindingState.REQUIRED:
                    raise StateTransitionError(
                        "mapped GPU fast-path waiter is not required"
                    )
                continue
            if binding.state != BindingState.RETAINED:
                raise StateTransitionError(
                    f"invalid GPU fast-path waiter state: {binding.state.value}"
                )
            mapping_generation = self._next_execution_mapping_generation(
                binding.binding_id
            )
            mapping_id = self._execution_mapping_id(
                binding.handle,
                gpu.replica_id,
                mapping_generation,
            )
            drafts.append(
                self._execution_map_draft(
                    binding,
                    gpu.replica_id,
                    mapping_id=mapping_id,
                    timestamp_ns=timestamp_ns,
                    parent_event_id=binding.bind_event_id,
                    reason=reason,
                )
            )
            drafts.append(
                self._binding_state_draft(
                    binding,
                    BindingState.REQUIRED,
                    timestamp_ns=timestamp_ns,
                    reason=reason,
                )
            )
            pending.append((binding, mapping_id, mapping_generation))
        events = self._ledger.append_batch(drafts)
        with self._committed_runtime_apply("publish_waiter_mappings"):
            events_by_operation = {event.operation_id: event for event in events}
            for binding, mapping_id, mapping_generation in pending:
                if binding.execution_ref is None:
                    raise StateTransitionError("request waiter lost execution identity")
                self._execution_mappings[binding.execution_ref] = ExecutionMapping(
                    mapping_id=mapping_id,
                    execution_ref=binding.execution_ref,
                    binding_id=binding.binding_id,
                    block_key=binding.block_key,
                    gpu_replica=gpu.replica_id,
                    location_version=block.location_version,
                    map_event_id=events_by_operation[mapping_id].event_id,
                )
                self._execution_mapping_generations[binding.binding_id] = (
                    mapping_generation
                )
                binding.transition(BindingState.REQUIRED)

    def _schedule_transfer(
        self,
        block: BlockRecord,
        source: ReplicaRecord,
        target_replica: ReplicaId,
        *,
        transfer_id: str,
        action: LedgerAction,
        direction: TransferDirection,
        waiter_bindings: tuple[OwnerBinding, ...],
        timestamp_ns: int,
        reason: str,
    ) -> TransferCommand:
        require_text("transfer_id", transfer_id)
        if transfer_id in self._transfers:
            raise IdentityError(f"transfer ID already used: {transfer_id}")
        self._validate_new_slot(target_replica)
        drafts = [
            self._allocation_draft(
                block.block_key,
                target_replica,
                byte_capacity=source.byte_capacity,
                payload_size=source.payload_size,
                payload_digest=source.payload_digest,
                timestamp_ns=timestamp_ns,
                reason="transfer_target_reservation",
                local_id="target-allocation",
            ),
            EventDraft(
                action=action,
                status=LedgerStatus.SCHEDULED,
                reason=reason,
                timestamp_ns=timestamp_ns,
                operation_id=transfer_id,
                local_id="transfer-scheduled",
                parent_local_id="target-allocation",
                block_key=block.block_key,
                blocks=(target_replica,),
                transfer_id=transfer_id,
                source_tier=source.replica_id.tier,
                target_tier=target_replica.tier,
                byte_count=source.payload_size,
                payload_digest=source.payload_digest,
                waiter_binding_ids_after=(),
            ),
        ]
        waiter_ids: set[str] = set()
        for binding in sorted(waiter_bindings, key=lambda item: item.binding_id):
            drafts.append(
                self._binding_state_draft(
                    binding,
                    BindingState.WAITING,
                    timestamp_ns=timestamp_ns,
                    reason="h2d_waiter_prepared",
                )
            )
            waiter_ids.add(binding.binding_id)
            drafts.append(
                EventDraft(
                    action=LedgerAction.WAITER_JOIN,
                    status=LedgerStatus.COMPLETED,
                    reason="h2d_waiter_joined",
                    timestamp_ns=timestamp_ns,
                    operation_id=(f"{transfer_id}:{binding.binding_id}:waiter_join"),
                    parent_local_id="transfer-scheduled",
                    workflow=binding.workflow,
                    request_id=binding.request_id,
                    node_id=binding.node_id,
                    block_key=binding.block_key,
                    binding_id=binding.binding_id,
                    binding_kind=binding.kind,
                    transfer_id=transfer_id,
                    execution_ref=binding.execution_ref,
                    waiter_binding_ids_after=tuple(sorted(waiter_ids)),
                )
            )
        state_after = self._planned_block_state(
            block,
            inflight_transfer_id=transfer_id,
            inflight_direction=direction,
            reclaimed=False,
        )
        drafts.append(
            self._block_state_draft(
                block.block_key,
                state_after,
                timestamp_ns=timestamp_ns,
                reason="transfer_scheduled_state",
            )
        )
        events = self._ledger.append_batch(drafts)
        with self._committed_runtime_apply("schedule_transfer"):
            reservation = ReplicaReservation(
                replica_id=target_replica,
                block_key=block.block_key,
                byte_capacity=source.byte_capacity,
                payload_size=source.payload_size,
                payload_digest=source.payload_digest,
                allocate_event_id=events[0].event_id,
                transfer_id=transfer_id,
            )
            transfer = Transfer(
                transfer_id=transfer_id,
                direction=direction,
                block_key=block.block_key,
                source_replica=source.replica_id,
                target_replica=target_replica,
                declared_bytes=source.payload_size,
                payload_digest=source.payload_digest,
                started_ns=timestamp_ns,
                scheduled_event_id=events[1].event_id,
                ledger_action=action,
                waiter_binding_ids={binding.binding_id for binding in waiter_bindings},
            )
            self._reservations[target_replica] = reservation
            self._transfers[transfer_id] = transfer
            self._transfer_waiter_history.update(transfer.waiter_binding_ids)
            block.inflight_transfer_id = transfer_id
            block.inflight_direction = direction
            self._occupy_new_slot(target_replica)
            for binding in waiter_bindings:
                self._prepare_waiter(binding)
        return self._transfer_command(transfer)

    @staticmethod
    def _transfer_command(transfer: Transfer) -> TransferCommand:
        return TransferCommand(
            transfer_id=transfer.transfer_id,
            direction=transfer.direction,
            action=transfer.ledger_action,
            block_key=transfer.block_key,
            source_replica=transfer.source_replica,
            target_replica=transfer.target_replica,
            byte_count=transfer.declared_bytes,
            payload_digest=transfer.payload_digest,
        )

    def _validate_transfer_commit(
        self,
        transfer: Transfer,
    ) -> tuple[BlockRecord, ReplicaReservation, ReplicaRecord]:
        block = self._block(transfer.block_key)
        if block.inflight_transfer_id != transfer.transfer_id:
            raise StateTransitionError(
                "stale transfer no longer owns block inflight state"
            )
        try:
            reservation = self._reservations[transfer.target_replica]
        except KeyError as exc:
            raise StateTransitionError("stale transfer target reservation") from exc
        if reservation.transfer_id != transfer.transfer_id:
            raise StateTransitionError("target reservation belongs to another transfer")
        if self._slot_occupants.get(transfer.target_replica.physical_slot) != (
            transfer.target_replica
        ):
            raise StateTransitionError("target slot generation changed")
        source = block.replicas.get(transfer.source_replica.tier)
        if source is None or source.replica_id != transfer.source_replica:
            raise StateTransitionError("source replica generation changed")
        if (
            source.payload_size != transfer.declared_bytes
            or source.payload_digest != transfer.payload_digest
        ):
            raise StateTransitionError("source payload changed after scheduling")
        return block, reservation, source

    def _finalize_transfer_success(
        self,
        transfer: Transfer,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str,
    ) -> None:
        block, reservation, _ = self._validate_transfer_commit(transfer)
        mapping_id = self._content_mapping_id(block.block_key, transfer.target_replica)
        if mapping_id in self._content_mappings:
            raise StateTransitionError("target content mapping already exists")
        waiter_bindings: list[OwnerBinding] = []
        inactive_waiters: list[OwnerBinding] = []
        if transfer.direction == TransferDirection.H2D:
            for binding_id in sorted(transfer.waiter_binding_ids):
                binding = self._bindings[binding_id]
                if (
                    binding.active
                    and binding.state == BindingState.WAITING
                    and binding.block_key == block.block_key
                ):
                    if self._binding_node_is_running(binding):
                        waiter_bindings.append(binding)
                    else:
                        inactive_waiters.append(binding)
        drafts: list[EventDraft] = [
            self._transfer_terminal_draft(
                transfer,
                LedgerStatus.COMPLETED,
                timestamp_ns=timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
                reason="transfer_completed",
            ),
            EventDraft(
                action=LedgerAction.MAP,
                status=LedgerStatus.COMPLETED,
                reason="transferred_content_published",
                timestamp_ns=timestamp_ns,
                operation_id=mapping_id,
                parent_event_id=reservation.allocate_event_id,
                block_key=block.block_key,
                blocks=(transfer.target_replica,),
                mapping_id=mapping_id,
                payload_size=reservation.payload_size,
                payload_digest=transfer.payload_digest,
            ),
        ]
        for binding in inactive_waiters:
            drafts.append(
                self._binding_state_draft(
                    binding,
                    BindingState.RETAINED,
                    timestamp_ns=timestamp_ns,
                    reason="h2d_waiter_node_inactive",
                )
            )
        exec_pending: list[tuple[OwnerBinding, str, int]] = []
        for binding in waiter_bindings:
            mapping = self._mapping_for_binding(binding)
            if mapping is not None:
                raise StateTransitionError("waiting owner already has an execution map")
            mapping_generation = self._next_execution_mapping_generation(
                binding.binding_id
            )
            exec_mapping_id = self._execution_mapping_id(
                binding.handle,
                transfer.target_replica,
                mapping_generation,
            )
            drafts.append(
                self._execution_map_draft(
                    binding,
                    transfer.target_replica,
                    mapping_id=exec_mapping_id,
                    timestamp_ns=timestamp_ns,
                    parent_event_id=binding.bind_event_id,
                    reason="h2d_waiter_published",
                )
            )
            drafts.append(
                self._binding_state_draft(
                    binding,
                    BindingState.REQUIRED,
                    timestamp_ns=timestamp_ns,
                    reason="h2d_waiter_published",
                )
            )
            exec_pending.append((binding, exec_mapping_id, mapping_generation))
        final_replicas = {replica.replica_id for replica in block.replicas.values()}
        final_replicas.add(transfer.target_replica)
        location_version = block.location_version + int(
            transfer.target_replica.tier == Tier.GPU
        )
        drafts.append(
            self._block_state_draft(
                block.block_key,
                self._planned_block_state(
                    block,
                    replicas=final_replicas,
                    location_version=location_version,
                    reclaimed=False,
                ),
                timestamp_ns=timestamp_ns,
                reason="transfer_completed_state",
            )
        )
        events = self._ledger.append_batch(drafts)
        with self._committed_runtime_apply("complete_transfer"):
            transfer.terminate(
                TransferState.COMPLETED,
                timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
            )
            replica = ReplicaRecord(
                replica_id=transfer.target_replica,
                byte_capacity=reservation.byte_capacity,
                payload_size=reservation.payload_size,
                payload_digest=reservation.payload_digest,
                allocate_event_id=reservation.allocate_event_id,
                mapping_id=mapping_id,
            )
            block.replicas[transfer.target_replica.tier] = replica
            if transfer.target_replica.tier == Tier.GPU:
                block.location_version += 1
            self._content_mappings[mapping_id] = ContentMapping(
                mapping_id=mapping_id,
                block_key=block.block_key,
                replica_id=transfer.target_replica,
                map_event_id=events[1].event_id,
            )
            self._reservations.pop(transfer.target_replica)
            block.inflight_transfer_id = None
            block.inflight_direction = None
            for binding in inactive_waiters:
                binding.transition(BindingState.RETAINED)
            events_by_operation = {event.operation_id: event for event in events}
            for binding, exec_mapping_id, mapping_generation in exec_pending:
                if binding.execution_ref is None:
                    raise StateTransitionError("waiter lost execution identity")
                self._execution_mappings[binding.execution_ref] = ExecutionMapping(
                    mapping_id=exec_mapping_id,
                    execution_ref=binding.execution_ref,
                    binding_id=binding.binding_id,
                    block_key=binding.block_key,
                    gpu_replica=transfer.target_replica,
                    location_version=block.location_version,
                    map_event_id=events_by_operation[exec_mapping_id].event_id,
                )
                self._execution_mapping_generations[binding.binding_id] = (
                    mapping_generation
                )
                binding.transition(BindingState.REQUIRED)

    def _finalize_transfer_failure(
        self,
        transfer: Transfer,
        state: TransferState,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str | None,
        error: str,
        cleanup_reason: str,
    ) -> None:
        block, reservation, _ = self._validate_transfer_commit(transfer)
        status = {
            TransferState.FAILED: LedgerStatus.FAILED,
            TransferState.CANCELLED: LedgerStatus.CANCELLED,
        }[state]
        drafts = [
            self._transfer_terminal_draft(
                transfer,
                status,
                timestamp_ns=timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
                reason=error,
                error=error,
            ),
            self._eviction_draft(
                block.block_key,
                reservation,
                timestamp_ns=timestamp_ns,
                reason=cleanup_reason,
            ),
        ]
        for binding_id in sorted(transfer.waiter_binding_ids):
            binding = self._bindings[binding_id]
            if binding.active and binding.state == BindingState.WAITING:
                drafts.append(
                    self._binding_state_draft(
                        binding,
                        BindingState.RETAINED,
                        timestamp_ns=timestamp_ns,
                        reason="h2d_transfer_terminal",
                    )
                )
        drafts.append(
            self._block_state_draft(
                block.block_key,
                self._planned_block_state(
                    block,
                    reclaimed=False,
                ),
                timestamp_ns=timestamp_ns,
                reason="transfer_failed_state",
            )
        )
        self._ledger.append_batch(drafts)
        with self._committed_runtime_apply("terminate_transfer_failure"):
            transfer.terminate(
                state,
                timestamp_ns,
                observed_bytes=observed_bytes,
                observed_digest=observed_digest,
                error=error,
            )
            self._reservations.pop(transfer.target_replica)
            self._free_slot(transfer.target_replica)
            block.inflight_transfer_id = None
            block.inflight_direction = None
            for binding_id in transfer.waiter_binding_ids:
                binding = self._bindings[binding_id]
                if binding.active and binding.state == BindingState.WAITING:
                    binding.transition(BindingState.RETAINED)

    @staticmethod
    def _transfer_terminal_draft(
        transfer: Transfer,
        status: LedgerStatus,
        *,
        timestamp_ns: int,
        observed_bytes: int,
        observed_digest: str | None,
        reason: str,
        error: str | None = None,
    ) -> EventDraft:
        return EventDraft(
            action=transfer.ledger_action,
            status=status,
            reason=reason,
            timestamp_ns=timestamp_ns,
            operation_id=transfer.transfer_id,
            parent_event_id=transfer.scheduled_event_id,
            block_key=transfer.block_key,
            blocks=(transfer.target_replica,),
            transfer_id=transfer.transfer_id,
            source_tier=transfer.source_replica.tier,
            target_tier=transfer.target_replica.tier,
            byte_count=transfer.declared_bytes,
            observed_byte_count=observed_bytes,
            payload_digest=transfer.payload_digest,
            observed_digest=observed_digest,
            error=error,
            waiter_binding_ids_after=tuple(sorted(transfer.waiter_binding_ids)),
        )

    def _remove_waiter(self, binding_id: str) -> None:
        for transfer in self._transfers.values():
            if transfer.active:
                transfer.waiter_binding_ids.discard(binding_id)

    def _active_waiter_transfer(self, binding_id: str) -> Transfer:
        matches = [
            transfer
            for transfer in self._transfers.values()
            if transfer.active and binding_id in transfer.waiter_binding_ids
        ]
        if len(matches) != 1:
            raise StateTransitionError(
                "waiting binding must belong to exactly one active transfer"
            )
        return matches[0]

    def _validate_terminal_timestamp(self, timestamp_ns: int) -> None:
        if timestamp_ns < self._ledger.last_timestamp_ns:
            raise StateTransitionError("terminal timestamp predates ledger state")

    def _plan_binding_release(
        self,
        binding: OwnerBinding,
        *,
        timestamp_ns: int,
        reason: str,
        waiter_sets: dict[str, set[str]] | None = None,
    ) -> _BindingReleasePlan:
        if not binding.active:
            raise StateTransitionError("cannot plan an inactive binding release")
        if timestamp_ns < binding.created_ns:
            raise StateTransitionError("binding release predates creation")
        active_leases = tuple(
            sorted(
                (
                    self._leases[lease_id]
                    for lease_id in self._block(binding.block_key).lease_ids
                    if self._leases[lease_id].active
                    and self._leases[lease_id].binding_id == binding.binding_id
                ),
                key=lambda lease: lease.lease_id,
            )
        )
        if any(timestamp_ns < lease.registered_ns for lease in active_leases):
            raise StateTransitionError("release predates an attached lease")
        mapping = self._mapping_for_binding(binding)
        if (
            binding.execution_ref is not None
            and self._execution_owners.get(binding.execution_ref) != binding.binding_id
        ):
            raise StateTransitionError("binding execution-owner index is corrupt")
        drafts = [
            self._lease_terminal_draft(
                binding,
                lease,
                LeaseState.CANCELLED,
                timestamp_ns=timestamp_ns,
                error=reason,
            )
            for lease in active_leases
        ]
        if mapping is not None:
            drafts.append(
                self._execution_unmap_draft(
                    binding,
                    mapping,
                    timestamp_ns=timestamp_ns,
                    reason=reason,
                )
            )
        if binding.state == BindingState.WAITING:
            transfer = self._active_waiter_transfer(binding.binding_id)
            if waiter_sets is None:
                waiter_ids = set(transfer.waiter_binding_ids)
            else:
                waiter_ids = waiter_sets.setdefault(
                    transfer.transfer_id,
                    set(transfer.waiter_binding_ids),
                )
            if binding.binding_id not in waiter_ids:
                raise StateTransitionError("waiting release lost its transfer edge")
            waiter_ids.remove(binding.binding_id)
            drafts.append(
                self._waiter_draft(
                    transfer,
                    binding,
                    LedgerAction.WAITER_LEAVE,
                    waiter_ids,
                    timestamp_ns=timestamp_ns,
                    reason=reason,
                )
            )
        drafts.append(
            EventDraft(
                action=LedgerAction.RELEASE,
                status=LedgerStatus.COMPLETED,
                reason=reason,
                timestamp_ns=timestamp_ns,
                operation_id=binding.binding_id,
                parent_event_id=binding.bind_event_id,
                workflow=binding.workflow,
                request_id=binding.request_id,
                node_id=binding.node_id,
                block_key=binding.block_key,
                binding_id=binding.binding_id,
                binding_kind=binding.kind,
                binding_state_before=binding.state,
                binding_state_after=BindingState.RELEASED,
                execution_ref=binding.execution_ref,
            )
        )
        return _BindingReleasePlan(
            binding=binding,
            leases=active_leases,
            mapping=mapping,
            drafts=tuple(drafts),
            timestamp_ns=timestamp_ns,
            reason=reason,
        )

    def _commit_binding_release(self, plan: _BindingReleasePlan) -> None:
        binding = plan.binding
        block = self._block(binding.block_key)
        for lease in plan.leases:
            lease.terminate(
                LeaseState.CANCELLED,
                plan.timestamp_ns,
                error=plan.reason,
            )
            block.lease_ids.discard(lease.lease_id)
        if plan.mapping is not None:
            self._execution_mappings.pop(plan.mapping.execution_ref)
        if binding.execution_ref is not None:
            owner = self._execution_owners.get(binding.execution_ref)
            if owner != binding.binding_id:
                raise StateTransitionError("execution owner changed during release")
            self._execution_owners.pop(binding.execution_ref)
        binding.release(plan.timestamp_ns)
        self._workflow(binding.workflow).binding_ids.discard(binding.binding_id)
        block.binding_ids.discard(binding.binding_id)
        self._remove_waiter(binding.binding_id)

    def _release_workflow_bindings(
        self,
        workflow: WorkflowRecord,
        *,
        timestamp_ns: int,
    ) -> None:
        active = [
            self._bindings[binding_id]
            for binding_id in workflow.binding_ids
            if self._bindings[binding_id].active
        ]
        active.sort(
            key=lambda binding: (
                0 if binding.kind == BindingKind.WORKFLOW_RETENTION else 1,
                binding.binding_id,
            )
        )
        waiter_sets: dict[str, set[str]] = {}
        plans = [
            self._plan_binding_release(
                binding,
                timestamp_ns=timestamp_ns,
                reason="workflow_terminal",
                waiter_sets=waiter_sets,
            )
            for binding in active
        ]
        self._ledger.append_batch(draft for plan in plans for draft in plan.drafts)
        with self._committed_runtime_apply("release_workflow_bindings"):
            for plan in plans:
                self._commit_binding_release(plan)

    def _fail_workflow_locked(
        self,
        record: WorkflowRecord,
        *,
        timestamp_ns: int,
        error: str,
        failed_node: NodeRecord | None = None,
    ) -> None:
        self._validate_terminal_timestamp(timestamp_ns)
        drafts: list[EventDraft] = []
        running: list[NodeRecord] = []
        if failed_node is not None:
            if failed_node.scheduled_event_id is None:
                raise StateTransitionError("failed node lacks a scheduled event")
            drafts.append(
                EventDraft(
                    action=LedgerAction.NODE,
                    status=LedgerStatus.FAILED,
                    reason="node_failed",
                    timestamp_ns=timestamp_ns,
                    operation_id=self._node_operation(
                        record.spec.key, failed_node.node_id
                    ),
                    parent_event_id=failed_node.scheduled_event_id,
                    workflow=record.spec.key,
                    node_id=failed_node.node_id,
                    error=error,
                )
            )
        for node in record.nodes.values():
            if node.status == NodeStatus.RUNNING and node is not failed_node:
                if node.scheduled_event_id is None:
                    raise StateTransitionError("running node lacks a scheduled event")
                drafts.append(
                    EventDraft(
                        action=LedgerAction.NODE,
                        status=LedgerStatus.CANCELLED,
                        reason="workflow_failed",
                        timestamp_ns=timestamp_ns,
                        operation_id=self._node_operation(
                            record.spec.key, node.node_id
                        ),
                        parent_event_id=node.scheduled_event_id,
                        workflow=record.spec.key,
                        node_id=node.node_id,
                        error=error,
                    )
                )
                running.append(node)
        active = [
            self._bindings[binding_id]
            for binding_id in record.binding_ids
            if self._bindings[binding_id].active
        ]
        active.sort(
            key=lambda binding: (
                0 if binding.kind == BindingKind.WORKFLOW_RETENTION else 1,
                binding.binding_id,
            )
        )
        waiter_sets: dict[str, set[str]] = {}
        plans = [
            self._plan_binding_release(
                binding,
                timestamp_ns=timestamp_ns,
                reason="workflow_terminal",
                waiter_sets=waiter_sets,
            )
            for binding in active
        ]
        release_drafts = [draft for plan in plans for draft in plan.drafts]
        self._ledger.append_batch((*release_drafts, *drafts))
        with self._committed_runtime_apply("fail_workflow"):
            if failed_node is not None:
                failed_node.status = NodeStatus.FAILED
                failed_node.terminal_ns = timestamp_ns
                failed_node.error = error
            for node in running:
                node.status = NodeStatus.SKIPPED
                node.terminal_ns = timestamp_ns
                node.error = error
            for node in record.nodes.values():
                if node.status in {NodeStatus.PENDING, NodeStatus.READY}:
                    node.status = NodeStatus.SKIPPED
                    node.terminal_ns = timestamp_ns
                    node.error = error
            for plan in plans:
                self._commit_binding_release(plan)
            record.status = WorkflowStatus.FAILED
            record.terminal_ns = timestamp_ns
            record.error = error

    def _audit_workflows(
        self,
        issues: list[str],
        active_bindings: dict[str, OwnerBinding],
    ) -> None:
        for key, workflow in self._workflows.items():
            expected = {
                binding_id
                for binding_id, binding in active_bindings.items()
                if binding.workflow == key
            }
            if workflow.binding_ids != expected:
                issues.append(f"workflow binding index mismatch: {key}")
            for node in workflow.nodes.values():
                parent_states = [
                    workflow.nodes[parent].status for parent in node.predecessors
                ]
                if node.status in {
                    NodeStatus.READY,
                    NodeStatus.RUNNING,
                    NodeStatus.DONE,
                } and any(status != NodeStatus.DONE for status in parent_states):
                    issues.append(f"node dependency violation: {key}/{node.node_id}")
                if (
                    node.status == NodeStatus.RUNNING
                    and node.scheduled_event_id is None
                ):
                    issues.append(f"running node lacks event: {key}/{node.node_id}")
            if workflow.status == WorkflowStatus.DONE and any(
                node.status != NodeStatus.DONE for node in workflow.nodes.values()
            ):
                issues.append(f"completed workflow has non-done nodes: {key}")
            if workflow.status == WorkflowStatus.ACTIVE and any(
                node.status in {NodeStatus.FAILED, NodeStatus.SKIPPED}
                for node in workflow.nodes.values()
            ):
                issues.append(f"active workflow has terminal-failure nodes: {key}")
            if workflow.status != WorkflowStatus.ACTIVE and workflow.binding_ids:
                issues.append(f"terminal workflow retains bindings: {key}")

    def _audit_blocks(
        self,
        issues: list[str],
        active_bindings: dict[str, OwnerBinding],
        active_leases: dict[str, Lease],
        active_transfers: dict[str, Transfer],
    ) -> None:
        expected_execution_history: dict[ExecutionRef, tuple[BlockKey, str]] = {}
        for binding in self._bindings.values():
            if binding.execution_ref is None:
                continue
            identity = (binding.block_key, binding.binding_id)
            prior = expected_execution_history.setdefault(
                binding.execution_ref,
                identity,
            )
            if prior != identity:
                issues.append(f"execution reference reused: {binding.execution_ref}")
        if self._execution_history != expected_execution_history:
            issues.append("execution history index mismatch")

        expected_execution_owners = {
            binding.execution_ref: binding.binding_id
            for binding in active_bindings.values()
            if binding.kind == BindingKind.REQUEST and binding.execution_ref is not None
        }
        if self._execution_owners != expected_execution_owners:
            issues.append("execution owner index mismatch")

        expected_mapping_generations: dict[str, int] = {}
        for event in self._ledger.events:
            if event.action == LedgerAction.EXEC_MAP and event.binding_id is not None:
                expected_mapping_generations[event.binding_id] = (
                    expected_mapping_generations.get(event.binding_id, 0) + 1
                )
        if self._execution_mapping_generations != expected_mapping_generations:
            issues.append("execution mapping generation index mismatch")

        for key, block in self._blocks.items():
            expected_bindings = {
                binding_id
                for binding_id, binding in active_bindings.items()
                if binding.block_key == key
            }
            expected_leases = {
                lease_id
                for lease_id, lease in active_leases.items()
                if lease.block_key == key
            }
            if block.binding_ids != expected_bindings:
                issues.append(f"block binding index mismatch: {key.content_digest}")
            if block.lease_ids != expected_leases:
                issues.append(f"block lease index mismatch: {key.content_digest}")
            inflight = [
                transfer
                for transfer in active_transfers.values()
                if transfer.block_key == key
            ]
            if len(inflight) > 1:
                issues.append(f"multiple block transfers: {key.content_digest}")
            expected_inflight = inflight[0].transfer_id if inflight else None
            if block.inflight_transfer_id != expected_inflight:
                issues.append(f"block transfer index mismatch: {key.content_digest}")
            if block.reclaimed and (
                block.replicas
                or block.binding_ids
                or block.lease_ids
                or block.inflight_transfer_id
            ):
                issues.append(f"reclaimed block retains state: {key.content_digest}")
            for replica in block.replicas.values():
                if (
                    replica.payload_size != block.payload_size
                    or replica.payload_digest != block.payload_digest
                ):
                    issues.append(
                        f"replica payload conflicts with block: {replica.replica_id}"
                    )
                mapping = self._content_mappings.get(replica.mapping_id)
                if (
                    mapping is None
                    or mapping.block_key != key
                    or mapping.replica_id != replica.replica_id
                ):
                    issues.append(f"replica content map mismatch: {replica.replica_id}")
                if self._slot_occupants.get(replica.replica_id.physical_slot) != (
                    replica.replica_id
                ):
                    issues.append(f"replica slot mismatch: {replica.replica_id}")
        for binding in active_bindings.values():
            workflow = self._workflows.get(binding.workflow)
            block = self._blocks.get(binding.block_key)
            if workflow is None or binding.binding_id not in workflow.binding_ids:
                issues.append(f"binding missing workflow edge: {binding.binding_id}")
            if block is None or binding.binding_id not in block.binding_ids:
                issues.append(f"binding missing block edge: {binding.binding_id}")
            mapping = (
                self._execution_mappings.get(binding.execution_ref)
                if binding.execution_ref is not None
                else None
            )
            if mapping is not None and mapping.binding_id != binding.binding_id:
                issues.append(
                    f"execution map has conflicting owner: {mapping.mapping_id}"
                )
            if binding.kind == BindingKind.REQUEST:
                if binding.state == BindingState.REQUIRED and mapping is None:
                    issues.append(
                        f"required binding lacks execution map: {binding.binding_id}"
                    )
                if binding.state == BindingState.RETAINED and mapping is not None:
                    issues.append(
                        f"retained binding has execution map: {binding.binding_id}"
                    )
                if binding.state == BindingState.WAITING:
                    is_waiter = any(
                        transfer.direction == TransferDirection.H2D
                        and binding.binding_id in transfer.waiter_binding_ids
                        for transfer in active_transfers.values()
                    )
                    if mapping is not None or not is_waiter:
                        issues.append(
                            f"waiting binding lacks H2D edge: {binding.binding_id}"
                        )
            elif binding.state != BindingState.RETAINED or mapping is not None:
                issues.append(f"invalid retention binding state: {binding.binding_id}")
        for lease in active_leases.values():
            binding = self._bindings.get(lease.binding_id)
            if (
                binding is None
                or not binding.active
                or binding.kind != BindingKind.WORKFLOW_RETENTION
                or binding.block_key != lease.block_key
            ):
                issues.append(f"invalid active lease owner: {lease.lease_id}")
        for execution_ref, mapping in self._execution_mappings.items():
            binding = self._bindings.get(mapping.binding_id)
            block = self._blocks.get(mapping.block_key)
            gpu = block.replicas.get(Tier.GPU) if block else None
            if (
                binding is None
                or not binding.active
                or binding.execution_ref != execution_ref
                or binding.state != BindingState.REQUIRED
            ):
                issues.append(f"invalid execution map owner: {mapping.mapping_id}")
            if (
                gpu is None
                or gpu.replica_id != mapping.gpu_replica
                or block is None
                or mapping.location_version != block.location_version
            ):
                issues.append(f"stale execution map: {mapping.mapping_id}")
        for transfer in active_transfers.values():
            reservation = self._reservations.get(transfer.target_replica)
            if reservation is None or reservation.transfer_id != transfer.transfer_id:
                issues.append(f"transfer reservation mismatch: {transfer.transfer_id}")
            block = self._blocks.get(transfer.block_key)
            if block is None or block.inflight_direction != transfer.direction:
                issues.append(
                    f"transfer direction index mismatch: {transfer.transfer_id}"
                )
            if (
                reservation is not None
                and block is not None
                and (
                    reservation.payload_size != block.payload_size
                    or reservation.payload_digest != block.payload_digest
                )
            ):
                issues.append(f"reservation payload mismatch: {transfer.transfer_id}")
        for mapping_id, mapping in self._content_mappings.items():
            block = self._blocks.get(mapping.block_key)
            replica = block.replicas.get(mapping.replica_id.tier) if block else None
            if (
                replica is None
                or replica.replica_id != mapping.replica_id
                or replica.mapping_id != mapping_id
            ):
                issues.append(f"orphan content mapping: {mapping_id}")
        for replica_id, reservation in self._reservations.items():
            transfer = active_transfers.get(reservation.transfer_id)
            if transfer is None or transfer.target_replica != replica_id:
                issues.append(f"orphan transfer reservation: {replica_id}")

    def _audit_slots(self, issues: list[str]) -> None:
        expected: dict[tuple[Tier, str, str], ReplicaId] = {}
        for block in self._blocks.values():
            for replica in block.replicas.values():
                slot = replica.replica_id.physical_slot
                if slot in expected:
                    issues.append(f"duplicate live physical slot: {slot}")
                expected[slot] = replica.replica_id
        for reservation in self._reservations.values():
            slot = reservation.replica_id.physical_slot
            if slot in expected:
                issues.append(f"reservation aliases a live physical slot: {slot}")
            expected[slot] = reservation.replica_id
        if self._slot_occupants != expected:
            issues.append("slot occupant index mismatch")
        for slot, replica in self._slot_occupants.items():
            if self._slot_generations.get(slot) != replica.generation:
                issues.append(f"slot generation index mismatch: {slot}")

    def _live_replica_count(self) -> int:
        return sum(len(block.replicas) for block in self._blocks.values())
