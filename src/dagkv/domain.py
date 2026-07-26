"""Canonical identities and runtime state for DAGKV."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from graphlib import CycleError, TopologicalSorter

_SHA256 = re.compile(r"[0-9a-f]{64}")


class DAGKVError(Exception):
    """Base class for fail-closed DAGKV errors."""


class IdentityError(DAGKVError, ValueError):
    """Raised when an identity is incomplete or internally inconsistent."""


class StateTransitionError(DAGKVError, RuntimeError):
    """Raised when a lifecycle transition would violate an invariant."""


class TransferIntegrityError(DAGKVError, RuntimeError):
    """Raised when a physical transfer reports inconsistent payload data."""


class Tier(StrEnum):
    """Physical KV-cache storage tier."""

    GPU = "GPU"
    CPU = "CPU"


class WorkflowStatus(StrEnum):
    """Terminal-aware workflow state."""

    ACTIVE = "ACTIVE"
    DONE = "DONE"
    FAILED = "FAILED"


class NodeStatus(StrEnum):
    """Runtime state of one node in a workflow DAG."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class BindingKind(StrEnum):
    """Logical owner lifetime represented by a binding."""

    REQUEST = "request"
    WORKFLOW_RETENTION = "workflow_retention"


class BindingState(StrEnum):
    """Availability state of an active logical owner."""

    REQUIRED = "REQUIRED"
    WAITING = "WAITING"
    RETAINED = "RETAINED"
    RELEASED = "RELEASED"


class LeaseState(StrEnum):
    """TTL lease state."""

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TransferDirection(StrEnum):
    """Direction of one physical KV payload copy."""

    D2H = "D2H"
    H2D = "H2D"


class TransferState(StrEnum):
    """Physical transfer state."""

    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResidencyState(StrEnum):
    """Derived block residency state."""

    GPU_ONLY = "GPU_ONLY"
    CPU_ONLY = "CPU_ONLY"
    GPU_AND_CPU = "GPU_AND_CPU"
    D2H_COPYING = "D2H_COPYING"
    H2D_COPYING = "H2D_COPYING"
    ABSENT = "ABSENT"
    FREED = "FREED"


class LedgerAction(StrEnum):
    """Lifecycle action recorded by the append-only ledger."""

    ALLOCATE = "allocate"
    MAP = "map"
    BIND = "bind"
    BIND_STATE = "bind_state"
    RELEASE = "release"
    LEASE = "lease"
    SAVE = "save"
    LOAD = "load"
    PREFETCH = "prefetch"
    UNMAP = "unmap"
    EVICT = "evict"
    EXEC_MAP = "exec_map"
    EXEC_UNMAP = "exec_unmap"
    WAITER_JOIN = "waiter_join"
    WAITER_LEAVE = "waiter_leave"
    BLOCK_STATE = "block_state"
    STREAM_SEAL = "stream_seal"
    NODE = "node"


class LedgerStatus(StrEnum):
    """Lifecycle event status."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def require_text(name: str, value: str) -> None:
    """Validate a required non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{name} is required")


def require_sha256(name: str, value: str) -> None:
    """Validate the canonical lowercase representation of a SHA-256 digest."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityError(f"{name} must be a lowercase SHA-256 digest")


def require_optional_sha256(name: str, value: str | None) -> None:
    """Validate an optional canonical SHA-256 digest."""

    if value is not None:
        require_sha256(name, value)


@dataclass(frozen=True, order=True, slots=True)
class WorkflowKey:
    """Identity of one workflow DAG instance."""

    workflow_id: str
    epoch: int

    def __post_init__(self) -> None:
        require_text("workflow_id", self.workflow_id)
        if self.epoch < 0:
            raise IdentityError("workflow epoch must be non-negative")


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One immutable node in a workflow DAG."""

    node_id: str
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text("node_id", self.node_id)
        for predecessor in self.predecessors:
            require_text("predecessor", predecessor)
        if len(self.predecessors) != len(set(self.predecessors)):
            raise IdentityError(f"duplicate predecessor for node {self.node_id}")
        if self.node_id in self.predecessors:
            raise IdentityError(f"node {self.node_id} cannot depend on itself")


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """Immutable validated DAG topology."""

    key: WorkflowKey
    nodes: tuple[WorkflowNode, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise IdentityError("workflow requires at least one node")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise IdentityError("workflow node IDs must be unique")
        known = set(node_ids)
        graph = {node.node_id: set(node.predecessors) for node in self.nodes}
        unknown = sorted(
            predecessor
            for node in self.nodes
            for predecessor in node.predecessors
            if predecessor not in known
        )
        if unknown:
            raise IdentityError(f"unknown workflow predecessors: {unknown}")
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            raise IdentityError("workflow graph must be acyclic") from exc

    @property
    def node_ids(self) -> frozenset[str]:
        """Return all node identities."""

        return frozenset(node.node_id for node in self.nodes)

    def successors(self, node_id: str) -> tuple[str, ...]:
        """Return direct successors in declared node order."""

        return tuple(
            node.node_id for node in self.nodes if node_id in node.predecessors
        )


@dataclass(slots=True)
class NodeRecord:
    """Mutable runtime state for one DAG node."""

    workflow: WorkflowKey
    node_id: str
    predecessors: tuple[str, ...]
    successors: tuple[str, ...]
    status: NodeStatus
    started_ns: int | None = None
    terminal_ns: int | None = None
    error: str | None = None
    scheduled_event_id: str | None = None


@dataclass(slots=True)
class WorkflowRecord:
    """Runtime state and derived indices for one workflow."""

    spec: WorkflowSpec
    nodes: dict[str, NodeRecord]
    status: WorkflowStatus = WorkflowStatus.ACTIVE
    binding_ids: set[str] = field(default_factory=set)
    terminal_ns: int | None = None
    error: str | None = None


@dataclass(frozen=True, order=True, slots=True)
class BlockKey:
    """Complete immutable identity of one KV content block."""

    content_digest: str
    parent_digest: str | None
    model_fingerprint: str
    tokenizer_fingerprint: str
    adapter_fingerprint: str | None
    block_size_tokens: int
    kv_dtype: str
    cache_salt: str | None = None

    def __post_init__(self) -> None:
        require_sha256("content_digest", self.content_digest)
        require_optional_sha256("parent_digest", self.parent_digest)
        require_text("model_fingerprint", self.model_fingerprint)
        require_text("tokenizer_fingerprint", self.tokenizer_fingerprint)
        if self.adapter_fingerprint is not None:
            require_text("adapter_fingerprint", self.adapter_fingerprint)
        require_text("kv_dtype", self.kv_dtype)
        if self.cache_salt is not None:
            require_text("cache_salt", self.cache_salt)
        if self.block_size_tokens <= 0:
            raise IdentityError("block_size_tokens must be positive")


@dataclass(frozen=True, order=True, slots=True)
class ReplicaId:
    """Generation-safe identity of one physical allocation slot."""

    tier: Tier
    device_id: str
    slot_id: str
    generation: int

    def __post_init__(self) -> None:
        require_text("device_id", self.device_id)
        require_text("slot_id", self.slot_id)
        if self.generation < 1:
            raise IdentityError("replica generation must start at one")

    @property
    def physical_slot(self) -> tuple[Tier, str, str]:
        """Return the reusable slot identity without its generation."""

        return (self.tier, self.device_id, self.slot_id)


@dataclass(frozen=True, order=True, slots=True)
class ExecutionRef:
    """Request-local logical block identity used by an engine adapter."""

    workflow: WorkflowKey
    request_id: str
    sequence_id: str
    logical_block_index: int

    def __post_init__(self) -> None:
        require_text("request_id", self.request_id)
        require_text("sequence_id", self.sequence_id)
        if self.logical_block_index < 0:
            raise IdentityError("logical_block_index must be non-negative")


@dataclass(frozen=True, order=True, slots=True)
class BindingHandle:
    """Caller-visible capability used for owner-scoped operations."""

    workflow: WorkflowKey
    request_id: str
    binding_id: str

    def __post_init__(self) -> None:
        require_text("request_id", self.request_id)
        require_text("binding_id", self.binding_id)


@dataclass(slots=True)
class OwnerBinding:
    """One logical owner-to-block relationship."""

    handle: BindingHandle
    node_id: str
    block_key: BlockKey
    kind: BindingKind
    execution_ref: ExecutionRef | None
    created_ns: int
    state: BindingState
    released_ns: int | None = None
    bind_event_id: str | None = None

    def __post_init__(self) -> None:
        require_text("node_id", self.node_id)
        if self.created_ns < 0:
            raise IdentityError("binding created_ns must be non-negative")
        if self.kind == BindingKind.REQUEST:
            if self.execution_ref is None:
                raise IdentityError("request binding requires an execution reference")
            if self.execution_ref.workflow != self.handle.workflow:
                raise IdentityError("execution reference belongs to another workflow")
            if self.execution_ref.request_id != self.handle.request_id:
                raise IdentityError("execution reference belongs to another request")
        elif self.execution_ref is not None:
            raise IdentityError("retention binding cannot own an execution reference")
        if self.kind == BindingKind.WORKFLOW_RETENTION and self.state not in {
            BindingState.RETAINED,
            BindingState.RELEASED,
        }:
            raise IdentityError("retention binding must use retained state")

    @property
    def binding_id(self) -> str:
        """Return the globally unique binding ID."""

        return self.handle.binding_id

    @property
    def workflow(self) -> WorkflowKey:
        """Return the binding workflow."""

        return self.handle.workflow

    @property
    def request_id(self) -> str:
        """Return the binding request identity."""

        return self.handle.request_id

    @property
    def active(self) -> bool:
        """Return whether the binding still owns the block."""

        return self.state != BindingState.RELEASED

    def transition(self, state: BindingState) -> bool:
        """Move an active request binding to another availability state."""

        if self.kind != BindingKind.REQUEST:
            raise StateTransitionError("retention binding state is fixed")
        if state == BindingState.RELEASED:
            raise StateTransitionError("use release() to close a binding")
        if not self.active:
            raise StateTransitionError("released binding cannot transition")
        if self.state == state:
            return False
        self.state = state
        return True

    def release(self, timestamp_ns: int) -> bool:
        """Close the binding once; later owner-validated calls are idempotent."""

        if not self.active:
            return False
        if timestamp_ns < self.created_ns:
            raise StateTransitionError("binding release predates creation")
        self.state = BindingState.RELEASED
        self.released_ns = timestamp_ns
        return True


@dataclass(slots=True)
class Lease:
    """Explicit TTL protection owned by a retention binding."""

    lease_id: str
    binding_id: str
    block_key: BlockKey
    registered_ns: int
    deadline_ns: int
    reason: str
    state: LeaseState = LeaseState.ACTIVE
    terminal_ns: int | None = None
    error: str | None = None
    scheduled_event_id: str | None = None

    def __post_init__(self) -> None:
        require_text("lease_id", self.lease_id)
        require_text("binding_id", self.binding_id)
        require_text("lease reason", self.reason)
        if self.registered_ns < 0:
            raise IdentityError("lease registered_ns must be non-negative")
        if self.deadline_ns < self.registered_ns:
            raise IdentityError("lease deadline must not predate registration")

    @property
    def active(self) -> bool:
        """Return whether the lease is open."""

        return self.state == LeaseState.ACTIVE

    def terminate(
        self,
        state: LeaseState,
        timestamp_ns: int,
        *,
        error: str | None = None,
    ) -> bool:
        """Record one exact terminal, allowing only an exact replay."""

        if state == LeaseState.ACTIVE:
            raise StateTransitionError("lease terminal must be non-active")
        if not self.active:
            if (
                self.state == state
                and self.terminal_ns == timestamp_ns
                and self.error == error
            ):
                return False
            raise StateTransitionError("conflicting lease terminal replay")
        if timestamp_ns < self.registered_ns:
            raise StateTransitionError("lease terminal predates registration")
        if state == LeaseState.EXPIRED and timestamp_ns < self.deadline_ns:
            raise StateTransitionError("lease cannot expire before its deadline")
        self.state = state
        self.terminal_ns = timestamp_ns
        self.error = error
        return True


@dataclass(slots=True)
class ContentMapping:
    """Content-to-allocation mapping, independent from request execution maps."""

    mapping_id: str
    block_key: BlockKey
    replica_id: ReplicaId
    map_event_id: str

    def __post_init__(self) -> None:
        require_text("mapping_id", self.mapping_id)
        require_text("map_event_id", self.map_event_id)


@dataclass(slots=True)
class ReplicaRecord:
    """One published physical replica."""

    replica_id: ReplicaId
    byte_capacity: int
    payload_size: int
    payload_digest: str
    allocate_event_id: str
    mapping_id: str

    def __post_init__(self) -> None:
        if self.byte_capacity <= 0:
            raise IdentityError("replica byte_capacity must be positive")
        if self.payload_size <= 0 or self.payload_size > self.byte_capacity:
            raise IdentityError("replica payload_size must fit positive capacity")
        require_sha256("payload_digest", self.payload_digest)
        require_text("allocate_event_id", self.allocate_event_id)
        require_text("mapping_id", self.mapping_id)


@dataclass(slots=True)
class ReplicaReservation:
    """Allocated target slot that is private to one in-flight transfer."""

    replica_id: ReplicaId
    block_key: BlockKey
    byte_capacity: int
    payload_size: int
    payload_digest: str
    allocate_event_id: str
    transfer_id: str

    def __post_init__(self) -> None:
        if self.byte_capacity <= 0:
            raise IdentityError("reservation byte_capacity must be positive")
        if self.payload_size <= 0 or self.payload_size > self.byte_capacity:
            raise IdentityError("reservation payload_size must fit capacity")
        require_sha256("payload_digest", self.payload_digest)
        require_text("allocate_event_id", self.allocate_event_id)
        require_text("transfer_id", self.transfer_id)


@dataclass(slots=True)
class ExecutionMapping:
    """Request-local mapping to a published GPU replica."""

    mapping_id: str
    execution_ref: ExecutionRef
    binding_id: str
    block_key: BlockKey
    gpu_replica: ReplicaId
    location_version: int
    map_event_id: str


@dataclass(slots=True)
class Transfer:
    """One generation-bound physical copy and its exact terminal result."""

    transfer_id: str
    direction: TransferDirection
    block_key: BlockKey
    source_replica: ReplicaId
    target_replica: ReplicaId
    declared_bytes: int
    payload_digest: str
    started_ns: int
    scheduled_event_id: str
    ledger_action: LedgerAction
    waiter_binding_ids: set[str] = field(default_factory=set)
    state: TransferState = TransferState.SCHEDULED
    observed_bytes: int | None = None
    observed_digest: str | None = None
    terminal_ns: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        require_text("transfer_id", self.transfer_id)
        require_sha256("payload_digest", self.payload_digest)
        require_text("scheduled_event_id", self.scheduled_event_id)
        if self.declared_bytes <= 0:
            raise IdentityError("transfer declared_bytes must be positive")
        if self.started_ns < 0:
            raise IdentityError("transfer started_ns must be non-negative")
        if self.source_replica == self.target_replica:
            raise IdentityError("transfer source and target must differ")
        if self.direction == TransferDirection.D2H:
            valid = (
                self.source_replica.tier == Tier.GPU
                and self.target_replica.tier == Tier.CPU
                and self.ledger_action == LedgerAction.SAVE
            )
        else:
            valid = (
                self.source_replica.tier == Tier.CPU
                and self.target_replica.tier == Tier.GPU
                and self.ledger_action in {LedgerAction.LOAD, LedgerAction.PREFETCH}
            )
        if not valid:
            raise IdentityError("transfer direction, tiers, and action disagree")

    @property
    def active(self) -> bool:
        """Return whether the transfer awaits its terminal callback."""

        return self.state == TransferState.SCHEDULED

    def terminate(
        self,
        state: TransferState,
        timestamp_ns: int,
        *,
        observed_bytes: int,
        observed_digest: str | None,
        error: str | None = None,
    ) -> bool:
        """Record one exact terminal, allowing only an exact replay."""

        if state == TransferState.SCHEDULED:
            raise StateTransitionError("transfer terminal must be non-scheduled")
        if observed_digest is not None:
            require_sha256("observed_digest", observed_digest)
        if not self.active:
            if (
                self.state == state
                and self.terminal_ns == timestamp_ns
                and self.observed_bytes == observed_bytes
                and self.observed_digest == observed_digest
                and self.error == error
            ):
                return False
            raise StateTransitionError("conflicting transfer terminal replay")
        if timestamp_ns < self.started_ns:
            raise StateTransitionError("transfer terminal predates scheduling")
        if observed_bytes < 0:
            raise TransferIntegrityError("observed transfer bytes must be non-negative")
        self.state = state
        self.observed_bytes = observed_bytes
        self.observed_digest = observed_digest
        self.terminal_ns = timestamp_ns
        self.error = error
        return True


@dataclass(frozen=True, slots=True)
class TransferCommand:
    """Immutable command consumed by an external DMA adapter."""

    transfer_id: str
    direction: TransferDirection
    action: LedgerAction
    block_key: BlockKey
    source_replica: ReplicaId
    target_replica: ReplicaId
    byte_count: int
    payload_digest: str


@dataclass(slots=True)
class BlockRecord:
    """Canonical block state plus explicitly audited derived indices."""

    block_key: BlockKey
    payload_size: int | None = None
    payload_digest: str | None = None
    location_version: int = 0
    replicas: dict[Tier, ReplicaRecord] = field(default_factory=dict)
    binding_ids: set[str] = field(default_factory=set)
    lease_ids: set[str] = field(default_factory=set)
    inflight_transfer_id: str | None = None
    inflight_direction: TransferDirection | None = None
    reclaimed: bool = False

    def __post_init__(self) -> None:
        if (self.payload_size is None) != (self.payload_digest is None):
            raise IdentityError("block payload size and digest must appear together")
        if self.payload_size is not None and self.payload_size <= 0:
            raise IdentityError("block payload_size must be positive")
        if self.payload_digest is not None:
            require_sha256("payload_digest", self.payload_digest)

    @property
    def residency(self) -> ResidencyState:
        """Derive residency from published replicas and transfer state."""

        if self.reclaimed:
            return ResidencyState.FREED
        if self.inflight_transfer_id is not None:
            if self.inflight_direction == TransferDirection.D2H:
                return ResidencyState.D2H_COPYING
            if self.inflight_direction == TransferDirection.H2D:
                return ResidencyState.H2D_COPYING
            raise StateTransitionError("inflight transfer is missing its direction")
        gpu = Tier.GPU in self.replicas
        cpu = Tier.CPU in self.replicas
        if gpu and cpu:
            return ResidencyState.GPU_AND_CPU
        if gpu:
            return ResidencyState.GPU_ONLY
        if cpu:
            return ResidencyState.CPU_ONLY
        return ResidencyState.ABSENT


@dataclass(frozen=True, slots=True)
class BlockStateSnapshot:
    """Exact block state at one atomic lifecycle batch boundary."""

    location_version: int
    residency: ResidencyState
    replicas: tuple[ReplicaId, ...]
    inflight_transfer_id: str | None
    inflight_direction: TransferDirection | None
    reclaimed: bool

    def __post_init__(self) -> None:
        if type(self.location_version) is not int or self.location_version < 0:
            raise IdentityError("block location_version must be non-negative")
        if not isinstance(self.replicas, tuple):
            raise IdentityError("block replicas must be a tuple")
        if self.replicas != tuple(sorted(self.replicas)) or len(self.replicas) != len(
            set(self.replicas)
        ):
            raise IdentityError("block replicas must be sorted and unique")
        if len({replica.tier for replica in self.replicas}) != len(self.replicas):
            raise IdentityError("block snapshot has duplicate replica tiers")
        if self.reclaimed != (self.residency == ResidencyState.FREED):
            raise IdentityError("block reclaimed flag disagrees with residency")
        inflight = self.inflight_transfer_id is not None
        if inflight != (self.inflight_direction is not None):
            raise IdentityError("block inflight identity and direction must co-occur")
        if inflight:
            require_text("inflight_transfer_id", self.inflight_transfer_id or "")
        expected_copy = {
            TransferDirection.D2H: ResidencyState.D2H_COPYING,
            TransferDirection.H2D: ResidencyState.H2D_COPYING,
        }.get(self.inflight_direction)
        if expected_copy is not None and self.residency != expected_copy:
            raise IdentityError("block transfer direction disagrees with residency")
        if expected_copy is None and self.residency in {
            ResidencyState.D2H_COPYING,
            ResidencyState.H2D_COPYING,
        }:
            raise IdentityError("copying residency lacks an inflight transfer")
        tiers = {replica.tier for replica in self.replicas}
        if self.reclaimed and self.replicas:
            raise IdentityError("reclaimed block retains published replicas")
        if self.inflight_direction == TransferDirection.D2H and Tier.GPU not in tiers:
            raise IdentityError("D2H block snapshot lacks its GPU source")
        if self.inflight_direction == TransferDirection.H2D and Tier.CPU not in tiers:
            raise IdentityError("H2D block snapshot lacks its CPU source")
        if not inflight and not self.reclaimed:
            expected_residency = {
                frozenset(): ResidencyState.ABSENT,
                frozenset({Tier.GPU}): ResidencyState.GPU_ONLY,
                frozenset({Tier.CPU}): ResidencyState.CPU_ONLY,
                frozenset({Tier.GPU, Tier.CPU}): ResidencyState.GPU_AND_CPU,
            }[frozenset(tiers)]
            if self.residency != expected_residency:
                raise IdentityError("block replicas disagree with residency")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Immutable row in the DAGKV lifecycle ledger."""

    schema_version: str
    sequence: int
    event_id: str
    batch_id: str
    batch_index: int
    batch_size: int
    parent_event_id: str | None
    run_id: str
    phase: str
    source: str
    workflow: WorkflowKey | None
    request_id: str | None
    node_id: str | None
    operation_id: str
    action: LedgerAction
    status: LedgerStatus
    reason: str
    timestamp_ns: int
    source_tier: Tier | None = None
    target_tier: Tier | None = None
    block_key: BlockKey | None = None
    blocks: tuple[ReplicaId, ...] = ()
    binding_id: str | None = None
    binding_kind: BindingKind | None = None
    binding_state_before: BindingState | None = None
    binding_state_after: BindingState | None = None
    lease_id: str | None = None
    lease_deadline_ns: int | None = None
    transfer_id: str | None = None
    mapping_id: str | None = None
    execution_ref: ExecutionRef | None = None
    payload_size: int = 0
    byte_count: int = 0
    observed_byte_count: int = 0
    payload_digest: str | None = None
    observed_digest: str | None = None
    error: str | None = None
    waiter_binding_ids_after: tuple[str, ...] | None = None
    block_state_after: BlockStateSnapshot | None = None

    @property
    def block_count(self) -> int:
        """Return the exact number of physical block references."""

        return len(self.blocks)


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Immutable result of a runtime plus ledger conservation audit."""

    issues: tuple[str, ...]
    active_bindings: int
    active_leases: int
    content_mappings: int
    execution_mappings: int
    live_replicas: int
    reservations: int
    inflight_transfers: int

    @property
    def passed(self) -> bool:
        """Return whether the auditor found no issue."""

        return not self.issues
