"""Transactional append-only lifecycle event storage."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import RLock

from dagkv.domain import (
    BindingKind,
    BlockKey,
    DAGKVError,
    ExecutionRef,
    LedgerAction,
    LedgerStatus,
    LifecycleEvent,
    ReplicaId,
    StateTransitionError,
    Tier,
    WorkflowKey,
    require_optional_sha256,
    require_text,
)


@dataclass(frozen=True, slots=True)
class EventDraft:
    """Validated input row for one atomic ledger append batch."""

    action: LedgerAction
    status: LedgerStatus
    reason: str
    timestamp_ns: int
    operation_id: str
    local_id: str | None = None
    parent_event_id: str | None = None
    parent_local_id: str | None = None
    workflow: WorkflowKey | None = None
    request_id: str | None = None
    node_id: str | None = None
    source_tier: Tier | None = None
    target_tier: Tier | None = None
    block_key: BlockKey | None = None
    blocks: tuple[ReplicaId, ...] = ()
    binding_id: str | None = None
    binding_kind: BindingKind | None = None
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


@dataclass(slots=True)
class _ReferenceState:
    """Cross-family live references used for atomic validation and replay."""

    seen_execution_refs: set[ExecutionRef] = field(default_factory=set)
    seen_node_refs: set[tuple[WorkflowKey, str]] = field(default_factory=set)
    live_node_refs: set[tuple[WorkflowKey, str]] = field(default_factory=set)
    allocations: dict[ReplicaId, BlockKey] = field(default_factory=dict)
    block_tier_allocations: dict[tuple[BlockKey, Tier], ReplicaId] = field(
        default_factory=dict
    )
    slot_generations: dict[tuple[Tier, str, str], int] = field(default_factory=dict)
    slot_occupants: dict[tuple[Tier, str, str], ReplicaId] = field(default_factory=dict)
    block_payload_digests: dict[BlockKey, str] = field(default_factory=dict)
    block_payload_sizes: dict[BlockKey, int] = field(default_factory=dict)
    mapped_allocations: set[ReplicaId] = field(default_factory=set)
    binding_blocks: dict[str, BlockKey] = field(default_factory=dict)
    binding_nodes: dict[str, tuple[WorkflowKey, str]] = field(default_factory=dict)
    content_locations: dict[tuple[BlockKey, ReplicaId], str] = field(
        default_factory=dict
    )
    execution_locations: dict[
        str,
        tuple[ExecutionRef, BlockKey, ReplicaId, str],
    ] = field(default_factory=dict)
    execution_refs: dict[ExecutionRef, str] = field(default_factory=dict)
    lease_bindings: dict[str, str] = field(default_factory=dict)
    transfers: dict[str, tuple[BlockKey, ReplicaId, Tier]] = field(default_factory=dict)
    transfer_targets: dict[ReplicaId, LedgerStatus] = field(default_factory=dict)
    transfer_target_actions: dict[ReplicaId, LedgerAction] = field(default_factory=dict)
    h2d_required_allocations: set[ReplicaId] = field(default_factory=set)

    def clone(self) -> _ReferenceState:
        return _ReferenceState(
            seen_execution_refs=set(self.seen_execution_refs),
            seen_node_refs=set(self.seen_node_refs),
            live_node_refs=set(self.live_node_refs),
            allocations=dict(self.allocations),
            block_tier_allocations=dict(self.block_tier_allocations),
            slot_generations=dict(self.slot_generations),
            slot_occupants=dict(self.slot_occupants),
            block_payload_digests=dict(self.block_payload_digests),
            block_payload_sizes=dict(self.block_payload_sizes),
            mapped_allocations=set(self.mapped_allocations),
            binding_blocks=dict(self.binding_blocks),
            binding_nodes=dict(self.binding_nodes),
            content_locations=dict(self.content_locations),
            execution_locations=dict(self.execution_locations),
            execution_refs=dict(self.execution_refs),
            lease_bindings=dict(self.lease_bindings),
            transfers=dict(self.transfers),
            transfer_targets=dict(self.transfer_targets),
            transfer_target_actions=dict(self.transfer_target_actions),
            h2d_required_allocations=set(self.h2d_required_allocations),
        )


class EventLedger:
    """Create ordered immutable events with atomic multi-row commits."""

    SCHEMA_VERSION = "dagkv_lifecycle_event_v1"

    def __init__(
        self,
        *,
        run_id: str,
        phase: str,
        source: str,
        mutation_guard: Callable[[], None] | None = None,
    ) -> None:
        require_text("run_id", run_id)
        require_text("phase", phase)
        require_text("source", source)
        self.run_id = run_id
        self.phase = phase
        self.source = source
        self._mutation_guard = mutation_guard
        self._events: list[LifecycleEvent] = []
        self._events_by_id: dict[str, LifecycleEvent] = {}
        self._seen_identities = self._empty_identity_registry()
        self._live_identities = self._empty_identity_registry()
        self._references = _ReferenceState()
        self._lock = RLock()

    @property
    def events(self) -> tuple[LifecycleEvent, ...]:
        """Return an immutable snapshot of all rows."""

        with self._lock:
            return tuple(self._events)

    @property
    def last_timestamp_ns(self) -> int:
        """Return the last committed timestamp, or -1 for an empty ledger."""

        with self._lock:
            return self._events[-1].timestamp_ns if self._events else -1

    def append(self, draft: EventDraft) -> LifecycleEvent:
        """Append one row through the same validation path as a batch."""

        return self.append_batch((draft,))[0]

    def append_batch(self, drafts: Iterable[EventDraft]) -> tuple[LifecycleEvent, ...]:
        """Validate and append every row atomically.

        Local parent references make allocation-plus-map and similar compound
        transitions atomic without exposing predicted event IDs to callers.
        """

        batch = tuple(drafts)
        if self._mutation_guard is not None:
            self._mutation_guard()
        if not batch:
            return ()
        with self._lock:
            local_ids = [draft.local_id for draft in batch if draft.local_id]
            if len(local_ids) != len(set(local_ids)):
                raise StateTransitionError("duplicate local event ID in batch")

            pending: list[LifecycleEvent] = []
            local_events: dict[str, LifecycleEvent] = {}
            available = dict(self._events_by_id)
            seen_identities = {
                name: set(identities)
                for name, identities in self._seen_identities.items()
            }
            live_identities = {
                name: set(identities)
                for name, identities in self._live_identities.items()
            }
            references = self._references.clone()
            last_timestamp = self._events[-1].timestamp_ns if self._events else -1

            for offset, draft in enumerate(batch):
                parent_id = self._resolve_parent(draft, local_events)
                sequence = len(self._events) + offset
                event = LifecycleEvent(
                    schema_version=self.SCHEMA_VERSION,
                    sequence=sequence,
                    event_id=f"evt-{sequence:012d}",
                    parent_event_id=parent_id,
                    run_id=self.run_id,
                    phase=self.phase,
                    source=self.source,
                    workflow=draft.workflow,
                    request_id=draft.request_id,
                    node_id=draft.node_id,
                    operation_id=draft.operation_id,
                    action=draft.action,
                    status=draft.status,
                    reason=draft.reason,
                    timestamp_ns=draft.timestamp_ns,
                    source_tier=draft.source_tier,
                    target_tier=draft.target_tier,
                    block_key=draft.block_key,
                    blocks=draft.blocks,
                    binding_id=draft.binding_id,
                    binding_kind=draft.binding_kind,
                    lease_id=draft.lease_id,
                    lease_deadline_ns=draft.lease_deadline_ns,
                    transfer_id=draft.transfer_id,
                    mapping_id=draft.mapping_id,
                    execution_ref=draft.execution_ref,
                    payload_size=draft.payload_size,
                    byte_count=draft.byte_count,
                    observed_byte_count=draft.observed_byte_count,
                    payload_digest=draft.payload_digest,
                    observed_digest=draft.observed_digest,
                    error=draft.error,
                )
                parent = available.get(parent_id) if parent_id else None
                self._validate_event(event, parent, last_timestamp)
                self._apply_identity_transition(
                    event,
                    seen=seen_identities,
                    live=live_identities,
                )
                self._apply_reference_transition(
                    event,
                    live=live_identities,
                    references=references,
                )
                pending.append(event)
                available[event.event_id] = event
                if draft.local_id:
                    local_events[draft.local_id] = event
                last_timestamp = event.timestamp_ns

            self._events.extend(pending)
            self._events_by_id.update((event.event_id, event) for event in pending)
            self._seen_identities = seen_identities
            self._live_identities = live_identities
            self._references = references
            return tuple(pending)

    @staticmethod
    def _empty_identity_registry() -> dict[str, set[str | ReplicaId]]:
        return {
            "allocation": set(),
            "content mapping": set(),
            "binding": set(),
            "lease": set(),
            "transfer": set(),
            "execution mapping": set(),
            "node": set(),
        }

    @staticmethod
    def _identity_transition(
        event: LifecycleEvent,
    ) -> tuple[str, str | ReplicaId, bool] | None:
        replica = event.blocks[0] if event.blocks else None
        if event.action == LedgerAction.ALLOCATE and replica is not None:
            return ("allocation", replica, True)
        if event.action == LedgerAction.EVICT and replica is not None:
            return ("allocation", replica, False)
        if event.action == LedgerAction.MAP and event.mapping_id is not None:
            return ("content mapping", event.mapping_id, True)
        if event.action == LedgerAction.UNMAP and event.mapping_id is not None:
            return ("content mapping", event.mapping_id, False)
        if event.action == LedgerAction.BIND and event.binding_id is not None:
            return ("binding", event.binding_id, True)
        if event.action == LedgerAction.RELEASE and event.binding_id is not None:
            return ("binding", event.binding_id, False)
        if event.action == LedgerAction.LEASE and event.lease_id is not None:
            return (
                "lease",
                event.lease_id,
                event.status == LedgerStatus.SCHEDULED,
            )
        if (
            event.action
            in {LedgerAction.SAVE, LedgerAction.LOAD, LedgerAction.PREFETCH}
            and event.transfer_id is not None
        ):
            return (
                "transfer",
                event.transfer_id,
                event.status == LedgerStatus.SCHEDULED,
            )
        if event.action == LedgerAction.EXEC_MAP and event.mapping_id is not None:
            return ("execution mapping", event.mapping_id, True)
        if event.action == LedgerAction.EXEC_UNMAP and event.mapping_id is not None:
            return ("execution mapping", event.mapping_id, False)
        if event.action == LedgerAction.NODE:
            return (
                "node",
                event.operation_id,
                event.status == LedgerStatus.SCHEDULED,
            )
        return None

    @classmethod
    def _apply_identity_transition(
        cls,
        event: LifecycleEvent,
        *,
        seen: dict[str, set[str | ReplicaId]],
        live: dict[str, set[str | ReplicaId]],
    ) -> None:
        transition = cls._identity_transition(event)
        if transition is None:
            return
        category, identity, opens = transition
        if opens:
            if identity in seen[category]:
                raise StateTransitionError(
                    f"{category} identity already used: {identity}"
                )
            seen[category].add(identity)
            live[category].add(identity)
            return
        if identity not in live[category]:
            raise StateTransitionError(
                f"{category} terminal without live identity: {identity}"
            )
        live[category].remove(identity)

    @staticmethod
    def _apply_reference_transition(
        event: LifecycleEvent,
        *,
        live: dict[str, set[str | ReplicaId]],
        references: _ReferenceState,
    ) -> None:
        replica = event.blocks[0] if event.blocks else None
        block_key = event.block_key

        if event.action == LedgerAction.NODE:
            if event.workflow is None or event.node_id is None:
                raise StateTransitionError("node reference identity is incomplete")
            node_ref = (event.workflow, event.node_id)
            if event.status == LedgerStatus.SCHEDULED:
                if node_ref in references.seen_node_refs:
                    raise StateTransitionError("workflow node lifecycle already used")
                references.seen_node_refs.add(node_ref)
                references.live_node_refs.add(node_ref)
                return
            if node_ref not in references.live_node_refs:
                raise StateTransitionError("node terminal has no live node reference")
            if any(
                references.binding_nodes.get(binding_id) == node_ref
                for _, _, _, binding_id in references.execution_locations.values()
            ):
                raise StateTransitionError(
                    "node terminal still has a live execution mapping"
                )
            references.live_node_refs.remove(node_ref)
            return

        if event.action == LedgerAction.ALLOCATE:
            if block_key is None or replica is None or event.payload_digest is None:
                raise StateTransitionError("allocation identity is incomplete")
            canonical_digest = references.block_payload_digests.setdefault(
                block_key,
                event.payload_digest,
            )
            if canonical_digest != event.payload_digest:
                raise StateTransitionError("block payload digest changed")
            canonical_size = references.block_payload_sizes.setdefault(
                block_key,
                event.payload_size,
            )
            if canonical_size != event.payload_size:
                raise StateTransitionError("block payload size changed")
            block_tier = (block_key, replica.tier)
            if block_tier in references.block_tier_allocations:
                raise StateTransitionError("block tier already has a live allocation")
            slot = replica.physical_slot
            if slot in references.slot_occupants:
                raise StateTransitionError("physical slot is already occupied")
            expected_generation = references.slot_generations.get(slot, 0) + 1
            if replica.generation != expected_generation:
                raise StateTransitionError(
                    "allocation has a stale or skipped slot generation"
                )
            references.allocations[replica] = block_key
            references.block_tier_allocations[block_tier] = replica
            references.slot_generations[slot] = replica.generation
            references.slot_occupants[slot] = replica
            if replica.tier == Tier.GPU and any(
                mapped_block == block_key and mapped_replica.tier == Tier.CPU
                for mapped_block, mapped_replica in references.content_locations
            ):
                references.h2d_required_allocations.add(replica)
            return

        if event.action == LedgerAction.BIND:
            if block_key is None or event.binding_id is None:
                raise StateTransitionError("binding block identity is incomplete")
            if not any(
                mapped_block == block_key
                for mapped_block, _ in references.content_locations
            ):
                raise StateTransitionError("binding has no live content location")
            if event.workflow is None or event.node_id is None:
                raise StateTransitionError("binding node identity is incomplete")
            if (
                event.binding_kind == BindingKind.REQUEST
                and event.execution_ref is not None
            ):
                if event.execution_ref in references.seen_execution_refs:
                    raise StateTransitionError(
                        f"execution reference already used: {event.execution_ref}"
                    )
                references.seen_execution_refs.add(event.execution_ref)
            references.binding_blocks[event.binding_id] = block_key
            references.binding_nodes[event.binding_id] = (
                event.workflow,
                event.node_id,
            )
            return

        if event.action == LedgerAction.MAP:
            if block_key is None or replica is None or event.mapping_id is None:
                raise StateTransitionError("content map identity is incomplete")
            location = (block_key, replica)
            if replica not in live["allocation"]:
                raise StateTransitionError("content map allocation is not live")
            if references.allocations.get(replica) != block_key:
                raise StateTransitionError("content map allocation changed block")
            if event.payload_digest != references.block_payload_digests.get(block_key):
                raise StateTransitionError("content map payload digest changed")
            if event.payload_size != references.block_payload_sizes.get(block_key):
                raise StateTransitionError("content map payload size changed")
            transfer_status = references.transfer_targets.get(replica)
            if transfer_status in {LedgerStatus.FAILED, LedgerStatus.CANCELLED}:
                raise StateTransitionError(
                    "failed transfer target cannot publish content"
                )
            if replica in references.h2d_required_allocations and not (
                transfer_status == LedgerStatus.COMPLETED
                and references.transfer_target_actions.get(replica)
                in {LedgerAction.LOAD, LedgerAction.PREFETCH}
            ):
                raise StateTransitionError(
                    "CPU-only block must publish GPU through completed H2D"
                )
            if replica in references.mapped_allocations:
                raise StateTransitionError(
                    "allocation content was already published once"
                )
            if location in references.content_locations:
                raise StateTransitionError(
                    "physical content location is already mapped"
                )
            if any(
                mapped_block == block_key and mapped_replica.tier == replica.tier
                for mapped_block, mapped_replica in references.content_locations
            ):
                raise StateTransitionError(
                    "block tier already has a live content mapping"
                )
            if any(target == replica for _, target, _ in references.transfers.values()):
                raise StateTransitionError(
                    "transfer target cannot publish before its terminal"
                )
            references.content_locations[location] = event.mapping_id
            references.mapped_allocations.add(replica)
            return

        if event.action == LedgerAction.UNMAP:
            if block_key is None or replica is None or event.mapping_id is None:
                raise StateTransitionError("content unmap identity is incomplete")
            location = (block_key, replica)
            if references.content_locations.get(location) != event.mapping_id:
                raise StateTransitionError("content unmap has no live location")
            if any(
                (mapped_block, mapped_replica) == location
                for _, mapped_block, mapped_replica, _ in (
                    references.execution_locations.values()
                )
            ):
                raise StateTransitionError(
                    "content unmap still has a live execution mapping"
                )
            if any(
                transfer_block == block_key
                and (target == replica or source_tier == replica.tier)
                for transfer_block, target, source_tier in (
                    references.transfers.values()
                )
            ):
                raise StateTransitionError("content unmap races a live transfer")
            other_locations = any(
                mapped_block == block_key and mapped_replica != replica
                for mapped_block, mapped_replica in references.content_locations
            )
            if not other_locations and block_key in references.binding_blocks.values():
                raise StateTransitionError(
                    "last content location still has a live binding"
                )
            references.content_locations.pop(location)
            return

        if event.action == LedgerAction.EVICT and replica is not None:
            if any(
                mapped_replica == replica
                for _, mapped_replica in references.content_locations
            ):
                raise StateTransitionError("allocation eviction still has content map")
            if any(
                target == replica
                or (
                    block_key is not None
                    and transfer_block == block_key
                    and source_tier == replica.tier
                )
                for transfer_block, target, source_tier in (
                    references.transfers.values()
                )
            ):
                raise StateTransitionError("allocation eviction races a live transfer")
            if block_key is None or references.allocations.get(replica) != block_key:
                raise StateTransitionError("allocation eviction changed block")
            block_tier = (block_key, replica.tier)
            if references.block_tier_allocations.get(block_tier) != replica:
                raise StateTransitionError("block tier allocation index changed")
            if references.slot_occupants.get(replica.physical_slot) != replica:
                raise StateTransitionError("physical slot occupant changed")
            references.allocations.pop(replica)
            references.block_tier_allocations.pop(block_tier)
            references.slot_occupants.pop(replica.physical_slot)
            references.h2d_required_allocations.discard(replica)
            return

        if event.action in {
            LedgerAction.SAVE,
            LedgerAction.LOAD,
            LedgerAction.PREFETCH,
        }:
            if (
                block_key is None
                or replica is None
                or event.transfer_id is None
                or event.source_tier is None
            ):
                raise StateTransitionError("transfer reference identity is incomplete")
            transfer_ref = (block_key, replica, event.source_tier)
            if event.status == LedgerStatus.SCHEDULED:
                if replica not in live["allocation"]:
                    raise StateTransitionError("transfer target allocation is not live")
                if references.allocations.get(replica) != block_key:
                    raise StateTransitionError(
                        "transfer target allocation changed block"
                    )
                if (block_key, replica) in references.content_locations:
                    raise StateTransitionError("transfer target is already published")
                if event.payload_digest != references.block_payload_digests.get(
                    block_key
                ):
                    raise StateTransitionError("transfer payload digest changed")
                if event.byte_count != references.block_payload_sizes.get(block_key):
                    raise StateTransitionError("transfer payload size changed")
                if replica in references.transfer_targets:
                    raise StateTransitionError(
                        "allocation was already used as a transfer target"
                    )
                if any(
                    target == replica for _, target, _ in references.transfers.values()
                ):
                    raise StateTransitionError(
                        "transfer target already belongs to a live transfer"
                    )
                if any(
                    transfer_block == block_key
                    for transfer_block, _, _ in references.transfers.values()
                ):
                    raise StateTransitionError("block already has a live transfer")
                source_is_live = any(
                    mapped_block == block_key
                    and mapped_replica.tier == event.source_tier
                    for mapped_block, mapped_replica in (references.content_locations)
                )
                if not source_is_live:
                    raise StateTransitionError("transfer source content is not live")
                references.transfers[event.transfer_id] = transfer_ref
                references.transfer_targets[replica] = LedgerStatus.SCHEDULED
                references.transfer_target_actions[replica] = event.action
                return
            if references.transfers.get(event.transfer_id) != transfer_ref:
                raise StateTransitionError("transfer terminal has no live reference")
            if references.transfer_target_actions.get(replica) != event.action:
                raise StateTransitionError("transfer target action changed")
            if replica not in live["allocation"]:
                raise StateTransitionError("transfer target allocation was released")
            references.transfers.pop(event.transfer_id)
            references.transfer_targets[replica] = event.status
            return

        if event.action == LedgerAction.EXEC_MAP:
            if (
                block_key is None
                or replica is None
                or event.mapping_id is None
                or event.execution_ref is None
                or event.binding_id is None
            ):
                raise StateTransitionError("execution map identity is incomplete")
            if replica.tier != Tier.GPU:
                raise StateTransitionError("execution map requires a GPU replica")
            if event.binding_id not in live["binding"]:
                raise StateTransitionError("execution map binding is not live")
            node_ref = (event.workflow, event.node_id)
            if references.binding_nodes.get(event.binding_id) != node_ref:
                raise StateTransitionError("execution map binding changed node")
            if node_ref not in references.live_node_refs:
                raise StateTransitionError(
                    "execution map requires a live running node lifecycle"
                )
            if (block_key, replica) not in references.content_locations:
                raise StateTransitionError(
                    "execution map has no live physical content mapping"
                )
            if event.execution_ref in references.execution_refs:
                raise StateTransitionError(
                    "execution reference already has a live mapping"
                )
            references.execution_locations[event.mapping_id] = (
                event.execution_ref,
                block_key,
                replica,
                event.binding_id,
            )
            references.execution_refs[event.execution_ref] = event.mapping_id
            return

        if event.action == LedgerAction.EXEC_UNMAP:
            if (
                block_key is None
                or replica is None
                or event.mapping_id is None
                or event.execution_ref is None
                or event.binding_id is None
            ):
                raise StateTransitionError("execution unmap identity is incomplete")
            expected = (
                event.execution_ref,
                block_key,
                replica,
                event.binding_id,
            )
            if references.execution_locations.get(event.mapping_id) != expected:
                raise StateTransitionError("execution unmap has no live reference")
            if references.execution_refs.get(event.execution_ref) != event.mapping_id:
                raise StateTransitionError("execution reference index changed")
            references.execution_locations.pop(event.mapping_id)
            references.execution_refs.pop(event.execution_ref)
            return

        if event.action == LedgerAction.LEASE and event.lease_id is not None:
            if event.binding_id is None:
                raise StateTransitionError("lease binding identity is incomplete")
            if event.status == LedgerStatus.SCHEDULED:
                if event.binding_id not in live["binding"]:
                    raise StateTransitionError("lease binding is not live")
                references.lease_bindings[event.lease_id] = event.binding_id
                return
            if references.lease_bindings.get(event.lease_id) != event.binding_id:
                raise StateTransitionError("lease terminal has no live binding")
            references.lease_bindings.pop(event.lease_id)
            return

        if event.action == LedgerAction.RELEASE and event.binding_id is not None:
            if event.binding_id in references.lease_bindings.values():
                raise StateTransitionError("binding release still has a live lease")
            if any(
                binding_id == event.binding_id
                for _, _, _, binding_id in references.execution_locations.values()
            ):
                raise StateTransitionError(
                    "binding release still has a live execution mapping"
                )
            if references.binding_blocks.get(event.binding_id) != block_key:
                raise StateTransitionError("binding release changed block")
            references.binding_blocks.pop(event.binding_id)
            references.binding_nodes.pop(event.binding_id)

    def _resolve_parent(
        self,
        draft: EventDraft,
        local_events: dict[str, LifecycleEvent],
    ) -> str | None:
        if draft.parent_event_id and draft.parent_local_id:
            raise StateTransitionError("event cannot have two parent references")
        if draft.parent_local_id:
            try:
                return local_events[draft.parent_local_id].event_id
            except KeyError as exc:
                raise StateTransitionError(
                    f"unknown or forward local parent: {draft.parent_local_id}"
                ) from exc
        if draft.parent_event_id and draft.parent_event_id not in self._events_by_id:
            raise StateTransitionError(f"unknown parent event: {draft.parent_event_id}")
        return draft.parent_event_id

    def _validate_event(
        self,
        event: LifecycleEvent,
        parent: LifecycleEvent | None,
        last_timestamp: int,
    ) -> None:
        require_text("event reason", event.reason)
        require_text("operation_id", event.operation_id)
        require_optional_sha256("payload_digest", event.payload_digest)
        require_optional_sha256("observed_digest", event.observed_digest)
        if event.timestamp_ns < 0 or event.timestamp_ns < last_timestamp:
            raise StateTransitionError("ledger timestamps must be non-decreasing")
        if (
            event.payload_size < 0
            or event.byte_count < 0
            or event.observed_byte_count < 0
        ):
            raise StateTransitionError("event bytes must be non-negative")
        self._validate_shape(event)
        self._validate_parent(event, parent)

    def _validate_shape(self, event: LifecycleEvent) -> None:
        completed_only = {
            LedgerAction.ALLOCATE,
            LedgerAction.MAP,
            LedgerAction.UNMAP,
            LedgerAction.BIND,
            LedgerAction.RELEASE,
            LedgerAction.EVICT,
            LedgerAction.EXEC_MAP,
            LedgerAction.EXEC_UNMAP,
        }
        transfers = {
            LedgerAction.SAVE,
            LedgerAction.LOAD,
            LedgerAction.PREFETCH,
        }
        if event.action in completed_only and event.status != LedgerStatus.COMPLETED:
            raise StateTransitionError(f"invalid status for {event.action.value}")
        if event.action == LedgerAction.LEASE and event.status not in {
            LedgerStatus.SCHEDULED,
            LedgerStatus.COMPLETED,
            LedgerStatus.FAILED,
            LedgerStatus.CANCELLED,
        }:
            raise StateTransitionError("invalid lease status")
        if event.action in transfers and event.status not in {
            LedgerStatus.SCHEDULED,
            LedgerStatus.COMPLETED,
            LedgerStatus.FAILED,
            LedgerStatus.CANCELLED,
        }:
            raise StateTransitionError("invalid transfer status")
        if event.action == LedgerAction.NODE and event.status not in {
            LedgerStatus.SCHEDULED,
            LedgerStatus.COMPLETED,
            LedgerStatus.FAILED,
            LedgerStatus.CANCELLED,
        }:
            raise StateTransitionError("invalid node status")

        kv_actions = set(LedgerAction) - {LedgerAction.NODE}
        if event.action in kv_actions and event.block_key is None:
            raise StateTransitionError("KV event requires a block identity")
        physical_actions = {
            LedgerAction.ALLOCATE,
            LedgerAction.MAP,
            LedgerAction.UNMAP,
            LedgerAction.EVICT,
            LedgerAction.SAVE,
            LedgerAction.LOAD,
            LedgerAction.PREFETCH,
            LedgerAction.EXEC_MAP,
            LedgerAction.EXEC_UNMAP,
        }
        expected_blocks = 1 if event.action in physical_actions else 0
        if event.block_count != expected_blocks:
            raise StateTransitionError(
                f"{event.action.value} requires {expected_blocks} physical block refs"
            )
        if event.action in {LedgerAction.MAP, LedgerAction.UNMAP}:
            require_text("mapping_id", event.mapping_id or "")
        payload_metadata_actions = {
            LedgerAction.ALLOCATE,
            LedgerAction.MAP,
            LedgerAction.UNMAP,
            LedgerAction.EVICT,
        }
        if event.action in payload_metadata_actions:
            if event.payload_size <= 0:
                raise StateTransitionError(
                    f"{event.action.value} requires a positive payload size"
                )
        elif event.payload_size != 0:
            raise StateTransitionError(
                "non-content event carries physical payload size"
            )
        if event.action in {LedgerAction.EXEC_MAP, LedgerAction.EXEC_UNMAP}:
            require_text("execution mapping_id", event.mapping_id or "")
            if event.execution_ref is None:
                raise StateTransitionError("execution map requires execution_ref")
        if event.execution_ref is not None and (
            event.workflow != event.execution_ref.workflow
            or event.request_id != event.execution_ref.request_id
        ):
            raise StateTransitionError(
                "event identity disagrees with its execution reference"
            )
        if event.action in {LedgerAction.BIND, LedgerAction.RELEASE}:
            require_text("binding_id", event.binding_id or "")
            if event.binding_kind is None:
                raise StateTransitionError("binding event requires binding_kind")
            if event.workflow is None:
                raise StateTransitionError("binding event requires workflow identity")
            require_text("request_id", event.request_id or "")
            require_text("node_id", event.node_id or "")
            if (
                event.binding_kind == BindingKind.REQUEST
                and event.execution_ref is None
            ):
                raise StateTransitionError(
                    "request binding requires an execution reference"
                )
            if (
                event.binding_kind == BindingKind.WORKFLOW_RETENTION
                and event.execution_ref is not None
            ):
                raise StateTransitionError(
                    "retention binding cannot carry an execution reference"
                )
        if event.action in {LedgerAction.EXEC_MAP, LedgerAction.EXEC_UNMAP}:
            require_text("execution binding_id", event.binding_id or "")
            if event.binding_kind != BindingKind.REQUEST:
                raise StateTransitionError("execution map requires a request binding")
            if event.workflow is None:
                raise StateTransitionError("execution map requires workflow identity")
            require_text("request_id", event.request_id or "")
            require_text("node_id", event.node_id or "")
        if event.action == LedgerAction.LEASE:
            require_text("lease_id", event.lease_id or "")
            require_text("binding_id", event.binding_id or "")
            if event.binding_kind != BindingKind.WORKFLOW_RETENTION:
                raise StateTransitionError("lease requires a retention binding")
            if event.workflow is None:
                raise StateTransitionError("lease requires workflow identity")
            require_text("request_id", event.request_id or "")
            require_text("node_id", event.node_id or "")
            if event.lease_deadline_ns is None:
                raise StateTransitionError("lease requires a deadline")
            if event.lease_deadline_ns < 0:
                raise StateTransitionError("lease deadline must be non-negative")
            if (
                event.status == LedgerStatus.SCHEDULED
                and event.lease_deadline_ns < event.timestamp_ns
            ):
                raise StateTransitionError("lease deadline predates registration")
            if (
                event.status == LedgerStatus.COMPLETED
                and event.timestamp_ns < event.lease_deadline_ns
            ):
                raise StateTransitionError("lease expired before its deadline")
        elif event.lease_deadline_ns is not None:
            raise StateTransitionError("non-lease event carries a lease deadline")
        if event.action in transfers:
            require_text("transfer_id", event.transfer_id or "")
            if event.source_tier is None or event.target_tier is None:
                raise StateTransitionError("transfer requires both tiers")
            expected_tiers = {
                LedgerAction.SAVE: (Tier.GPU, Tier.CPU),
                LedgerAction.LOAD: (Tier.CPU, Tier.GPU),
                LedgerAction.PREFETCH: (Tier.CPU, Tier.GPU),
            }[event.action]
            if (event.source_tier, event.target_tier) != expected_tiers:
                raise StateTransitionError(
                    "transfer action and tier direction disagree"
                )
            if event.blocks[0].tier != event.target_tier:
                raise StateTransitionError(
                    "transfer target replica tier disagrees with event"
                )
            if event.byte_count <= 0 or event.payload_digest is None:
                raise StateTransitionError("transfer requires bytes and payload digest")
            if event.status == LedgerStatus.SCHEDULED and (
                event.observed_byte_count != 0 or event.observed_digest is not None
            ):
                raise StateTransitionError(
                    "scheduled transfer has terminal observations"
                )
            if event.status == LedgerStatus.COMPLETED and (
                event.observed_byte_count != event.byte_count
                or event.observed_digest != event.payload_digest
            ):
                raise StateTransitionError("completed transfer payload is inconsistent")
        if event.action == LedgerAction.ALLOCATE:
            if event.byte_count <= 0:
                raise StateTransitionError("allocation capacity must be positive")
            if event.payload_size > event.byte_count:
                raise StateTransitionError("allocation payload exceeds capacity")
            if event.payload_digest is None:
                raise StateTransitionError("allocation requires payload digest")
        if event.action == LedgerAction.NODE:
            require_text("node_id", event.node_id or "")
            if event.workflow is None:
                raise StateTransitionError("node event requires workflow identity")

    def _validate_parent(
        self,
        event: LifecycleEvent,
        parent: LifecycleEvent | None,
    ) -> None:
        terminal = {
            LedgerStatus.COMPLETED,
            LedgerStatus.FAILED,
            LedgerStatus.CANCELLED,
        }
        expected: tuple[LedgerAction, LedgerStatus] | None
        if event.action in {LedgerAction.ALLOCATE, LedgerAction.BIND}:
            expected = None
        elif event.action == LedgerAction.NODE:
            expected = (
                None
                if event.status == LedgerStatus.SCHEDULED
                else (LedgerAction.NODE, LedgerStatus.SCHEDULED)
            )
        elif event.action in {LedgerAction.MAP, LedgerAction.EVICT}:
            expected = (LedgerAction.ALLOCATE, LedgerStatus.COMPLETED)
        elif event.action == LedgerAction.UNMAP:
            expected = (LedgerAction.MAP, LedgerStatus.COMPLETED)
        elif event.action in {LedgerAction.RELEASE, LedgerAction.EXEC_MAP}:
            expected = (LedgerAction.BIND, LedgerStatus.COMPLETED)
        elif event.action == LedgerAction.EXEC_UNMAP:
            expected = (LedgerAction.EXEC_MAP, LedgerStatus.COMPLETED)
        elif event.action == LedgerAction.LEASE:
            expected = (
                (LedgerAction.BIND, LedgerStatus.COMPLETED)
                if event.status == LedgerStatus.SCHEDULED
                else (LedgerAction.LEASE, LedgerStatus.SCHEDULED)
            )
        elif event.action in {
            LedgerAction.SAVE,
            LedgerAction.LOAD,
            LedgerAction.PREFETCH,
        }:
            expected = (
                (LedgerAction.ALLOCATE, LedgerStatus.COMPLETED)
                if event.status == LedgerStatus.SCHEDULED
                else (event.action, LedgerStatus.SCHEDULED)
            )
        else:
            raise StateTransitionError(f"unsupported action: {event.action.value}")

        if expected is None:
            if parent is not None:
                raise StateTransitionError("open event cannot have a parent")
            return
        if parent is None or (parent.action, parent.status) != expected:
            raise StateTransitionError(
                f"invalid parent for {event.action.value}/{event.status.value}"
            )
        if event.timestamp_ns < parent.timestamp_ns:
            raise StateTransitionError("event predates its parent")

        paired_actions = {
            LedgerAction.EVICT,
            LedgerAction.UNMAP,
            LedgerAction.RELEASE,
            LedgerAction.EXEC_UNMAP,
        }
        paired_terminal = (
            event.action in paired_actions
            or (
                event.action in {LedgerAction.LEASE, LedgerAction.NODE}
                and event.status in terminal
            )
            or (
                event.action
                in {LedgerAction.SAVE, LedgerAction.LOAD, LedgerAction.PREFETCH}
                and event.status in terminal
            )
        )
        if paired_terminal and event.operation_id != parent.operation_id:
            raise StateTransitionError("terminal operation identity changed")
        if event.block_key is not None and parent.block_key != event.block_key:
            raise StateTransitionError("parent block identity changed")
        same_resource_actions = {
            LedgerAction.MAP,
            LedgerAction.UNMAP,
            LedgerAction.EVICT,
            LedgerAction.EXEC_UNMAP,
        }
        if event.action in {
            LedgerAction.SAVE,
            LedgerAction.LOAD,
            LedgerAction.PREFETCH,
        }:
            same_resource_actions.add(event.action)
        if event.action in same_resource_actions and parent.blocks != event.blocks:
            raise StateTransitionError("parent physical resource changed")
        if event.binding_id and parent.binding_id != event.binding_id:
            raise StateTransitionError("parent binding identity changed")
        if (
            event.lease_id
            and event.status != LedgerStatus.SCHEDULED
            and parent.lease_id != event.lease_id
        ):
            raise StateTransitionError("parent lease identity changed")
        if (
            event.action == LedgerAction.LEASE
            and event.status != LedgerStatus.SCHEDULED
            and parent.lease_deadline_ns != event.lease_deadline_ns
        ):
            raise StateTransitionError("parent lease deadline changed")
        if (
            event.transfer_id
            and event.status != LedgerStatus.SCHEDULED
            and parent.transfer_id != event.transfer_id
        ):
            raise StateTransitionError("parent transfer identity changed")
        if (
            event.mapping_id
            and event.action in {LedgerAction.UNMAP, LedgerAction.EXEC_UNMAP}
            and parent.mapping_id != event.mapping_id
        ):
            raise StateTransitionError("parent mapping identity changed")
        if event.workflow is not None and parent.workflow != event.workflow:
            raise StateTransitionError("parent workflow identity changed")
        binding_lineage_actions = {
            LedgerAction.RELEASE,
            LedgerAction.EXEC_MAP,
            LedgerAction.EXEC_UNMAP,
            LedgerAction.LEASE,
        }
        if event.action in binding_lineage_actions and (
            event.workflow != parent.workflow
            or event.request_id != parent.request_id
            or event.node_id != parent.node_id
            or event.binding_kind != parent.binding_kind
            or event.execution_ref != parent.execution_ref
        ):
            raise StateTransitionError("parent binding lineage changed")
        if (
            event.action == LedgerAction.NODE
            and event.status in terminal
            and event.node_id != parent.node_id
        ):
            raise StateTransitionError("parent node identity changed")
        if (
            event.action
            in {LedgerAction.SAVE, LedgerAction.LOAD, LedgerAction.PREFETCH}
            and event.status != LedgerStatus.SCHEDULED
            and (
                event.source_tier != parent.source_tier
                or event.target_tier != parent.target_tier
                or event.byte_count != parent.byte_count
                or event.payload_digest != parent.payload_digest
            )
        ):
            raise StateTransitionError("transfer terminal geometry changed")
        if event.action == LedgerAction.EVICT and (
            event.byte_count != parent.byte_count
            or event.payload_size != parent.payload_size
            or event.payload_digest != parent.payload_digest
        ):
            raise StateTransitionError("eviction allocation metadata changed")
        if event.action == LedgerAction.MAP and (
            event.payload_size != parent.payload_size
            or event.payload_digest != parent.payload_digest
        ):
            raise StateTransitionError("content map payload identity changed")
        if event.action == LedgerAction.UNMAP and (
            event.payload_size != parent.payload_size
            or event.payload_digest != parent.payload_digest
        ):
            raise StateTransitionError("unmap payload identity changed")

    def audit(self, *, require_quiescent: bool = False) -> tuple[str, ...]:
        """Replay independent lifecycle conservation checks."""

        with self._lock:
            events = tuple(self._events)
        issues: list[str] = []
        allocations: dict[ReplicaId, LifecycleEvent] = {}
        mappings: dict[str, LifecycleEvent] = {}
        bindings: dict[str, LifecycleEvent] = {}
        leases: dict[str, LifecycleEvent] = {}
        transfers: dict[str, LifecycleEvent] = {}
        exec_maps: dict[str, LifecycleEvent] = {}
        nodes: dict[str, LifecycleEvent] = {}
        seen_allocations: set[ReplicaId] = set()
        seen_mappings: set[str] = set()
        seen_bindings: set[str] = set()
        seen_leases: set[str] = set()
        seen_transfers: set[str] = set()
        seen_exec_maps: set[str] = set()
        seen_nodes: set[str] = set()
        observed_event_ids: set[str] = set()
        replay_events: dict[str, LifecycleEvent] = {}
        replay_seen_identities = self._empty_identity_registry()
        replay_live_identities = self._empty_identity_registry()
        replay_references = _ReferenceState()
        last_timestamp = -1

        for expected_sequence, event in enumerate(events):
            envelope_valid = True
            if event.sequence != expected_sequence:
                issues.append(f"ledger sequence gap at {event.event_id}")
                envelope_valid = False
            expected_event_id = f"evt-{expected_sequence:012d}"
            if event.event_id != expected_event_id:
                issues.append(
                    f"event ID mismatch at sequence {expected_sequence}: "
                    f"{event.event_id} != {expected_event_id}"
                )
                envelope_valid = False
            if event.event_id in observed_event_ids:
                issues.append(f"duplicate ledger event ID: {event.event_id}")
                envelope_valid = False
            observed_event_ids.add(event.event_id)
            expected_envelope = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "phase": self.phase,
                "source": self.source,
            }
            for field_name, expected_value in expected_envelope.items():
                observed_value = getattr(event, field_name)
                if observed_value != expected_value:
                    issues.append(
                        f"ledger {field_name} changed at {event.event_id}: "
                        f"{observed_value} != {expected_value}"
                    )
                    envelope_valid = False
            parent = (
                replay_events.get(event.parent_event_id)
                if event.parent_event_id is not None
                else None
            )
            if event.parent_event_id is not None and parent is None:
                issues.append(
                    f"unknown parent event at {event.event_id}: {event.parent_event_id}"
                )
                envelope_valid = False
            structurally_valid = envelope_valid
            if structurally_valid:
                try:
                    self._validate_event(event, parent, last_timestamp)
                except DAGKVError as exc:
                    issues.append(f"invalid event {event.event_id}: {exc}")
                    structurally_valid = False
            accepted = False
            if structurally_valid:
                candidate_seen = {
                    name: set(identities)
                    for name, identities in replay_seen_identities.items()
                }
                candidate_live = {
                    name: set(identities)
                    for name, identities in replay_live_identities.items()
                }
                candidate_references = replay_references.clone()
                try:
                    self._apply_identity_transition(
                        event,
                        seen=candidate_seen,
                        live=candidate_live,
                    )
                    self._apply_reference_transition(
                        event,
                        live=candidate_live,
                        references=candidate_references,
                    )
                except DAGKVError as exc:
                    issues.append(
                        f"invalid lifecycle references at {event.event_id}: {exc}"
                    )
                else:
                    replay_seen_identities = candidate_seen
                    replay_live_identities = candidate_live
                    replay_references = candidate_references
                    accepted = True
            if not accepted:
                continue
            replay_events[event.event_id] = event
            last_timestamp = event.timestamp_ns
            replica = event.blocks[0] if event.blocks else None
            if event.action == LedgerAction.ALLOCATE and replica is not None:
                if replica in seen_allocations:
                    issues.append(f"reused allocation identity: {replica}")
                seen_allocations.add(replica)
                if replica in allocations:
                    issues.append(f"duplicate allocation open: {replica}")
                allocations[replica] = event
            elif event.action == LedgerAction.EVICT and replica is not None:
                if allocations.pop(replica, None) is None:
                    issues.append(f"evict without allocation: {replica}")
            elif event.action == LedgerAction.MAP and event.mapping_id:
                if event.mapping_id in seen_mappings:
                    issues.append(
                        f"reused content mapping identity: {event.mapping_id}"
                    )
                seen_mappings.add(event.mapping_id)
                if event.mapping_id in mappings:
                    issues.append(f"duplicate content map: {event.mapping_id}")
                mappings[event.mapping_id] = event
            elif event.action == LedgerAction.UNMAP and event.mapping_id:
                if mappings.pop(event.mapping_id, None) is None:
                    issues.append(f"unmap without content map: {event.mapping_id}")
            elif event.action == LedgerAction.BIND and event.binding_id:
                if event.binding_id in seen_bindings:
                    issues.append(f"reused binding identity: {event.binding_id}")
                seen_bindings.add(event.binding_id)
                if event.binding_id in bindings:
                    issues.append(f"duplicate binding: {event.binding_id}")
                bindings[event.binding_id] = event
            elif event.action == LedgerAction.RELEASE and event.binding_id:
                if bindings.pop(event.binding_id, None) is None:
                    issues.append(f"release without binding: {event.binding_id}")
            elif event.action == LedgerAction.LEASE and event.lease_id:
                if event.status == LedgerStatus.SCHEDULED:
                    if event.lease_id in seen_leases:
                        issues.append(f"reused lease identity: {event.lease_id}")
                    seen_leases.add(event.lease_id)
                    if event.lease_id in leases:
                        issues.append(f"duplicate lease: {event.lease_id}")
                    leases[event.lease_id] = event
                elif leases.pop(event.lease_id, None) is None:
                    issues.append(f"lease terminal without open: {event.lease_id}")
            elif (
                event.action
                in {
                    LedgerAction.SAVE,
                    LedgerAction.LOAD,
                    LedgerAction.PREFETCH,
                }
                and event.transfer_id
            ):
                if event.status == LedgerStatus.SCHEDULED:
                    if event.transfer_id in seen_transfers:
                        issues.append(f"reused transfer identity: {event.transfer_id}")
                    seen_transfers.add(event.transfer_id)
                    if event.transfer_id in transfers:
                        issues.append(f"duplicate transfer: {event.transfer_id}")
                    transfers[event.transfer_id] = event
                else:
                    if transfers.pop(event.transfer_id, None) is None:
                        issues.append(
                            f"transfer terminal without open: {event.transfer_id}"
                        )
                    if event.observed_byte_count > event.byte_count:
                        issues.append(f"transfer byte overrun: {event.transfer_id}")
                    if (
                        event.status == LedgerStatus.COMPLETED
                        and event.observed_byte_count != event.byte_count
                    ):
                        issues.append(f"short completed transfer: {event.transfer_id}")
            elif event.action == LedgerAction.EXEC_MAP and event.mapping_id:
                if event.mapping_id in seen_exec_maps:
                    issues.append(
                        f"reused execution mapping identity: {event.mapping_id}"
                    )
                seen_exec_maps.add(event.mapping_id)
                if event.mapping_id in exec_maps:
                    issues.append(f"duplicate execution map: {event.mapping_id}")
                exec_maps[event.mapping_id] = event
            elif event.action == LedgerAction.EXEC_UNMAP and event.mapping_id:
                if exec_maps.pop(event.mapping_id, None) is None:
                    issues.append(f"exec unmap without map: {event.mapping_id}")
            elif event.action == LedgerAction.NODE:
                key = event.operation_id
                if event.status == LedgerStatus.SCHEDULED:
                    if key in seen_nodes:
                        issues.append(f"reused node identity: {key}")
                    seen_nodes.add(key)
                    if key in nodes:
                        issues.append(f"duplicate node start: {key}")
                    nodes[key] = event
                elif nodes.pop(key, None) is None:
                    issues.append(f"node terminal without start: {key}")

        if require_quiescent:
            live = {
                "allocations": len(allocations),
                "content mappings": len(mappings),
                "bindings": len(bindings),
                "leases": len(leases),
                "transfers": len(transfers),
                "execution mappings": len(exec_maps),
                "running nodes": len(nodes),
            }
            issues.extend(
                f"live {name}: {count}" for name, count in live.items() if count
            )
        return tuple(dict.fromkeys(issues))

    def live_counts(self) -> dict[str, int]:
        """Return ledger-side live resource counts for runtime reconciliation."""

        counts = {
            "allocations": 0,
            "content_mappings": 0,
            "bindings": 0,
            "leases": 0,
            "transfers": 0,
            "execution_mappings": 0,
        }
        with self._lock:
            events = tuple(self._events)
        for event in events:
            if event.action == LedgerAction.ALLOCATE:
                counts["allocations"] += 1
            elif event.action == LedgerAction.EVICT:
                counts["allocations"] -= 1
            elif event.action == LedgerAction.MAP:
                counts["content_mappings"] += 1
            elif event.action == LedgerAction.UNMAP:
                counts["content_mappings"] -= 1
            elif event.action == LedgerAction.BIND:
                counts["bindings"] += 1
            elif event.action == LedgerAction.RELEASE:
                counts["bindings"] -= 1
            elif event.action == LedgerAction.LEASE:
                counts["leases"] += 1 if event.status == LedgerStatus.SCHEDULED else -1
            elif event.action in {
                LedgerAction.SAVE,
                LedgerAction.LOAD,
                LedgerAction.PREFETCH,
            }:
                counts["transfers"] += (
                    1 if event.status == LedgerStatus.SCHEDULED else -1
                )
            elif event.action == LedgerAction.EXEC_MAP:
                counts["execution_mappings"] += 1
            elif event.action == LedgerAction.EXEC_UNMAP:
                counts["execution_mappings"] -= 1
        return counts
