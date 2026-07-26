"""Canonical C1-B lifecycle sidecar and independent evidence gate."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from dagkv.c1_trace import (
    AtomicCutoffView,
    CutoffPayload,
    DemandIntentPayload,
    EvidenceRole,
    H2DExecMapService,
    H2DFailedService,
    ObservationTerminalPayload,
    RequestCancelledService,
    ResidentExecMapService,
    ReuseEpochPayload,
    ServiceDisposition,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceValidationError,
    TraceValidationReceipt,
    ValidatedObservation,
    ValidatedTrace,
    WorkflowTopologyPayload,
    canonical_json,
    parse_canonical_dataclass,
    trace_stream_digest,
    validate_trace,
)
from dagkv.domain import (
    BindingKind,
    BindingState,
    BlockKey,
    BlockStateSnapshot,
    LedgerAction,
    LedgerStatus,
    LifecycleEvent,
    Tier,
    WorkflowKey,
    WorkflowSpec,
    require_sha256,
    require_text,
)
from dagkv.ledger import EventLedger

LIFECYCLE_SIDECAR_SCHEMA_VERSION = "dagkv.m3.lifecycle_sidecar.v1"
LIFECYCLE_CLOCK_DOMAIN = "campaign_monotonic_ns"
MAX_LIFECYCLE_BYTES = 64 * 1024 * 1024


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def lifecycle_stream_digest(events: tuple[LifecycleEvent, ...]) -> str:
    """Hash the exact canonical JSONL lifecycle stream represented by events."""

    if not isinstance(events, tuple):
        raise TraceValidationError("lifecycle events must be a tuple")
    hasher = sha256()
    for event in events:
        if type(event) is not LifecycleEvent:
            raise TraceValidationError("lifecycle stream contains a non-event value")
        hasher.update(canonical_json(event))
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class LifecycleClosure:
    """Durable terminal for one complete lifecycle event stream."""

    closed_through_ns: int
    event_count: int
    last_event_id: str | None
    last_event_timestamp_ns: int | None
    last_batch_id: str | None
    event_stream_digest: str

    def __post_init__(self) -> None:
        _require_int("lifecycle closed_through_ns", self.closed_through_ns)
        _require_int("lifecycle event_count", self.event_count)
        require_sha256("lifecycle event_stream_digest", self.event_stream_digest)
        if self.event_count == 0:
            if any(
                value is not None
                for value in (
                    self.last_event_id,
                    self.last_event_timestamp_ns,
                    self.last_batch_id,
                )
            ):
                raise TraceValidationError("empty lifecycle closure has a last event")
            return
        require_text("lifecycle last_event_id", self.last_event_id or "")
        require_text("lifecycle last_batch_id", self.last_batch_id or "")
        if self.last_event_timestamp_ns is None:
            raise TraceValidationError("lifecycle closure lacks its last timestamp")
        _require_int(
            "lifecycle last_event_timestamp_ns",
            self.last_event_timestamp_ns,
        )
        if self.closed_through_ns < self.last_event_timestamp_ns:
            raise TraceValidationError("lifecycle closure predates its final event")


def make_lifecycle_closure(
    events: tuple[LifecycleEvent, ...],
) -> LifecycleClosure:
    """Construct a closure from a sole-writer sealed lifecycle stream."""

    last = events[-1] if events else None
    if last is not None and last.batch_index != last.batch_size - 1:
        raise TraceValidationError("lifecycle closure truncates an atomic batch")
    if last is None or last.action != LedgerAction.STREAM_SEAL:
        raise TraceValidationError("lifecycle closure lacks a final stream seal")
    return LifecycleClosure(
        closed_through_ns=last.timestamp_ns,
        event_count=len(events),
        last_event_id=last.event_id if last is not None else None,
        last_event_timestamp_ns=last.timestamp_ns if last is not None else None,
        last_batch_id=last.batch_id if last is not None else None,
        event_stream_digest=lifecycle_stream_digest(events),
    )


@dataclass(frozen=True, slots=True)
class ClosedLifecycleArtifact:
    """Content-addressed lifecycle evidence for one trace pair."""

    schema_version: str
    artifact_id: str
    trace_pair_id: str
    run_id: str
    phase: str
    source: str
    clock_domain: str
    implementation_digest: str
    environment_digest: str
    events: tuple[LifecycleEvent, ...]
    closure: LifecycleClosure

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_SIDECAR_SCHEMA_VERSION:
            raise TraceValidationError("unsupported lifecycle sidecar schema")
        for name in ("artifact_id", "trace_pair_id", "run_id", "phase", "source"):
            require_text(name, getattr(self, name))
        if self.clock_domain != LIFECYCLE_CLOCK_DOMAIN:
            raise TraceValidationError("unsupported lifecycle clock domain")
        require_sha256("implementation_digest", self.implementation_digest)
        require_sha256("environment_digest", self.environment_digest)
        if not isinstance(self.events, tuple):
            raise TraceValidationError("lifecycle artifact events must be a tuple")
        if self.closure.event_count != len(self.events):
            raise TraceValidationError("lifecycle closure event count differs")
        if self.closure.event_stream_digest != lifecycle_stream_digest(self.events):
            raise TraceValidationError("lifecycle closure stream digest differs")
        last = self.events[-1] if self.events else None
        expected_last = (
            (
                last.event_id,
                last.timestamp_ns,
                last.batch_id,
            )
            if last is not None
            else (None, None, None)
        )
        observed_last = (
            self.closure.last_event_id,
            self.closure.last_event_timestamp_ns,
            self.closure.last_batch_id,
        )
        if observed_last != expected_last:
            raise TraceValidationError("lifecycle closure final event differs")
        if last is None or last.action != LedgerAction.STREAM_SEAL:
            raise TraceValidationError("lifecycle artifact lacks a final stream seal")
        if self.closure.closed_through_ns != last.timestamp_ns:
            raise TraceValidationError("lifecycle closure is not bound to its seal")
        issues = EventLedger.audit_detached(
            self.events,
            run_id=self.run_id,
            phase=self.phase,
            source=self.source,
            require_complete_state=True,
        )
        if issues:
            raise TraceValidationError(f"lifecycle replay failed: {issues[0]}")


@dataclass(frozen=True, slots=True)
class LoadedLifecycleArtifact:
    artifact: ClosedLifecycleArtifact
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256("loaded lifecycle digest", self.digest)
        _require_int("loaded lifecycle size", self.size_bytes, minimum=1)


def _open_parent(path: Path) -> int:
    if not path.is_absolute():
        raise TraceValidationError("lifecycle path must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise TraceValidationError("lifecycle parent must be a non-symlink directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(parent, flags)
        opened = os.fstat(descriptor)
        linked = parent.stat(follow_symlinks=False)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise TraceValidationError("cannot open lifecycle parent safely") from exc
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        os.close(descriptor)
        raise TraceValidationError("lifecycle parent changed while opening")
    return descriptor


def write_lifecycle_artifact(
    path: Path,
    artifact: ClosedLifecycleArtifact,
) -> str:
    """Create, fsync, and read back one canonical lifecycle artifact."""

    raw = canonical_json(artifact)
    if len(raw) > MAX_LIFECYCLE_BYTES:
        raise TraceValidationError("lifecycle artifact exceeds the size limit")
    parsed = parse_canonical_dataclass(
        raw,
        ClosedLifecycleArtifact,
        artifact_name="lifecycle artifact",
        max_bytes=MAX_LIFECYCLE_BYTES,
    )
    if parsed != artifact:
        raise TraceValidationError("lifecycle artifact changes during canonical replay")
    parent_descriptor = _open_parent(path)
    descriptor: int | None = None
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            path.name,
            flags,
            0o640,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise OSError("lifecycle write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or opened.st_size != len(raw)
        ):
            raise OSError("lifecycle output identity changed")
        observed = bytearray()
        read_offset = 0
        while read_offset < len(raw):
            chunk = os.pread(descriptor, len(raw) - read_offset, read_offset)
            if not chunk:
                raise OSError("lifecycle output ended during readback")
            observed.extend(chunk)
            read_offset += len(chunk)
        if bytes(observed) != raw:
            raise OSError("lifecycle output differs from staged bytes")
    except FileExistsError as exc:
        raise TraceValidationError("lifecycle artifact is create-only") from exc
    except OSError as exc:
        raise TraceCommitIndeterminateError(
            "lifecycle artifact durability is indeterminate"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return sha256(raw).hexdigest()


def _read_stable_file(path: Path) -> bytes:
    if not path.is_absolute():
        raise TraceValidationError("lifecycle artifact path must be absolute")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise TraceValidationError("lifecycle artifact is missing") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise TraceValidationError(
            "lifecycle artifact must be a singly linked regular non-symlink file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TraceValidationError("cannot open lifecycle artifact safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise TraceValidationError("lifecycle artifact changed while opening")
        raw = bytearray()
        while len(raw) <= MAX_LIFECYCLE_BYTES:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_LIFECYCLE_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise TraceValidationError("lifecycle artifact changed while reading")
        linked = path.lstat()
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
            or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise TraceValidationError("lifecycle artifact path changed while reading")
    except OSError as exc:
        raise TraceValidationError("cannot read lifecycle artifact safely") from exc
    finally:
        os.close(descriptor)
    if not raw or len(raw) > MAX_LIFECYCLE_BYTES:
        raise TraceValidationError("lifecycle artifact has an invalid size")
    return bytes(raw)


def load_lifecycle_artifact(path: Path) -> LoadedLifecycleArtifact:
    """Load one stable canonical lifecycle artifact and its file identity."""

    raw = _read_stable_file(path)
    artifact = parse_canonical_dataclass(
        raw,
        ClosedLifecycleArtifact,
        artifact_name="lifecycle artifact",
        max_bytes=MAX_LIFECYCLE_BYTES,
    )
    return LoadedLifecycleArtifact(
        artifact=artifact,
        digest=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


@dataclass(frozen=True, slots=True)
class _BindingProjection:
    open_event: LifecycleEvent
    state: BindingState


@dataclass(frozen=True, slots=True)
class _LifecycleProjection:
    blocks: dict[BlockKey, BlockStateSnapshot]
    bindings: dict[str, _BindingProjection]
    terminal_nodes: frozenset[tuple[object, str]]
    seen_exec_map_binding_ids: frozenset[str]
    seen_waiter_binding_ids: frozenset[str]


def _project_lifecycle(events: tuple[LifecycleEvent, ...]) -> _LifecycleProjection:
    blocks: dict[BlockKey, BlockStateSnapshot] = {}
    bindings: dict[str, _BindingProjection] = {}
    terminal_nodes: set[tuple[object, str]] = set()
    seen_exec_map_binding_ids: set[str] = set()
    seen_waiter_binding_ids: set[str] = set()
    for event in events:
        if event.action == LedgerAction.BLOCK_STATE:
            if event.block_key is None or event.block_state_after is None:
                raise TraceValidationError("block state event is incomplete")
            blocks[event.block_key] = event.block_state_after
        elif event.action == LedgerAction.BIND:
            if event.binding_id is None or event.binding_state_after is None:
                raise TraceValidationError("binding open event is incomplete")
            bindings[event.binding_id] = _BindingProjection(
                open_event=event,
                state=event.binding_state_after,
            )
        elif event.action == LedgerAction.BIND_STATE:
            if event.binding_id is None or event.binding_state_after is None:
                raise TraceValidationError("binding state event is incomplete")
            opened = bindings[event.binding_id].open_event
            bindings[event.binding_id] = _BindingProjection(
                open_event=opened,
                state=event.binding_state_after,
            )
        elif event.action == LedgerAction.RELEASE and event.binding_id is not None:
            bindings.pop(event.binding_id, None)
        elif event.action == LedgerAction.EXEC_MAP and event.binding_id is not None:
            seen_exec_map_binding_ids.add(event.binding_id)
        elif event.action == LedgerAction.WAITER_JOIN and event.binding_id is not None:
            seen_waiter_binding_ids.add(event.binding_id)
        elif (
            event.action == LedgerAction.NODE
            and event.status != LedgerStatus.SCHEDULED
            and event.workflow is not None
            and event.node_id is not None
        ):
            terminal_nodes.add((event.workflow, event.node_id))
    return _LifecycleProjection(
        blocks=blocks,
        bindings=bindings,
        terminal_nodes=frozenset(terminal_nodes),
        seen_exec_map_binding_ids=frozenset(seen_exec_map_binding_ids),
        seen_waiter_binding_ids=frozenset(seen_waiter_binding_ids),
    )


def _payload[T](record: object, expected: type[T]) -> T:
    payload = getattr(record, "payload", None)
    if type(payload) is not expected:
        raise TraceValidationError(f"expected {expected.__name__} payload")
    return payload


def _verify_prefix(
    events: tuple[LifecycleEvent, ...],
    *,
    event_count: int,
    last_event_id: str | None,
    last_timestamp_ns: int | None,
    prefix_name: str,
) -> tuple[LifecycleEvent, ...]:
    _require_int(f"{prefix_name} event count", event_count)
    if event_count > len(events):
        raise TraceValidationError(f"{prefix_name} exceeds lifecycle closure")
    prefix = events[:event_count]
    if not prefix:
        if last_event_id is not None or last_timestamp_ns is not None:
            raise TraceValidationError(f"empty {prefix_name} has a last event")
        return prefix
    last = prefix[-1]
    if last.batch_index != last.batch_size - 1:
        raise TraceValidationError(f"{prefix_name} truncates an atomic batch")
    if last.event_id != last_event_id or last.timestamp_ns != last_timestamp_ns:
        raise TraceValidationError(f"{prefix_name} watermark differs from lifecycle")
    return prefix


class CanonicalLifecycleEvidenceGate:
    """Issue a lifecycle receipt after replay and service reconstruction."""

    def __init__(
        self,
        path: Path,
        expected_artifact_digest: str,
        verifier_digest: str,
    ) -> None:
        if not path.is_absolute():
            raise TraceValidationError("lifecycle verifier path must be absolute")
        require_sha256("expected lifecycle artifact digest", expected_artifact_digest)
        require_sha256("lifecycle verifier digest", verifier_digest)
        self._path = path
        self._expected_artifact_digest = expected_artifact_digest
        self._verifier_digest = verifier_digest

    def verify_lifecycle(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        replayed = validate_trace(trace.records)
        loaded = load_lifecycle_artifact(self._path)
        artifact = loaded.artifact
        header = _payload(replayed.header, TraceHeaderPayload)
        if loaded.digest != self._expected_artifact_digest:
            raise TraceValidationError(
                "lifecycle artifact differs from the expected external digest"
            )
        if (
            artifact.trace_pair_id != header.trace_pair_id
            or artifact.run_id != replayed.header.run_id
        ):
            raise TraceValidationError("lifecycle artifact belongs to another trace")
        if (
            artifact.implementation_digest != header.implementation_digest
            or artifact.environment_digest != header.environment_digest
        ):
            raise TraceValidationError("lifecycle implementation binding differs")

        topologies = {
            _payload(record, WorkflowTopologyPayload).workflow_spec.key: _payload(
                record,
                WorkflowTopologyPayload,
            ).workflow_spec
            for record in replayed.topologies
        }
        event_by_id = {event.event_id: event for event in artifact.events}
        verified: list[str] = []
        for observation in replayed.observations:
            terminal = _payload(observation.terminal, ObservationTerminalPayload)
            if terminal.status != TerminalStatus.COMPLETE:
                continue
            self._verify_complete_observation(
                observation,
                artifact,
                topologies=topologies,
                event_by_id=event_by_id,
            )
            verified.append(observation.observation_id)
        return TraceValidationReceipt(
            role=EvidenceRole.LIFECYCLE,
            trace_pair_id=header.trace_pair_id,
            trace_digest=trace_stream_digest(replayed.records),
            artifact_digest=loaded.digest,
            verifier_digest=self._verifier_digest,
            verified_observation_ids=tuple(sorted(verified)),
        )

    def _verify_complete_observation(
        self,
        observation: ValidatedObservation,
        artifact: ClosedLifecycleArtifact,
        *,
        topologies: dict[WorkflowKey, WorkflowSpec],
        event_by_id: dict[str, LifecycleEvent],
    ) -> None:
        cutoff = _payload(observation.cutoff, CutoffPayload)
        terminal = _payload(observation.terminal, ObservationTerminalPayload)
        cutoff_prefix = _verify_prefix(
            artifact.events,
            event_count=cutoff.lifecycle_event_count,
            last_event_id=cutoff.last_event_id,
            last_timestamp_ns=cutoff.last_event_timestamp_ns,
            prefix_name="cutoff lifecycle prefix",
        )
        projection = _project_lifecycle(cutoff_prefix)
        owner_workflows = sorted({owner.workflow for owner in cutoff.snapshot.owners})
        try:
            owner_specs = tuple(topologies[workflow] for workflow in owner_workflows)
        except KeyError as exc:
            raise TraceValidationError("cutoff owner lacks workflow topology") from exc
        cutoff_view = AtomicCutoffView(
            snapshot=cutoff.snapshot,
            owner_specs=owner_specs,
            lifecycle_prefix=cutoff_prefix,
            cutoff_ns=cutoff.cutoff_ns,
            horizon_duration_ns=cutoff.horizon_duration_ns,
            deadline_ns=cutoff.deadline_ns,
        )
        if cutoff_view.view_digest != cutoff.atomic_cutoff_view_digest:
            raise TraceValidationError("cutoff digest differs from lifecycle view")
        block_state = projection.blocks.get(cutoff.snapshot.block_key)
        if block_state is None or (
            block_state.location_version != cutoff.snapshot.location_version
            or block_state.residency != cutoff.snapshot.residency
        ):
            raise TraceValidationError("cutoff block state differs from lifecycle")
        expected_owner_ids = {owner.binding_id for owner in cutoff.snapshot.owners}
        observed_owner_ids = {
            binding_id
            for binding_id, binding in projection.bindings.items()
            if binding.open_event.block_key == cutoff.snapshot.block_key
            and binding.open_event.binding_kind == BindingKind.WORKFLOW_RETENTION
            and binding.state == BindingState.RETAINED
        }
        if observed_owner_ids != expected_owner_ids:
            raise TraceValidationError("cutoff retention owners differ from lifecycle")
        for owner in cutoff.snapshot.owners:
            binding = projection.bindings[owner.binding_id].open_event
            if (
                binding.workflow != owner.workflow
                or binding.timestamp_ns != owner.created_ns
            ):
                raise TraceValidationError(
                    "cutoff owner identity differs from lifecycle"
                )
            spec = topologies.get(owner.workflow)
            if spec is None:
                raise TraceValidationError("cutoff owner lacks workflow topology")
            eligible = tuple(
                sorted(
                    node_id
                    for node_id in spec.node_ids
                    if (owner.workflow, node_id) not in projection.terminal_nodes
                )
            )
            if eligible != owner.eligible_node_ids:
                raise TraceValidationError(
                    "cutoff eligible nodes differ from lifecycle"
                )

        intent_projections: dict[str, _LifecycleProjection] = {}
        for intent_record in observation.intents:
            intent = _payload(intent_record, DemandIntentPayload)
            prefix = _verify_prefix(
                artifact.events,
                event_count=intent.pre_service_event_count,
                last_event_id=intent.pre_service_last_event_id,
                last_timestamp_ns=intent.pre_service_last_timestamp_ns,
                prefix_name="pre-service lifecycle prefix",
            )
            intent_projection = _project_lifecycle(prefix)
            self._verify_intent_bindings(
                intent,
                intent_projection,
                cutoff_owner_ids=expected_owner_ids,
            )
            intent_projections[intent_record.record_id] = intent_projection
        intent_by_id = {record.record_id: record for record in observation.intents}
        for epoch_record in observation.epochs:
            epoch = _payload(epoch_record, ReuseEpochPayload)
            for service in epoch.service_terminals:
                intent_record = intent_by_id[service.intent_record_id]
                intent = _payload(intent_record, DemandIntentPayload)
                if terminal.label_available_ns is None:
                    raise TraceValidationError(
                        "complete terminal lacks label availability"
                    )
                self._verify_service(
                    service,
                    intent,
                    event_by_id,
                    pre_service_projection=intent_projections[intent_record.record_id],
                    label_available_ns=terminal.label_available_ns,
                )

        _verify_prefix(
            artifact.events,
            event_count=terminal.last_verified_event_count,
            last_event_id=terminal.last_verified_event_id,
            last_timestamp_ns=terminal.last_verified_event_timestamp_ns,
            prefix_name="terminal lifecycle prefix",
        )
        if terminal.last_verified_event_count != artifact.closure.event_count:
            raise TraceValidationError("complete terminal does not close lifecycle EOF")
        if terminal.label_available_ns is None:
            raise TraceValidationError("complete terminal lacks label availability")
        if artifact.closure.closed_through_ns < max(
            cutoff.deadline_ns,
            terminal.label_available_ns,
        ):
            raise TraceValidationError(
                "lifecycle closure does not cover the label window"
            )

    @staticmethod
    def _verify_intent_bindings(
        intent: DemandIntentPayload,
        projection: _LifecycleProjection,
        *,
        cutoff_owner_ids: set[str],
    ) -> None:
        if intent.retention_binding_id not in cutoff_owner_ids:
            raise TraceValidationError("demand retention owner differs from cutoff")
        retention = projection.bindings.get(intent.retention_binding_id)
        if retention is None or (
            retention.state != BindingState.RETAINED
            or retention.open_event.binding_kind != BindingKind.WORKFLOW_RETENTION
            or retention.open_event.workflow != intent.workflow
            or retention.open_event.block_key != intent.block_key
        ):
            raise TraceValidationError("demand retention binding differs at service")
        request = projection.bindings.get(intent.request_binding_id)
        if request is None or (
            request.state != BindingState.RETAINED
            or request.open_event.binding_kind != BindingKind.REQUEST
            or request.open_event.workflow != intent.workflow
            or request.open_event.node_id != intent.node_id
            or request.open_event.execution_ref != intent.execution_ref
            or request.open_event.block_key != intent.block_key
        ):
            raise TraceValidationError("demand request binding differs at service")
        if intent.request_binding_id in (
            projection.seen_exec_map_binding_ids | projection.seen_waiter_binding_ids
        ):
            raise TraceValidationError(
                "demand request binding has prior service history"
            )

    @staticmethod
    def _verify_service(
        service: object,
        intent: DemandIntentPayload,
        event_by_id: dict[str, LifecycleEvent],
        *,
        pre_service_projection: _LifecycleProjection,
        label_available_ns: int,
    ) -> None:
        if isinstance(service, ResidentExecMapService):
            event = event_by_id.get(service.exec_map_event_id)
            CanonicalLifecycleEvidenceGate._verify_exec_map(event, intent)
            block_state = pre_service_projection.blocks.get(intent.block_key)
            if event is None or (
                event.sequence < intent.pre_service_event_count
                or event.timestamp_ns != intent.scheduled_access_ns
                or event.timestamp_ns > label_available_ns
                or block_state is None
                or not event.blocks
                or event.blocks[0] not in block_state.replicas
                or event.blocks[0].tier != Tier.GPU
            ):
                raise TraceValidationError("resident service predates demand intent")
            return
        if isinstance(service, RequestCancelledService):
            event = event_by_id.get(service.release_event_id)
            if event is None or (
                event.action != LedgerAction.RELEASE
                or event.binding_id != intent.request_binding_id
                or event.block_key != intent.block_key
                or event.workflow != intent.workflow
                or event.node_id != intent.node_id
                or event.sequence < intent.pre_service_event_count
                or event.timestamp_ns < intent.scheduled_access_ns
                or event.timestamp_ns > label_available_ns
            ):
                raise TraceValidationError("request cancellation event differs")
            prior_service = any(
                intent.pre_service_event_count <= candidate.sequence < event.sequence
                and (
                    (
                        candidate.action == LedgerAction.EXEC_MAP
                        and candidate.binding_id == intent.request_binding_id
                    )
                    or (
                        candidate.action in {LedgerAction.LOAD, LedgerAction.PREFETCH}
                        and candidate.status != LedgerStatus.SCHEDULED
                        and candidate.waiter_binding_ids_after is not None
                        and intent.request_binding_id
                        in candidate.waiter_binding_ids_after
                    )
                )
                for candidate in event_by_id.values()
            )
            if prior_service:
                raise TraceValidationError("request was serviced before cancellation")
            return
        if not isinstance(service, (H2DExecMapService, H2DFailedService)):
            raise TraceValidationError("unknown lifecycle service terminal")
        scheduled = event_by_id.get(service.transfer_scheduled_event_id)
        join = event_by_id.get(service.waiter_join_event_id)
        terminal = event_by_id.get(service.transfer_terminal_event_id)
        if scheduled is None or join is None or terminal is None:
            raise TraceValidationError("H2D service references an unknown transfer")
        if (
            scheduled.action not in {LedgerAction.LOAD, LedgerAction.PREFETCH}
            or scheduled.status != LedgerStatus.SCHEDULED
            or scheduled.transfer_id != service.transfer_id
            or scheduled.block_key != intent.block_key
        ):
            raise TraceValidationError("H2D scheduled event differs from demand")
        if (
            join.action != LedgerAction.WAITER_JOIN
            or join.transfer_id != service.transfer_id
            or join.parent_event_id != scheduled.event_id
            or join.binding_id != intent.request_binding_id
            or join.workflow != intent.workflow
            or join.node_id != intent.node_id
            or join.execution_ref != intent.execution_ref
            or join.block_key != intent.block_key
            or join.sequence < intent.pre_service_event_count
            or join.sequence <= scheduled.sequence
            or join.timestamp_ns != intent.scheduled_access_ns
        ):
            raise TraceValidationError("H2D waiter join differs from demand")
        expected_terminal_status = (
            {
                ServiceDisposition.H2D_FAILED: LedgerStatus.FAILED,
                ServiceDisposition.H2D_CANCELLED: LedgerStatus.CANCELLED,
            }[service.disposition]
            if isinstance(service, H2DFailedService)
            else LedgerStatus.COMPLETED
        )
        if (
            terminal.action != scheduled.action
            or terminal.status != expected_terminal_status
            or terminal.transfer_id != service.transfer_id
            or terminal.parent_event_id != scheduled.event_id
            or terminal.block_key != intent.block_key
            or terminal.sequence <= join.sequence
            or terminal.timestamp_ns < intent.scheduled_access_ns
            or terminal.timestamp_ns > label_available_ns
            or terminal.waiter_binding_ids_after != service.waiter_binding_ids
            or intent.request_binding_id not in service.waiter_binding_ids
        ):
            raise TraceValidationError("H2D terminal provenance differs")
        if isinstance(service, H2DExecMapService):
            event = event_by_id.get(service.exec_map_event_id)
            CanonicalLifecycleEvidenceGate._verify_exec_map(event, intent)
            if event is None or (
                event.sequence <= terminal.sequence
                or event.batch_id != terminal.batch_id
                or event.timestamp_ns != terminal.timestamp_ns
                or event.blocks != terminal.blocks
                or event.timestamp_ns > label_available_ns
            ):
                raise TraceValidationError("H2D execution map predates terminal")

    @staticmethod
    def _verify_exec_map(
        event: LifecycleEvent | None,
        intent: DemandIntentPayload,
    ) -> None:
        if event is None or (
            event.action != LedgerAction.EXEC_MAP
            or event.binding_id != intent.request_binding_id
            or event.workflow != intent.workflow
            or event.node_id != intent.node_id
            or event.execution_ref != intent.execution_ref
            or event.block_key != intent.block_key
        ):
            raise TraceValidationError("execution-map service differs from demand")
