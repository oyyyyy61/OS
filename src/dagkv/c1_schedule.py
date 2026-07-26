"""Closed schedule evidence and independent C1-B demand replay."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from dagkv.c1_trace import (
    MAX_TRACE_BYTES,
    CutoffPayload,
    DemandIntentPayload,
    EvidenceRole,
    NaturalTraceWatermarkPayload,
    ObservationTerminalPayload,
    ReplayScheduleWatermarkPayload,
    ReuseEpochPayload,
    ScheduleProducerKind,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceValidationError,
    TraceValidationReceipt,
    ValidatedObservation,
    ValidatedTrace,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
    trace_stream_digest,
    validate_trace,
)
from dagkv.domain import (
    BlockKey,
    ExecutionRef,
    WorkflowKey,
    require_sha256,
    require_text,
)

SCHEDULE_SIDECAR_SCHEMA_VERSION = "dagkv.m3.schedule_sidecar.v1"
SCHEDULE_EVENT_ORDER_RULE = "timestamp_then_ordinal_v1"
SCHEDULE_CLOCK_DOMAIN = "campaign_monotonic_ns"
MAX_SCHEDULE_BYTES = MAX_TRACE_BYTES


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TraceValidationError(f"{name} must be a tuple")


def _require_sorted_unique(name: str, values: tuple[str, ...]) -> None:
    _require_tuple(name, values)
    for value in values:
        require_text(name, value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


def schedule_stream_digest(values: tuple[object, ...]) -> str:
    """Hash an ordered canonical JSONL prefix, including row delimiters."""

    _require_tuple("schedule digest values", values)
    hasher = sha256()
    for value in values:
        hasher.update(canonical_json(value))
        hasher.update(b"\n")
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class ScheduleDemandEvent:
    """One predeclared logical demand, before any cache policy runs."""

    event_ordinal: int
    schedule_event_id: str
    scheduled_access_ns: int
    claim_id: str
    retention_binding_id: str
    request_binding_id: str
    workflow: WorkflowKey
    node_id: str
    execution_ref: ExecutionRef
    block_key: BlockKey
    reuse_epoch_id: str
    source_record_id: str
    source_record_digest: str

    def __post_init__(self) -> None:
        _require_int("event_ordinal", self.event_ordinal)
        _require_int("scheduled_access_ns", self.scheduled_access_ns)
        for name in (
            "schedule_event_id",
            "claim_id",
            "retention_binding_id",
            "request_binding_id",
            "node_id",
            "reuse_epoch_id",
            "source_record_id",
        ):
            require_text(name, getattr(self, name))
        require_sha256("source_record_digest", self.source_record_digest)
        if self.execution_ref.workflow != self.workflow:
            raise TraceValidationError("schedule execution belongs to another workflow")


@dataclass(frozen=True, slots=True)
class ScheduleEpoch:
    """Frozen reference-demand equivalence class."""

    reuse_epoch_id: str
    access_ns: int
    block_key: BlockKey
    schedule_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text("reuse_epoch_id", self.reuse_epoch_id)
        _require_int("epoch access_ns", self.access_ns)
        _require_sorted_unique("schedule_event_ids", self.schedule_event_ids)
        if not self.schedule_event_ids:
            raise TraceValidationError("schedule epoch must contain an event")


@dataclass(frozen=True, slots=True)
class ScheduleCheckpoint:
    """Content-addressed proof that a schedule prefix is closed through time."""

    checkpoint_id: str
    closed_through_ns: int
    consumed_event_count: int
    last_schedule_event_id: str | None
    event_prefix_digest: str
    closed_epoch_count: int
    epoch_prefix_digest: str

    def __post_init__(self) -> None:
        require_text("checkpoint_id", self.checkpoint_id)
        _require_int("closed_through_ns", self.closed_through_ns)
        _require_int("consumed_event_count", self.consumed_event_count)
        if self.consumed_event_count == 0:
            if self.last_schedule_event_id is not None:
                raise TraceValidationError("empty checkpoint has a last event")
        else:
            require_text("last_schedule_event_id", self.last_schedule_event_id)
        require_sha256("event_prefix_digest", self.event_prefix_digest)
        _require_int("closed_epoch_count", self.closed_epoch_count)
        require_sha256("epoch_prefix_digest", self.epoch_prefix_digest)


@dataclass(frozen=True, slots=True)
class ReplayScheduleClosure:
    """EOF conservation for a finite replay plan."""

    declared_plan_event_count: int
    plan_event_digest: str
    final_consumed_plan_event_count: int

    def __post_init__(self) -> None:
        _require_int("declared_plan_event_count", self.declared_plan_event_count)
        require_sha256("plan_event_digest", self.plan_event_digest)
        _require_int(
            "final_consumed_plan_event_count",
            self.final_consumed_plan_event_count,
        )


@dataclass(frozen=True, slots=True)
class NaturalScheduleClosure:
    """EOF and instrumentation state for one sealed natural source."""

    source_eof_record_count: int
    source_eof_digest: str
    capture_start_ns: int
    capture_end_ns: int
    dropped_record_count: int
    clean_eof: bool

    def __post_init__(self) -> None:
        _require_int("source_eof_record_count", self.source_eof_record_count)
        require_sha256("source_eof_digest", self.source_eof_digest)
        _require_int("capture_start_ns", self.capture_start_ns)
        _require_int("capture_end_ns", self.capture_end_ns)
        if self.capture_end_ns < self.capture_start_ns:
            raise TraceValidationError("natural capture interval regresses")
        _require_int("dropped_record_count", self.dropped_record_count)
        if type(self.clean_eof) is not bool:
            raise TraceValidationError("clean_eof must be a bool")


ScheduleClosure = ReplayScheduleClosure | NaturalScheduleClosure


def make_schedule_checkpoint(
    checkpoint_id: str,
    closed_through_ns: int,
    events: tuple[ScheduleDemandEvent, ...],
    epochs: tuple[ScheduleEpoch, ...],
) -> ScheduleCheckpoint:
    """Derive one exact checkpoint from immutable event and epoch prefixes."""

    require_text("checkpoint_id", checkpoint_id)
    _require_int("closed_through_ns", closed_through_ns)
    _require_tuple("schedule events", events)
    _require_tuple("schedule epochs", epochs)
    event_prefix = tuple(
        event for event in events if event.scheduled_access_ns <= closed_through_ns
    )
    epoch_prefix = tuple(
        epoch for epoch in epochs if epoch.access_ns <= closed_through_ns
    )
    return ScheduleCheckpoint(
        checkpoint_id=checkpoint_id,
        closed_through_ns=closed_through_ns,
        consumed_event_count=len(event_prefix),
        last_schedule_event_id=(
            event_prefix[-1].schedule_event_id if event_prefix else None
        ),
        event_prefix_digest=schedule_stream_digest(event_prefix),
        closed_epoch_count=len(epoch_prefix),
        epoch_prefix_digest=schedule_stream_digest(epoch_prefix),
    )


@dataclass(frozen=True, slots=True)
class ClosedScheduleArtifact:
    """Canonical schedule whose file digest is bound by the trace header."""

    schema_version: str
    artifact_id: str
    trace_pair_id: str
    run_id: str
    schedule_id: str
    schedule_case_id: str
    producer_kind: ScheduleProducerKind
    producer_id: str
    source_artifact_digest: str
    source_schema_digest: str
    source_case_digest: str
    clock_domain: str
    event_order_rule: str
    events: tuple[ScheduleDemandEvent, ...]
    epochs: tuple[ScheduleEpoch, ...]
    checkpoints: tuple[ScheduleCheckpoint, ...]
    closure: ScheduleClosure
    final_event_digest: str
    final_epoch_digest: str
    final_checkpoint_id: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_SIDECAR_SCHEMA_VERSION:
            raise TraceValidationError("unsupported schedule sidecar schema")
        for name in (
            "artifact_id",
            "trace_pair_id",
            "run_id",
            "schedule_id",
            "schedule_case_id",
            "producer_id",
            "final_checkpoint_id",
        ):
            require_text(name, getattr(self, name))
        if type(self.producer_kind) is not ScheduleProducerKind:
            raise TraceValidationError("producer_kind must be a ScheduleProducerKind")
        require_sha256("source_artifact_digest", self.source_artifact_digest)
        require_sha256("source_schema_digest", self.source_schema_digest)
        require_sha256("source_case_digest", self.source_case_digest)
        if self.clock_domain != SCHEDULE_CLOCK_DOMAIN:
            raise TraceValidationError("unsupported schedule clock domain")
        if self.event_order_rule != SCHEDULE_EVENT_ORDER_RULE:
            raise TraceValidationError("unsupported schedule event order rule")
        for name, value in (
            ("events", self.events),
            ("epochs", self.epochs),
            ("checkpoints", self.checkpoints),
        ):
            _require_tuple(name, value)

        self._validate_events()
        self._validate_epochs()
        self._validate_checkpoints()
        self._validate_closure()
        require_sha256("final_event_digest", self.final_event_digest)
        require_sha256("final_epoch_digest", self.final_epoch_digest)
        if self.final_event_digest != schedule_stream_digest(self.events):
            raise TraceValidationError(
                "final event digest differs from schedule events"
            )
        if self.final_epoch_digest != schedule_stream_digest(self.epochs):
            raise TraceValidationError(
                "final epoch digest differs from schedule epochs"
            )
        if self.final_checkpoint_id != self.checkpoints[-1].checkpoint_id:
            raise TraceValidationError("final checkpoint identity is inconsistent")
        final_checkpoint = self.checkpoints[-1]
        if final_checkpoint.consumed_event_count != len(
            self.events
        ) or final_checkpoint.closed_epoch_count != len(self.epochs):
            raise TraceValidationError("final checkpoint does not close the schedule")

    def _validate_events(self) -> None:
        event_ids: set[str] = set()
        prior_timestamp = -1
        for ordinal, event in enumerate(self.events):
            if type(event) is not ScheduleDemandEvent:
                raise TraceValidationError("schedule contains a non-event value")
            if event.event_ordinal != ordinal:
                raise TraceValidationError("schedule event ordinals are not contiguous")
            if event.scheduled_access_ns < prior_timestamp:
                raise TraceValidationError("schedule event timestamps regress")
            if event.schedule_event_id in event_ids:
                raise TraceValidationError("schedule event identity is duplicated")
            event_ids.add(event.schedule_event_id)
            prior_timestamp = event.scheduled_access_ns

    def _validate_epochs(self) -> None:
        event_by_id = {event.schedule_event_id: event for event in self.events}
        epoch_ids: set[str] = set()
        assigned_event_ids: set[str] = set()
        prior_key: tuple[int, str] | None = None
        for epoch in self.epochs:
            if type(epoch) is not ScheduleEpoch:
                raise TraceValidationError("schedule contains a non-epoch value")
            key = (epoch.access_ns, epoch.reuse_epoch_id)
            if prior_key is not None and key <= prior_key:
                raise TraceValidationError("schedule epochs are not strictly ordered")
            prior_key = key
            if epoch.reuse_epoch_id in epoch_ids:
                raise TraceValidationError("schedule epoch identity is duplicated")
            epoch_ids.add(epoch.reuse_epoch_id)
            for event_id in epoch.schedule_event_ids:
                try:
                    event = event_by_id[event_id]
                except KeyError as exc:
                    raise TraceValidationError(
                        "schedule epoch references an unknown event"
                    ) from exc
                if event_id in assigned_event_ids:
                    raise TraceValidationError(
                        "schedule event belongs to multiple epochs"
                    )
                if (
                    event.reuse_epoch_id != epoch.reuse_epoch_id
                    or event.scheduled_access_ns != epoch.access_ns
                    or event.block_key != epoch.block_key
                ):
                    raise TraceValidationError(
                        "schedule epoch contradicts one of its events"
                    )
                assigned_event_ids.add(event_id)
        if assigned_event_ids != set(event_by_id):
            raise TraceValidationError("schedule epochs do not partition every event")

    def _validate_checkpoints(self) -> None:
        if not self.checkpoints:
            raise TraceValidationError("schedule requires at least one checkpoint")
        checkpoint_ids: set[str] = set()
        prior_closed_ns = -1
        for checkpoint in self.checkpoints:
            if type(checkpoint) is not ScheduleCheckpoint:
                raise TraceValidationError("schedule contains a non-checkpoint value")
            if checkpoint.checkpoint_id in checkpoint_ids:
                raise TraceValidationError("schedule checkpoint identity is duplicated")
            checkpoint_ids.add(checkpoint.checkpoint_id)
            if checkpoint.closed_through_ns <= prior_closed_ns:
                raise TraceValidationError(
                    "schedule checkpoints are not strictly time ordered"
                )
            expected = make_schedule_checkpoint(
                checkpoint.checkpoint_id,
                checkpoint.closed_through_ns,
                self.events,
                self.epochs,
            )
            if checkpoint != expected:
                raise TraceValidationError("schedule checkpoint prefix is inconsistent")
            prior_closed_ns = checkpoint.closed_through_ns

    def _validate_closure(self) -> None:
        final_event_digest = schedule_stream_digest(self.events)
        if self.producer_kind == ScheduleProducerKind.REPLAY:
            if type(self.closure) is not ReplayScheduleClosure:
                raise TraceValidationError("replay schedule has the wrong closure")
            if (
                self.closure.declared_plan_event_count != len(self.events)
                or self.closure.final_consumed_plan_event_count != len(self.events)
                or self.closure.plan_event_digest != final_event_digest
            ):
                raise TraceValidationError("replay plan closure is inconsistent")
            return
        if self.producer_kind == ScheduleProducerKind.SEALED_NATURAL_TRACE:
            if type(self.closure) is not NaturalScheduleClosure:
                raise TraceValidationError("natural schedule has the wrong closure")
            return
        raise TraceValidationError("unsupported schedule producer kind")


@dataclass(frozen=True, slots=True)
class LoadedScheduleArtifact:
    artifact: ClosedScheduleArtifact
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256("loaded schedule digest", self.digest)
        _require_int("loaded schedule size", self.size_bytes, minimum=1)


def _open_parent(path: Path) -> int:
    if not path.is_absolute():
        raise TraceValidationError("schedule path must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise TraceValidationError("schedule parent must be a non-symlink directory")
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
        raise TraceValidationError("cannot open schedule parent safely") from exc
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        os.close(descriptor)
        raise TraceValidationError("schedule parent changed while opening")
    return descriptor


def write_schedule_artifact(path: Path, artifact: ClosedScheduleArtifact) -> str:
    """Create, fsync, and read back one canonical schedule artifact."""

    raw = canonical_json(artifact)
    if len(raw) > MAX_SCHEDULE_BYTES:
        raise TraceValidationError("schedule artifact exceeds the size limit")
    parsed = parse_canonical_dataclass(
        raw,
        ClosedScheduleArtifact,
        artifact_name="schedule artifact",
        max_bytes=MAX_SCHEDULE_BYTES,
    )
    if parsed != artifact:
        raise TraceValidationError("schedule artifact changes during canonical replay")
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
                raise OSError("schedule write made no progress")
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
            raise OSError("schedule output identity changed")
        observed = bytearray()
        read_offset = 0
        while read_offset < len(raw):
            chunk = os.pread(descriptor, len(raw) - read_offset, read_offset)
            if not chunk:
                raise OSError("schedule output ended during readback")
            observed.extend(chunk)
            read_offset += len(chunk)
        if bytes(observed) != raw:
            raise OSError("schedule output differs from staged bytes")
    except FileExistsError as exc:
        raise TraceValidationError("schedule artifact is create-only") from exc
    except OSError as exc:
        raise TraceCommitIndeterminateError(
            "schedule artifact durability is indeterminate"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return sha256(raw).hexdigest()


def _read_stable_file(
    path: Path,
    *,
    artifact_name: str,
    max_bytes: int,
) -> bytes:
    require_text("artifact_name", artifact_name)
    _require_int("max_bytes", max_bytes, minimum=1)
    if not path.is_absolute():
        raise TraceValidationError(f"{artifact_name} path must be absolute")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise TraceValidationError(f"{artifact_name} is missing") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise TraceValidationError(
            f"{artifact_name} path must be a singly linked regular non-symlink file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TraceValidationError(f"cannot open {artifact_name} safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise TraceValidationError(f"{artifact_name} changed while opening")
        raw = bytearray()
        while len(raw) <= max_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, max_bytes + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1:
            raise TraceValidationError(
                f"{artifact_name} link state changed while reading"
            )
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
            raise TraceValidationError(f"{artifact_name} changed while reading")
        linked = path.lstat()
        if (
            not stat.S_ISREG(linked.st_mode)
            or linked.st_nlink != 1
            or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise TraceValidationError(f"{artifact_name} path changed while reading")
    except OSError as exc:
        raise TraceValidationError(f"cannot read {artifact_name} safely") from exc
    finally:
        os.close(descriptor)
    if not raw or len(raw) > max_bytes:
        raise TraceValidationError(f"{artifact_name} has an invalid size")
    return bytes(raw)


def load_schedule_artifact(path: Path) -> LoadedScheduleArtifact:
    """Load one stable canonical schedule and return its exact file identity."""

    raw = _read_stable_file(
        path,
        artifact_name="schedule artifact",
        max_bytes=MAX_SCHEDULE_BYTES,
    )
    artifact = parse_canonical_dataclass(
        raw,
        ClosedScheduleArtifact,
        artifact_name="schedule artifact",
        max_bytes=MAX_SCHEDULE_BYTES,
    )
    return LoadedScheduleArtifact(
        artifact=artifact,
        digest=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def _payload[T](record: object, expected: type[T]) -> T:
    payload = getattr(record, "payload", None)
    if type(payload) is not expected:
        raise TraceValidationError(f"expected {expected.__name__} payload")
    return payload


class CanonicalScheduleEvidenceGate:
    """Authorize labels only after exact schedule and demand reconstruction."""

    def __init__(
        self,
        path: Path,
        verifier_digest: str,
        *,
        producer_source_path: Path | None = None,
    ) -> None:
        if not path.is_absolute():
            raise TraceValidationError("schedule verifier path must be absolute")
        require_sha256("schedule verifier digest", verifier_digest)
        if producer_source_path is not None and not producer_source_path.is_absolute():
            raise TraceValidationError("producer source path must be absolute")
        self._path = path
        self._verifier_digest = verifier_digest
        self._producer_source_path = producer_source_path

    def verify_schedule(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        replayed = validate_trace(trace.records)
        loaded = load_schedule_artifact(self._path)
        artifact = loaded.artifact
        header = _payload(replayed.header, TraceHeaderPayload)
        if loaded.digest != header.schedule_digest:
            raise TraceValidationError("schedule artifact differs from trace header")
        if (
            artifact.trace_pair_id != header.trace_pair_id
            or artifact.run_id != replayed.header.run_id
            or artifact.schedule_id != replayed.header.schedule_id
            or artifact.schedule_case_id != replayed.header.schedule_case_id
        ):
            raise TraceValidationError("schedule artifact belongs to another trace")
        source_case_digests = {
            _payload(topology, WorkflowTopologyPayload).source_case_digest
            for topology in replayed.topologies
        }
        if source_case_digests != {artifact.source_case_digest}:
            raise TraceValidationError(
                "schedule source case differs from workflow topology"
            )
        if artifact.source_artifact_digest != header.source_digest:
            raise TraceValidationError(
                "schedule source artifact differs from trace header"
            )
        if isinstance(artifact.closure, NaturalScheduleClosure):
            if artifact.source_artifact_digest != artifact.closure.source_eof_digest:
                raise TraceValidationError(
                    "natural source closure differs from schedule source artifact"
                )
            self._verify_natural_source(artifact)

        verified: list[str] = []
        for observation in replayed.observations:
            terminal = _payload(observation.terminal, ObservationTerminalPayload)
            if terminal.status != TerminalStatus.COMPLETE:
                continue
            self._verify_complete_observation(observation, artifact, loaded.digest)
            verified.append(observation.observation_id)
        return TraceValidationReceipt(
            role=EvidenceRole.SCHEDULE,
            trace_pair_id=header.trace_pair_id,
            trace_digest=trace_stream_digest(replayed.records),
            artifact_digest=loaded.digest,
            verifier_digest=self._verifier_digest,
            verified_observation_ids=tuple(sorted(verified)),
        )

    def _verify_natural_source(self, artifact: ClosedScheduleArtifact) -> None:
        source_path = self._producer_source_path
        if source_path is None:
            raise TraceValidationError(
                "natural schedule requires its sealed producer source"
            )
        closure = artifact.closure
        if type(closure) is not NaturalScheduleClosure:
            raise TraceValidationError(
                "natural source verifier received replay closure"
            )
        raw = _read_stable_file(
            source_path,
            artifact_name="natural producer source",
            max_bytes=MAX_SCHEDULE_BYTES,
        )
        if sha256(raw).hexdigest() != closure.source_eof_digest:
            raise TraceValidationError("natural producer source EOF digest differs")
        if not raw.endswith(b"\n") or b"\r" in raw:
            raise TraceValidationError("natural producer source framing is invalid")
        lines = raw[:-1].split(b"\n")
        if any(not line for line in lines):
            raise TraceValidationError(
                "natural producer source contains a blank record"
            )
        if len(lines) != closure.source_eof_record_count:
            raise TraceValidationError("natural producer source EOF count differs")
        source_record_digests = tuple(sha256(line).hexdigest() for line in lines)
        if len(source_record_digests) != len(set(source_record_digests)):
            raise TraceValidationError("natural producer source record is duplicated")
        if any(
            event.source_record_digest not in source_record_digests
            for event in artifact.events
        ):
            raise TraceValidationError(
                "normalized schedule event lacks an exact natural source record"
            )

    def _verify_complete_observation(
        self,
        observation: ValidatedObservation,
        artifact: ClosedScheduleArtifact,
        artifact_digest: str,
    ) -> None:
        cutoff = _payload(observation.cutoff, CutoffPayload)
        if observation.watermark is None:
            raise TraceValidationError(
                "complete observation lacks a schedule watermark"
            )
        watermark = observation.watermark.payload
        if not isinstance(
            watermark,
            (ReplayScheduleWatermarkPayload, NaturalTraceWatermarkPayload),
        ):
            raise TraceValidationError("observation has an unknown schedule watermark")
        producer_artifact_digest = (
            artifact.closure.plan_event_digest
            if isinstance(artifact.closure, ReplayScheduleClosure)
            else artifact.closure.source_eof_digest
        )
        if (
            watermark.producer_kind != artifact.producer_kind
            or watermark.producer_id != artifact.producer_id
            or watermark.producer_artifact_digest != producer_artifact_digest
            or watermark.schedule_digest != artifact_digest
        ):
            raise TraceValidationError("watermark producer binding is inconsistent")
        checkpoint_by_id = {
            checkpoint.checkpoint_id: checkpoint for checkpoint in artifact.checkpoints
        }
        try:
            checkpoint = checkpoint_by_id[watermark.checkpoint_id]
        except KeyError as exc:
            raise TraceValidationError(
                "watermark references an unknown checkpoint"
            ) from exc
        if (
            watermark.checkpoint_digest != canonical_digest(checkpoint)
            or watermark.consumed_event_count != checkpoint.consumed_event_count
            or watermark.last_schedule_event_id != checkpoint.last_schedule_event_id
            or watermark.max_closed_timestamp_ns != checkpoint.closed_through_ns
            or watermark.event_prefix_digest != checkpoint.event_prefix_digest
            or watermark.closed_epoch_count != checkpoint.closed_epoch_count
            or watermark.epoch_prefix_digest != checkpoint.epoch_prefix_digest
        ):
            raise TraceValidationError("watermark differs from its exact checkpoint")
        if checkpoint.closed_through_ns <= cutoff.deadline_ns:
            raise TraceValidationError("schedule checkpoint does not exceed deadline")
        self._verify_producer_closure(watermark, artifact.closure, cutoff, checkpoint)

        owners = {owner.binding_id: owner for owner in cutoff.snapshot.owners}
        expected_events = tuple(
            event
            for event in artifact.events
            if cutoff.cutoff_ns < event.scheduled_access_ns <= cutoff.deadline_ns
            and event.block_key == cutoff.snapshot.block_key
            and event.retention_binding_id in owners
            and owners[event.retention_binding_id].workflow == event.workflow
            and event.node_id in owners[event.retention_binding_id].eligible_node_ids
        )
        self._verify_demand_bijection(observation, artifact, expected_events)

    @staticmethod
    def _verify_producer_closure(
        watermark: ReplayScheduleWatermarkPayload | NaturalTraceWatermarkPayload,
        closure: ScheduleClosure,
        cutoff: CutoffPayload,
        checkpoint: ScheduleCheckpoint,
    ) -> None:
        if isinstance(watermark, ReplayScheduleWatermarkPayload):
            if type(closure) is not ReplayScheduleClosure:
                raise TraceValidationError("replay watermark has natural closure")
            return
        if type(closure) is not NaturalScheduleClosure:
            raise TraceValidationError("natural watermark has replay closure")
        if (
            watermark.source_eof_record_count != closure.source_eof_record_count
            or watermark.source_eof_digest != closure.source_eof_digest
            or watermark.capture_start_ns != closure.capture_start_ns
            or watermark.capture_end_ns != closure.capture_end_ns
            or watermark.dropped_record_count != closure.dropped_record_count
            or watermark.clean_eof != closure.clean_eof
        ):
            raise TraceValidationError("natural watermark differs from sealed EOF")
        if (
            not closure.clean_eof
            or closure.dropped_record_count != 0
            or closure.capture_start_ns > cutoff.cutoff_ns
            or closure.capture_end_ns < checkpoint.closed_through_ns
        ):
            raise TraceValidationError(
                "natural source does not cover the closed window"
            )
        raise TraceValidationError(
            "natural source total normalization verifier is not implemented"
        )

    @staticmethod
    def _verify_demand_bijection(
        observation: ValidatedObservation,
        artifact: ClosedScheduleArtifact,
        expected_events: tuple[ScheduleDemandEvent, ...],
    ) -> None:
        intent_by_schedule_id: dict[str, tuple[object, DemandIntentPayload]] = {}
        for record in observation.intents:
            intent = _payload(record, DemandIntentPayload)
            if intent.schedule_event_id in intent_by_schedule_id:
                raise TraceValidationError("trace duplicates a scheduled demand")
            intent_by_schedule_id[intent.schedule_event_id] = (record, intent)
        expected_by_id = {event.schedule_event_id: event for event in expected_events}
        if set(intent_by_schedule_id) != set(expected_by_id):
            raise TraceValidationError(
                "trace demand intents are not a bijection with scheduled events"
            )
        for schedule_event_id, event in expected_by_id.items():
            _, intent = intent_by_schedule_id[schedule_event_id]
            if (
                intent.scheduled_access_ns != event.scheduled_access_ns
                or intent.claim_id != event.claim_id
                or intent.retention_binding_id != event.retention_binding_id
                or intent.request_binding_id != event.request_binding_id
                or intent.workflow != event.workflow
                or intent.node_id != event.node_id
                or intent.execution_ref != event.execution_ref
                or intent.block_key != event.block_key
                or intent.reuse_epoch_id != event.reuse_epoch_id
            ):
                raise TraceValidationError(
                    "demand intent differs from its scheduled event"
                )

        selected_ids = set(expected_by_id)
        expected_epochs: dict[str, ScheduleEpoch] = {}
        for epoch in artifact.epochs:
            member_ids = set(epoch.schedule_event_ids)
            selected_members = member_ids & selected_ids
            if selected_members and selected_members != member_ids:
                raise TraceValidationError(
                    "schedule epoch crosses the observation selection boundary"
                )
            if selected_members:
                expected_epochs[epoch.reuse_epoch_id] = epoch
        observed_epochs: dict[str, ReuseEpochPayload] = {}
        for record in observation.epochs:
            epoch = _payload(record, ReuseEpochPayload)
            if epoch.reuse_epoch_id in observed_epochs:
                raise TraceValidationError("trace duplicates a schedule epoch")
            observed_epochs[epoch.reuse_epoch_id] = epoch
        if set(observed_epochs) != set(expected_epochs):
            raise TraceValidationError(
                "trace reuse epochs are not a bijection with schedule epochs"
            )
        for epoch_id, expected_epoch in expected_epochs.items():
            observed_epoch = observed_epochs[epoch_id]
            expected_intent_ids = tuple(
                sorted(
                    intent_by_schedule_id[event_id][0].record_id
                    for event_id in expected_epoch.schedule_event_ids
                )
            )
            if (
                observed_epoch.access_ns != expected_epoch.access_ns
                or observed_epoch.block_key != expected_epoch.block_key
                or observed_epoch.demand_intent_record_ids != expected_intent_ids
            ):
                raise TraceValidationError(
                    "trace reuse epoch differs from the frozen schedule"
                )
