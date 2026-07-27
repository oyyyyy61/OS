"""Operation-typed durable commits for the C1-B canonical trace."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from threading import RLock

from dagkv.c1_schedule import (
    ClosedScheduleArtifact,
    NaturalScheduleClosure,
    ReplayScheduleClosure,
    ScheduleDemandEvent,
)
from dagkv.c1_trace import (
    TRACE_SCHEMA_VERSION,
    AbstainedAttemptPayload,
    AtomicCutoffView,
    CutoffPayload,
    DemandIntentPayload,
    DurableTraceWriter,
    NaturalTraceWatermarkPayload,
    ObservationTerminalPayload,
    PredictedAttemptPayload,
    PreServiceDemandView,
    ReplayScheduleWatermarkPayload,
    ReuseEpochPayload,
    ServiceTerminal,
    TerminalReason,
    TerminalStatus,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceRecord,
    TraceRecordType,
    TraceStreamClosure,
    TraceValidationError,
    WaiterIdentity,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
    encode_trace_record,
    load_trace_jsonl,
    parse_canonical_dataclass,
    parse_trace_record,
    trace_stream_digest,
)
from dagkv.domain import TransferCommand, WorkflowKey, require_sha256, require_text


class TraceOperationKind(StrEnum):
    """Closed operation vocabulary for the formal C1 trace writer."""

    PREAMBLE = "PREAMBLE"
    CUTOFF_ATTEMPT = "CUTOFF_ATTEMPT"
    DEMAND_INTENT = "DEMAND_INTENT"
    OBSERVATION_CLOSE = "OBSERVATION_CLOSE"


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_sorted_unique(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TraceValidationError(f"{name} must be a tuple")
    for value in values:
        require_text(name, value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


def _require_basename(name: str, value: str) -> None:
    require_text(name, value)
    if "\x00" in value or value in {".", ".."} or Path(value).name != value:
        raise TraceValidationError(f"{name} must be a plain basename")


def _safe_failure_detail(operation: str, exc: BaseException) -> str:
    """Format diagnostics without weakening a fail-closed poison boundary."""

    try:
        return f"{operation}: {type(exc).__name__}: {exc}"
    except BaseException:
        return f"{operation}: unprintable failure"


@dataclass(frozen=True, slots=True)
class TraceEnvelope:
    """Immutable identity repeated on every canonical trace record."""

    trace_id: str
    run_id: str
    schedule_id: str
    schedule_case_id: str

    def __post_init__(self) -> None:
        for name in ("trace_id", "run_id", "schedule_id", "schedule_case_id"):
            require_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class TracePreambleRequest:
    operation_id: str
    header: TraceHeaderPayload
    topologies: tuple[WorkflowTopologyPayload, ...]

    def __post_init__(self) -> None:
        require_text("preamble operation_id", self.operation_id)
        if not isinstance(self.topologies, tuple) or not self.topologies:
            raise TraceValidationError("trace preamble requires topology payloads")
        workflows = tuple(item.workflow_spec.key for item in self.topologies)
        if workflows != tuple(sorted(workflows)) or len(workflows) != len(
            set(workflows)
        ):
            raise TraceValidationError(
                "trace preamble topologies must be workflow ordered and unique"
            )


AttemptPayload = PredictedAttemptPayload | AbstainedAttemptPayload


@dataclass(frozen=True, slots=True)
class CutoffCommitRequest:
    operation_id: str
    observation_id: str
    attempt: AttemptPayload

    def __post_init__(self) -> None:
        require_text("cutoff operation_id", self.operation_id)
        require_text("cutoff observation_id", self.observation_id)
        if not isinstance(
            self.attempt,
            (PredictedAttemptPayload, AbstainedAttemptPayload),
        ):
            raise TraceValidationError("cutoff request has an invalid attempt")


@dataclass(frozen=True, slots=True)
class DemandCommitRequest:
    operation_id: str
    observation_id: str
    schedule_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text("demand operation_id", self.operation_id)
        require_text("demand observation_id", self.observation_id)
        _require_sorted_unique("demand schedule_event_ids", self.schedule_event_ids)
        if not self.schedule_event_ids:
            raise TraceValidationError("demand commit requires schedule events")


ScheduleWatermarkPayload = ReplayScheduleWatermarkPayload | NaturalTraceWatermarkPayload


@dataclass(frozen=True, slots=True)
class ObservationTerminalSpec:
    status: TerminalStatus
    reason: TerminalReason
    label_available_ns: int | None
    last_verified_event_count: int
    last_verified_event_id: str | None
    last_verified_event_timestamp_ns: int | None

    def __post_init__(self) -> None:
        if type(self.status) is not TerminalStatus:
            raise TraceValidationError("terminal spec status has the wrong type")
        if type(self.reason) is not TerminalReason:
            raise TraceValidationError("terminal spec reason has the wrong type")
        _require_int(
            "terminal spec last_verified_event_count",
            self.last_verified_event_count,
        )
        if self.last_verified_event_count == 0:
            if (
                self.last_verified_event_id is not None
                or self.last_verified_event_timestamp_ns is not None
            ):
                raise TraceValidationError(
                    "empty terminal lifecycle prefix has a last event"
                )
        else:
            require_text(
                "terminal spec last_verified_event_id",
                self.last_verified_event_id,
            )
            if self.last_verified_event_timestamp_ns is None:
                raise TraceValidationError(
                    "terminal lifecycle prefix lacks its timestamp"
                )
            _require_int(
                "terminal spec last_verified_event_timestamp_ns",
                self.last_verified_event_timestamp_ns,
            )


@dataclass(frozen=True, slots=True)
class ObservationCloseRequest:
    operation_id: str
    observation_id: str
    services: tuple[ServiceTerminal, ...]
    watermark: ScheduleWatermarkPayload | None
    terminal: ObservationTerminalSpec

    def __post_init__(self) -> None:
        require_text("close operation_id", self.operation_id)
        require_text("close observation_id", self.observation_id)
        if not isinstance(self.services, tuple):
            raise TraceValidationError("close services must be a tuple")
        terminal_ids = tuple(service.intent_record_id for service in self.services)
        if terminal_ids != tuple(sorted(terminal_ids)) or len(terminal_ids) != len(
            set(terminal_ids)
        ):
            raise TraceValidationError(
                "close services must be intent-record ordered and unique"
            )
        if self.watermark is not None and not isinstance(
            self.watermark,
            (ReplayScheduleWatermarkPayload, NaturalTraceWatermarkPayload),
        ):
            raise TraceValidationError("close request has an invalid watermark")


@dataclass(frozen=True, slots=True)
class TraceOperationCommit:
    """Persistent identity of one operation-sized append to the trace."""

    trace_pair_id: str
    kind: TraceOperationKind
    operation_id: str
    request_digest: str
    commit_index: int
    sequence_start: int
    sequence_end: int
    record_ids: tuple[str, ...]
    byte_start: int
    byte_end: int
    batch_digest: str
    prior_stream_digest: str
    committed_stream_digest: str
    runtime_event_count: int
    runtime_view_digest: str

    def __post_init__(self) -> None:
        require_text("commit trace_pair_id", self.trace_pair_id)
        if type(self.kind) is not TraceOperationKind:
            raise TraceValidationError("commit kind has the wrong type")
        require_text("commit operation_id", self.operation_id)
        for name in (
            "request_digest",
            "batch_digest",
            "prior_stream_digest",
            "committed_stream_digest",
            "runtime_view_digest",
        ):
            require_sha256(name, getattr(self, name))
        _require_int("commit_index", self.commit_index)
        _require_int("sequence_start", self.sequence_start)
        _require_int("sequence_end", self.sequence_end, minimum=1)
        if self.sequence_end <= self.sequence_start:
            raise TraceValidationError("commit sequence interval is empty")
        if not isinstance(self.record_ids, tuple) or len(self.record_ids) != (
            self.sequence_end - self.sequence_start
        ):
            raise TraceValidationError("commit record IDs do not cover its sequence")
        if len(self.record_ids) != len(set(self.record_ids)):
            raise TraceValidationError("commit record IDs are duplicated")
        for record_id in self.record_ids:
            require_text("commit record_id", record_id)
        _require_int("byte_start", self.byte_start)
        _require_int("byte_end", self.byte_end, minimum=1)
        if self.byte_end <= self.byte_start:
            raise TraceValidationError("commit byte interval is empty")
        _require_int("runtime_event_count", self.runtime_event_count)


@dataclass(frozen=True, slots=True)
class WriterIssuedCommitReceipt:
    """Opaque in-process capability backed by one durable operation commit."""

    commit: TraceOperationCommit
    receipt_id: str

    def __post_init__(self) -> None:
        require_sha256("typed receipt_id", self.receipt_id)
        if self.receipt_id != canonical_digest(self.commit):
            raise TraceValidationError("typed receipt identity differs from its commit")


@dataclass(frozen=True, slots=True)
class SealedTraceReceipt:
    """Final durable trace closure plus every operation boundary."""

    trace_pair_id: str
    trace_basename: str
    closure: TraceStreamClosure
    operations: tuple[TraceOperationCommit, ...]

    def __post_init__(self) -> None:
        require_text("sealed trace_pair_id", self.trace_pair_id)
        _require_basename("sealed trace_basename", self.trace_basename)
        if not isinstance(self.operations, tuple) or not self.operations:
            raise TraceValidationError("sealed trace lacks operation commits")
        if tuple(item.commit_index for item in self.operations) != tuple(
            range(len(self.operations))
        ):
            raise TraceValidationError("trace commit indices are not contiguous")
        operation_ids = tuple(item.operation_id for item in self.operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise TraceValidationError("trace operation identity is duplicated")
        if any(
            operation.trace_pair_id != self.trace_pair_id
            for operation in self.operations
        ):
            raise TraceValidationError("trace operation belongs to another trace pair")
        first = self.operations[0]
        if (
            first.sequence_start != 0
            or first.byte_start != 0
            or first.prior_stream_digest != sha256(b"").hexdigest()
        ):
            raise TraceValidationError("trace commit chain does not start at zero")
        for prior, current in zip(
            self.operations,
            self.operations[1:],
            strict=False,
        ):
            if (
                prior.sequence_end != current.sequence_start
                or prior.byte_end != current.byte_start
                or prior.committed_stream_digest != current.prior_stream_digest
            ):
                raise TraceValidationError("trace operation chain is discontinuous")
        final = self.operations[-1]
        if (
            final.sequence_end != self.closure.record_count
            or final.byte_end != self.closure.size_bytes
            or final.committed_stream_digest != self.closure.stream_digest
            or self.operations[0].record_ids[0] != self.closure.first_record_id
            or final.record_ids[-1] != self.closure.last_record_id
        ):
            raise TraceValidationError("trace closure differs from its commit chain")


@dataclass(frozen=True, slots=True)
class CommittedCutoff:
    """Atomic runtime view, model attempt, and exact durable trace capability."""

    view: AtomicCutoffView
    attempt: AttemptPayload
    receipt: WriterIssuedCommitReceipt


@dataclass(frozen=True, slots=True)
class CommittedDemandDispatch:
    """Service dispatch bound to the durable pre-service demand operation."""

    command: TransferCommand | None
    receipt: WriterIssuedCommitReceipt
    replayed: bool

    def __post_init__(self) -> None:
        if type(self.replayed) is not bool:
            raise TraceValidationError("demand replay marker must be a bool")


class CanonicalTraceCommitter:
    """Sole operation-typed owner of one create-only canonical trace writer."""

    def __init__(
        self,
        path: Path,
        *,
        envelope: TraceEnvelope,
        schedule: ClosedScheduleArtifact,
        schedule_artifact_digest: str,
    ) -> None:
        require_sha256("schedule_artifact_digest", schedule_artifact_digest)
        schedule_raw = canonical_json(schedule)
        frozen_schedule = parse_canonical_dataclass(
            schedule_raw,
            ClosedScheduleArtifact,
            artifact_name="canonical committer schedule",
            max_bytes=64 * 1024 * 1024,
        )
        frozen_envelope = parse_canonical_dataclass(
            canonical_json(envelope),
            TraceEnvelope,
            artifact_name="canonical committer envelope",
            max_bytes=1024 * 1024,
        )
        observed_schedule_digest = sha256(schedule_raw).hexdigest()
        if observed_schedule_digest != schedule_artifact_digest:
            raise TraceValidationError("frozen schedule digest differs from its bytes")
        if (
            frozen_schedule.run_id != frozen_envelope.run_id
            or frozen_schedule.schedule_id != frozen_envelope.schedule_id
            or frozen_schedule.schedule_case_id != frozen_envelope.schedule_case_id
        ):
            raise TraceValidationError("trace envelope differs from frozen schedule")
        self._path = path
        self._envelope = frozen_envelope
        self._schedule = frozen_schedule
        self._schedule_artifact_digest = schedule_artifact_digest
        self._lock = RLock()
        self._writer = DurableTraceWriter(self._path)
        self._stream_hasher = sha256()
        self._bytes_written = 0
        self._next_sequence = 0
        self._records: list[TraceRecord] = []
        self._operations: dict[
            str,
            tuple[str, str, WriterIssuedCommitReceipt],
        ] = {}
        self._commits: list[TraceOperationCommit] = []
        self._header_payload: TraceHeaderPayload | None = None
        self._header_record_id: str | None = None
        self._topology_payloads: dict[WorkflowKey, WorkflowTopologyPayload] = {}
        self._topology_record_ids: dict[WorkflowKey, str] = {}
        self._observation_heads: dict[str, str] = {}
        self._observation_cutoffs: dict[str, CutoffPayload] = {}
        self._observation_intents: dict[str, dict[str, TraceRecord]] = {}
        self._closed_observations: set[str] = set()
        self._poisoned_reason: str | None = None
        self._sealed: SealedTraceReceipt | None = None
        self._sealed_receipt_digest: str | None = None
        self._event_by_id = {
            event.schedule_event_id: event for event in self._schedule.events
        }
        self._epoch_by_id = {
            epoch.reuse_epoch_id: epoch for epoch in self._schedule.epochs
        }

    @property
    def path(self) -> Path:
        return self._path

    @property
    def envelope(self) -> TraceEnvelope:
        with self._lock:
            return deepcopy(self._envelope)

    @property
    def schedule(self) -> ClosedScheduleArtifact:
        with self._lock:
            return deepcopy(self._schedule)

    @property
    def schedule_artifact_digest(self) -> str:
        return self._schedule_artifact_digest

    @property
    def run_id(self) -> str:
        return self._envelope.run_id

    @property
    def trace_pair_id(self) -> str:
        with self._lock:
            if self._header_payload is not None:
                return self._header_payload.trace_pair_id
            return self._schedule.trace_pair_id

    @property
    def poisoned_reason(self) -> str | None:
        with self._lock:
            return self._poisoned_reason

    @property
    def records(self) -> tuple[TraceRecord, ...]:
        with self._lock:
            return deepcopy(tuple(self._records))

    @property
    def commits(self) -> tuple[TraceOperationCommit, ...]:
        with self._lock:
            return deepcopy(tuple(self._commits))

    def _guard_active(self) -> None:
        if self._poisoned_reason is not None:
            raise TraceCommitIndeterminateError(
                f"typed trace committer is poisoned: {self._poisoned_reason}"
            )
        if self._sealed is not None:
            raise TraceValidationError("typed trace committer is sealed")

    @staticmethod
    def _canonical_snapshot[T](value: T, expected: type[T], name: str) -> T:
        return parse_canonical_dataclass(
            canonical_json(value),
            expected,
            artifact_name=name,
            max_bytes=64 * 1024 * 1024,
        )

    def operation_committed(
        self,
        operation_id: str,
        kind: TraceOperationKind,
    ) -> bool:
        """Check an operation cache entry while revalidating its exact receipt."""

        require_text("operation_id", operation_id)
        if type(kind) is not TraceOperationKind:
            raise TraceValidationError("operation kind has the wrong type")
        with self._lock:
            cached = self._operations.get(operation_id)
            if cached is None:
                return False
            _, expected_receipt_id, receipt = cached
            self._assert_cached_receipt_integrity(
                operation_id,
                expected_receipt_id,
                receipt,
            )
            if receipt.commit.kind != kind:
                raise TraceValidationError(
                    "trace operation identity belongs to another kind"
                )
            return True

    def _poison(self, operation: str, exc: BaseException) -> None:
        self._poisoned_reason = "canonical trace attempt poisoned"
        self._poisoned_reason = _safe_failure_detail(operation, exc)

    def _request_digest(self, request: object, view_digest: str | None = None) -> str:
        hasher = sha256(canonical_json(request))
        if view_digest is not None:
            require_sha256("operation view_digest", view_digest)
            hasher.update(b"\n")
            hasher.update(view_digest.encode("ascii"))
        return hasher.hexdigest()

    def _cached(
        self,
        *,
        kind: TraceOperationKind,
        operation_id: str,
        request_digest: str,
    ) -> WriterIssuedCommitReceipt | None:
        cached = self._operations.get(operation_id)
        if cached is None:
            return None
        prior_digest, expected_receipt_id, receipt = cached
        self._assert_cached_receipt_integrity(
            operation_id,
            expected_receipt_id,
            receipt,
        )
        if prior_digest != request_digest or receipt.commit.kind != kind:
            raise TraceValidationError(
                "trace operation ID was reused for another typed request"
            )
        return receipt

    def _assert_cached_receipt_integrity(
        self,
        operation_id: str,
        expected_receipt_id: str,
        receipt: WriterIssuedCommitReceipt,
    ) -> None:
        try:
            intact = (
                type(receipt) is WriterIssuedCommitReceipt
                and type(receipt.commit) is TraceOperationCommit
                and receipt.receipt_id == expected_receipt_id
                and canonical_digest(receipt.commit) == expected_receipt_id
            )
        except BaseException as exc:
            error = TraceCommitIndeterminateError(
                "writer-issued receipt changed after durable commit"
            )
            self._poison(operation_id, error)
            if not isinstance(exc, Exception):
                raise
            raise error from exc
        if not intact:
            error = TraceCommitIndeterminateError(
                "writer-issued receipt changed after durable commit"
            )
            self._poison(operation_id, error)
            raise error

    def _record_id(
        self,
        operation_id: str,
        record_type: TraceRecordType,
        local_index: int,
        payload: object,
    ) -> str:
        hasher = sha256()
        for value in (
            self._envelope.trace_id,
            operation_id,
            record_type.value,
            str(local_index),
        ):
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\x00")
        hasher.update(canonical_json(payload))
        return f"c1-{hasher.hexdigest()}"

    def _record(
        self,
        *,
        operation_id: str,
        local_index: int,
        record_type: TraceRecordType,
        parent_record_id: str | None,
        observation_id: str | None,
        payload: object,
    ) -> TraceRecord:
        return TraceRecord(
            schema_version=TRACE_SCHEMA_VERSION,
            record_type=record_type,
            trace_id=self._envelope.trace_id,
            run_id=self._envelope.run_id,
            schedule_id=self._envelope.schedule_id,
            schedule_case_id=self._envelope.schedule_case_id,
            sequence=self._next_sequence + local_index,
            record_id=self._record_id(
                operation_id,
                record_type,
                local_index,
                payload,
            ),
            parent_record_id=parent_record_id,
            observation_id=observation_id,
            payload=payload,  # type: ignore[arg-type]
        )

    def _commit_records(
        self,
        *,
        kind: TraceOperationKind,
        operation_id: str,
        request_digest: str,
        records: tuple[TraceRecord, ...],
        runtime_event_count: int,
        runtime_view_digest: str,
        apply_committed: Callable[[tuple[TraceRecord, ...]], None],
    ) -> WriterIssuedCommitReceipt:
        self._guard_active()
        _require_int("runtime_event_count", runtime_event_count)
        require_sha256("runtime_view_digest", runtime_view_digest)
        if not records:
            raise TraceValidationError("typed trace operation produced no records")
        canonical_records = tuple(
            parse_trace_record(encode_trace_record(record)) for record in records
        )
        encoded = b"".join(
            encode_trace_record(record) + b"\n" for record in canonical_records
        )
        prior_stream_digest = self._stream_hasher.hexdigest()
        staged_hasher = self._stream_hasher.copy()
        staged_hasher.update(encoded)
        commit = TraceOperationCommit(
            trace_pair_id=self.trace_pair_id,
            kind=kind,
            operation_id=operation_id,
            request_digest=request_digest,
            commit_index=len(self._commits),
            sequence_start=self._next_sequence,
            sequence_end=self._next_sequence + len(canonical_records),
            record_ids=tuple(record.record_id for record in canonical_records),
            byte_start=self._bytes_written,
            byte_end=self._bytes_written + len(encoded),
            batch_digest=sha256(encoded).hexdigest(),
            prior_stream_digest=prior_stream_digest,
            committed_stream_digest=staged_hasher.hexdigest(),
            runtime_event_count=runtime_event_count,
            runtime_view_digest=runtime_view_digest,
        )
        receipt = WriterIssuedCommitReceipt(
            commit=deepcopy(commit),
            receipt_id=canonical_digest(commit),
        )
        write_attempted = False
        try:
            write_attempted = True
            base_receipt = self._writer.append_durable(
                canonical_records,
                event_count=runtime_event_count,
                view_digest=runtime_view_digest,
            )
            if (
                base_receipt.record_ids != commit.record_ids
                or base_receipt.batch_digest != commit.batch_digest
                or base_receipt.event_count != runtime_event_count
                or base_receipt.view_digest != runtime_view_digest
            ):
                raise TraceCommitIndeterminateError(
                    "raw writer receipt differs from the typed operation"
                )
            self._operations[operation_id] = (
                request_digest,
                receipt.receipt_id,
                receipt,
            )
            self._commits.append(commit)
            self._records.extend(canonical_records)
            self._stream_hasher = staged_hasher
            self._bytes_written += len(encoded)
            self._next_sequence += len(canonical_records)
            apply_committed(canonical_records)
        except BaseException as exc:
            if write_attempted or self._writer.poisoned:
                self._poison(operation_id, exc)
            raise
        return receipt

    def verify_receipt(
        self,
        receipt: WriterIssuedCommitReceipt,
        *,
        kind: TraceOperationKind,
        operation_id: str,
        runtime_event_count: int,
        runtime_view_digest: str,
    ) -> None:
        """Accept only the exact receipt object issued by this committer."""

        with self._lock:
            self._verify_receipt_locked(
                receipt,
                kind=kind,
                operation_id=operation_id,
                runtime_event_count=runtime_event_count,
                runtime_view_digest=runtime_view_digest,
            )

    def _verify_receipt_locked(
        self,
        receipt: WriterIssuedCommitReceipt,
        *,
        kind: TraceOperationKind,
        operation_id: str,
        runtime_event_count: int,
        runtime_view_digest: str,
    ) -> None:
        cached = self._operations.get(operation_id)
        if cached is None or cached[2] is not receipt:
            raise TraceValidationError("trace receipt was not issued by this committer")
        request_digest, expected_receipt_id, _ = cached
        self._assert_cached_receipt_integrity(
            operation_id,
            expected_receipt_id,
            receipt,
        )
        commit = receipt.commit
        if (
            commit.request_digest != request_digest
            or commit.operation_id != operation_id
            or commit.trace_pair_id != self.trace_pair_id
        ):
            error = TraceCommitIndeterminateError(
                "writer-issued receipt changed after durable commit"
            )
            self._poison(operation_id, error)
            raise error
        if (
            commit.kind != kind
            or commit.runtime_event_count != runtime_event_count
            or commit.runtime_view_digest != runtime_view_digest
        ):
            raise TraceValidationError(
                "trace receipt differs from its runtime operation"
            )

    def commit_preamble(
        self,
        request: TracePreambleRequest,
    ) -> WriterIssuedCommitReceipt:
        with self._lock:
            snapshot = self._canonical_snapshot(
                request,
                TracePreambleRequest,
                "trace preamble request",
            )
            return self._commit_preamble_locked(snapshot)

    def _commit_preamble_locked(
        self,
        request: TracePreambleRequest,
    ) -> WriterIssuedCommitReceipt:
        self._guard_active()
        request_digest = self._request_digest(request)
        cached = self._cached(
            kind=TraceOperationKind.PREAMBLE,
            operation_id=request.operation_id,
            request_digest=request_digest,
        )
        if cached is not None:
            return cached
        if self._records:
            raise TraceValidationError("trace preamble is already committed")
        if (
            request.header.trace_pair_id != self._schedule.trace_pair_id
            or request.header.schedule_digest != self._schedule_artifact_digest
            or request.header.source_digest != self._schedule.source_artifact_digest
        ):
            raise TraceValidationError("trace header differs from frozen schedule")
        for topology in request.topologies:
            if (
                topology.branch_grammar_digest != request.header.branch_grammar_digest
                or topology.source_case_digest != self._schedule.source_case_digest
            ):
                raise TraceValidationError(
                    "trace topology differs from header or schedule source"
                )
        header = self._record(
            operation_id=request.operation_id,
            local_index=0,
            record_type=TraceRecordType.TRACE_HEADER,
            parent_record_id=None,
            observation_id=None,
            payload=request.header,
        )
        records = [header]
        for index, topology in enumerate(request.topologies, start=1):
            records.append(
                self._record(
                    operation_id=request.operation_id,
                    local_index=index,
                    record_type=TraceRecordType.WORKFLOW_TOPOLOGY,
                    parent_record_id=header.record_id,
                    observation_id=None,
                    payload=topology,
                )
            )
        receipt = self._commit_records(
            kind=TraceOperationKind.PREAMBLE,
            operation_id=request.operation_id,
            request_digest=request_digest,
            records=tuple(records),
            runtime_event_count=0,
            runtime_view_digest=canonical_digest(request),
            apply_committed=self._apply_preamble,
        )
        return receipt

    def _apply_preamble(
        self,
        records: tuple[TraceRecord, ...],
    ) -> None:
        header = records[0]
        if not isinstance(header.payload, TraceHeaderPayload):
            raise TraceValidationError("committed preamble header changed type")
        self._header_payload = header.payload
        self._header_record_id = header.record_id
        for topology, record in zip(
            (item.payload for item in records[1:]),
            records[1:],
            strict=True,
        ):
            if not isinstance(topology, WorkflowTopologyPayload):
                raise TraceValidationError("committed topology changed type")
            workflow = topology.workflow_spec.key
            self._topology_payloads[workflow] = topology
            self._topology_record_ids[workflow] = record.record_id

    def commit_cutoff(
        self,
        request: CutoffCommitRequest,
        view: AtomicCutoffView,
    ) -> WriterIssuedCommitReceipt:
        with self._lock:
            request_snapshot = self._canonical_snapshot(
                request,
                CutoffCommitRequest,
                "trace cutoff request",
            )
            view_snapshot = self._canonical_snapshot(
                view,
                AtomicCutoffView,
                "trace cutoff runtime view",
            )
            return self._commit_cutoff_locked(request_snapshot, view_snapshot)

    def _commit_cutoff_locked(
        self,
        request: CutoffCommitRequest,
        view: AtomicCutoffView,
    ) -> WriterIssuedCommitReceipt:
        self._guard_active()
        request_digest = self._request_digest(request, view.view_digest)
        cached = self._cached(
            kind=TraceOperationKind.CUTOFF_ATTEMPT,
            operation_id=request.operation_id,
            request_digest=request_digest,
        )
        if cached is not None:
            return cached
        if self._header_record_id is None:
            raise TraceValidationError("cutoff precedes the trace preamble")
        if (
            request.observation_id in self._observation_heads
            or request.observation_id in self._closed_observations
        ):
            raise TraceValidationError("observation cutoff is already committed")
        topology_ids: list[str] = []
        for spec in view.owner_specs:
            topology = self._topology_payloads.get(spec.key)
            if topology is None or topology.workflow_spec != spec:
                raise TraceValidationError(
                    "atomic cutoff owner topology differs from the trace preamble"
                )
            topology_ids.append(self._topology_record_ids[spec.key])
        context = request.attempt.context
        last_event = view.lifecycle_prefix[-1] if view.lifecycle_prefix else None
        cutoff_payload = CutoffPayload(
            topology_record_ids=tuple(sorted(topology_ids)),
            snapshot=view.snapshot,
            cutoff_ns=view.cutoff_ns,
            horizon_duration_ns=view.horizon_duration_ns,
            deadline_ns=view.deadline_ns,
            lifecycle_event_count=len(view.lifecycle_prefix),
            last_event_id=last_event.event_id if last_event is not None else None,
            last_event_timestamp_ns=(
                last_event.timestamp_ns if last_event is not None else None
            ),
            atomic_cutoff_view_digest=view.view_digest,
            feature_view_digest=context.feature_view_digest,
        )
        cutoff = self._record(
            operation_id=request.operation_id,
            local_index=0,
            record_type=TraceRecordType.CUTOFF,
            parent_record_id=self._header_record_id,
            observation_id=request.observation_id,
            payload=cutoff_payload,
        )
        attempt = self._record(
            operation_id=request.operation_id,
            local_index=1,
            record_type=TraceRecordType.FORECAST_ATTEMPT,
            parent_record_id=cutoff.record_id,
            observation_id=request.observation_id,
            payload=request.attempt,
        )
        receipt = self._commit_records(
            kind=TraceOperationKind.CUTOFF_ATTEMPT,
            operation_id=request.operation_id,
            request_digest=request_digest,
            records=(cutoff, attempt),
            runtime_event_count=len(view.lifecycle_prefix),
            runtime_view_digest=view.view_digest,
            apply_committed=lambda committed_records: self._apply_cutoff(
                request.observation_id,
                committed_records,
            ),
        )
        return receipt

    def _apply_cutoff(
        self,
        observation_id: str,
        records: tuple[TraceRecord, ...],
    ) -> None:
        cutoff_payload = records[0].payload
        if not isinstance(cutoff_payload, CutoffPayload):
            raise TraceValidationError("committed cutoff changed type")
        self._observation_heads[observation_id] = records[-1].record_id
        self._observation_cutoffs[observation_id] = cutoff_payload
        self._observation_intents[observation_id] = {}

    def commit_demands(
        self,
        request: DemandCommitRequest,
        view: PreServiceDemandView,
    ) -> WriterIssuedCommitReceipt:
        with self._lock:
            request_snapshot = self._canonical_snapshot(
                request,
                DemandCommitRequest,
                "trace demand request",
            )
            view_snapshot = self._canonical_snapshot(
                view,
                PreServiceDemandView,
                "trace demand runtime view",
            )
            return self._commit_demands_locked(request_snapshot, view_snapshot)

    def _commit_demands_locked(
        self,
        request: DemandCommitRequest,
        view: PreServiceDemandView,
    ) -> WriterIssuedCommitReceipt:
        self._guard_active()
        request_digest = self._request_digest(request, view.view_digest)
        cached = self._cached(
            kind=TraceOperationKind.DEMAND_INTENT,
            operation_id=request.operation_id,
            request_digest=request_digest,
        )
        if cached is not None:
            return cached
        if request.operation_id != view.demand_commit_id:
            raise TraceValidationError(
                "demand operation differs from runtime commit ID"
            )
        head = self._observation_heads.get(request.observation_id)
        if head is None or request.observation_id in self._closed_observations:
            raise TraceValidationError("demand lacks an open committed observation")
        try:
            schedule_events = tuple(
                self._event_by_id[event_id] for event_id in request.schedule_event_ids
            )
        except KeyError as exc:
            raise TraceValidationError(
                "demand names an unknown schedule event"
            ) from exc
        waiter_by_binding = {waiter.binding_id: waiter for waiter in view.waiters}
        schedule_binding_ids = tuple(
            sorted(event.request_binding_id for event in schedule_events)
        )
        if len(schedule_binding_ids) != len(set(schedule_binding_ids)) or set(
            schedule_binding_ids
        ) != set(waiter_by_binding):
            raise TraceValidationError(
                "frozen schedule events and runtime waiters are not bijective"
            )
        existing = self._observation_intents[request.observation_id]
        records: list[TraceRecord] = []
        current_parent = head
        for local_index, event in enumerate(schedule_events):
            waiter = waiter_by_binding[event.request_binding_id]
            self._validate_schedule_waiter(event, waiter, view)
            if event.schedule_event_id in existing:
                raise TraceValidationError(
                    "schedule event already has a demand intent in this observation"
                )
            payload = DemandIntentPayload(
                schedule_event_id=event.schedule_event_id,
                scheduled_access_ns=event.scheduled_access_ns,
                claim_id=event.claim_id,
                retention_binding_id=event.retention_binding_id,
                request_binding_id=event.request_binding_id,
                workflow=event.workflow,
                node_id=event.node_id,
                execution_ref=event.execution_ref,
                block_key=event.block_key,
                reuse_epoch_id=event.reuse_epoch_id,
                pre_service_event_count=view.runtime_event_count,
                pre_service_last_event_id=view.last_event_id,
                pre_service_last_timestamp_ns=view.last_event_timestamp_ns,
            )
            record = self._record(
                operation_id=request.operation_id,
                local_index=local_index,
                record_type=TraceRecordType.DEMAND_INTENT,
                parent_record_id=current_parent,
                observation_id=request.observation_id,
                payload=payload,
            )
            records.append(record)
            current_parent = record.record_id
        receipt = self._commit_records(
            kind=TraceOperationKind.DEMAND_INTENT,
            operation_id=request.operation_id,
            request_digest=request_digest,
            records=tuple(records),
            runtime_event_count=view.runtime_event_count,
            runtime_view_digest=view.view_digest,
            apply_committed=lambda committed_records: self._apply_demands(
                request.observation_id,
                schedule_events,
                committed_records,
            ),
        )
        return receipt

    def _apply_demands(
        self,
        observation_id: str,
        schedule_events: tuple[ScheduleDemandEvent, ...],
        records: tuple[TraceRecord, ...],
    ) -> None:
        self._observation_heads[observation_id] = records[-1].record_id
        existing = self._observation_intents[observation_id]
        for event, record in zip(schedule_events, records, strict=True):
            existing[event.schedule_event_id] = record

    @staticmethod
    def _validate_schedule_waiter(
        event: ScheduleDemandEvent,
        waiter: WaiterIdentity,
        view: PreServiceDemandView,
    ) -> None:
        if (
            event.scheduled_access_ns != view.timestamp_ns
            or event.block_key != view.block_key
            or event.workflow != waiter.workflow
            or event.request_binding_id != waiter.binding_id
            or event.node_id != waiter.node_id
            or event.execution_ref != waiter.execution_ref
        ):
            raise TraceValidationError(
                "frozen schedule event differs from its runtime waiter"
            )

    def _validate_watermark(self, watermark: ScheduleWatermarkPayload) -> None:
        checkpoint_by_id = {
            checkpoint.checkpoint_id: checkpoint
            for checkpoint in self._schedule.checkpoints
        }
        try:
            checkpoint = checkpoint_by_id[watermark.checkpoint_id]
        except KeyError as exc:
            raise TraceValidationError(
                "watermark names an unknown frozen checkpoint"
            ) from exc
        common = {
            "producer_id": self._schedule.producer_id,
            "schedule_digest": self._schedule_artifact_digest,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_digest": canonical_digest(checkpoint),
            "consumed_event_count": checkpoint.consumed_event_count,
            "last_schedule_event_id": checkpoint.last_schedule_event_id,
            "max_closed_timestamp_ns": checkpoint.closed_through_ns,
            "event_prefix_digest": checkpoint.event_prefix_digest,
            "closed_epoch_count": checkpoint.closed_epoch_count,
            "epoch_prefix_digest": checkpoint.epoch_prefix_digest,
        }
        closure = self._schedule.closure
        if isinstance(closure, ReplayScheduleClosure):
            expected: ScheduleWatermarkPayload = ReplayScheduleWatermarkPayload(
                producer_kind=self._schedule.producer_kind,
                producer_artifact_digest=closure.plan_event_digest,
                **common,
            )
        elif isinstance(closure, NaturalScheduleClosure):
            expected = NaturalTraceWatermarkPayload(
                producer_kind=self._schedule.producer_kind,
                producer_artifact_digest=closure.source_eof_digest,
                source_eof_record_count=closure.source_eof_record_count,
                source_eof_digest=closure.source_eof_digest,
                capture_start_ns=closure.capture_start_ns,
                capture_end_ns=closure.capture_end_ns,
                dropped_record_count=closure.dropped_record_count,
                clean_eof=closure.clean_eof,
                **common,
            )
        else:  # pragma: no cover - schedule construction closes this union
            raise TraceValidationError("frozen schedule has an unknown closure")
        if watermark != expected:
            raise TraceValidationError(
                "watermark differs from its exact frozen schedule checkpoint"
            )

    def close_observation(
        self,
        request: ObservationCloseRequest,
    ) -> WriterIssuedCommitReceipt:
        with self._lock:
            snapshot = self._canonical_snapshot(
                request,
                ObservationCloseRequest,
                "trace observation close request",
            )
            return self._close_observation_locked(snapshot)

    def _close_observation_locked(
        self,
        request: ObservationCloseRequest,
    ) -> WriterIssuedCommitReceipt:
        self._guard_active()
        request_digest = self._request_digest(request)
        cached = self._cached(
            kind=TraceOperationKind.OBSERVATION_CLOSE,
            operation_id=request.operation_id,
            request_digest=request_digest,
        )
        if cached is not None:
            return cached
        head = self._observation_heads.get(request.observation_id)
        if head is None or request.observation_id in self._closed_observations:
            raise TraceValidationError("close request lacks an open observation")
        cutoff = self._observation_cutoffs[request.observation_id]
        intents_by_schedule = self._observation_intents[request.observation_id]
        intents_by_record = {
            record.record_id: record for record in intents_by_schedule.values()
        }
        services = {service.intent_record_id: service for service in request.services}
        if not set(services).issubset(intents_by_record):
            raise TraceValidationError("close service names another observation")
        if request.watermark is not None:
            self._validate_watermark(request.watermark)
        if request.terminal.status == TerminalStatus.COMPLETE:
            expected_schedule_ids = {
                event.schedule_event_id
                for event in self._schedule.events
                if self._eligible_schedule_event(event, cutoff)
            }
            if set(intents_by_schedule) != expected_schedule_ids:
                raise TraceValidationError(
                    "complete observation omits or adds a frozen schedule demand"
                )
            if set(services) != set(intents_by_record):
                raise TraceValidationError(
                    "complete observation lacks a service for every demand"
                )
            if request.watermark is None:
                raise TraceValidationError("complete observation lacks a watermark")

        records: list[TraceRecord] = []
        current_parent = head
        local_index = 0
        serviced_record_ids: set[str] = set()
        for epoch in self._schedule.epochs:
            relevant_ids = tuple(
                sorted(
                    intents_by_schedule[event_id].record_id
                    for event_id in epoch.schedule_event_ids
                    if event_id in intents_by_schedule
                )
            )
            if not relevant_ids:
                continue
            if set(epoch.schedule_event_ids) != {
                event_id
                for event_id in epoch.schedule_event_ids
                if event_id in intents_by_schedule
            }:
                raise TraceValidationError(
                    "observation only partially intersects a frozen reuse epoch"
                )
            present_services = set(relevant_ids) & set(services)
            if not present_services:
                continue
            if present_services != set(relevant_ids):
                raise TraceValidationError(
                    "reuse epoch has only a partial set of service terminals"
                )
            payload = ReuseEpochPayload(
                reuse_epoch_id=epoch.reuse_epoch_id,
                access_ns=epoch.access_ns,
                block_key=self._event_by_id[epoch.schedule_event_ids[0]].block_key,
                demand_intent_record_ids=relevant_ids,
                service_terminals=tuple(services[item] for item in relevant_ids),
            )
            record = self._record(
                operation_id=request.operation_id,
                local_index=local_index,
                record_type=TraceRecordType.REUSE_EPOCH,
                parent_record_id=current_parent,
                observation_id=request.observation_id,
                payload=payload,
            )
            records.append(record)
            current_parent = record.record_id
            local_index += 1
            serviced_record_ids.update(relevant_ids)
        if serviced_record_ids != set(services):
            raise TraceValidationError("service terminals do not form complete epochs")

        watermark_record_id: str | None = None
        if request.watermark is not None:
            watermark_record = self._record(
                operation_id=request.operation_id,
                local_index=local_index,
                record_type=TraceRecordType.SCHEDULE_WATERMARK,
                parent_record_id=current_parent,
                observation_id=request.observation_id,
                payload=request.watermark,
            )
            records.append(watermark_record)
            current_parent = watermark_record.record_id
            watermark_record_id = watermark_record.record_id
            local_index += 1
        unresolved = tuple(sorted(set(intents_by_record) - serviced_record_ids))
        terminal_payload = ObservationTerminalPayload(
            status=request.terminal.status,
            reason=request.terminal.reason,
            label_available_ns=request.terminal.label_available_ns,
            schedule_watermark_record_id=watermark_record_id,
            last_verified_event_count=request.terminal.last_verified_event_count,
            last_verified_event_id=request.terminal.last_verified_event_id,
            last_verified_event_timestamp_ns=(
                request.terminal.last_verified_event_timestamp_ns
            ),
            unresolved_demand_intent_record_ids=unresolved,
        )
        terminal_record = self._record(
            operation_id=request.operation_id,
            local_index=local_index,
            record_type=TraceRecordType.OBSERVATION_TERMINAL,
            parent_record_id=current_parent,
            observation_id=request.observation_id,
            payload=terminal_payload,
        )
        records.append(terminal_record)
        receipt = self._commit_records(
            kind=TraceOperationKind.OBSERVATION_CLOSE,
            operation_id=request.operation_id,
            request_digest=request_digest,
            records=tuple(records),
            runtime_event_count=request.terminal.last_verified_event_count,
            runtime_view_digest=canonical_digest(request),
            apply_committed=lambda committed_records: self._apply_observation_close(
                request.observation_id,
                committed_records[-1].record_id,
            ),
        )
        return receipt

    def _apply_observation_close(
        self,
        observation_id: str,
        terminal_record_id: str,
    ) -> None:
        self._observation_heads[observation_id] = terminal_record_id
        self._closed_observations.add(observation_id)

    @staticmethod
    def _eligible_schedule_event(
        event: ScheduleDemandEvent,
        cutoff: CutoffPayload,
    ) -> bool:
        owners = {owner.binding_id: owner for owner in cutoff.snapshot.owners}
        owner = owners.get(event.retention_binding_id)
        return (
            cutoff.cutoff_ns < event.scheduled_access_ns <= cutoff.deadline_ns
            and event.block_key == cutoff.snapshot.block_key
            and owner is not None
            and owner.workflow == event.workflow
            and event.node_id in owner.eligible_node_ids
        )

    def seal_trace(self) -> SealedTraceReceipt:
        """Finalize the trace after every opened observation has a terminal."""

        with self._lock:
            if self._poisoned_reason is not None:
                raise TraceCommitIndeterminateError(
                    f"typed trace committer is poisoned: {self._poisoned_reason}"
                )
            if self._sealed is not None:
                self._assert_sealed_receipt_integrity(self._sealed)
                return self._sealed
            return self._seal_trace_locked()

    def _assert_sealed_receipt_integrity(
        self,
        receipt: SealedTraceReceipt,
    ) -> None:
        try:
            intact = (
                self._sealed is receipt
                and self._sealed_receipt_digest is not None
                and type(receipt) is SealedTraceReceipt
                and canonical_digest(receipt) == self._sealed_receipt_digest
            )
        except BaseException as exc:
            self._poison("sealed receipt", exc)
            if not isinstance(exc, Exception):
                raise
            raise TraceCommitIndeterminateError(
                "sealed trace receipt changed after publication"
            ) from exc
        if not intact:
            error = TraceCommitIndeterminateError(
                "sealed trace receipt changed after publication"
            )
            self._poison("sealed receipt", error)
            raise error
        try:
            records = load_trace_jsonl(self._path)
            closure = receipt.closure
            file_intact = (
                len(records) == closure.record_count
                and records[0].record_id == closure.first_record_id
                and records[-1].record_id == closure.last_record_id
                and trace_stream_digest(records) == closure.stream_digest
                and sum(len(encode_trace_record(record)) + 1 for record in records)
                == closure.size_bytes
            )
        except BaseException as exc:
            self._poison("sealed trace file", exc)
            if not isinstance(exc, Exception):
                raise
            raise TraceCommitIndeterminateError(
                "sealed trace file changed after publication"
            ) from exc
        if not file_intact:
            error = TraceCommitIndeterminateError(
                "sealed trace file changed after publication"
            )
            self._poison("sealed trace file", error)
            raise error

    def snapshot_sealed_receipt(
        self,
        receipt: SealedTraceReceipt,
    ) -> SealedTraceReceipt:
        """Return a detached canonical copy of this committer's final receipt."""

        with self._lock:
            self._assert_sealed_receipt_integrity(receipt)
            raw = canonical_json(receipt)
            snapshot = parse_canonical_dataclass(
                raw,
                SealedTraceReceipt,
                artifact_name="sealed trace receipt",
                max_bytes=64 * 1024 * 1024,
            )
            self._assert_sealed_receipt_integrity(receipt)
            return snapshot

    def _seal_trace_locked(self) -> SealedTraceReceipt:
        self._guard_active()
        if (
            sha256(canonical_json(self._schedule)).hexdigest()
            != self._schedule_artifact_digest
        ):
            error = TraceCommitIndeterminateError(
                "frozen schedule changed before trace seal"
            )
            self._poison("seal_trace", error)
            raise error
        if len(self._operations) != len(self._commits):
            raise TraceCommitIndeterminateError(
                "trace operation cache differs from its commit chain"
            )
        for commit in self._commits:
            try:
                _, expected_receipt_id, receipt = self._operations[commit.operation_id]
            except KeyError as exc:
                self._poison("seal_trace", exc)
                raise TraceCommitIndeterminateError(
                    "trace commit lacks its writer-issued receipt"
                ) from exc
            self._assert_cached_receipt_integrity(
                commit.operation_id,
                expected_receipt_id,
                receipt,
            )
            if canonical_digest(commit) != expected_receipt_id:
                error = TraceCommitIndeterminateError(
                    "internal trace commit differs from its issued receipt"
                )
                self._poison("seal_trace", error)
                raise error
        if not self._observation_heads:
            raise TraceValidationError("trace has no observations")
        open_observations = sorted(
            set(self._observation_heads) - self._closed_observations
        )
        if open_observations:
            raise TraceValidationError(
                f"trace has open observations: {open_observations}"
            )
        seal_attempted = False
        try:
            seal_attempted = True
            closure = self._writer.seal()
            sealed = SealedTraceReceipt(
                trace_pair_id=self.trace_pair_id,
                trace_basename=self._path.name,
                closure=closure,
                operations=deepcopy(tuple(self._commits)),
            )
            self._writer.close()
            self._sealed = sealed
            self._sealed_receipt_digest = canonical_digest(sealed)
            self._assert_sealed_receipt_integrity(sealed)
        except BaseException as exc:
            if seal_attempted or self._writer.poisoned or self._writer.sealed:
                self._poison("seal_trace", exc)
            raise
        return sealed

    def abort(self) -> None:
        """Close an incomplete attempt without making it resumable or valid."""

        with self._lock:
            if self._sealed is not None:
                return
            if self._poisoned_reason is None:
                self._poisoned_reason = "trace attempt aborted before final seal"
            try:
                self._writer.close(finalize=False)
            except BaseException as exc:
                self._poisoned_reason = (
                    f"{self._poisoned_reason}; abort close failed: "
                    f"{_safe_failure_detail('trace close', exc)}"
                )
                raise
