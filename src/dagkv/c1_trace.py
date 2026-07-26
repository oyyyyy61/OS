"""Fail-closed C1-B trace records and durable append-only storage."""

from __future__ import annotations

import json
import os
import stat
import types
import weakref
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, get_args, get_origin, get_type_hints

from dagkv.c1_leases import (
    ForecastSource,
    SharedLeaseForecast,
    SharedLeasePolicySnapshot,
    aggregate_shared_lease,
)
from dagkv.domain import (
    BindingState,
    BlockKey,
    DAGKVError,
    ExecutionRef,
    IdentityError,
    LedgerAction,
    LifecycleEvent,
    ReplicaId,
    ResidencyState,
    Tier,
    WorkflowKey,
    WorkflowSpec,
    require_sha256,
    require_text,
)

TRACE_SCHEMA_VERSION = "dagkv.m3.c1_trace.v2"
MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_TRACE_LINE_BYTES = 4 * 1024 * 1024


class TraceValidationError(DAGKVError, ValueError):
    """Raised when a C1 trace violates its closed schema or state machine."""


class TraceDurabilityError(DAGKVError, OSError):
    """Raised when a trace batch is known not to have been committed."""


class TraceCommitIndeterminateError(TraceDurabilityError):
    """Raised when a failed write may have reached durable storage."""


class _FailFastLock:
    """Reject accidental multi-writer use instead of hiding ordering races."""

    def __init__(self) -> None:
        self._lock = Lock()

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise TraceDurabilityError("concurrent trace writer operation rejected")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._lock.release()


class TraceRecordType(StrEnum):
    TRACE_HEADER = "trace_header"
    WORKFLOW_TOPOLOGY = "workflow_topology"
    CUTOFF = "cutoff"
    FORECAST_ATTEMPT = "forecast_attempt"
    DEMAND_INTENT = "demand_intent"
    REUSE_EPOCH = "reuse_epoch"
    SCHEDULE_WATERMARK = "schedule_watermark"
    OBSERVATION_TERMINAL = "observation_terminal"


class ForecastAttemptStatus(StrEnum):
    PREDICTED = "PREDICTED"
    ABSTAINED = "ABSTAINED"


class AbstentionReason(StrEnum):
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    UNKNOWN_BRANCH_GRAMMAR = "UNKNOWN_BRANCH_GRAMMAR"
    SUPPORT_CAP_EXCEEDED = "SUPPORT_CAP_EXCEEDED"
    OOD_CONTEXT = "OOD_CONTEXT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_CATALOG = "INVALID_CATALOG"


class ServiceDisposition(StrEnum):
    RESIDENT_EXEC_MAP = "RESIDENT_EXEC_MAP"
    H2D_EXEC_MAP = "H2D_EXEC_MAP"
    H2D_FAILED = "H2D_FAILED"
    H2D_CANCELLED = "H2D_CANCELLED"
    REQUEST_CANCELLED = "REQUEST_CANCELLED"


class ScheduleProducerKind(StrEnum):
    REPLAY = "REPLAY"
    SEALED_NATURAL_TRACE = "SEALED_NATURAL_TRACE"


class TerminalStatus(StrEnum):
    COMPLETE = "COMPLETE"
    CENSORED = "CENSORED"
    UNIDENTIFIABLE = "UNIDENTIFIABLE"
    INVALID = "INVALID"


class TerminalReason(StrEnum):
    WINDOW_COMPLETE = "WINDOW_COMPLETE"
    TRACE_TRUNCATED = "TRACE_TRUNCATED"
    INSTRUMENTATION_FAILURE = "INSTRUMENTATION_FAILURE"
    MISSING_CANCELLATION = "MISSING_CANCELLATION"
    POLICY_CHANGED_SCHEDULE = "POLICY_CHANGED_SCHEDULE"
    MISSING_DEMAND_INTENT = "MISSING_DEMAND_INTENT"
    AMBIGUOUS_OWNER = "AMBIGUOUS_OWNER"
    UNRESOLVED_EPOCH = "UNRESOLVED_EPOCH"
    UNKNOWN_BRANCH_GRAMMAR = "UNKNOWN_BRANCH_GRAMMAR"
    PRE_SERVICE_ORDER_VIOLATION = "PRE_SERVICE_ORDER_VIOLATION"
    SUPPORT_VIOLATION = "SUPPORT_VIOLATION"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"


class EvidenceRole(StrEnum):
    LIFECYCLE = "LIFECYCLE"
    SCHEDULE = "SCHEDULE"


_CENSORED_REASONS = frozenset(
    {
        TerminalReason.TRACE_TRUNCATED,
        TerminalReason.INSTRUMENTATION_FAILURE,
        TerminalReason.MISSING_CANCELLATION,
        TerminalReason.POLICY_CHANGED_SCHEDULE,
    }
)
_UNIDENTIFIABLE_REASONS = frozenset(
    {
        TerminalReason.MISSING_DEMAND_INTENT,
        TerminalReason.AMBIGUOUS_OWNER,
        TerminalReason.UNRESOLVED_EPOCH,
        TerminalReason.UNKNOWN_BRANCH_GRAMMAR,
    }
)
_INVALID_REASONS = frozenset(
    {
        TerminalReason.PRE_SERVICE_ORDER_VIOLATION,
        TerminalReason.SUPPORT_VIOLATION,
        TerminalReason.INVALID_PROVENANCE,
    }
)


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None:
        require_text(name, value)


def _require_enum(name: str, value: Any, expected: type[StrEnum]) -> None:
    if type(value) is not expected:
        raise TraceValidationError(f"{name} must be a {expected.__name__}")


def _require_sorted_unique(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TraceValidationError(f"{name} must be a tuple")
    for value in values:
        require_text(name, value)
    if tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


def _require_unique(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TraceValidationError(f"{name} must be a tuple")
    for value in values:
        require_text(name, value)
    if len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be unique")


def _validate_lifecycle_prefix(prefix: tuple[LifecycleEvent, ...]) -> None:
    if not isinstance(prefix, tuple):
        raise TraceValidationError("lifecycle prefix must be a tuple")
    envelope: tuple[str, str, str] | None = None
    event_ids: set[str] = set()
    last_timestamp_ns = -1
    for sequence, event in enumerate(prefix):
        if type(event) is not LifecycleEvent:
            raise TraceValidationError("lifecycle prefix contains a non-event value")
        if event.sequence != sequence:
            raise TraceValidationError("lifecycle prefix sequence is not contiguous")
        if event.event_id in event_ids:
            raise TraceValidationError("lifecycle prefix event ID is duplicated")
        event_ids.add(event.event_id)
        current_envelope = (event.run_id, event.phase, event.source)
        if envelope is None:
            envelope = current_envelope
        elif current_envelope != envelope:
            raise TraceValidationError("lifecycle prefix envelope changes")
        if event.timestamp_ns < last_timestamp_ns:
            raise TraceValidationError("lifecycle prefix timestamps regress")
        last_timestamp_ns = event.timestamp_ns


@dataclass(frozen=True, slots=True)
class TraceHeaderPayload:
    trace_pair_id: str
    source_digest: str
    schedule_digest: str
    split_manifest_digest: str
    branch_grammar_digest: str
    feature_contract_digest: str
    implementation_digest: str
    environment_digest: str

    def __post_init__(self) -> None:
        require_text("trace_pair_id", self.trace_pair_id)
        for name in (
            "source_digest",
            "schedule_digest",
            "split_manifest_digest",
            "branch_grammar_digest",
            "feature_contract_digest",
            "implementation_digest",
            "environment_digest",
        ):
            require_sha256(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class WorkflowTopologyPayload:
    workflow_spec: WorkflowSpec
    workflow_template_digest: str
    source_case_digest: str
    split_component_id: str
    branch_grammar_digest: str

    def __post_init__(self) -> None:
        require_sha256("workflow_template_digest", self.workflow_template_digest)
        require_sha256("source_case_digest", self.source_case_digest)
        require_text("split_component_id", self.split_component_id)
        require_sha256("branch_grammar_digest", self.branch_grammar_digest)


@dataclass(frozen=True, slots=True)
class CutoffPayload:
    topology_record_ids: tuple[str, ...]
    snapshot: SharedLeasePolicySnapshot
    cutoff_ns: int
    horizon_duration_ns: int
    deadline_ns: int
    lifecycle_event_count: int
    last_event_id: str | None
    last_event_timestamp_ns: int | None
    atomic_cutoff_view_digest: str
    feature_view_digest: str

    def __post_init__(self) -> None:
        _require_sorted_unique("topology_record_ids", self.topology_record_ids)
        if not self.topology_record_ids:
            raise TraceValidationError("cutoff requires at least one topology record")
        _require_int("cutoff_ns", self.cutoff_ns)
        _require_int("horizon_duration_ns", self.horizon_duration_ns, minimum=1)
        _require_int("deadline_ns", self.deadline_ns)
        if self.deadline_ns != self.cutoff_ns + self.horizon_duration_ns:
            raise TraceValidationError(
                "cutoff deadline differs from cutoff plus horizon"
            )
        _require_int("lifecycle_event_count", self.lifecycle_event_count)
        if self.snapshot.runtime_event_count != self.lifecycle_event_count:
            raise TraceValidationError(
                "cutoff snapshot event count differs from prefix"
            )
        if self.lifecycle_event_count == 0:
            if (
                self.last_event_id is not None
                or self.last_event_timestamp_ns is not None
            ):
                raise TraceValidationError("empty lifecycle prefix has a last event")
        else:
            require_text("last_event_id", self.last_event_id)
            if self.last_event_timestamp_ns is None:
                raise TraceValidationError("non-empty lifecycle prefix lacks timestamp")
            _require_int("last_event_timestamp_ns", self.last_event_timestamp_ns)
            if self.last_event_timestamp_ns > self.cutoff_ns:
                raise TraceValidationError("cutoff predates the lifecycle prefix")
        require_sha256(
            "atomic_cutoff_view_digest",
            self.atomic_cutoff_view_digest,
        )
        require_sha256("feature_view_digest", self.feature_view_digest)


@dataclass(frozen=True, slots=True)
class ForecastAttemptContext:
    feature_view_digest: str
    information_cutoff_digest: str
    model_artifact_digest: str
    predictor_digest: str
    dependence_digest: str
    outcome_catalog_digest: str
    grouping_rules_digest: str
    model_inputs_digest: str

    def __post_init__(self) -> None:
        for dataclass_field in fields(self):
            require_sha256(
                dataclass_field.name,
                getattr(self, dataclass_field.name),
            )


@dataclass(frozen=True, slots=True)
class PredictedAttemptPayload:
    status: ForecastAttemptStatus
    context: ForecastAttemptContext
    forecast: SharedLeaseForecast

    def __post_init__(self) -> None:
        _require_enum("forecast attempt status", self.status, ForecastAttemptStatus)
        if self.status != ForecastAttemptStatus.PREDICTED:
            raise TraceValidationError("predicted attempt has the wrong status")
        if self.forecast.source != ForecastSource.PREDICTED:
            raise TraceValidationError("online trace requires a predicted forecast")
        if self.forecast.predictor_digest != self.context.predictor_digest:
            raise TraceValidationError("forecast predictor digest differs from context")
        if self.forecast.dependence_digest != self.context.dependence_digest:
            raise TraceValidationError(
                "forecast dependence digest differs from context"
            )


@dataclass(frozen=True, slots=True)
class AbstainedAttemptPayload:
    status: ForecastAttemptStatus
    context: ForecastAttemptContext
    reason: AbstentionReason

    def __post_init__(self) -> None:
        _require_enum("forecast attempt status", self.status, ForecastAttemptStatus)
        _require_enum("abstention reason", self.reason, AbstentionReason)
        if self.status != ForecastAttemptStatus.ABSTAINED:
            raise TraceValidationError("abstained attempt has the wrong status")


@dataclass(frozen=True, slots=True)
class DemandIntentPayload:
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
    pre_service_event_count: int
    pre_service_last_event_id: str | None
    pre_service_last_timestamp_ns: int | None

    def __post_init__(self) -> None:
        for name in (
            "schedule_event_id",
            "claim_id",
            "retention_binding_id",
            "request_binding_id",
            "node_id",
            "reuse_epoch_id",
        ):
            require_text(name, getattr(self, name))
        _require_int("scheduled_access_ns", self.scheduled_access_ns)
        _require_int("pre_service_event_count", self.pre_service_event_count)
        if self.execution_ref.workflow != self.workflow:
            raise TraceValidationError("demand execution belongs to another workflow")
        if self.pre_service_event_count == 0:
            if (
                self.pre_service_last_event_id is not None
                or self.pre_service_last_timestamp_ns is not None
            ):
                raise TraceValidationError("empty pre-service prefix has a last event")
        else:
            require_text("pre_service_last_event_id", self.pre_service_last_event_id)
            if self.pre_service_last_timestamp_ns is None:
                raise TraceValidationError("pre-service prefix lacks a timestamp")
            _require_int(
                "pre_service_last_timestamp_ns", self.pre_service_last_timestamp_ns
            )
            if self.pre_service_last_timestamp_ns > self.scheduled_access_ns:
                raise TraceValidationError(
                    "demand intent predates its lifecycle prefix"
                )


@dataclass(frozen=True, slots=True)
class ResidentExecMapService:
    intent_record_id: str
    disposition: ServiceDisposition
    exec_map_event_id: str

    def __post_init__(self) -> None:
        _require_enum("service disposition", self.disposition, ServiceDisposition)
        require_text("intent_record_id", self.intent_record_id)
        require_text("exec_map_event_id", self.exec_map_event_id)
        if self.disposition != ServiceDisposition.RESIDENT_EXEC_MAP:
            raise TraceValidationError("resident service has the wrong disposition")


@dataclass(frozen=True, slots=True)
class H2DExecMapService:
    intent_record_id: str
    disposition: ServiceDisposition
    transfer_id: str
    transfer_scheduled_event_id: str
    transfer_terminal_event_id: str
    exec_map_event_id: str
    waiter_binding_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("service disposition", self.disposition, ServiceDisposition)
        for name in (
            "intent_record_id",
            "transfer_id",
            "transfer_scheduled_event_id",
            "transfer_terminal_event_id",
            "exec_map_event_id",
        ):
            require_text(name, getattr(self, name))
        if self.disposition != ServiceDisposition.H2D_EXEC_MAP:
            raise TraceValidationError("H2D success has the wrong disposition")
        _require_sorted_unique("waiter_binding_ids", self.waiter_binding_ids)
        if not self.waiter_binding_ids:
            raise TraceValidationError("H2D success requires waiter provenance")


@dataclass(frozen=True, slots=True)
class H2DFailedService:
    intent_record_id: str
    disposition: ServiceDisposition
    transfer_id: str
    transfer_scheduled_event_id: str
    transfer_terminal_event_id: str
    waiter_binding_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("service disposition", self.disposition, ServiceDisposition)
        for name in (
            "intent_record_id",
            "transfer_id",
            "transfer_scheduled_event_id",
            "transfer_terminal_event_id",
        ):
            require_text(name, getattr(self, name))
        if self.disposition not in {
            ServiceDisposition.H2D_FAILED,
            ServiceDisposition.H2D_CANCELLED,
        }:
            raise TraceValidationError("H2D failure has the wrong disposition")
        _require_sorted_unique("waiter_binding_ids", self.waiter_binding_ids)
        if not self.waiter_binding_ids:
            raise TraceValidationError("H2D failure requires waiter provenance")


@dataclass(frozen=True, slots=True)
class RequestCancelledService:
    intent_record_id: str
    disposition: ServiceDisposition
    release_event_id: str

    def __post_init__(self) -> None:
        _require_enum("service disposition", self.disposition, ServiceDisposition)
        require_text("intent_record_id", self.intent_record_id)
        require_text("release_event_id", self.release_event_id)
        if self.disposition != ServiceDisposition.REQUEST_CANCELLED:
            raise TraceValidationError("request cancellation has wrong disposition")


ServiceTerminal = (
    ResidentExecMapService
    | H2DExecMapService
    | H2DFailedService
    | RequestCancelledService
)


@dataclass(frozen=True, slots=True)
class ReuseEpochPayload:
    reuse_epoch_id: str
    access_ns: int
    block_key: BlockKey
    demand_intent_record_ids: tuple[str, ...]
    service_terminals: tuple[ServiceTerminal, ...]

    def __post_init__(self) -> None:
        require_text("reuse_epoch_id", self.reuse_epoch_id)
        _require_int("epoch access_ns", self.access_ns)
        _require_sorted_unique(
            "demand_intent_record_ids", self.demand_intent_record_ids
        )
        if not self.demand_intent_record_ids:
            raise TraceValidationError("reuse epoch requires demand intents")
        terminal_ids = tuple(
            terminal.intent_record_id for terminal in self.service_terminals
        )
        if tuple(sorted(terminal_ids)) != terminal_ids:
            raise TraceValidationError("service terminals must be intent-ID ordered")
        if terminal_ids != self.demand_intent_record_ids:
            raise TraceValidationError("service terminals do not close every intent")


@dataclass(frozen=True, slots=True)
class ReplayScheduleWatermarkPayload:
    producer_kind: ScheduleProducerKind
    producer_id: str
    producer_artifact_digest: str
    schedule_digest: str
    checkpoint_id: str
    checkpoint_digest: str
    consumed_event_count: int
    last_schedule_event_id: str | None
    max_closed_timestamp_ns: int
    event_prefix_digest: str
    closed_epoch_count: int
    epoch_prefix_digest: str

    def __post_init__(self) -> None:
        _require_enum(
            "schedule producer kind", self.producer_kind, ScheduleProducerKind
        )
        if self.producer_kind != ScheduleProducerKind.REPLAY:
            raise TraceValidationError("replay watermark has the wrong producer kind")
        require_text("producer_id", self.producer_id)
        require_sha256("producer_artifact_digest", self.producer_artifact_digest)
        require_sha256("schedule_digest", self.schedule_digest)
        require_text("checkpoint_id", self.checkpoint_id)
        require_sha256("checkpoint_digest", self.checkpoint_digest)
        _require_int("consumed_event_count", self.consumed_event_count)
        if self.consumed_event_count == 0:
            if self.last_schedule_event_id is not None:
                raise TraceValidationError("empty schedule prefix has a last event")
        else:
            require_text("last_schedule_event_id", self.last_schedule_event_id)
        _require_int("max_closed_timestamp_ns", self.max_closed_timestamp_ns)
        require_sha256("event_prefix_digest", self.event_prefix_digest)
        _require_int("closed_epoch_count", self.closed_epoch_count)
        require_sha256("epoch_prefix_digest", self.epoch_prefix_digest)


@dataclass(frozen=True, slots=True)
class NaturalTraceWatermarkPayload:
    producer_kind: ScheduleProducerKind
    producer_id: str
    producer_artifact_digest: str
    schedule_digest: str
    checkpoint_id: str
    checkpoint_digest: str
    consumed_event_count: int
    last_schedule_event_id: str | None
    max_closed_timestamp_ns: int
    event_prefix_digest: str
    closed_epoch_count: int
    epoch_prefix_digest: str
    source_eof_record_count: int
    source_eof_digest: str
    capture_start_ns: int
    capture_end_ns: int
    dropped_record_count: int
    clean_eof: bool

    def __post_init__(self) -> None:
        _require_enum(
            "schedule producer kind", self.producer_kind, ScheduleProducerKind
        )
        if self.producer_kind != ScheduleProducerKind.SEALED_NATURAL_TRACE:
            raise TraceValidationError("natural watermark has wrong producer kind")
        require_text("producer_id", self.producer_id)
        require_sha256("producer_artifact_digest", self.producer_artifact_digest)
        require_sha256("schedule_digest", self.schedule_digest)
        require_text("checkpoint_id", self.checkpoint_id)
        require_sha256("checkpoint_digest", self.checkpoint_digest)
        _require_int("consumed_event_count", self.consumed_event_count)
        if self.consumed_event_count == 0:
            if self.last_schedule_event_id is not None:
                raise TraceValidationError("empty schedule prefix has a last event")
        else:
            require_text("last_schedule_event_id", self.last_schedule_event_id)
        _require_int("max_closed_timestamp_ns", self.max_closed_timestamp_ns)
        require_sha256("event_prefix_digest", self.event_prefix_digest)
        _require_int("closed_epoch_count", self.closed_epoch_count)
        require_sha256("epoch_prefix_digest", self.epoch_prefix_digest)
        _require_int("source_eof_record_count", self.source_eof_record_count)
        require_sha256("source_eof_digest", self.source_eof_digest)
        _require_int("capture_start_ns", self.capture_start_ns)
        _require_int("capture_end_ns", self.capture_end_ns)
        if self.capture_end_ns < self.capture_start_ns:
            raise TraceValidationError("natural capture interval regresses")
        _require_int("dropped_record_count", self.dropped_record_count)
        if type(self.clean_eof) is not bool:
            raise TraceValidationError("clean_eof must be a bool")


ScheduleWatermarkPayload = ReplayScheduleWatermarkPayload | NaturalTraceWatermarkPayload


@dataclass(frozen=True, slots=True)
class ObservationTerminalPayload:
    status: TerminalStatus
    reason: TerminalReason
    label_available_ns: int | None
    schedule_watermark_record_id: str | None
    last_verified_event_count: int
    last_verified_event_id: str | None
    last_verified_event_timestamp_ns: int | None
    unresolved_demand_intent_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("terminal status", self.status, TerminalStatus)
        _require_enum("terminal reason", self.reason, TerminalReason)
        _require_int("last_verified_event_count", self.last_verified_event_count)
        _require_sorted_unique(
            "unresolved_demand_intent_record_ids",
            self.unresolved_demand_intent_record_ids,
        )
        if self.last_verified_event_count == 0:
            if (
                self.last_verified_event_id is not None
                or self.last_verified_event_timestamp_ns is not None
            ):
                raise TraceValidationError("empty verified prefix has a last event")
        else:
            require_text("last_verified_event_id", self.last_verified_event_id)
            if self.last_verified_event_timestamp_ns is None:
                raise TraceValidationError("verified prefix lacks a timestamp")
            _require_int(
                "last_verified_event_timestamp_ns",
                self.last_verified_event_timestamp_ns,
            )

        if self.status == TerminalStatus.COMPLETE:
            if self.reason != TerminalReason.WINDOW_COMPLETE:
                raise TraceValidationError("complete terminal has a failure reason")
            if self.label_available_ns is None:
                raise TraceValidationError("complete terminal lacks label time")
            _require_int("label_available_ns", self.label_available_ns)
            require_text(
                "schedule_watermark_record_id", self.schedule_watermark_record_id
            )
            if self.unresolved_demand_intent_record_ids:
                raise TraceValidationError("complete terminal has unresolved intents")
            return

        if self.label_available_ns is not None:
            raise TraceValidationError("non-complete terminal exposes a label")
        _require_optional_text(
            "schedule_watermark_record_id", self.schedule_watermark_record_id
        )
        allowed = {
            TerminalStatus.CENSORED: _CENSORED_REASONS,
            TerminalStatus.UNIDENTIFIABLE: _UNIDENTIFIABLE_REASONS,
            TerminalStatus.INVALID: _INVALID_REASONS,
        }[self.status]
        if self.reason not in allowed:
            raise TraceValidationError("terminal status and reason are incompatible")


TracePayload = (
    TraceHeaderPayload
    | WorkflowTopologyPayload
    | CutoffPayload
    | PredictedAttemptPayload
    | AbstainedAttemptPayload
    | DemandIntentPayload
    | ReuseEpochPayload
    | ReplayScheduleWatermarkPayload
    | NaturalTraceWatermarkPayload
    | ObservationTerminalPayload
)


_PAYLOAD_TYPES: dict[TraceRecordType, tuple[type[Any], ...]] = {
    TraceRecordType.TRACE_HEADER: (TraceHeaderPayload,),
    TraceRecordType.WORKFLOW_TOPOLOGY: (WorkflowTopologyPayload,),
    TraceRecordType.CUTOFF: (CutoffPayload,),
    TraceRecordType.FORECAST_ATTEMPT: (
        PredictedAttemptPayload,
        AbstainedAttemptPayload,
    ),
    TraceRecordType.DEMAND_INTENT: (DemandIntentPayload,),
    TraceRecordType.REUSE_EPOCH: (ReuseEpochPayload,),
    TraceRecordType.SCHEDULE_WATERMARK: (
        ReplayScheduleWatermarkPayload,
        NaturalTraceWatermarkPayload,
    ),
    TraceRecordType.OBSERVATION_TERMINAL: (ObservationTerminalPayload,),
}


@dataclass(frozen=True, slots=True)
class TraceRecord:
    schema_version: str
    record_type: TraceRecordType
    trace_id: str
    run_id: str
    schedule_id: str
    schedule_case_id: str
    sequence: int
    record_id: str
    parent_record_id: str | None
    observation_id: str | None
    payload: TracePayload

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise TraceValidationError("unsupported C1 trace schema")
        for name in ("trace_id", "run_id", "schedule_id", "schedule_case_id"):
            require_text(name, getattr(self, name))
        _require_int("sequence", self.sequence)
        require_text("record_id", self.record_id)
        _require_optional_text("parent_record_id", self.parent_record_id)
        _require_enum("record_type", self.record_type, TraceRecordType)
        allowed = _PAYLOAD_TYPES.get(self.record_type, ())
        if type(self.payload) not in allowed:
            raise TraceValidationError("record type and payload type differ")
        if self.record_type in {
            TraceRecordType.TRACE_HEADER,
            TraceRecordType.WORKFLOW_TOPOLOGY,
        }:
            if self.observation_id is not None:
                raise TraceValidationError("non-observation record has observation_id")
        else:
            require_text("observation_id", self.observation_id)
        if self.record_type == TraceRecordType.TRACE_HEADER and (
            self.sequence != 0 or self.parent_record_id is not None
        ):
            raise TraceValidationError("trace header must be the root sequence")


@dataclass(frozen=True, slots=True)
class WaiterIdentity:
    binding_id: str
    workflow: WorkflowKey
    request_id: str
    node_id: str
    execution_ref: ExecutionRef
    state: BindingState

    def __post_init__(self) -> None:
        for name in ("binding_id", "request_id", "node_id"):
            require_text(name, getattr(self, name))
        if self.execution_ref.workflow != self.workflow:
            raise TraceValidationError("waiter execution belongs to another workflow")
        if self.execution_ref.request_id != self.request_id:
            raise TraceValidationError("waiter execution belongs to another request")
        _require_enum("waiter state", self.state, BindingState)


@dataclass(frozen=True, slots=True)
class AtomicCutoffView:
    snapshot: SharedLeasePolicySnapshot
    owner_specs: tuple[WorkflowSpec, ...]
    lifecycle_prefix: tuple[LifecycleEvent, ...]
    cutoff_ns: int
    horizon_duration_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.owner_specs, tuple):
            raise TraceValidationError("atomic cutoff owner specs must be a tuple")
        _validate_lifecycle_prefix(self.lifecycle_prefix)
        _require_int("cutoff_ns", self.cutoff_ns)
        _require_int("horizon_duration_ns", self.horizon_duration_ns, minimum=1)
        if self.deadline_ns != self.cutoff_ns + self.horizon_duration_ns:
            raise TraceValidationError("atomic cutoff deadline is inconsistent")
        if self.snapshot.runtime_event_count != len(self.lifecycle_prefix):
            raise TraceValidationError("atomic cutoff prefix differs from snapshot")
        if (
            self.lifecycle_prefix
            and self.lifecycle_prefix[-1].timestamp_ns > self.cutoff_ns
        ):
            raise TraceValidationError("atomic cutoff predates its lifecycle prefix")
        owner_workflows = {owner.workflow for owner in self.snapshot.owners}
        if not owner_workflows:
            raise TraceValidationError(
                "atomic cutoff requires an active retention owner"
            )
        spec_workflows = {spec.key for spec in self.owner_specs}
        if owner_workflows != spec_workflows:
            raise TraceValidationError("atomic cutoff owner topology set differs")
        if tuple(sorted(spec_workflows)) != tuple(
            spec.key for spec in self.owner_specs
        ):
            raise TraceValidationError("atomic cutoff owner specs are not ordered")
        specs = {spec.key: spec for spec in self.owner_specs}
        for owner in self.snapshot.owners:
            if not set(owner.eligible_node_ids).issubset(
                specs[owner.workflow].node_ids
            ):
                raise TraceValidationError(
                    "cutoff owner names an unknown eligible node"
                )

    @property
    def view_digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True, slots=True)
class PreServiceDemandView:
    block_key: BlockKey
    demand_commit_id: str
    target_replica: ReplicaId
    action: LedgerAction
    transfer_id: str
    timestamp_ns: int
    lifecycle_prefix: tuple[LifecycleEvent, ...]
    runtime_event_count: int
    last_event_id: str | None
    last_event_timestamp_ns: int | None
    location_version: int
    residency: ResidencyState
    waiters: tuple[WaiterIdentity, ...]

    def __post_init__(self) -> None:
        require_text("demand_commit_id", self.demand_commit_id)
        _require_enum("pre-service action", self.action, LedgerAction)
        _require_enum("pre-service residency", self.residency, ResidencyState)
        _require_enum("pre-service target tier", self.target_replica.tier, Tier)
        if self.target_replica.tier != Tier.GPU:
            raise TraceValidationError("pre-service target must be a GPU replica")
        if not isinstance(self.waiters, tuple) or not self.waiters:
            raise TraceValidationError("pre-service demand requires request waiters")
        _validate_lifecycle_prefix(self.lifecycle_prefix)
        if self.action not in {LedgerAction.LOAD, LedgerAction.PREFETCH}:
            raise TraceValidationError("pre-service view action is not H2D")
        require_text("transfer_id", self.transfer_id)
        _require_int("timestamp_ns", self.timestamp_ns)
        _require_int("runtime_event_count", self.runtime_event_count)
        if self.runtime_event_count != len(self.lifecycle_prefix):
            raise TraceValidationError("pre-service prefix event count differs")
        _require_int("location_version", self.location_version)
        if self.runtime_event_count == 0:
            if (
                self.last_event_id is not None
                or self.last_event_timestamp_ns is not None
            ):
                raise TraceValidationError("empty pre-service view has a last event")
        else:
            require_text("last_event_id", self.last_event_id)
            if self.last_event_timestamp_ns is None:
                raise TraceValidationError("pre-service view lacks last timestamp")
            _require_int("last_event_timestamp_ns", self.last_event_timestamp_ns)
            if self.last_event_timestamp_ns > self.timestamp_ns:
                raise TraceValidationError("pre-service timestamp predates lifecycle")
        waiter_ids = tuple(waiter.binding_id for waiter in self.waiters)
        if tuple(sorted(waiter_ids)) != waiter_ids or len(waiter_ids) != len(
            set(waiter_ids)
        ):
            raise TraceValidationError("pre-service waiters must be ordered and unique")
        if any(waiter.state != BindingState.RETAINED for waiter in self.waiters):
            raise TraceValidationError(
                "pre-service demand waiters must be retained before service"
            )

    @property
    def view_digest(self) -> str:
        return canonical_digest(self)


@dataclass(frozen=True, slots=True)
class DurableCommitReceipt:
    record_ids: tuple[str, ...]
    event_count: int
    view_digest: str
    batch_digest: str

    def __post_init__(self) -> None:
        _require_unique("receipt record_ids", self.record_ids)
        if not self.record_ids:
            raise TraceValidationError("durable receipt requires record IDs")
        _require_int("receipt event_count", self.event_count)
        require_sha256("receipt view_digest", self.view_digest)
        require_sha256("receipt batch_digest", self.batch_digest)


class CutoffCommitter(Protocol):
    def commit_cutoff(self, view: AtomicCutoffView) -> DurableCommitReceipt:
        """Publish a cutoff batch; formal runs require a writer-issued receipt."""


class DemandCommitter(Protocol):
    def commit_demands(self, view: PreServiceDemandView) -> DurableCommitReceipt:
        """Publish pre-service intents; formal runs require a writer receipt."""


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    raise TraceValidationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    """Encode one supported value as canonical JSON without a line terminator."""

    try:
        encoded = json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TraceValidationError(f"cannot encode canonical JSON: {exc}") from exc
    return encoded.encode("ascii")


def canonical_digest(value: Any) -> str:
    """Return the SHA-256 identity of one canonical structured value."""

    return sha256(canonical_json(value)).hexdigest()


def _decode_value(expected: Any, value: Any, *, path: str) -> Any:
    origin = get_origin(expected)
    if origin is types.UnionType:
        failures: list[str] = []
        for candidate in get_args(expected):
            try:
                return _decode_value(candidate, value, path=path)
            except (IdentityError, TraceValidationError, TypeError, ValueError) as exc:
                failures.append(str(exc))
        raise TraceValidationError(
            f"{path} matches no closed variant: {'; '.join(failures)}"
        )
    if origin is tuple:
        if not isinstance(value, list):
            raise TraceValidationError(f"{path} must be an array")
        arguments = get_args(expected)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_value(arguments[0], item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise TraceValidationError(f"{path} has the wrong tuple length")
        return tuple(
            _decode_value(argument, item, path=f"{path}[{index}]")
            for index, (argument, item) in enumerate(zip(arguments, value, strict=True))
        )
    if expected is type(None):
        if value is not None:
            raise TraceValidationError(f"{path} must be null")
        return None
    if expected is str:
        if type(value) is not str:
            raise TraceValidationError(f"{path} must be a string")
        return value
    if expected is int:
        if type(value) is not int:
            raise TraceValidationError(f"{path} must be an integer")
        return value
    if expected is bool:
        if type(value) is not bool:
            raise TraceValidationError(f"{path} must be a bool")
        return value
    if isinstance(expected, type) and issubclass(expected, StrEnum):
        if type(value) is not str:
            raise TraceValidationError(f"{path} must be an enum string")
        try:
            return expected(value)
        except ValueError as exc:
            raise TraceValidationError(f"{path} has an unknown enum value") from exc
    if isinstance(expected, type) and is_dataclass(expected):
        return _decode_dataclass(expected, value, path=path)
    raise TraceValidationError(f"{path} has unsupported type metadata")


def _decode_dataclass[T](expected: type[T], value: Any, *, path: str) -> T:
    if not isinstance(value, dict):
        raise TraceValidationError(f"{path} must be an object")
    expected_fields = {field.name for field in fields(expected)}
    observed_fields = set(value)
    if observed_fields != expected_fields:
        missing = sorted(expected_fields - observed_fields)
        unknown = sorted(observed_fields - expected_fields)
        raise TraceValidationError(
            f"{path} field set differs: missing={missing}, unknown={unknown}"
        )
    hints = get_type_hints(expected)
    decoded = {
        name: _decode_value(hints[name], value[name], path=f"{path}.{name}")
        for name in expected_fields
    }
    try:
        return expected(**decoded)
    except (IdentityError, TraceValidationError, TypeError, ValueError) as exc:
        raise TraceValidationError(f"invalid {path}: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TraceValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_float(value: str) -> None:
    raise TraceValidationError(f"floating JSON value is forbidden: {value}")


def _reject_constant(value: str) -> None:
    raise TraceValidationError(f"non-finite JSON value is forbidden: {value}")


def encode_trace_record(record: TraceRecord) -> bytes:
    """Encode one trace record without the JSONL line terminator."""

    return canonical_json(record)


def parse_canonical_dataclass[T](
    raw: bytes,
    expected: type[T],
    *,
    artifact_name: str,
    max_bytes: int,
) -> T:
    """Parse one exact closed dataclass from canonical JSON bytes."""

    require_text("artifact_name", artifact_name)
    _require_int("max_bytes", max_bytes, minimum=1)
    if not raw or len(raw) > max_bytes:
        raise TraceValidationError(f"{artifact_name} has an invalid size")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or b"\n" in raw:
        raise TraceValidationError(f"{artifact_name} framing is not canonical")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise TraceValidationError(f"{artifact_name} is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"{artifact_name} is invalid JSON: {exc}") from exc
    except TraceValidationError:
        raise
    except (RecursionError, ValueError) as exc:
        raise TraceValidationError(f"{artifact_name} exceeds parser limits") from exc
    decoded = _decode_dataclass(expected, value, path=artifact_name)
    if canonical_json(decoded) != raw:
        raise TraceValidationError(f"{artifact_name} bytes are not canonical")
    return decoded


def parse_trace_record(raw: bytes) -> TraceRecord:
    """Parse one exact canonical trace record without a line terminator."""

    return parse_canonical_dataclass(
        raw,
        TraceRecord,
        artifact_name="trace record",
        max_bytes=MAX_TRACE_LINE_BYTES,
    )


@dataclass(frozen=True, slots=True)
class ValidatedObservation:
    observation_id: str
    cutoff: TraceRecord
    attempt: TraceRecord
    intents: tuple[TraceRecord, ...]
    epochs: tuple[TraceRecord, ...]
    watermark: TraceRecord | None
    terminal: TraceRecord


@dataclass(frozen=True, slots=True)
class TraceValidationReceipt:
    """Content-addressed result from one independent evidence verifier."""

    role: EvidenceRole
    trace_pair_id: str
    trace_digest: str
    artifact_digest: str
    verifier_digest: str
    verified_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("validation role", self.role, EvidenceRole)
        require_text("validation trace_pair_id", self.trace_pair_id)
        require_sha256("validation trace_digest", self.trace_digest)
        require_sha256("validation artifact_digest", self.artifact_digest)
        require_sha256("validation verifier_digest", self.verifier_digest)
        _require_sorted_unique(
            "validated observation IDs", self.verified_observation_ids
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ValidatedTrace:
    header: TraceRecord
    topologies: tuple[TraceRecord, ...]
    observations: tuple[ValidatedObservation, ...]
    records: tuple[TraceRecord, ...]
    lifecycle_validation: TraceValidationReceipt | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    schedule_validation: TraceValidationReceipt | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _authorization_token: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


def _make_label_authorization_guard() -> tuple[Any, Any]:
    registry: dict[
        object,
        tuple[
            weakref.ReferenceType[ValidatedTrace],
            str,
            TraceValidationReceipt,
            TraceValidationReceipt,
        ],
    ] = {}

    def issue(
        trace: ValidatedTrace,
        lifecycle_receipt: TraceValidationReceipt,
        schedule_receipt: TraceValidationReceipt,
    ) -> ValidatedTrace:
        authorized = replace(trace)
        object.__setattr__(authorized, "lifecycle_validation", lifecycle_receipt)
        object.__setattr__(authorized, "schedule_validation", schedule_receipt)
        token = object()

        def discard(
            _: weakref.ReferenceType[ValidatedTrace],
            *,
            expired_token: object = token,
        ) -> None:
            registry.pop(expired_token, None)

        registry[token] = (
            weakref.ref(authorized, discard),
            trace_stream_digest(authorized.records),
            lifecycle_receipt,
            schedule_receipt,
        )
        object.__setattr__(authorized, "_authorization_token", token)
        return authorized

    def accepts(trace: ValidatedTrace) -> bool:
        token = trace._authorization_token
        if type(token) is not object:
            return False
        authorization = registry.get(token)
        if authorization is None:
            return False
        trace_reference, trace_digest, lifecycle_receipt, schedule_receipt = (
            authorization
        )
        return (
            trace_reference() is trace
            and trace_digest == trace_stream_digest(trace.records)
            and lifecycle_receipt is trace.lifecycle_validation
            and schedule_receipt is trace.schedule_validation
        )

    return issue, accepts


_issue_label_authorization, _has_label_authorization = _make_label_authorization_guard()


class LifecycleEvidenceGate(Protocol):
    def verify_lifecycle(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        """Replay lifecycle evidence and authorize exact complete observations."""


class ScheduleEvidenceGate(Protocol):
    def verify_schedule(self, trace: ValidatedTrace) -> TraceValidationReceipt:
        """Replay schedule evidence and authorize exact complete observations."""


@dataclass(frozen=True, slots=True)
class DemandLabel:
    observation_id: str
    first_demand: int
    epoch_count: int
    repeat_count: int


@dataclass(slots=True)
class _ObservationState:
    cutoff: TraceRecord
    latest_record_id: str
    attempt: TraceRecord | None = None
    intents: dict[str, TraceRecord] | None = None
    epochs: list[TraceRecord] | None = None
    closed_intents: set[str] | None = None
    intent_semantic_keys: set[tuple[Any, ...]] | None = None
    schedule_claim_keys: set[tuple[str, str]] | None = None
    execution_refs: set[ExecutionRef] | None = None
    request_binding_ids: set[str] | None = None
    epoch_ids: set[str] | None = None
    request_terminal_event_ids: set[str] | None = None
    watermark: TraceRecord | None = None
    terminal: TraceRecord | None = None

    def __post_init__(self) -> None:
        self.intents = {} if self.intents is None else self.intents
        self.epochs = [] if self.epochs is None else self.epochs
        self.closed_intents = (
            set() if self.closed_intents is None else self.closed_intents
        )
        self.intent_semantic_keys = (
            set() if self.intent_semantic_keys is None else self.intent_semantic_keys
        )
        self.schedule_claim_keys = (
            set() if self.schedule_claim_keys is None else self.schedule_claim_keys
        )
        self.execution_refs = (
            set() if self.execution_refs is None else self.execution_refs
        )
        self.request_binding_ids = (
            set() if self.request_binding_ids is None else self.request_binding_ids
        )
        self.epoch_ids = set() if self.epoch_ids is None else self.epoch_ids
        self.request_terminal_event_ids = (
            set()
            if self.request_terminal_event_ids is None
            else self.request_terminal_event_ids
        )


class TraceStateMachine:
    """Validate record order and observation-local closure incrementally."""

    def __init__(self) -> None:
        self._records: list[TraceRecord] = []
        self._record_ids: set[str] = set()
        self._header: TraceRecord | None = None
        self._topologies: dict[str, TraceRecord] = {}
        self._workflow_topologies: dict[WorkflowKey, TraceRecord] = {}
        self._observations: dict[str, _ObservationState] = {}
        self._observation_units: set[tuple[str, str, BlockKey, int, int, int]] = set()
        self._envelope: tuple[str, str, str, str] | None = None

    def push(self, record: TraceRecord) -> None:
        if record.sequence != len(self._records):
            raise TraceValidationError("trace sequence is not contiguous")
        if record.record_id in self._record_ids:
            raise TraceValidationError("trace record_id is duplicated")
        envelope = (
            record.trace_id,
            record.run_id,
            record.schedule_id,
            record.schedule_case_id,
        )
        if self._envelope is None:
            self._envelope = envelope
        elif envelope != self._envelope:
            raise TraceValidationError("trace envelope changes between records")

        if record.record_type == TraceRecordType.TRACE_HEADER:
            self._push_header(record)
        elif record.record_type == TraceRecordType.WORKFLOW_TOPOLOGY:
            self._push_topology(record)
        elif record.record_type == TraceRecordType.CUTOFF:
            self._push_cutoff(record)
        else:
            self._push_observation_record(record)
        self._records.append(record)
        self._record_ids.add(record.record_id)

    def _push_header(self, record: TraceRecord) -> None:
        if self._header is not None or self._records:
            raise TraceValidationError("trace header is not the unique first row")
        self._header = record

    def _push_topology(self, record: TraceRecord) -> None:
        if self._header is None or self._observations:
            raise TraceValidationError("topology must precede every observation")
        if record.parent_record_id != self._header.record_id:
            raise TraceValidationError("topology parent is not the trace header")
        payload = _payload(record, WorkflowTopologyPayload)
        header = _payload(self._header, TraceHeaderPayload)
        if payload.branch_grammar_digest != header.branch_grammar_digest:
            raise TraceValidationError("topology branch grammar differs from header")
        workflow = payload.workflow_spec.key
        if workflow in self._workflow_topologies:
            raise TraceValidationError("workflow topology is duplicated")
        self._topologies[record.record_id] = record
        self._workflow_topologies[workflow] = record

    def _push_cutoff(self, record: TraceRecord) -> None:
        if self._header is None or not self._topologies:
            raise TraceValidationError("cutoff lacks header or topology")
        if record.parent_record_id != self._header.record_id:
            raise TraceValidationError("cutoff parent is not the trace header")
        observation_id = _observation_id(record)
        if observation_id in self._observations:
            raise TraceValidationError("observation cutoff is duplicated")
        payload = _payload(record, CutoffPayload)
        observation_unit = (
            record.run_id,
            record.schedule_case_id,
            payload.snapshot.block_key,
            payload.lifecycle_event_count,
            payload.cutoff_ns,
            payload.horizon_duration_ns,
        )
        if observation_unit in self._observation_units:
            raise TraceValidationError("statistical observation unit is duplicated")
        if any(item not in self._topologies for item in payload.topology_record_ids):
            raise TraceValidationError("cutoff references an unknown topology")
        owner_binding_ids = tuple(owner.binding_id for owner in payload.snapshot.owners)
        if tuple(sorted(owner_binding_ids)) != owner_binding_ids:
            raise TraceValidationError("cutoff owners are not binding-ID ordered")
        owner_workflows = {owner.workflow for owner in payload.snapshot.owners}
        topology_specs = {
            _payload(
                self._topologies[item], WorkflowTopologyPayload
            ).workflow_spec.key: _payload(
                self._topologies[item], WorkflowTopologyPayload
            ).workflow_spec
            for item in payload.topology_record_ids
        }
        if not owner_workflows or set(topology_specs) != owner_workflows:
            raise TraceValidationError(
                "cutoff topology set differs from active owner workflows"
            )
        for owner in payload.snapshot.owners:
            if owner.created_ns > payload.cutoff_ns:
                raise TraceValidationError("cutoff owner was created after the cutoff")
            if tuple(sorted(owner.eligible_node_ids)) != owner.eligible_node_ids:
                raise TraceValidationError("eligible cutoff nodes are not ordered")
            if not set(owner.eligible_node_ids).issubset(
                topology_specs[owner.workflow].node_ids
            ):
                raise TraceValidationError(
                    "cutoff owner names an unknown eligible node"
                )
        self._observation_units.add(observation_unit)
        self._observations[observation_id] = _ObservationState(
            cutoff=record,
            latest_record_id=record.record_id,
        )

    def _push_observation_record(self, record: TraceRecord) -> None:
        observation_id = _observation_id(record)
        try:
            state = self._observations[observation_id]
        except KeyError as exc:
            raise TraceValidationError(
                "observation record precedes its cutoff"
            ) from exc
        if state.terminal is not None:
            raise TraceValidationError("record follows an observation terminal")
        if record.parent_record_id != state.latest_record_id:
            raise TraceValidationError("observation parent chain is broken")

        if record.record_type == TraceRecordType.FORECAST_ATTEMPT:
            self._push_attempt(state, record)
        elif record.record_type == TraceRecordType.DEMAND_INTENT:
            self._push_intent(state, record)
        elif record.record_type == TraceRecordType.REUSE_EPOCH:
            self._push_epoch(state, record)
        elif record.record_type == TraceRecordType.SCHEDULE_WATERMARK:
            self._push_watermark(state, record)
        elif record.record_type == TraceRecordType.OBSERVATION_TERMINAL:
            self._push_terminal(state, record)
        else:
            raise TraceValidationError("unsupported observation record type")
        state.latest_record_id = record.record_id

    def _push_attempt(self, state: _ObservationState, record: TraceRecord) -> None:
        if state.attempt is not None:
            raise TraceValidationError("forecast attempt is duplicated")
        cutoff = _payload(state.cutoff, CutoffPayload)
        if not isinstance(
            record.payload,
            (PredictedAttemptPayload, AbstainedAttemptPayload),
        ):
            raise TraceValidationError("forecast attempt has the wrong payload")
        context = record.payload.context
        if context.feature_view_digest != cutoff.feature_view_digest:
            raise TraceValidationError("forecast feature view differs from cutoff")
        if context.information_cutoff_digest != cutoff.atomic_cutoff_view_digest:
            raise TraceValidationError("forecast information cutoff identity differs")
        if isinstance(record.payload, PredictedAttemptPayload):
            forecast = record.payload.forecast
            if forecast.block_key != cutoff.snapshot.block_key:
                raise TraceValidationError("forecast block differs from cutoff")
            if forecast.runtime_event_count != cutoff.lifecycle_event_count:
                raise TraceValidationError("forecast event count differs from cutoff")
            if forecast.generated_ns != cutoff.cutoff_ns:
                raise TraceValidationError("forecast generation differs from cutoff")
            if forecast.horizon_ns != cutoff.deadline_ns:
                raise TraceValidationError("forecast deadline differs from cutoff")
            try:
                aggregate_shared_lease(cutoff.snapshot, forecast)
            except IdentityError as exc:
                raise TraceValidationError(
                    f"invalid snapshot-bound forecast: {exc}"
                ) from exc
        state.attempt = record

    def _push_intent(self, state: _ObservationState, record: TraceRecord) -> None:
        if state.attempt is None or state.epochs or state.watermark is not None:
            raise TraceValidationError("demand intent violates observation order")
        payload = _payload(record, DemandIntentPayload)
        cutoff = _payload(state.cutoff, CutoffPayload)
        if not cutoff.cutoff_ns < payload.scheduled_access_ns <= cutoff.deadline_ns:
            raise TraceValidationError("demand intent falls outside the primary window")
        if payload.block_key != cutoff.snapshot.block_key:
            raise TraceValidationError("demand intent block differs from cutoff")
        if payload.pre_service_event_count < cutoff.lifecycle_event_count:
            raise TraceValidationError("demand pre-service prefix predates cutoff")
        owners = {owner.binding_id: owner for owner in cutoff.snapshot.owners}
        owner = owners.get(payload.retention_binding_id)
        if owner is None:
            raise TraceValidationError("demand lacks an active cutoff owner")
        if owner.workflow != payload.workflow:
            raise TraceValidationError("demand crosses retention workflow")
        if payload.node_id not in owner.eligible_node_ids:
            raise TraceValidationError("demand targets an ineligible cutoff node")
        assert state.intents is not None
        assert state.intent_semantic_keys is not None
        assert state.schedule_claim_keys is not None
        assert state.execution_refs is not None
        assert state.request_binding_ids is not None
        if record.record_id in state.intents:
            raise TraceValidationError("demand intent is duplicated")
        semantic_key = (
            payload.schedule_event_id,
            payload.execution_ref,
            payload.block_key,
            payload.claim_id,
            payload.request_binding_id,
        )
        if semantic_key in state.intent_semantic_keys:
            raise TraceValidationError("demand intent semantic identity is duplicated")
        schedule_claim_key = (payload.schedule_event_id, payload.claim_id)
        if schedule_claim_key in state.schedule_claim_keys:
            raise TraceValidationError(
                "demand schedule-event and claim identity is duplicated"
            )
        if payload.execution_ref in state.execution_refs:
            raise TraceValidationError("demand execution reference is duplicated")
        if payload.request_binding_id in state.request_binding_ids:
            raise TraceValidationError("demand request binding is duplicated")
        state.intents[record.record_id] = record
        state.intent_semantic_keys.add(semantic_key)
        state.schedule_claim_keys.add(schedule_claim_key)
        state.execution_refs.add(payload.execution_ref)
        state.request_binding_ids.add(payload.request_binding_id)

    def _push_epoch(self, state: _ObservationState, record: TraceRecord) -> None:
        if state.attempt is None or state.watermark is not None:
            raise TraceValidationError("reuse epoch violates observation order")
        payload = _payload(record, ReuseEpochPayload)
        assert state.intents is not None
        assert state.closed_intents is not None
        assert state.epoch_ids is not None
        assert state.request_terminal_event_ids is not None
        cutoff = _payload(state.cutoff, CutoffPayload)
        if payload.block_key != cutoff.snapshot.block_key:
            raise TraceValidationError("reuse epoch block differs from cutoff")
        if payload.reuse_epoch_id in state.epoch_ids:
            raise TraceValidationError("reuse epoch identity is duplicated")
        if state.epochs:
            prior = _payload(state.epochs[-1], ReuseEpochPayload)
            if payload.access_ns < prior.access_ns:
                raise TraceValidationError("reuse epochs are not time ordered")
        for intent_id, service in zip(
            payload.demand_intent_record_ids,
            payload.service_terminals,
            strict=True,
        ):
            intent_record = state.intents.get(intent_id)
            if intent_record is None:
                raise TraceValidationError("reuse epoch references an unknown intent")
            if intent_id in state.closed_intents:
                raise TraceValidationError("demand intent belongs to multiple epochs")
            intent = _payload(intent_record, DemandIntentPayload)
            if (
                intent.reuse_epoch_id != payload.reuse_epoch_id
                or intent.scheduled_access_ns != payload.access_ns
                or intent.block_key != payload.block_key
            ):
                raise TraceValidationError("reuse epoch contradicts its demand intent")
            if isinstance(service, (H2DExecMapService, H2DFailedService)) and (
                intent.request_binding_id not in service.waiter_binding_ids
            ):
                raise TraceValidationError("H2D service omits its request waiter")
            terminal_event_id = None
            if isinstance(service, (ResidentExecMapService, H2DExecMapService)):
                terminal_event_id = service.exec_map_event_id
            elif isinstance(service, RequestCancelledService):
                terminal_event_id = service.release_event_id
            if terminal_event_id is not None:
                if terminal_event_id in state.request_terminal_event_ids:
                    raise TraceValidationError(
                        "request service terminal event is reused across intents"
                    )
                state.request_terminal_event_ids.add(terminal_event_id)
        state.closed_intents.update(payload.demand_intent_record_ids)
        state.epoch_ids.add(payload.reuse_epoch_id)
        assert state.epochs is not None
        state.epochs.append(record)

    def _push_watermark(self, state: _ObservationState, record: TraceRecord) -> None:
        if state.attempt is None or state.watermark is not None:
            raise TraceValidationError("schedule watermark violates observation order")
        header = _payload(self._required_header(), TraceHeaderPayload)
        payload = _schedule_watermark(record)
        if payload.schedule_digest != header.schedule_digest:
            raise TraceValidationError("watermark schedule digest differs from header")
        state.watermark = record

    def _push_terminal(self, state: _ObservationState, record: TraceRecord) -> None:
        if state.attempt is None:
            raise TraceValidationError("terminal precedes forecast attempt")
        payload = _payload(record, ObservationTerminalPayload)
        assert state.intents is not None
        assert state.closed_intents is not None
        cutoff = _payload(state.cutoff, CutoffPayload)
        required_event_count = max(
            [cutoff.lifecycle_event_count]
            + [
                _payload(intent, DemandIntentPayload).pre_service_event_count
                for intent in state.intents.values()
            ]
        )
        if payload.last_verified_event_count < required_event_count:
            raise TraceValidationError(
                "terminal verified prefix omits demand provenance"
            )
        if payload.status == TerminalStatus.COMPLETE:
            if state.watermark is None:
                raise TraceValidationError("complete observation lacks watermark")
            watermark = _schedule_watermark(state.watermark)
            if payload.schedule_watermark_record_id != state.watermark.record_id:
                raise TraceValidationError("terminal references another watermark")
            if watermark.max_closed_timestamp_ns <= cutoff.deadline_ns:
                raise TraceValidationError("watermark does not exceed the deadline")
            if payload.label_available_ns is None or payload.label_available_ns < (
                watermark.max_closed_timestamp_ns
            ):
                raise TraceValidationError("label predates schedule completeness")
            if set(state.intents) != state.closed_intents:
                raise TraceValidationError("complete observation has unclosed intents")
        else:
            unresolved = set(payload.unresolved_demand_intent_record_ids)
            expected_unresolved = set(state.intents) - state.closed_intents
            if unresolved != expected_unresolved:
                raise TraceValidationError(
                    "terminal unresolved intents differ from the open intent set"
                )
            if state.watermark is None:
                if payload.schedule_watermark_record_id is not None:
                    raise TraceValidationError(
                        "terminal references an absent watermark"
                    )
            elif payload.schedule_watermark_record_id != state.watermark.record_id:
                raise TraceValidationError("terminal omits or changes its watermark")
        state.terminal = record

    def _required_header(self) -> TraceRecord:
        if self._header is None:
            raise TraceValidationError("trace header is missing")
        return self._header

    def finalize(self) -> ValidatedTrace:
        header = self._required_header()
        if not self._topologies:
            raise TraceValidationError("trace topology is missing")
        if not self._observations:
            raise TraceValidationError("trace has no observations")
        validated: list[ValidatedObservation] = []
        for observation_id, state in self._observations.items():
            if state.attempt is None or state.terminal is None:
                raise TraceValidationError("observation is not terminal")
            assert state.intents is not None
            assert state.epochs is not None
            validated.append(
                ValidatedObservation(
                    observation_id=observation_id,
                    cutoff=state.cutoff,
                    attempt=state.attempt,
                    intents=tuple(state.intents.values()),
                    epochs=tuple(state.epochs),
                    watermark=state.watermark,
                    terminal=state.terminal,
                )
            )
        return ValidatedTrace(
            header=header,
            topologies=tuple(self._topologies.values()),
            observations=tuple(validated),
            records=tuple(self._records),
        )


def _payload[T](record: TraceRecord, expected: type[T]) -> T:
    if not isinstance(record.payload, expected):
        raise TraceValidationError("trace record has the wrong payload")
    return record.payload


def _schedule_watermark(record: TraceRecord) -> ScheduleWatermarkPayload:
    if not isinstance(
        record.payload,
        (ReplayScheduleWatermarkPayload, NaturalTraceWatermarkPayload),
    ):
        raise TraceValidationError("trace record is not a schedule watermark")
    return record.payload


def _observation_id(record: TraceRecord) -> str:
    if record.observation_id is None:
        raise TraceValidationError("observation record lacks observation_id")
    return record.observation_id


def validate_trace(records: tuple[TraceRecord, ...]) -> ValidatedTrace:
    """Replay structural records without authorizing research labels."""

    machine = TraceStateMachine()
    for record in records:
        machine.push(record)
    return machine.finalize()


def trace_stream_digest(records: tuple[TraceRecord, ...]) -> str:
    """Digest the exact canonical JSONL byte stream represented by records."""

    return sha256(
        b"".join(encode_trace_record(record) + b"\n" for record in records)
    ).hexdigest()


def _complete_observation_ids(trace: ValidatedTrace) -> tuple[str, ...]:
    return tuple(
        sorted(
            observation.observation_id
            for observation in trace.observations
            if _payload(
                observation.terminal,
                ObservationTerminalPayload,
            ).status
            == TerminalStatus.COMPLETE
        )
    )


def _validate_evidence_receipt(
    trace: ValidatedTrace,
    receipt: TraceValidationReceipt,
    *,
    evidence_name: str,
    expected_role: EvidenceRole,
) -> None:
    if type(receipt) is not TraceValidationReceipt:
        raise TraceValidationError(f"{evidence_name} gate returned an invalid receipt")
    if receipt.role != expected_role:
        raise TraceValidationError(f"{evidence_name} gate returned the wrong role")
    header = _payload(trace.header, TraceHeaderPayload)
    if receipt.trace_pair_id != header.trace_pair_id:
        raise TraceValidationError(f"{evidence_name} receipt names another trace pair")
    if receipt.trace_digest != trace_stream_digest(trace.records):
        raise TraceValidationError(
            f"{evidence_name} receipt names another trace stream"
        )
    if receipt.verified_observation_ids != _complete_observation_ids(trace):
        raise TraceValidationError(
            f"{evidence_name} receipt does not cover every complete observation"
        )


def validate_trace_for_labels(
    records: tuple[TraceRecord, ...],
    *,
    lifecycle_gate: LifecycleEvidenceGate,
    schedule_gate: ScheduleEvidenceGate,
) -> ValidatedTrace:
    """Authorize labels only after independent lifecycle and schedule replay."""

    trace = validate_trace(records)
    lifecycle_receipt = lifecycle_gate.verify_lifecycle(trace)
    schedule_receipt = schedule_gate.verify_schedule(trace)
    _validate_evidence_receipt(
        trace,
        lifecycle_receipt,
        evidence_name="lifecycle",
        expected_role=EvidenceRole.LIFECYCLE,
    )
    _validate_evidence_receipt(
        trace,
        schedule_receipt,
        evidence_name="schedule",
        expected_role=EvidenceRole.SCHEDULE,
    )
    header = _payload(trace.header, TraceHeaderPayload)
    if schedule_receipt.artifact_digest != header.schedule_digest:
        raise TraceValidationError("schedule receipt differs from the frozen schedule")
    return _issue_label_authorization(
        trace,
        lifecycle_receipt,
        schedule_receipt,
    )


def reconstruct_demand_label(
    trace: ValidatedTrace,
    observation_id: str,
) -> DemandLabel:
    """Derive policy-independent demand labels for one complete observation."""

    replayed = validate_trace(trace.records)
    for observation in replayed.observations:
        if observation.observation_id != observation_id:
            continue
        terminal = _payload(observation.terminal, ObservationTerminalPayload)
        if terminal.status != TerminalStatus.COMPLETE:
            raise TraceValidationError("labels are unavailable for non-complete data")
        if not _has_label_authorization(trace):
            raise TraceValidationError("labels require opaque evidence authorization")
        lifecycle_receipt = trace.lifecycle_validation
        schedule_receipt = trace.schedule_validation
        if lifecycle_receipt is None or schedule_receipt is None:
            raise TraceValidationError(
                "labels are unavailable before lifecycle and schedule verification"
            )
        _validate_evidence_receipt(
            replayed,
            lifecycle_receipt,
            evidence_name="lifecycle",
            expected_role=EvidenceRole.LIFECYCLE,
        )
        _validate_evidence_receipt(
            replayed,
            schedule_receipt,
            evidence_name="schedule",
            expected_role=EvidenceRole.SCHEDULE,
        )
        header = _payload(replayed.header, TraceHeaderPayload)
        if schedule_receipt.artifact_digest != header.schedule_digest:
            raise TraceValidationError(
                "schedule validation no longer matches the trace"
            )
        if observation_id not in lifecycle_receipt.verified_observation_ids:
            raise TraceValidationError(
                "lifecycle evidence did not authorize this label"
            )
        if observation_id not in schedule_receipt.verified_observation_ids:
            raise TraceValidationError("schedule evidence did not authorize this label")
        epoch_ids = {
            _payload(epoch, ReuseEpochPayload).reuse_epoch_id
            for epoch in observation.epochs
        }
        if len(epoch_ids) != len(observation.epochs):
            raise TraceValidationError(
                "validated label contains duplicate reuse epochs"
            )
        epoch_count = len(epoch_ids)
        return DemandLabel(
            observation_id=observation_id,
            first_demand=int(epoch_count > 0),
            epoch_count=epoch_count,
            repeat_count=max(0, epoch_count - int(epoch_count > 0)),
        )
    raise TraceValidationError(f"unknown observation: {observation_id}")


def _read_regular_file(path: Path) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise TraceValidationError(f"trace file is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TraceValidationError("trace path must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise TraceValidationError("trace file changed while opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_TRACE_BYTES + 1)
        after = os.fstat(descriptor)
        if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise TraceValidationError("trace file changed while reading")
        try:
            linked = path.lstat()
        except OSError as exc:
            raise TraceValidationError("trace path changed while reading") from exc
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise TraceValidationError("trace path changed while reading")
    finally:
        os.close(descriptor)
    if not raw or len(raw) > MAX_TRACE_BYTES:
        raise TraceValidationError("trace file has an invalid size")
    return raw


def load_trace_jsonl(path: Path) -> tuple[TraceRecord, ...]:
    """Load canonical JSONL records from one stable regular file."""

    raw = _read_regular_file(path)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise TraceValidationError("trace JSONL framing is invalid")
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        raise TraceValidationError("trace JSONL contains a blank line")
    records = tuple(parse_trace_record(line) for line in lines)
    validate_trace(records)
    return records


class DurableTraceWriter:
    """Create and durably append one fail-closed canonical trace stream.

    The writer never resumes a stream after a process or I/O failure. A formal
    runner must start a new create-only artifact when this writer is poisoned.
    """

    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise TraceDurabilityError("trace path must be absolute")
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise TraceDurabilityError("trace parent must be a non-symlink directory")
        self._lock = _FailFastLock()
        parent_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            parent_flags |= os.O_NOFOLLOW
        try:
            self._parent_descriptor = os.open(parent, parent_flags)
        except OSError as exc:
            raise TraceDurabilityError("cannot open trace parent safely") from exc
        parent_stat = os.fstat(self._parent_descriptor)
        try:
            current_parent = parent.stat(follow_symlinks=False)
        except OSError as exc:
            os.close(self._parent_descriptor)
            raise TraceDurabilityError("cannot revalidate trace parent") from exc
        if (parent_stat.st_dev, parent_stat.st_ino) != (
            current_parent.st_dev,
            current_parent.st_ino,
        ):
            os.close(self._parent_descriptor)
            raise TraceDurabilityError("trace parent changed while opening")

        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self._entry_name = path.name
        try:
            self._descriptor = os.open(
                self._entry_name,
                flags,
                0o640,
                dir_fd=self._parent_descriptor,
            )
        except FileExistsError as exc:
            os.close(self._parent_descriptor)
            raise TraceDurabilityError("trace output is create-only") from exc
        except OSError as exc:
            os.close(self._parent_descriptor)
            raise TraceDurabilityError("cannot create trace output safely") from exc
        self._state = TraceStateMachine()
        self._poisoned = False
        self._closed = False
        self._bytes_written = 0
        self._file_identity: tuple[int, int] | None = None
        self._expected_mtime_ns = 0
        self._expected_ctime_ns = 0
        self._stream_hasher = sha256()
        try:
            os.fsync(self._descriptor)
            os.fsync(self._parent_descriptor)
            current = self._checked_file_stat(expected_size=0)
            self._file_identity = (current.st_dev, current.st_ino)
            self._expected_mtime_ns = current.st_mtime_ns
            self._expected_ctime_ns = current.st_ctime_ns
        except OSError as exc:
            self._poisoned = True
            os.close(self._descriptor)
            os.close(self._parent_descriptor)
            self._closed = True
            raise TraceCommitIndeterminateError(
                "trace creation durability is indeterminate"
            ) from exc

    def _checked_file_stat(self, *, expected_size: int) -> os.stat_result:
        opened = os.fstat(self._descriptor)
        linked = os.stat(
            self._entry_name,
            dir_fd=self._parent_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise OSError("trace descriptor is not a singly linked regular file")
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise OSError("trace directory entry no longer names the open file")
        if (
            self._file_identity is not None
            and (
                opened.st_dev,
                opened.st_ino,
            )
            != self._file_identity
        ):
            raise OSError("trace file identity changed")
        if opened.st_size != expected_size:
            raise OSError("trace file size changed outside the writer")
        return opened

    def _checked_stream_digest(
        self,
        *,
        expected_size: int,
    ) -> tuple[os.stat_result, str]:
        before = self._checked_file_stat(expected_size=expected_size)
        hasher = sha256()
        offset = 0
        while offset < expected_size:
            chunk = os.pread(
                self._descriptor,
                min(1024 * 1024, expected_size - offset),
                offset,
            )
            if not chunk:
                raise OSError("trace file ended while verifying its committed prefix")
            hasher.update(chunk)
            offset += len(chunk)
        after = self._checked_file_stat(expected_size=expected_size)
        if (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise OSError("trace file changed while verifying its committed prefix")
        return after, hasher.hexdigest()

    def _assert_unchanged(self) -> None:
        current, stream_digest = self._checked_stream_digest(
            expected_size=self._bytes_written
        )
        if (
            current.st_mtime_ns != self._expected_mtime_ns
            or current.st_ctime_ns != self._expected_ctime_ns
        ):
            raise OSError("trace file metadata changed outside the writer")
        if stream_digest != self._stream_hasher.hexdigest():
            raise OSError("trace file content changed outside the writer")

    def append_durable(
        self,
        records: tuple[TraceRecord, ...],
        *,
        event_count: int,
        view_digest: str,
    ) -> DurableCommitReceipt:
        with self._lock:
            if self._closed:
                raise TraceDurabilityError("trace writer is closed")
            if self._poisoned:
                raise TraceCommitIndeterminateError("trace writer is poisoned")
            if not isinstance(records, tuple) or not records:
                raise TraceValidationError("trace batch must be a non-empty tuple")
            _require_int("batch event_count", event_count)
            require_sha256("batch view_digest", view_digest)

            canonical_records: list[TraceRecord] = []
            lines: list[bytes] = []
            for record in records:
                line = encode_trace_record(record)
                if len(line) > MAX_TRACE_LINE_BYTES:
                    raise TraceValidationError("trace record exceeds the line limit")
                parsed = parse_trace_record(line)
                canonical_records.append(parsed)
                lines.append(line + b"\n")
            encoded = b"".join(lines)
            if self._bytes_written + len(encoded) > MAX_TRACE_BYTES:
                raise TraceValidationError("trace stream exceeds the total size limit")

            staged = deepcopy(self._state)
            for record in canonical_records:
                staged.push(record)
            batch_digest = sha256(encoded).hexdigest()
            receipt = DurableCommitReceipt(
                record_ids=tuple(record.record_id for record in canonical_records),
                event_count=event_count,
                view_digest=view_digest,
                batch_digest=batch_digest,
            )
            staged_stream_hasher = self._stream_hasher.copy()
            staged_stream_hasher.update(encoded)
            written = 0
            try:
                self._assert_unchanged()
            except OSError as exc:
                self._poisoned = True
                raise TraceDurabilityError(
                    "trace file changed outside the writer"
                ) from exc
            try:
                while written < len(encoded):
                    count = os.write(self._descriptor, encoded[written:])
                    if count <= 0:
                        raise OSError("trace write made no progress")
                    written += count
                os.fsync(self._descriptor)
                os.fsync(self._parent_descriptor)
                committed_size = self._bytes_written + len(encoded)
                current, stream_digest = self._checked_stream_digest(
                    expected_size=committed_size
                )
                if stream_digest != staged_stream_hasher.hexdigest():
                    raise OSError("trace committed prefix differs from staged bytes")
            except OSError as exc:
                self._poisoned = True
                raise TraceCommitIndeterminateError(
                    "trace append durability is indeterminate"
                ) from exc
            self._bytes_written += written
            self._expected_mtime_ns = current.st_mtime_ns
            self._expected_ctime_ns = current.st_ctime_ns
            self._stream_hasher = staged_stream_hasher
            self._state = staged
            return receipt

    def close(self, *, finalize: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if finalize and not self._poisoned:
                    self._state.finalize()
                if not self._poisoned:
                    self._assert_unchanged()
                os.fsync(self._descriptor)
                os.fsync(self._parent_descriptor)
            finally:
                os.close(self._descriptor)
                os.close(self._parent_descriptor)
                self._closed = True

    def __enter__(self) -> DurableTraceWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(finalize=exc_type is None)
