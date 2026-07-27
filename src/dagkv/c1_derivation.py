"""Closed C1-B1 typed feature derivations and selector-bound replay."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum
from heapq import heappop, heappush
from pathlib import Path

from dagkv.c1_features import (
    MAX_FEATURE_ARTIFACT_BYTES,
    FeatureAvailabilityKind,
    FeatureClassification,
    FeatureContract,
    FeatureSourceSchema,
    FeatureValueKind,
    LoadedFeatureArtifact,
    _load_artifact,
    _write_artifact,
    validate_feature_contract,
)
from dagkv.c1_trace import (
    TraceValidationError,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
)
from dagkv.domain import require_sha256, require_text

DERIVATION_REGISTRY_SCHEMA_VERSION = "dagkv.m3.derivation_registry.v1"
DERIVATION_ENTRY_SCHEMA_VERSION = "dagkv.m3.derivation_entry.v1"
DERIVATION_SAFETY_CEILING_SCHEMA_VERSION = "dagkv.m3.derivation_safety_ceiling.v1"
FEATURE_VALUE_SCHEMA_VERSION = "dagkv.m3.canonical_feature_value.v1"
SOURCE_FEATURE_VALUE_SCHEMA_VERSION = "dagkv.m3.source_feature_value.v1"
DERIVED_FEATURE_VALUE_SCHEMA_VERSION = "dagkv.m3.derived_feature_value.v1"
MAX_DERIVATION_ENTRIES = 4_096
MAX_DERIVATION_DEPENDENCIES = 65_536
MAX_DERIVATION_GRAPH_VALUES = 65_536
MAX_DERIVATION_EXPANDED_EDGES = 65_536
MAX_DERIVATION_TEXT_BYTES = 4_096
MAX_DERIVATION_BUCKET_EDGES = 4_096
MAX_DERIVATION_TOTAL_TEXT_BYTES = 16 * 1024 * 1024


class DerivationOpcode(StrEnum):
    """The complete non-executable v1 expression language."""

    IDENTITY = "IDENTITY"
    COUNT = "COUNT"
    SUM_INT = "SUM_INT"
    MIN_INT = "MIN_INT"
    MAX_INT = "MAX_INT"
    SUB_INT = "SUB_INT"
    BOOL_ALL = "BOOL_ALL"
    BOOL_ANY = "BOOL_ANY"
    EQUAL = "EQUAL"
    CLAMP_INT = "CLAMP_INT"
    RIGHT_CLOSED_BUCKET_INT = "RIGHT_CLOSED_BUCKET_INT"


class DependencySelectorKind(StrEnum):
    """Closed selectors whose expansion is independent of caller choices."""

    SOURCE_EXACT_ALL = "SOURCE_EXACT_ALL"
    DERIVATION_EXACT_ONE = "DERIVATION_EXACT_ONE"


@dataclass(frozen=True, slots=True)
class CanonicalFeatureValue:
    """One exact scalar value without bool/int or enum/text coercion."""

    schema_version: str
    kind: FeatureValueKind
    bool_value: bool | None
    int_value: int | None
    text_value: str | None
    enum_type: str | None
    enum_value: str | None

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_VALUE_SCHEMA_VERSION:
            raise TraceValidationError("unsupported canonical feature-value schema")
        if type(self.kind) is not FeatureValueKind:
            raise TraceValidationError("canonical feature value has an invalid kind")
        active = {
            FeatureValueKind.ABSENT: (),
            FeatureValueKind.BOOL: ("bool_value",),
            FeatureValueKind.INT: ("int_value",),
            FeatureValueKind.TEXT: ("text_value",),
            FeatureValueKind.ENUM: ("enum_type", "enum_value"),
        }[self.kind]
        values = {
            "bool_value": self.bool_value,
            "int_value": self.int_value,
            "text_value": self.text_value,
            "enum_type": self.enum_type,
            "enum_value": self.enum_value,
        }
        if any(values[name] is not None for name in values.keys() - set(active)):
            raise TraceValidationError("canonical feature value has inactive payload")
        if self.kind == FeatureValueKind.ABSENT:
            return
        if self.kind == FeatureValueKind.BOOL:
            if type(self.bool_value) is not bool:
                raise TraceValidationError("BOOL feature value has the wrong type")
            return
        if self.kind == FeatureValueKind.INT:
            if type(self.int_value) is not int:
                raise TraceValidationError("INT feature value has the wrong type")
            return
        if self.kind == FeatureValueKind.TEXT:
            if type(self.text_value) is not str:
                raise TraceValidationError("TEXT feature value has the wrong type")
            if len(self.text_value.encode("utf-8")) > MAX_DERIVATION_TEXT_BYTES:
                raise TraceValidationError("TEXT feature value exceeds the byte limit")
            return
        require_text("feature enum type", self.enum_type or "")
        require_text("feature enum value", self.enum_value or "")
        if (
            len(self.enum_type.encode("utf-8")) > MAX_DERIVATION_TEXT_BYTES
            or len(self.enum_value.encode("utf-8")) > MAX_DERIVATION_TEXT_BYTES
        ):
            raise TraceValidationError("ENUM feature value exceeds the byte limit")


def absent_feature_value() -> CanonicalFeatureValue:
    return CanonicalFeatureValue(
        schema_version=FEATURE_VALUE_SCHEMA_VERSION,
        kind=FeatureValueKind.ABSENT,
        bool_value=None,
        int_value=None,
        text_value=None,
        enum_type=None,
        enum_value=None,
    )


def bool_feature_value(value: bool) -> CanonicalFeatureValue:
    return CanonicalFeatureValue(
        schema_version=FEATURE_VALUE_SCHEMA_VERSION,
        kind=FeatureValueKind.BOOL,
        bool_value=value,
        int_value=None,
        text_value=None,
        enum_type=None,
        enum_value=None,
    )


def int_feature_value(value: int) -> CanonicalFeatureValue:
    return CanonicalFeatureValue(
        schema_version=FEATURE_VALUE_SCHEMA_VERSION,
        kind=FeatureValueKind.INT,
        bool_value=None,
        int_value=value,
        text_value=None,
        enum_type=None,
        enum_value=None,
    )


def text_feature_value(value: str) -> CanonicalFeatureValue:
    return CanonicalFeatureValue(
        schema_version=FEATURE_VALUE_SCHEMA_VERSION,
        kind=FeatureValueKind.TEXT,
        bool_value=None,
        int_value=None,
        text_value=value,
        enum_type=None,
        enum_value=None,
    )


def enum_feature_value(enum_type: str, enum_value: str) -> CanonicalFeatureValue:
    return CanonicalFeatureValue(
        schema_version=FEATURE_VALUE_SCHEMA_VERSION,
        kind=FeatureValueKind.ENUM,
        bool_value=None,
        int_value=None,
        text_value=None,
        enum_type=enum_type,
        enum_value=enum_value,
    )


@dataclass(frozen=True, slots=True)
class DerivationDependencySelector:
    kind: DependencySelectorKind
    source_schema: FeatureSourceSchema | None
    field_path: str | None
    availability_rule_id: str | None
    derivation_id: str | None

    def __post_init__(self) -> None:
        if type(self.kind) is not DependencySelectorKind:
            raise TraceValidationError("dependency selector has an invalid kind")
        if self.kind == DependencySelectorKind.SOURCE_EXACT_ALL:
            if type(self.source_schema) is not FeatureSourceSchema:
                raise TraceValidationError("source selector has an invalid schema")
            _require_bounded_text(
                "source selector field path",
                self.field_path or "",
            )
            require_sha256(
                "source selector availability-rule ID",
                self.availability_rule_id or "",
            )
            if self.derivation_id is not None:
                raise TraceValidationError("source selector names a derivation")
            return
        if any(
            value is not None
            for value in (
                self.source_schema,
                self.field_path,
                self.availability_rule_id,
            )
        ):
            raise TraceValidationError("derived selector contains source fields")
        require_sha256("derived selector derivation ID", self.derivation_id or "")


def source_exact_all_selector(
    source_schema: FeatureSourceSchema,
    field_path: str,
    availability_rule_id: str,
) -> DerivationDependencySelector:
    return DerivationDependencySelector(
        kind=DependencySelectorKind.SOURCE_EXACT_ALL,
        source_schema=source_schema,
        field_path=field_path,
        availability_rule_id=availability_rule_id,
        derivation_id=None,
    )


def derivation_exact_one_selector(
    derivation_id: str,
) -> DerivationDependencySelector:
    return DerivationDependencySelector(
        kind=DependencySelectorKind.DERIVATION_EXACT_ONE,
        source_schema=None,
        field_path=None,
        availability_rule_id=None,
        derivation_id=derivation_id,
    )


@dataclass(frozen=True, slots=True)
class DerivationDependencySlot:
    slot_id: str
    selector: DerivationDependencySelector
    allowed_kinds: tuple[FeatureValueKind, ...]

    def __post_init__(self) -> None:
        _require_bounded_text("derivation dependency slot ID", self.slot_id)
        if type(self.selector) is not DerivationDependencySelector:
            raise TraceValidationError("dependency slot has an invalid selector")
        self.selector.__post_init__()
        if type(self.allowed_kinds) is not tuple or not self.allowed_kinds:
            raise TraceValidationError("dependency slot requires allowed kinds")
        if any(type(kind) is not FeatureValueKind for kind in self.allowed_kinds):
            raise TraceValidationError("dependency slot has an invalid value kind")
        kind_values = tuple(kind.value for kind in self.allowed_kinds)
        if kind_values != tuple(sorted(kind_values)) or len(kind_values) != len(
            set(kind_values)
        ):
            raise TraceValidationError(
                "dependency-slot value kinds must be sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class DerivationParameters:
    clamp_low: int | None = None
    clamp_high: int | None = None
    bucket_edges: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.clamp_low is not None and type(self.clamp_low) is not int:
            raise TraceValidationError("clamp lower bound has the wrong type")
        if self.clamp_high is not None and type(self.clamp_high) is not int:
            raise TraceValidationError("clamp upper bound has the wrong type")
        if type(self.bucket_edges) is not tuple or any(
            type(edge) is not int for edge in self.bucket_edges
        ):
            raise TraceValidationError("bucket edges must be exact integers")
        if len(self.bucket_edges) > MAX_DERIVATION_BUCKET_EDGES:
            raise TraceValidationError("bucket-edge ceiling exceeded")
        if self.bucket_edges != tuple(sorted(self.bucket_edges)) or len(
            self.bucket_edges
        ) != len(set(self.bucket_edges)):
            raise TraceValidationError("bucket edges must be sorted and unique")


@dataclass(frozen=True, slots=True)
class DerivationEntry:
    schema_version: str
    derivation_id: str
    derivation_key: str
    opcode: DerivationOpcode
    dependency_slots: tuple[DerivationDependencySlot, ...]
    output_kind: FeatureValueKind
    parameters: DerivationParameters

    def __post_init__(self) -> None:
        if self.schema_version != DERIVATION_ENTRY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported derivation-entry schema")
        require_sha256("derivation ID", self.derivation_id)
        _require_bounded_text("derivation key", self.derivation_key)
        if type(self.opcode) is not DerivationOpcode:
            raise TraceValidationError("derivation entry has an invalid opcode")
        if type(self.dependency_slots) is not tuple or any(
            type(slot) is not DerivationDependencySlot for slot in self.dependency_slots
        ):
            raise TraceValidationError("derivation entry has invalid dependency slots")
        if len(self.dependency_slots) > MAX_DERIVATION_DEPENDENCIES:
            raise TraceValidationError("derivation dependency-slot ceiling exceeded")
        for slot in self.dependency_slots:
            slot.__post_init__()
        slot_ids = tuple(slot.slot_id for slot in self.dependency_slots)
        if len(slot_ids) != len(set(slot_ids)):
            raise TraceValidationError("derivation dependency slot IDs are duplicated")
        selectors = tuple(slot.selector for slot in self.dependency_slots)
        if len(selectors) != len(set(selectors)):
            raise TraceValidationError("derivation dependency selectors are duplicated")
        if type(self.output_kind) is not FeatureValueKind:
            raise TraceValidationError("derivation entry has an invalid output kind")
        if type(self.parameters) is not DerivationParameters:
            raise TraceValidationError("derivation entry has invalid parameters")
        self.parameters.__post_init__()
        _validate_entry_semantics(self)
        if self.derivation_id != canonical_digest(_derivation_identity(self)):
            raise TraceValidationError("derivation ID differs from its definition")


@dataclass(frozen=True, slots=True)
class DerivationSafetyCeiling:
    schema_version: str
    max_entries: int
    max_declared_dependencies: int
    max_graph_values: int
    max_expanded_edges: int
    max_text_bytes: int
    max_bucket_edges: int
    max_total_text_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != DERIVATION_SAFETY_CEILING_SCHEMA_VERSION:
            raise TraceValidationError("unsupported derivation safety ceiling")
        for name in (
            "max_entries",
            "max_declared_dependencies",
            "max_graph_values",
            "max_expanded_edges",
            "max_text_bytes",
            "max_bucket_edges",
            "max_total_text_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TraceValidationError(
                    f"derivation safety {name} must be a positive integer"
                )


def _current_derivation_safety_ceiling() -> DerivationSafetyCeiling:
    return DerivationSafetyCeiling(
        schema_version=DERIVATION_SAFETY_CEILING_SCHEMA_VERSION,
        max_entries=MAX_DERIVATION_ENTRIES,
        max_declared_dependencies=MAX_DERIVATION_DEPENDENCIES,
        max_graph_values=MAX_DERIVATION_GRAPH_VALUES,
        max_expanded_edges=MAX_DERIVATION_EXPANDED_EDGES,
        max_text_bytes=MAX_DERIVATION_TEXT_BYTES,
        max_bucket_edges=MAX_DERIVATION_BUCKET_EDGES,
        max_total_text_bytes=MAX_DERIVATION_TOTAL_TEXT_BYTES,
    )


@dataclass(frozen=True, slots=True)
class DerivationRegistry:
    schema_version: str
    safety_ceiling: DerivationSafetyCeiling
    feature_contract_digest: str
    feature_contract: FeatureContract
    entries: tuple[DerivationEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != DERIVATION_REGISTRY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported derivation-registry schema")
        if (
            type(self.safety_ceiling) is not DerivationSafetyCeiling
            or self.safety_ceiling != _current_derivation_safety_ceiling()
        ):
            raise TraceValidationError("derivation-registry safety ceiling differs")
        require_sha256(
            "derivation-registry feature-contract digest", self.feature_contract_digest
        )
        if type(self.feature_contract) is not FeatureContract:
            raise TraceValidationError("derivation registry has an invalid contract")
        validate_feature_contract(self.feature_contract)
        if self.feature_contract_digest != canonical_digest(self.feature_contract):
            raise TraceValidationError(
                "derivation-registry feature-contract digest differs"
            )
        if type(self.entries) is not tuple or len(self.entries) > (
            self.safety_ceiling.max_entries
        ):
            raise TraceValidationError("derivation-registry entries are invalid")
        if any(type(entry) is not DerivationEntry for entry in self.entries):
            raise TraceValidationError("derivation registry has an invalid entry")
        for entry in self.entries:
            entry.__post_init__()
        entry_keys = tuple(entry.derivation_key for entry in self.entries)
        if entry_keys != tuple(sorted(entry_keys)) or len(entry_keys) != len(
            set(entry_keys)
        ):
            raise TraceValidationError(
                "derivation-registry entries must be key-sorted and unique"
            )
        entry_ids = tuple(entry.derivation_id for entry in self.entries)
        if len(entry_ids) != len(set(entry_ids)):
            raise TraceValidationError("derivation-registry IDs are duplicated")
        if sum(len(entry.dependency_slots) for entry in self.entries) > (
            self.safety_ceiling.max_declared_dependencies
        ):
            raise TraceValidationError(
                "derivation-registry dependency ceiling exceeded"
            )
        total_text_bytes = sum(
            len(entry.derivation_key.encode("utf-8"))
            + sum(
                len(slot.slot_id.encode("utf-8"))
                + len((slot.selector.field_path or "").encode("utf-8"))
                for slot in entry.dependency_slots
            )
            for entry in self.entries
        )
        if total_text_bytes > self.safety_ceiling.max_total_text_bytes:
            raise TraceValidationError("derivation-registry text ceiling exceeded")
        _validate_registry_bindings(self)


@dataclass(frozen=True, slots=True)
class SourceFeatureValue:
    """One source-bound raw value; provenance is replayed by feature_view.v1."""

    schema_version: str
    value_id: str
    feature_contract_digest: str
    source_schema: FeatureSourceSchema
    field_path: str
    source_artifact_id: str
    source_record_ref: str
    source_record_digest: str
    element_identity: str
    availability_kind: FeatureAvailabilityKind
    availability_rule_id: str
    sequence_rule_id: str
    committed_ns: int
    event_ns: int | None
    lookback_start_ns: int
    value: CanonicalFeatureValue
    value_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_FEATURE_VALUE_SCHEMA_VERSION:
            raise TraceValidationError("unsupported source-feature-value schema")
        require_sha256("source feature value ID", self.value_id)
        require_sha256("source feature contract digest", self.feature_contract_digest)
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("source feature has an invalid schema")
        _require_bounded_text("source feature field path", self.field_path)
        require_sha256("source feature artifact ID", self.source_artifact_id)
        _require_bounded_text(
            "source feature record reference",
            self.source_record_ref,
        )
        require_sha256("source feature record digest", self.source_record_digest)
        require_sha256("source feature element identity", self.element_identity)
        if type(self.availability_kind) is not FeatureAvailabilityKind:
            raise TraceValidationError("source feature has invalid availability")
        require_sha256("source feature availability-rule ID", self.availability_rule_id)
        require_sha256("source feature sequence-rule ID", self.sequence_rule_id)
        _require_nonnegative_int("source feature committed_ns", self.committed_ns)
        _require_nonnegative_int(
            "source feature lookback_start_ns", self.lookback_start_ns
        )
        if self.availability_kind == FeatureAvailabilityKind.WINDOWED_EVENT:
            if self.event_ns is None:
                raise TraceValidationError("windowed source feature lacks event_ns")
            _require_nonnegative_int("source feature event_ns", self.event_ns)
            if self.event_ns < self.lookback_start_ns:
                raise TraceValidationError(
                    "source feature predates its lookback window"
                )
            if self.event_ns > self.committed_ns:
                raise TraceValidationError("source feature event follows its commit")
        elif self.event_ns is not None:
            raise TraceValidationError("static source feature contains event_ns")
        if type(self.value) is not CanonicalFeatureValue:
            raise TraceValidationError("source feature has an invalid typed value")
        self.value.__post_init__()
        require_sha256("source feature value digest", self.value_digest)
        if self.value_digest != canonical_digest(self.value):
            raise TraceValidationError("source feature value digest differs")
        if self.value_id != canonical_digest(_source_value_identity(self)):
            raise TraceValidationError("source feature value ID differs")


@dataclass(frozen=True, slots=True)
class DerivedFeatureValue:
    schema_version: str
    registry_digest: str
    derivation_id: str
    dependency_value_ids: tuple[str, ...]
    output: CanonicalFeatureValue
    output_digest: str
    value_id: str

    def __post_init__(self) -> None:
        if self.schema_version != DERIVED_FEATURE_VALUE_SCHEMA_VERSION:
            raise TraceValidationError("unsupported derived-feature-value schema")
        require_sha256("derived-value registry digest", self.registry_digest)
        require_sha256("derived-value derivation ID", self.derivation_id)
        if (
            type(self.dependency_value_ids) is not tuple
            or len(self.dependency_value_ids) > MAX_DERIVATION_DEPENDENCIES
        ):
            raise TraceValidationError("derived-value dependencies are invalid")
        for value_id in self.dependency_value_ids:
            require_sha256("derived dependency value ID", value_id)
        if len(self.dependency_value_ids) != len(set(self.dependency_value_ids)):
            raise TraceValidationError("derived dependencies are duplicated")
        if type(self.output) is not CanonicalFeatureValue:
            raise TraceValidationError("derived feature has an invalid output")
        self.output.__post_init__()
        require_sha256("derived output digest", self.output_digest)
        if self.output_digest != canonical_digest(self.output):
            raise TraceValidationError("derived output digest differs")
        require_sha256("derived feature value ID", self.value_id)
        if self.value_id != canonical_digest(_derived_value_identity(self)):
            raise TraceValidationError("derived feature value ID differs")


@dataclass(frozen=True, slots=True)
class _DerivationIdentity:
    schema_version: str
    derivation_key: str
    opcode: DerivationOpcode
    dependency_slots: tuple[DerivationDependencySlot, ...]
    output_kind: FeatureValueKind
    parameters: DerivationParameters


@dataclass(frozen=True, slots=True)
class _SourceValueIdentity:
    schema_version: str
    feature_contract_digest: str
    field_path: str
    source_artifact_id: str
    source_record_ref: str
    element_identity: str
    typed_value_digest: str
    availability_kind: FeatureAvailabilityKind


@dataclass(frozen=True, slots=True)
class _DerivedValueIdentity:
    registry_digest: str
    derivation_id: str
    dependency_value_ids: tuple[str, ...]
    output: CanonicalFeatureValue
    output_digest: str


@dataclass(frozen=True, slots=True)
class _ResolvedFeatureValue:
    value_id: str
    value: CanonicalFeatureValue


def _require_nonnegative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise TraceValidationError(f"{name} must be a nonnegative integer")


def _require_bounded_text(name: str, value: str) -> None:
    require_text(name, value)
    if len(value.encode("utf-8")) > MAX_DERIVATION_TEXT_BYTES:
        raise TraceValidationError(f"{name} exceeds the byte limit")


def _derivation_identity(entry: DerivationEntry) -> _DerivationIdentity:
    return _DerivationIdentity(
        schema_version=entry.schema_version,
        derivation_key=entry.derivation_key,
        opcode=entry.opcode,
        dependency_slots=entry.dependency_slots,
        output_kind=entry.output_kind,
        parameters=entry.parameters,
    )


def _source_value_identity(value: SourceFeatureValue) -> _SourceValueIdentity:
    return _SourceValueIdentity(
        schema_version=value.schema_version,
        feature_contract_digest=value.feature_contract_digest,
        field_path=value.field_path,
        source_artifact_id=value.source_artifact_id,
        source_record_ref=value.source_record_ref,
        element_identity=value.element_identity,
        typed_value_digest=value.value_digest,
        availability_kind=value.availability_kind,
    )


def _derived_value_identity(value: DerivedFeatureValue) -> _DerivedValueIdentity:
    return _DerivedValueIdentity(
        registry_digest=value.registry_digest,
        derivation_id=value.derivation_id,
        dependency_value_ids=value.dependency_value_ids,
        output=value.output,
        output_digest=value.output_digest,
    )


def _parameters_are_empty(parameters: DerivationParameters) -> bool:
    return (
        parameters.clamp_low is None
        and parameters.clamp_high is None
        and not parameters.bucket_edges
    )


def _all_slots_have_kind(
    entry: DerivationEntry,
    allowed_kinds: tuple[FeatureValueKind, ...],
) -> bool:
    return all(slot.allowed_kinds == allowed_kinds for slot in entry.dependency_slots)


def _all_slots_are_exact_one(entry: DerivationEntry) -> bool:
    return all(
        slot.selector.kind == DependencySelectorKind.DERIVATION_EXACT_ONE
        for slot in entry.dependency_slots
    )


def _validate_entry_semantics(entry: DerivationEntry) -> None:
    integer = (FeatureValueKind.INT,)
    boolean = (FeatureValueKind.BOOL,)
    empty_parameters = _parameters_are_empty(entry.parameters)
    if entry.opcode == DerivationOpcode.IDENTITY:
        if (
            len(entry.dependency_slots) != 1
            or not _all_slots_are_exact_one(entry)
            or entry.dependency_slots[0].allowed_kinds != (entry.output_kind,)
            or not empty_parameters
        ):
            raise TraceValidationError("IDENTITY signature differs")
        return
    if entry.opcode == DerivationOpcode.COUNT:
        if entry.output_kind != FeatureValueKind.INT or not empty_parameters:
            raise TraceValidationError("COUNT signature differs")
        return
    if entry.opcode == DerivationOpcode.SUM_INT:
        minimum_slots = 0
        allowed_kinds = integer
    elif entry.opcode in {DerivationOpcode.MIN_INT, DerivationOpcode.MAX_INT}:
        minimum_slots = 1
        allowed_kinds = integer
    elif entry.opcode in {DerivationOpcode.BOOL_ALL, DerivationOpcode.BOOL_ANY}:
        if (
            not entry.dependency_slots
            or not _all_slots_have_kind(entry, boolean)
            or entry.output_kind != FeatureValueKind.BOOL
            or not empty_parameters
        ):
            raise TraceValidationError("boolean derivation signature differs")
        return
    elif entry.opcode == DerivationOpcode.SUB_INT:
        if (
            len(entry.dependency_slots) != 2
            or not _all_slots_are_exact_one(entry)
            or not _all_slots_have_kind(entry, integer)
        ):
            raise TraceValidationError("SUB_INT signature differs")
        minimum_slots = 2
        allowed_kinds = integer
    elif entry.opcode == DerivationOpcode.EQUAL:
        if (
            len(entry.dependency_slots) != 2
            or not _all_slots_are_exact_one(entry)
            or len(entry.dependency_slots[0].allowed_kinds) != 1
            or entry.dependency_slots[0].allowed_kinds
            != entry.dependency_slots[1].allowed_kinds
            or entry.output_kind != FeatureValueKind.BOOL
            or not empty_parameters
        ):
            raise TraceValidationError("EQUAL signature differs")
        return
    elif entry.opcode in {
        DerivationOpcode.CLAMP_INT,
        DerivationOpcode.RIGHT_CLOSED_BUCKET_INT,
    }:
        if (
            len(entry.dependency_slots) != 1
            or not _all_slots_are_exact_one(entry)
            or not _all_slots_have_kind(entry, integer)
        ):
            raise TraceValidationError("unary integer signature differs")
        minimum_slots = 1
        allowed_kinds = integer
    else:
        raise TraceValidationError("unsupported derivation opcode")

    if (
        len(entry.dependency_slots) < minimum_slots
        or not _all_slots_have_kind(entry, allowed_kinds)
        or entry.output_kind != FeatureValueKind.INT
    ):
        raise TraceValidationError("integer derivation signature differs")
    if entry.opcode == DerivationOpcode.CLAMP_INT:
        if (
            entry.parameters.clamp_low is None
            or entry.parameters.clamp_high is None
            or entry.parameters.clamp_low > entry.parameters.clamp_high
            or entry.parameters.bucket_edges
        ):
            raise TraceValidationError("CLAMP_INT parameters are invalid")
        return
    if entry.opcode == DerivationOpcode.RIGHT_CLOSED_BUCKET_INT:
        if (
            entry.parameters.clamp_low is not None
            or entry.parameters.clamp_high is not None
        ):
            raise TraceValidationError("RIGHT_CLOSED_BUCKET_INT parameters are invalid")
        return
    if not empty_parameters:
        raise TraceValidationError("integer derivation cannot have parameters")


def build_derivation_entry(
    *,
    derivation_key: str,
    opcode: DerivationOpcode,
    dependency_slots: tuple[DerivationDependencySlot, ...],
    output_kind: FeatureValueKind,
    parameters: DerivationParameters | None = None,
) -> DerivationEntry:
    actual_parameters = DerivationParameters() if parameters is None else parameters
    identity = _DerivationIdentity(
        schema_version=DERIVATION_ENTRY_SCHEMA_VERSION,
        derivation_key=derivation_key,
        opcode=opcode,
        dependency_slots=dependency_slots,
        output_kind=output_kind,
        parameters=actual_parameters,
    )
    return DerivationEntry(
        schema_version=identity.schema_version,
        derivation_id=canonical_digest(identity),
        derivation_key=identity.derivation_key,
        opcode=identity.opcode,
        dependency_slots=identity.dependency_slots,
        output_kind=identity.output_kind,
        parameters=identity.parameters,
    )


def _validate_registry_bindings(registry: DerivationRegistry) -> None:
    assignments = {
        (assignment.source_schema, assignment.field_path): assignment
        for assignment in registry.feature_contract.assignments
    }
    paths = {
        (path.source_schema, path.field_path): path
        for path in registry.feature_contract.field_path_catalog.paths
    }
    rules = {
        (rule.source_schema, rule.field_path): rule
        for rule in registry.feature_contract.availability_rule_catalog.rules
    }
    entries = {entry.derivation_id: entry for entry in registry.entries}
    for entry in registry.entries:
        for slot in entry.dependency_slots:
            selector = slot.selector
            if selector.kind == DependencySelectorKind.SOURCE_EXACT_ALL:
                key = (selector.source_schema, selector.field_path)
                assignment = assignments.get(key)
                path = paths.get(key)
                rule = rules.get(key)
                if (
                    assignment is None
                    or assignment.classification != FeatureClassification.ONLINE_ALLOWED
                    or assignment.availability_rule_id != selector.availability_rule_id
                    or rule is None
                    or rule.rule_id != selector.availability_rule_id
                ):
                    raise TraceValidationError(
                        "source selector is outside the exact online contract"
                    )
                if path is None or slot.allowed_kinds != path.value_kinds:
                    raise TraceValidationError(
                        "source selector expected types differ from its field path"
                    )
                continue
            target = entries.get(selector.derivation_id or "")
            if target is None:
                raise TraceValidationError("derived selector names an unknown entry")
            if slot.allowed_kinds != (target.output_kind,):
                raise TraceValidationError(
                    "derived selector expected type differs from its output"
                )
    _topological_derivation_ids(entries)


def _topological_derivation_ids(
    entries: dict[str, DerivationEntry],
) -> tuple[str, ...]:
    """Return a deterministic dependency-first order without Python recursion."""

    indegrees = {derivation_id: 0 for derivation_id in entries}
    dependents: dict[str, list[str]] = {derivation_id: [] for derivation_id in entries}
    for entry in entries.values():
        for slot in entry.dependency_slots:
            if slot.selector.kind != DependencySelectorKind.DERIVATION_EXACT_ONE:
                continue
            target_id = slot.selector.derivation_id or ""
            if target_id not in entries:
                raise TraceValidationError("derived selector names an unknown entry")
            indegrees[entry.derivation_id] += 1
            dependents[target_id].append(entry.derivation_id)

    ready: list[tuple[str, str]] = []
    for derivation_id, degree in indegrees.items():
        if degree == 0:
            heappush(ready, (entries[derivation_id].derivation_key, derivation_id))
    ordered: list[str] = []
    while ready:
        _, derivation_id = heappop(ready)
        ordered.append(derivation_id)
        for dependent_id in dependents[derivation_id]:
            indegrees[dependent_id] -= 1
            if indegrees[dependent_id] == 0:
                heappush(
                    ready,
                    (entries[dependent_id].derivation_key, dependent_id),
                )
    if len(ordered) != len(entries):
        raise TraceValidationError("derivation registry contains a cycle")
    return tuple(ordered)


def build_derivation_registry(
    feature_contract: FeatureContract,
    entries: tuple[DerivationEntry, ...],
) -> DerivationRegistry:
    validate_feature_contract(feature_contract)
    if type(entries) is not tuple:
        raise TraceValidationError("derivation-registry entries must be a tuple")
    if any(type(entry) is not DerivationEntry for entry in entries):
        raise TraceValidationError("derivation registry has an invalid entry")
    return DerivationRegistry(
        schema_version=DERIVATION_REGISTRY_SCHEMA_VERSION,
        safety_ceiling=_current_derivation_safety_ceiling(),
        feature_contract_digest=canonical_digest(feature_contract),
        feature_contract=feature_contract,
        entries=tuple(sorted(entries, key=lambda entry: entry.derivation_key)),
    )


def validate_derivation_registry(registry: DerivationRegistry) -> None:
    if type(registry) is not DerivationRegistry:
        raise TraceValidationError("derivation registry has the wrong type")
    raw = canonical_json(registry)
    replayed = parse_canonical_dataclass(
        raw,
        DerivationRegistry,
        artifact_name="derivation-registry dependency",
        max_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )
    if replayed != registry:
        raise TraceValidationError("derivation registry changes during replay")


def _validate_opcode_values(
    entry: DerivationEntry,
    dependencies: tuple[CanonicalFeatureValue, ...],
) -> None:
    entry.__post_init__()
    if len(dependencies) > MAX_DERIVATION_DEPENDENCIES:
        raise TraceValidationError("derivation dependency ceiling exceeded")
    for value in dependencies:
        if type(value) is not CanonicalFeatureValue:
            raise TraceValidationError("derivation dependency has an invalid value")
        value.__post_init__()
    count = len(dependencies)
    if entry.opcode == DerivationOpcode.IDENTITY:
        valid_count = count == 1
        allowed_kinds = (entry.output_kind,)
    elif entry.opcode == DerivationOpcode.COUNT:
        valid_count = True
        allowed_kinds = tuple(FeatureValueKind)
    elif entry.opcode == DerivationOpcode.SUM_INT:
        valid_count = True
        allowed_kinds = (FeatureValueKind.INT,)
    elif entry.opcode in {DerivationOpcode.MIN_INT, DerivationOpcode.MAX_INT}:
        valid_count = count >= 1
        allowed_kinds = (FeatureValueKind.INT,)
    elif entry.opcode == DerivationOpcode.SUB_INT:
        valid_count = count == 2
        allowed_kinds = (FeatureValueKind.INT,)
    elif entry.opcode in {DerivationOpcode.BOOL_ALL, DerivationOpcode.BOOL_ANY}:
        valid_count = count >= 1
        allowed_kinds = (FeatureValueKind.BOOL,)
    elif entry.opcode == DerivationOpcode.EQUAL:
        valid_count = count == 2
        allowed_kinds = entry.dependency_slots[0].allowed_kinds
    else:
        valid_count = count == 1
        allowed_kinds = (FeatureValueKind.INT,)
    if not valid_count:
        raise TraceValidationError("derivation dependency count differs")
    if any(value.kind not in allowed_kinds for value in dependencies):
        raise TraceValidationError("derivation dependency kind differs")
    if entry.opcode == DerivationOpcode.EQUAL and (
        dependencies[0].kind != dependencies[1].kind
    ):
        raise TraceValidationError("EQUAL dependencies have different types")
    if (
        entry.opcode == DerivationOpcode.EQUAL
        and dependencies[0].kind == FeatureValueKind.ENUM
        and dependencies[0].enum_type != dependencies[1].enum_type
    ):
        raise TraceValidationError("EQUAL enum dependencies have different types")


def _integer_payload(value: CanonicalFeatureValue) -> int:
    if value.kind != FeatureValueKind.INT or value.int_value is None:
        raise TraceValidationError("integer derivation lacks an exact integer")
    return value.int_value


def evaluate_derivation(
    entry: DerivationEntry,
    dependencies: tuple[CanonicalFeatureValue, ...],
) -> CanonicalFeatureValue:
    """Evaluate one closed opcode using exact Python integers and booleans."""

    if type(entry) is not DerivationEntry or type(dependencies) is not tuple:
        raise TraceValidationError("derivation evaluation input has the wrong type")
    _validate_opcode_values(entry, dependencies)
    if entry.opcode == DerivationOpcode.IDENTITY:
        output = dependencies[0]
    elif entry.opcode == DerivationOpcode.COUNT:
        output = int_feature_value(len(dependencies))
    elif entry.opcode == DerivationOpcode.SUM_INT:
        output = int_feature_value(
            sum(_integer_payload(value) for value in dependencies)
        )
    elif entry.opcode == DerivationOpcode.MIN_INT:
        output = int_feature_value(
            min(_integer_payload(value) for value in dependencies)
        )
    elif entry.opcode == DerivationOpcode.MAX_INT:
        output = int_feature_value(
            max(_integer_payload(value) for value in dependencies)
        )
    elif entry.opcode == DerivationOpcode.SUB_INT:
        output = int_feature_value(
            _integer_payload(dependencies[0]) - _integer_payload(dependencies[1])
        )
    elif entry.opcode == DerivationOpcode.BOOL_ALL:
        output = bool_feature_value(
            all(value.bool_value is True for value in dependencies)
        )
    elif entry.opcode == DerivationOpcode.BOOL_ANY:
        output = bool_feature_value(
            any(value.bool_value is True for value in dependencies)
        )
    elif entry.opcode == DerivationOpcode.EQUAL:
        output = bool_feature_value(dependencies[0] == dependencies[1])
    elif entry.opcode == DerivationOpcode.CLAMP_INT:
        low = entry.parameters.clamp_low
        high = entry.parameters.clamp_high
        assert low is not None and high is not None
        output = int_feature_value(
            min(max(_integer_payload(dependencies[0]), low), high)
        )
    elif entry.opcode == DerivationOpcode.RIGHT_CLOSED_BUCKET_INT:
        output = int_feature_value(
            bisect_left(
                entry.parameters.bucket_edges,
                _integer_payload(dependencies[0]),
            )
        )
    else:
        raise TraceValidationError("unsupported derivation opcode")
    if output.kind != entry.output_kind:
        raise TraceValidationError("derivation output kind differs")
    return output


def build_source_feature_value(
    *,
    feature_contract_digest: str,
    source_schema: FeatureSourceSchema,
    field_path: str,
    source_artifact_id: str,
    source_record_ref: str,
    source_record_digest: str,
    element_identity: str,
    availability_kind: FeatureAvailabilityKind,
    availability_rule_id: str,
    sequence_rule_id: str,
    committed_ns: int,
    event_ns: int | None,
    lookback_start_ns: int,
    value: CanonicalFeatureValue,
) -> SourceFeatureValue:
    value_digest = canonical_digest(value)
    identity = _SourceValueIdentity(
        schema_version=SOURCE_FEATURE_VALUE_SCHEMA_VERSION,
        feature_contract_digest=feature_contract_digest,
        field_path=field_path,
        source_artifact_id=source_artifact_id,
        source_record_ref=source_record_ref,
        element_identity=element_identity,
        typed_value_digest=value_digest,
        availability_kind=availability_kind,
    )
    return SourceFeatureValue(
        schema_version=identity.schema_version,
        value_id=canonical_digest(identity),
        feature_contract_digest=identity.feature_contract_digest,
        source_schema=source_schema,
        field_path=identity.field_path,
        source_artifact_id=identity.source_artifact_id,
        source_record_ref=identity.source_record_ref,
        source_record_digest=source_record_digest,
        element_identity=identity.element_identity,
        availability_kind=identity.availability_kind,
        availability_rule_id=availability_rule_id,
        sequence_rule_id=sequence_rule_id,
        committed_ns=committed_ns,
        event_ns=event_ns,
        lookback_start_ns=lookback_start_ns,
        value=value,
        value_digest=value_digest,
    )


def _source_value_sort_key(
    value: SourceFeatureValue,
) -> tuple[str, str, str, str]:
    return (
        value.source_schema.value,
        value.field_path,
        value.element_identity,
        value.value_id,
    )


def _validate_source_values(
    registry: DerivationRegistry,
    source_values: tuple[SourceFeatureValue, ...],
) -> None:
    if type(source_values) is not tuple or len(source_values) > (
        MAX_DERIVATION_GRAPH_VALUES
    ):
        raise TraceValidationError("source feature values have an invalid container")
    if any(type(value) is not SourceFeatureValue for value in source_values):
        raise TraceValidationError("source feature graph contains an invalid value")
    for value in source_values:
        value.__post_init__()
    if source_values != tuple(sorted(source_values, key=_source_value_sort_key)):
        raise TraceValidationError("source feature values are not canonically sorted")
    value_ids = tuple(value.value_id for value in source_values)
    if len(value_ids) != len(set(value_ids)):
        raise TraceValidationError("source feature value IDs are duplicated")
    leaf_keys = tuple(
        (value.source_schema, value.field_path, value.element_identity)
        for value in source_values
    )
    if len(leaf_keys) != len(set(leaf_keys)):
        raise TraceValidationError("source feature path elements are duplicated")
    if source_values:
        envelopes = {
            (
                value.feature_contract_digest,
                value.source_artifact_id,
                value.committed_ns,
                value.lookback_start_ns,
            )
            for value in source_values
        }
        if len(envelopes) != 1:
            raise TraceValidationError("source feature values cross cutoff envelopes")

    assignments = {
        (assignment.source_schema, assignment.field_path): assignment
        for assignment in registry.feature_contract.assignments
    }
    paths = {
        (path.source_schema, path.field_path): path
        for path in registry.feature_contract.field_path_catalog.paths
    }
    rules = {
        (rule.source_schema, rule.field_path): rule
        for rule in registry.feature_contract.availability_rule_catalog.rules
    }
    sequence_rules = {
        (rule.source_schema, rule.collection_path): rule
        for rule in registry.feature_contract.field_path_catalog.sequence_identity_rules
    }
    for value in source_values:
        if value.feature_contract_digest != registry.feature_contract_digest:
            raise TraceValidationError("source feature binds another contract")
        key = (value.source_schema, value.field_path)
        assignment = assignments.get(key)
        path = paths.get(key)
        rule = rules.get(key)
        if (
            assignment is None
            or assignment.classification != FeatureClassification.ONLINE_ALLOWED
            or assignment.availability_rule_id != value.availability_rule_id
            or rule is None
            or rule.rule_id != value.availability_rule_id
            or rule.availability_kind != value.availability_kind
        ):
            raise TraceValidationError("source feature is outside the online contract")
        if path is None or value.value.kind not in path.value_kinds:
            raise TraceValidationError(
                "source feature value kind differs from its path"
            )
        collection_paths = tuple(
            value.field_path[:index]
            for index in range(len(value.field_path))
            if value.field_path.startswith("[*]", index)
        )
        if len(collection_paths) != 1:
            raise TraceValidationError(
                "source feature path lacks one exact sequence identity"
            )
        sequence_rule = sequence_rules.get((value.source_schema, collection_paths[0]))
        if sequence_rule is None or value.sequence_rule_id != sequence_rule.rule_id:
            raise TraceValidationError("source feature sequence-rule ID differs")


def _build_derived_value(
    *,
    registry_digest: str,
    entry: DerivationEntry,
    dependencies: tuple[_ResolvedFeatureValue, ...],
) -> DerivedFeatureValue:
    dependency_ids = tuple(value.value_id for value in dependencies)
    if len(dependency_ids) != len(set(dependency_ids)):
        raise TraceValidationError("expanded derivation dependencies are duplicated")
    output = evaluate_derivation(
        entry,
        tuple(value.value for value in dependencies),
    )
    output_digest = canonical_digest(output)
    identity = _DerivedValueIdentity(
        registry_digest=registry_digest,
        derivation_id=entry.derivation_id,
        dependency_value_ids=dependency_ids,
        output=output,
        output_digest=output_digest,
    )
    return DerivedFeatureValue(
        schema_version=DERIVED_FEATURE_VALUE_SCHEMA_VERSION,
        registry_digest=identity.registry_digest,
        derivation_id=identity.derivation_id,
        dependency_value_ids=identity.dependency_value_ids,
        output=identity.output,
        output_digest=identity.output_digest,
        value_id=canonical_digest(identity),
    )


def build_derived_feature_graph(
    registry: DerivationRegistry,
    source_values: tuple[SourceFeatureValue, ...],
) -> tuple[DerivedFeatureValue, ...]:
    """Expand frozen selectors and compute exactly one node per registry entry."""

    validate_derivation_registry(registry)
    _validate_source_values(registry, source_values)
    if len(source_values) + len(registry.entries) > (
        registry.safety_ceiling.max_graph_values
    ):
        raise TraceValidationError("derived feature graph value ceiling exceeded")
    registry_digest = canonical_digest(registry)
    entries = {entry.derivation_id: entry for entry in registry.entries}
    source_index: dict[
        tuple[FeatureSourceSchema, str, str],
        list[SourceFeatureValue],
    ] = {}
    for value in source_values:
        source_index.setdefault(
            (
                value.source_schema,
                value.field_path,
                value.availability_rule_id,
            ),
            [],
        ).append(value)
    for values in source_index.values():
        values.sort(key=lambda value: (value.element_identity, value.value_id))

    nodes: dict[str, DerivedFeatureValue] = {}
    expanded_edges = 0
    for derivation_id in _topological_derivation_ids(entries):
        entry = entries[derivation_id]
        dependencies: list[_ResolvedFeatureValue] = []
        for slot in entry.dependency_slots:
            selector = slot.selector
            if selector.kind == DependencySelectorKind.SOURCE_EXACT_ALL:
                selected = source_index.get(
                    (
                        selector.source_schema,
                        selector.field_path or "",
                        selector.availability_rule_id or "",
                    ),
                    [],
                )
                expanded_edges += len(selected)
                if expanded_edges > registry.safety_ceiling.max_expanded_edges:
                    raise TraceValidationError(
                        "derived feature graph expanded-edge ceiling exceeded"
                    )
                if any(
                    value.value.kind not in slot.allowed_kinds for value in selected
                ):
                    raise TraceValidationError(
                        "expanded source dependency kind differs"
                    )
                dependencies.extend(
                    _ResolvedFeatureValue(value.value_id, value.value)
                    for value in selected
                )
                continue
            expanded_edges += 1
            if expanded_edges > registry.safety_ceiling.max_expanded_edges:
                raise TraceValidationError(
                    "derived feature graph expanded-edge ceiling exceeded"
                )
            target = nodes.get(selector.derivation_id or "")
            if target is None:
                raise TraceValidationError(
                    "derived selector target is unavailable in topological order"
                )
            if target.output.kind not in slot.allowed_kinds:
                raise TraceValidationError("expanded derived dependency kind differs")
            dependencies.append(_ResolvedFeatureValue(target.value_id, target.output))
        node = _build_derived_value(
            registry_digest=registry_digest,
            entry=entry,
            dependencies=tuple(dependencies),
        )
        nodes[derivation_id] = node
    return tuple(nodes[entry.derivation_id] for entry in registry.entries)


def validate_derived_feature_graph(
    registry: DerivationRegistry,
    source_values: tuple[SourceFeatureValue, ...],
    derived_values: tuple[DerivedFeatureValue, ...],
) -> tuple[DerivedFeatureValue, ...]:
    """Replay all selectors and require the complete registry graph exactly."""

    if type(registry) is not DerivationRegistry:
        raise TraceValidationError("derivation registry has the wrong type")
    if (
        type(derived_values) is not tuple
        or len(derived_values) > MAX_DERIVATION_ENTRIES
        or any(type(value) is not DerivedFeatureValue for value in derived_values)
    ):
        raise TraceValidationError("derived feature values have an invalid container")
    for value in derived_values:
        value.__post_init__()
    expected_ids = tuple(entry.derivation_id for entry in registry.entries)
    observed_ids = tuple(value.derivation_id for value in derived_values)
    if observed_ids != expected_ids:
        raise TraceValidationError(
            "derived feature graph does not cover each registry entry exactly"
        )
    expected = build_derived_feature_graph(registry, source_values)
    if derived_values != expected:
        raise TraceValidationError("derived feature graph differs during replay")
    return expected


def write_derivation_registry(path: Path, artifact: DerivationRegistry) -> str:
    return _write_artifact(
        path,
        artifact,
        DerivationRegistry,
        "derivation-registry artifact",
        validator=validate_derivation_registry,
    )


def load_derivation_registry(
    path: Path,
) -> LoadedFeatureArtifact[DerivationRegistry]:
    return _load_artifact(
        path,
        DerivationRegistry,
        "derivation-registry artifact",
        validator=validate_derivation_registry,
    )
