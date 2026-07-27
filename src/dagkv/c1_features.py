"""Closed C1-B1 feature schemas and exhaustive field classification."""

from __future__ import annotations

import os
import stat
import types
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from dagkv.c1_lifecycle import (
    LIFECYCLE_CLOCK_DOMAIN,
    LIFECYCLE_SIDECAR_SCHEMA_VERSION,
    ClosedLifecycleArtifact,
)
from dagkv.c1_schedule import (
    SCHEDULE_SIDECAR_SCHEMA_VERSION,
    ClosedScheduleArtifact,
)
from dagkv.c1_trace import (
    MAX_TRACE_BYTES,
    TRACE_SCHEMA_VERSION,
    TraceCommitIndeterminateError,
    TraceRecord,
    TraceRecordType,
    TraceValidationError,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
    trace_payload_schema_variants,
)
from dagkv.domain import require_sha256, require_text

FIELD_PATH_CATALOG_SCHEMA_VERSION = "dagkv.m3.field_path_catalog.v1"
AVAILABILITY_RULE_CATALOG_SCHEMA_VERSION = (
    "dagkv.m3.feature_availability_rule_catalog.v1"
)
AVAILABILITY_RULE_IDENTITY_SCHEMA_VERSION = (
    "dagkv.m3.feature_availability_rule_identity.v1"
)
FEATURE_CLASSIFICATION_PROFILE_SCHEMA_VERSION = (
    "dagkv.m3.feature_classification_profile.v1"
)
# Updated only after reviewing the complete generated path-to-class mapping.
FROZEN_FEATURE_CLASSIFICATION_PROFILE_DIGEST = (
    "b5d553978151ed684f7aaed581a8ee1098e2bfcc761a963a99fda84c7048a469"
)
FEATURE_CONTRACT_SCHEMA_VERSION = "dagkv.m3.feature_contract.v1"
FIELD_PATH_GENERATOR_IMPLEMENTATION = "closed_dataclass_paths_v1"
FIELD_PATH_NOTATION = "variant.dataclass_fields.sequence_wildcard_v1"
ELEMENT_IDENTITY_IMPLEMENTATION = "typed_fields_or_scalar_value_v1"
ELEMENT_IDENTITY_COLLISION_POLICY = "reject_duplicate_identity_v1"
MAX_FEATURE_ARTIFACT_BYTES = MAX_TRACE_BYTES
MAX_FEATURE_SCHEMA_ROOTS = 16
MAX_FEATURE_SCHEMA_PATHS = 16_384
MAX_FEATURE_SEQUENCE_RULES = 4_096
MAX_FEATURE_SCHEMA_DEPTH = 64
MAX_FEATURE_PATH_BYTES = 4_096
MAX_FEATURE_TOTAL_TEXT_BYTES = 16 * 1024 * 1024


class FeatureSourceSchema(StrEnum):
    """Closed source schemas whose leaves require classification."""

    TRACE = "TRACE"
    LIFECYCLE_SIDECAR = "LIFECYCLE_SIDECAR"
    SCHEDULE_SIDECAR = "SCHEDULE_SIDECAR"


class FeatureValueKind(StrEnum):
    """Scalar kinds admitted by the canonical source schemas."""

    ABSENT = "ABSENT"
    BOOL = "BOOL"
    ENUM = "ENUM"
    INT = "INT"
    TEXT = "TEXT"


class FeatureClassification(StrEnum):
    """Exhaustive feature-boundary classification."""

    ONLINE_ALLOWED = "ONLINE_ALLOWED"
    LABEL_ONLY = "LABEL_ONLY"
    PROVENANCE_ONLY = "PROVENANCE_ONLY"
    FORBIDDEN_PROXY = "FORBIDDEN_PROXY"


class FeatureAvailabilityKind(StrEnum):
    """Closed cutoff-availability semantics for one online path."""

    IMMUTABLE_STATIC = "IMMUTABLE_STATIC"
    WINDOWED_EVENT = "WINDOWED_EVENT"


class FeatureValueExtractor(StrEnum):
    """Closed source-value extraction implementations."""

    CLOSED_DATACLASS_FIELD_PATH = "CLOSED_DATACLASS_FIELD_PATH_V1"


class FeatureAvailabilityReceipt(StrEnum):
    """Closed evidence kinds later consumed by feature-view replay."""

    TRACE_CUTOFF_COMMIT_BOUNDARY = "TRACE_CUTOFF_COMMIT_BOUNDARY_V1"
    LIFECYCLE_CUTOFF_PREFIX = "LIFECYCLE_CUTOFF_PREFIX_V1"


@dataclass(frozen=True, slots=True)
class FeatureCatalogSafetyCeiling:
    schema_version: str
    max_schema_roots: int
    max_schema_paths: int
    max_sequence_rules: int
    max_schema_depth: int
    max_path_bytes: int
    max_total_text_bytes: int
    max_artifact_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != "dagkv.m3.feature_catalog_safety_ceiling.v1":
            raise TraceValidationError("unsupported feature-catalog safety ceiling")
        for name in (
            "max_schema_roots",
            "max_schema_paths",
            "max_sequence_rules",
            "max_schema_depth",
            "max_path_bytes",
            "max_total_text_bytes",
            "max_artifact_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TraceValidationError(
                    f"feature-catalog {name} must be a positive integer"
                )


def _current_safety_ceiling() -> FeatureCatalogSafetyCeiling:
    return FeatureCatalogSafetyCeiling(
        schema_version="dagkv.m3.feature_catalog_safety_ceiling.v1",
        max_schema_roots=MAX_FEATURE_SCHEMA_ROOTS,
        max_schema_paths=MAX_FEATURE_SCHEMA_PATHS,
        max_sequence_rules=MAX_FEATURE_SEQUENCE_RULES,
        max_schema_depth=MAX_FEATURE_SCHEMA_DEPTH,
        max_path_bytes=MAX_FEATURE_PATH_BYTES,
        max_total_text_bytes=MAX_FEATURE_TOTAL_TEXT_BYTES,
        max_artifact_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )


@dataclass(frozen=True, slots=True)
class SchemaRootBinding:
    source_schema: FeatureSourceSchema
    schema_version: str
    root_type: str
    schema_descriptor_digest: str

    def __post_init__(self) -> None:
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("schema-root source has the wrong type")
        require_text("schema-root version", self.schema_version)
        require_text("schema-root type", self.root_type)
        require_sha256("schema-root descriptor digest", self.schema_descriptor_digest)


@dataclass(frozen=True, slots=True)
class ElementIdentityVariant:
    variant_type: str
    identity_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text("element-identity variant type", self.variant_type)
        _require_sorted_unique_text(
            "element-identity fields",
            self.identity_fields,
        )
        if not self.identity_fields:
            raise TraceValidationError("element identity requires a field")


@dataclass(frozen=True, slots=True)
class SequenceIdentityRule:
    source_schema: FeatureSourceSchema
    collection_path: str
    rule_id: str
    implementation: str
    collision_policy: str
    variants: tuple[ElementIdentityVariant, ...]

    def __post_init__(self) -> None:
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("sequence-rule source has the wrong type")
        _require_schema_path("sequence-rule collection path", self.collection_path)
        require_sha256("sequence identity rule ID", self.rule_id)
        if self.implementation != ELEMENT_IDENTITY_IMPLEMENTATION:
            raise TraceValidationError("unsupported element-identity implementation")
        if self.collision_policy != ELEMENT_IDENTITY_COLLISION_POLICY:
            raise TraceValidationError("unsupported element-identity collision policy")
        _require_tuple("element-identity variants", self.variants)
        if not self.variants or any(
            type(variant) is not ElementIdentityVariant for variant in self.variants
        ):
            raise TraceValidationError("sequence rule has invalid variants")
        variant_types = tuple(variant.variant_type for variant in self.variants)
        if variant_types != tuple(sorted(variant_types)) or len(variant_types) != len(
            set(variant_types)
        ):
            raise TraceValidationError(
                "element-identity variants must be sorted and unique"
            )
        _validate_element_identity_variants(self.variants)
        identity = _SequenceRuleIdentity(
            source_schema=self.source_schema,
            collection_path=self.collection_path,
            implementation=self.implementation,
            collision_policy=self.collision_policy,
            variants=self.variants,
        )
        if self.rule_id != canonical_digest(identity):
            raise TraceValidationError("sequence identity rule ID differs")


@dataclass(frozen=True, slots=True)
class FieldPathEntry:
    source_schema: FeatureSourceSchema
    field_path: str
    value_kinds: tuple[FeatureValueKind, ...]

    def __post_init__(self) -> None:
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("field-path source has the wrong type")
        _require_schema_path("field path", self.field_path)
        _require_tuple("field-path value kinds", self.value_kinds)
        if not self.value_kinds or any(
            type(kind) is not FeatureValueKind for kind in self.value_kinds
        ):
            raise TraceValidationError("field path has invalid value kinds")
        values = tuple(kind.value for kind in self.value_kinds)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise TraceValidationError(
                "field-path value kinds must be sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class FieldPathCatalog:
    schema_version: str
    generator_implementation: str
    path_notation: str
    safety_ceiling: FeatureCatalogSafetyCeiling
    schema_roots: tuple[SchemaRootBinding, ...]
    sequence_identity_rules: tuple[SequenceIdentityRule, ...]
    paths: tuple[FieldPathEntry, ...]

    def __post_init__(self) -> None:
        _require_exact_runtime_type(FieldPathCatalog, self, "field-path catalog")
        if self.schema_version != FIELD_PATH_CATALOG_SCHEMA_VERSION:
            raise TraceValidationError("unsupported field-path-catalog schema")
        if self.generator_implementation != FIELD_PATH_GENERATOR_IMPLEMENTATION:
            raise TraceValidationError("unsupported field-path generator")
        if self.path_notation != FIELD_PATH_NOTATION:
            raise TraceValidationError("unsupported field-path notation")
        if type(self.safety_ceiling) is not FeatureCatalogSafetyCeiling:
            raise TraceValidationError("field-path catalog has an invalid ceiling")
        if self.safety_ceiling != _current_safety_ceiling():
            raise TraceValidationError("field-path catalog ceiling differs")
        _require_tuple("field-path schema roots", self.schema_roots)
        _require_tuple(
            "field-path sequence rules",
            self.sequence_identity_rules,
        )
        _require_tuple("field-path entries", self.paths)
        if len(self.schema_roots) > self.safety_ceiling.max_schema_roots:
            raise TraceValidationError("field-path schema-root ceiling exceeded")
        if len(self.paths) > self.safety_ceiling.max_schema_paths:
            raise TraceValidationError("field-path count ceiling exceeded")
        if len(self.sequence_identity_rules) > (self.safety_ceiling.max_sequence_rules):
            raise TraceValidationError("sequence-rule count ceiling exceeded")
        if any(type(root) is not SchemaRootBinding for root in self.schema_roots):
            raise TraceValidationError("field-path catalog has an invalid root")
        if any(
            type(rule) is not SequenceIdentityRule
            for rule in self.sequence_identity_rules
        ):
            raise TraceValidationError("field-path catalog has an invalid rule")
        if any(type(entry) is not FieldPathEntry for entry in self.paths):
            raise TraceValidationError("field-path catalog has an invalid path")
        root_keys = tuple(
            (root.source_schema.value, root.root_type) for root in self.schema_roots
        )
        if root_keys != tuple(sorted(root_keys)) or len(root_keys) != len(
            set(root_keys)
        ):
            raise TraceValidationError(
                "field-path schema roots must be sorted and unique"
            )
        rule_keys = tuple(
            (rule.source_schema.value, rule.collection_path)
            for rule in self.sequence_identity_rules
        )
        if rule_keys != tuple(sorted(rule_keys)) or len(rule_keys) != len(
            set(rule_keys)
        ):
            raise TraceValidationError(
                "sequence identity rules must be sorted and unique"
            )
        rule_ids = tuple(rule.rule_id for rule in self.sequence_identity_rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise TraceValidationError("sequence identity rule IDs are duplicated")
        if self.sequence_identity_rules != _current_catalog_components()[1]:
            raise TraceValidationError(
                "sequence identity rules differ from current schemas"
            )
        path_keys = tuple(
            (entry.source_schema.value, entry.field_path) for entry in self.paths
        )
        if path_keys != tuple(sorted(path_keys)) or len(path_keys) != len(
            set(path_keys)
        ):
            raise TraceValidationError("field paths must be sorted and unique")
        rule_paths = set(rule_keys)
        for entry in self.paths:
            segments = _wildcard_collection_paths(entry.field_path)
            for collection_path in segments:
                if (entry.source_schema.value, collection_path) not in rule_paths:
                    raise TraceValidationError(
                        "field-path wildcard lacks an element-identity rule"
                    )
        for rule in self.sequence_identity_rules:
            marker = f"{rule.collection_path}[*]"
            if not any(
                entry.source_schema == rule.source_schema
                and (
                    entry.field_path == marker
                    or entry.field_path.startswith(f"{marker}.")
                )
                for entry in self.paths
            ):
                raise TraceValidationError("sequence identity rule has no path")
        total_text_bytes = sum(
            len(value.encode("utf-8")) for value in _catalog_text_values(self)
        )
        if total_text_bytes > self.safety_ceiling.max_total_text_bytes:
            raise TraceValidationError("field-path total-text ceiling exceeded")


@dataclass(frozen=True, slots=True)
class FeatureAvailabilityRule:
    schema_version: str
    rule_id: str
    source_schema: FeatureSourceSchema
    source_schema_descriptor_digest: str
    field_path: str
    availability_kind: FeatureAvailabilityKind
    value_extractor: FeatureValueExtractor
    receipt_kind: FeatureAvailabilityReceipt
    event_time_path: str | None
    clock_domain: str

    def __post_init__(self) -> None:
        if self.schema_version != AVAILABILITY_RULE_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported feature-availability rule schema")
        require_sha256("feature-availability rule ID", self.rule_id)
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("availability-rule source has the wrong type")
        require_sha256(
            "availability-rule schema digest",
            self.source_schema_descriptor_digest,
        )
        _require_schema_path("availability-rule field path", self.field_path)
        if type(self.availability_kind) is not FeatureAvailabilityKind:
            raise TraceValidationError("availability rule has the wrong kind")
        if type(self.value_extractor) is not FeatureValueExtractor:
            raise TraceValidationError("availability rule has an invalid extractor")
        if type(self.receipt_kind) is not FeatureAvailabilityReceipt:
            raise TraceValidationError("availability rule has an invalid receipt")
        if self.event_time_path is not None:
            _require_schema_path(
                "availability-rule event-time path",
                self.event_time_path,
            )
        if self.clock_domain != LIFECYCLE_CLOCK_DOMAIN:
            raise TraceValidationError("availability rule has another clock domain")
        if self.availability_kind == FeatureAvailabilityKind.IMMUTABLE_STATIC:
            if (
                self.source_schema != FeatureSourceSchema.TRACE
                or self.receipt_kind
                != FeatureAvailabilityReceipt.TRACE_CUTOFF_COMMIT_BOUNDARY
                or self.event_time_path is not None
            ):
                raise TraceValidationError(
                    "static availability rule has incompatible evidence"
                )
        elif (
            self.source_schema != FeatureSourceSchema.LIFECYCLE_SIDECAR
            or self.receipt_kind != FeatureAvailabilityReceipt.LIFECYCLE_CUTOFF_PREFIX
            or self.event_time_path
            != "lifecycle_sidecar.ClosedLifecycleArtifact.events[*]."
            "LifecycleEvent.timestamp_ns"
        ):
            raise TraceValidationError(
                "windowed availability rule has incompatible evidence"
            )
        if self.rule_id != canonical_digest(_availability_rule_identity(self)):
            raise TraceValidationError("feature-availability rule ID differs")


@dataclass(frozen=True, slots=True)
class FeatureAvailabilityRuleCatalog:
    schema_version: str
    field_path_catalog_digest: str
    field_path_catalog: FieldPathCatalog
    online_allowlist_digest: str
    rules: tuple[FeatureAvailabilityRule, ...]

    def __post_init__(self) -> None:
        _require_exact_runtime_type(
            FeatureAvailabilityRuleCatalog,
            self,
            "feature-availability rule catalog",
        )
        if self.schema_version != AVAILABILITY_RULE_CATALOG_SCHEMA_VERSION:
            raise TraceValidationError(
                "unsupported feature-availability-rule-catalog schema"
            )
        require_sha256(
            "availability-rule-catalog field-path digest",
            self.field_path_catalog_digest,
        )
        if type(self.field_path_catalog) is not FieldPathCatalog:
            raise TraceValidationError(
                "availability-rule catalog has an invalid field catalog"
            )
        validate_field_path_catalog(self.field_path_catalog)
        if self.field_path_catalog_digest != canonical_digest(self.field_path_catalog):
            raise TraceValidationError(
                "availability-rule catalog field-path digest differs"
            )
        require_sha256(
            "availability-rule-catalog allowlist digest",
            self.online_allowlist_digest,
        )
        if self.online_allowlist_digest != _online_allowlist_digest():
            raise TraceValidationError("availability-rule-catalog allowlist differs")
        _require_tuple("feature availability rules", self.rules)
        if any(type(rule) is not FeatureAvailabilityRule for rule in self.rules):
            raise TraceValidationError("availability-rule catalog has an invalid rule")
        rule_keys = tuple(
            (rule.source_schema.value, rule.field_path) for rule in self.rules
        )
        if rule_keys != tuple(sorted(rule_keys)) or len(rule_keys) != len(
            set(rule_keys)
        ):
            raise TraceValidationError(
                "feature availability rules must be path-sorted and unique"
            )
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise TraceValidationError("feature availability rule IDs are duplicated")
        expected_keys = tuple(
            sorted((source.value, path) for source, path in _online_allowlist_keys())
        )
        if rule_keys != expected_keys:
            raise TraceValidationError(
                "availability rules differ from the frozen online allowlist"
            )
        root_digests = {
            root.source_schema: root.schema_descriptor_digest
            for root in self.field_path_catalog.schema_roots
        }
        catalog_keys = {
            (entry.source_schema, entry.field_path)
            for entry in self.field_path_catalog.paths
        }
        for rule in self.rules:
            if (rule.source_schema, rule.field_path) not in catalog_keys:
                raise TraceValidationError(
                    "availability rule names an unknown field path"
                )
            if root_digests.get(rule.source_schema) != (
                rule.source_schema_descriptor_digest
            ):
                raise TraceValidationError(
                    "availability rule binds another source schema"
                )


@dataclass(frozen=True, slots=True)
class FeatureFieldAssignment:
    source_schema: FeatureSourceSchema
    field_path: str
    classification: FeatureClassification
    availability_rule_id: str | None

    def __post_init__(self) -> None:
        if type(self.source_schema) is not FeatureSourceSchema:
            raise TraceValidationError("feature assignment source has the wrong type")
        _require_schema_path("feature assignment path", self.field_path)
        if type(self.classification) is not FeatureClassification:
            raise TraceValidationError("feature assignment classification is invalid")
        if self.availability_rule_id is not None:
            require_sha256(
                "feature assignment availability-rule ID",
                self.availability_rule_id,
            )
        if self.classification == FeatureClassification.ONLINE_ALLOWED:
            if self.availability_rule_id is None:
                raise TraceValidationError("online feature lacks an availability rule")
        elif self.availability_rule_id is not None:
            raise TraceValidationError(
                "non-online feature cannot bind an availability rule"
            )


@dataclass(frozen=True, slots=True)
class FeatureContract:
    schema_version: str
    field_path_catalog_digest: str
    field_path_catalog: FieldPathCatalog
    online_allowlist_digest: str
    availability_rule_catalog_digest: str
    availability_rule_catalog: FeatureAvailabilityRuleCatalog
    classification_profile_digest: str
    assignments: tuple[FeatureFieldAssignment, ...]

    def __post_init__(self) -> None:
        _require_exact_runtime_type(FeatureContract, self, "feature contract")
        if self.schema_version != FEATURE_CONTRACT_SCHEMA_VERSION:
            raise TraceValidationError("unsupported feature-contract schema")
        require_sha256(
            "feature-contract field-path-catalog digest",
            self.field_path_catalog_digest,
        )
        if type(self.field_path_catalog) is not FieldPathCatalog:
            raise TraceValidationError("feature contract has an invalid catalog")
        validate_field_path_catalog(self.field_path_catalog)
        if self.field_path_catalog_digest != canonical_digest(self.field_path_catalog):
            raise TraceValidationError("feature-contract catalog digest differs")
        require_sha256(
            "feature-contract online allowlist digest",
            self.online_allowlist_digest,
        )
        if self.online_allowlist_digest != _online_allowlist_digest():
            raise TraceValidationError("feature-contract online allowlist differs")
        require_sha256(
            "feature-contract availability-rule catalog digest",
            self.availability_rule_catalog_digest,
        )
        if type(self.availability_rule_catalog) is not FeatureAvailabilityRuleCatalog:
            raise TraceValidationError("feature contract has an invalid rule catalog")
        validate_feature_availability_rule_catalog(self.availability_rule_catalog)
        if self.availability_rule_catalog.field_path_catalog != (
            self.field_path_catalog
        ):
            raise TraceValidationError(
                "feature contract and availability rules bind different catalogs"
            )
        if self.availability_rule_catalog_digest != canonical_digest(
            self.availability_rule_catalog
        ):
            raise TraceValidationError(
                "feature-contract availability-rule catalog digest differs"
            )
        _require_tuple("feature-contract assignments", self.assignments)
        if any(
            type(assignment) is not FeatureFieldAssignment
            for assignment in self.assignments
        ):
            raise TraceValidationError("feature contract has an invalid assignment")
        assignment_keys = tuple(
            (assignment.source_schema.value, assignment.field_path)
            for assignment in self.assignments
        )
        if assignment_keys != tuple(sorted(assignment_keys)) or len(
            assignment_keys
        ) != len(set(assignment_keys)):
            raise TraceValidationError(
                "feature assignments must be path-sorted and unique"
            )
        expected_keys = tuple(
            (entry.source_schema.value, entry.field_path)
            for entry in self.field_path_catalog.paths
        )
        if assignment_keys != expected_keys:
            missing = sorted(set(expected_keys).difference(assignment_keys))
            extra = sorted(set(assignment_keys).difference(expected_keys))
            raise TraceValidationError(
                "feature assignments differ from the field-path catalog: "
                f"missing={missing}, extra={extra}"
            )
        rules_by_key = {
            (rule.source_schema, rule.field_path): rule
            for rule in self.availability_rule_catalog.rules
        }
        observed_online_keys = {
            (assignment.source_schema, assignment.field_path)
            for assignment in self.assignments
            if assignment.classification == FeatureClassification.ONLINE_ALLOWED
        }
        expected_online_keys = _online_allowlist_keys()
        if observed_online_keys != expected_online_keys:
            missing = sorted(
                (source.value, path)
                for source, path in expected_online_keys.difference(
                    observed_online_keys
                )
            )
            extra = sorted(
                (source.value, path)
                for source, path in observed_online_keys.difference(
                    expected_online_keys
                )
            )
            raise TraceValidationError(
                "online assignments differ from the frozen allowlist: "
                f"missing={missing}, extra={extra}"
            )
        for assignment in self.assignments:
            if assignment.classification != FeatureClassification.ONLINE_ALLOWED:
                continue
            rule = rules_by_key.get((assignment.source_schema, assignment.field_path))
            if rule is None or assignment.availability_rule_id != rule.rule_id:
                raise TraceValidationError(
                    "online feature differs from its exact availability rule"
                )
        require_sha256(
            "feature-contract classification-profile digest",
            self.classification_profile_digest,
        )
        expected_profile_digest = _classification_profile_digest(
            self.field_path_catalog
        )
        if (
            self.classification_profile_digest
            != FROZEN_FEATURE_CLASSIFICATION_PROFILE_DIGEST
            or self.classification_profile_digest != expected_profile_digest
        ):
            raise TraceValidationError(
                "feature-contract classification profile differs"
            )
        expected_assignments = build_feature_classification_assignments(
            self.field_path_catalog,
            self.availability_rule_catalog,
        )
        if self.assignments != expected_assignments:
            raise TraceValidationError(
                "feature assignments differ from the frozen classification profile"
            )


@dataclass(frozen=True, slots=True)
class LoadedFeatureArtifact[T]:
    artifact: T
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256("loaded feature artifact digest", self.digest)
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise TraceValidationError("loaded feature artifact size is invalid")


@dataclass(frozen=True, slots=True)
class _SchemaDescriptor:
    source_schema: FeatureSourceSchema
    schema_version: str
    generator_implementation: str
    safety_ceiling: FeatureCatalogSafetyCeiling
    roots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SequenceRuleIdentity:
    source_schema: FeatureSourceSchema
    collection_path: str
    implementation: str
    collision_policy: str
    variants: tuple[ElementIdentityVariant, ...]


@dataclass(frozen=True, slots=True)
class _AvailabilityRuleIdentity:
    schema_version: str
    source_schema: FeatureSourceSchema
    source_schema_descriptor_digest: str
    field_path: str
    availability_kind: FeatureAvailabilityKind
    value_extractor: FeatureValueExtractor
    receipt_kind: FeatureAvailabilityReceipt
    event_time_path: str | None
    clock_domain: str


@dataclass(frozen=True, slots=True)
class _ClassificationProfileEntry:
    source_schema: FeatureSourceSchema
    field_path: str
    value_kinds: tuple[FeatureValueKind, ...]
    classification: FeatureClassification


@dataclass(frozen=True, slots=True)
class _ClassificationProfile:
    schema_version: str
    field_path_catalog_digest: str
    entries: tuple[_ClassificationProfileEntry, ...]


def _availability_rule_identity(
    rule: FeatureAvailabilityRule,
) -> _AvailabilityRuleIdentity:
    return _AvailabilityRuleIdentity(
        schema_version=rule.schema_version,
        source_schema=rule.source_schema,
        source_schema_descriptor_digest=rule.source_schema_descriptor_digest,
        field_path=rule.field_path,
        availability_kind=rule.availability_kind,
        value_extractor=rule.value_extractor,
        receipt_kind=rule.receipt_kind,
        event_time_path=rule.event_time_path,
        clock_domain=rule.clock_domain,
    )


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TraceValidationError(f"{name} must be a tuple")


def _require_exact_runtime_type(
    expected: Any,
    value: object,
    path: str,
    *,
    depth: int = 0,
) -> None:
    if depth > MAX_FEATURE_SCHEMA_DEPTH:
        raise TraceValidationError("feature dependency depth ceiling exceeded")
    origin = get_origin(expected)
    if origin is types.UnionType:
        matches = 0
        for candidate in get_args(expected):
            try:
                _require_exact_runtime_type(
                    candidate,
                    value,
                    path,
                    depth=depth + 1,
                )
            except TraceValidationError:
                continue
            matches += 1
        if matches != 1:
            raise TraceValidationError(f"{path} has an invalid union value")
        return
    if origin is tuple:
        if type(value) is not tuple:
            raise TraceValidationError(f"{path} must be an exact tuple")
        arguments = get_args(expected)
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TraceValidationError(f"{path} has unsupported tuple metadata")
        for index, item in enumerate(value):
            _require_exact_runtime_type(
                arguments[0],
                item,
                f"{path}[{index}]",
                depth=depth + 1,
            )
        return
    if expected is type(None):
        if value is not None:
            raise TraceValidationError(f"{path} must be absent")
        return
    if expected in {str, int, bool}:
        if type(value) is not expected:
            raise TraceValidationError(f"{path} has the wrong scalar type")
        return
    if isinstance(expected, type) and issubclass(expected, StrEnum):
        if type(value) is not expected:
            raise TraceValidationError(f"{path} has the wrong enum type")
        return
    if isinstance(expected, type) and is_dataclass(expected):
        if type(value) is not expected:
            raise TraceValidationError(f"{path} has the wrong dataclass type")
        hints = get_type_hints(expected)
        for field in fields(expected):
            _require_exact_runtime_type(
                hints[field.name],
                getattr(value, field.name),
                f"{path}.{field.name}",
                depth=depth + 1,
            )
        return
    raise TraceValidationError(f"{path} has unsupported runtime metadata")


def _require_sorted_unique_text(name: str, values: tuple[str, ...]) -> None:
    _require_tuple(name, values)
    for value in values:
        require_text(name, value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


def _require_schema_path(name: str, value: str) -> None:
    require_text(name, value)
    if len(value.encode("utf-8")) > MAX_FEATURE_PATH_BYTES:
        raise TraceValidationError(f"{name} exceeds the byte ceiling")
    if "[" in value.replace("[*]", "") or "]" in value.replace("[*]", ""):
        raise TraceValidationError(f"{name} contains a runtime sequence index")
    if ".." in value or value.startswith(".") or value.endswith("."):
        raise TraceValidationError(f"{name} is not canonical")


def _wildcard_collection_paths(path: str) -> tuple[str, ...]:
    values: list[str] = []
    offset = 0
    while True:
        marker = path.find("[*]", offset)
        if marker < 0:
            return tuple(values)
        values.append(path[:marker])
        offset = marker + 3


def _catalog_text_values(catalog: FieldPathCatalog) -> tuple[str, ...]:
    values = [
        catalog.schema_version,
        catalog.generator_implementation,
        catalog.path_notation,
    ]
    for root in catalog.schema_roots:
        values.extend((root.schema_version, root.root_type))
    for rule in catalog.sequence_identity_rules:
        values.extend(
            (
                rule.collection_path,
                rule.implementation,
                rule.collision_policy,
            )
        )
        for variant in rule.variants:
            values.append(variant.variant_type)
            values.extend(variant.identity_fields)
    values.extend(entry.field_path for entry in catalog.paths)
    return tuple(values)


_ELEMENT_IDENTITY_FIELDS: dict[type[Any], tuple[str, ...]] = {}


def _register_element_identity(
    expected: type[Any],
    *identity_fields: str,
) -> None:
    _ELEMENT_IDENTITY_FIELDS[expected] = tuple(sorted(identity_fields))


def _initialize_element_identities() -> None:
    from dagkv.c1_leases import (
        DependenceGroup,
        JointOutcome,
        LeaseOwnerSnapshot,
        ReuseClaim,
    )
    from dagkv.c1_schedule import ScheduleCheckpoint, ScheduleDemandEvent, ScheduleEpoch
    from dagkv.c1_trace import (
        H2DExecMapService,
        H2DFailedService,
        RequestCancelledService,
        ResidentExecMapService,
    )
    from dagkv.domain import LifecycleEvent, ReplicaId, WorkflowNode, WorkflowSpec

    registrations = (
        (DependenceGroup, ("group_id",)),
        (JointOutcome, ("outcome_id",)),
        (LeaseOwnerSnapshot, ("binding_id",)),
        (ReuseClaim, ("claim_id",)),
        (ScheduleCheckpoint, ("checkpoint_id",)),
        (ScheduleDemandEvent, ("schedule_event_id",)),
        (ScheduleEpoch, ("reuse_epoch_id",)),
        (H2DExecMapService, ("intent_record_id",)),
        (H2DFailedService, ("intent_record_id",)),
        (RequestCancelledService, ("intent_record_id",)),
        (ResidentExecMapService, ("intent_record_id",)),
        (LifecycleEvent, ("event_id",)),
        (ReplicaId, ("device_id", "generation", "slot_id", "tier")),
        (WorkflowNode, ("node_id",)),
        (WorkflowSpec, ("key",)),
    )
    for expected, identity_fields in registrations:
        _register_element_identity(expected, *identity_fields)


_initialize_element_identities()


def _type_fingerprint(
    expected: Any,
    *,
    stack: tuple[type[Any], ...] = (),
    depth: int = 0,
) -> str:
    if depth > MAX_FEATURE_SCHEMA_DEPTH:
        raise TraceValidationError("feature schema depth ceiling exceeded")
    origin = get_origin(expected)
    if origin is types.UnionType:
        return (
            "union["
            + ",".join(
                _type_fingerprint(item, stack=stack, depth=depth + 1)
                for item in get_args(expected)
            )
            + "]"
        )
    if origin is tuple:
        arguments = get_args(expected)
        if len(arguments) != 2 or arguments[1] is not Ellipsis:
            raise TraceValidationError("fixed-length tuple metadata is unsupported")
        return (
            "tuple["
            + _type_fingerprint(arguments[0], stack=stack, depth=depth + 1)
            + ",...]"
        )
    if expected is type(None):
        return "none"
    if expected in {str, int, bool}:
        return expected.__name__
    if isinstance(expected, type) and issubclass(expected, StrEnum):
        members = ",".join(f"{item.name}={item.value}" for item in expected)
        return f"enum:{expected.__module__}.{expected.__qualname__}[{members}]"
    if isinstance(expected, type) and is_dataclass(expected):
        if expected in stack:
            raise TraceValidationError("recursive feature schema is unsupported")
        hints = get_type_hints(expected)
        field_descriptors = []
        for field in fields(expected):
            field_type = _type_fingerprint(
                hints[field.name],
                stack=(*stack, expected),
                depth=depth + 1,
            )
            field_descriptors.append(f"{field.name}:{field_type}")
        body = ";".join(field_descriptors)
        return f"dataclass:{expected.__module__}.{expected.__qualname__}{{{body}}}"
    raise TraceValidationError(f"unsupported feature schema metadata: {expected!r}")


def _trace_root_fingerprint(
    record_type: TraceRecordType,
    payload_type: type[Any],
) -> str:
    hints = get_type_hints(TraceRecord)
    field_descriptors = []
    for field in fields(TraceRecord):
        expected = payload_type if field.name == "payload" else hints[field.name]
        field_descriptors.append(f"{field.name}:{_type_fingerprint(expected)}")
    body = ";".join(field_descriptors)
    return (
        f"record_type={record_type.name}={record_type.value};"
        f"payload={payload_type.__module__}.{payload_type.__qualname__};"
        f"TraceRecord{{{body}}}"
    )


def _scalar_kind(expected: Any) -> FeatureValueKind | None:
    if expected is str:
        return FeatureValueKind.TEXT
    if expected is int:
        return FeatureValueKind.INT
    if expected is bool:
        return FeatureValueKind.BOOL
    if isinstance(expected, type) and issubclass(expected, StrEnum):
        return FeatureValueKind.ENUM
    return None


@dataclass(slots=True)
class _CatalogBuilder:
    paths: list[FieldPathEntry]
    rules: list[SequenceIdentityRule]

    def add_path(
        self,
        source_schema: FeatureSourceSchema,
        path: str,
        kinds: tuple[FeatureValueKind, ...],
    ) -> None:
        _require_schema_path("generated field path", path)
        ordered = tuple(sorted(set(kinds), key=lambda kind: kind.value))
        self.paths.append(FieldPathEntry(source_schema, path, ordered))
        if len(self.paths) > MAX_FEATURE_SCHEMA_PATHS:
            raise TraceValidationError("field-path count ceiling exceeded")

    def add_sequence_rule(
        self,
        source_schema: FeatureSourceSchema,
        path: str,
        element_type: Any,
    ) -> None:
        variants = _element_identity_variants(element_type)
        identity = _SequenceRuleIdentity(
            source_schema=source_schema,
            collection_path=path,
            implementation=ELEMENT_IDENTITY_IMPLEMENTATION,
            collision_policy=ELEMENT_IDENTITY_COLLISION_POLICY,
            variants=variants,
        )
        self.rules.append(
            SequenceIdentityRule(
                source_schema=source_schema,
                collection_path=path,
                rule_id=canonical_digest(identity),
                implementation=ELEMENT_IDENTITY_IMPLEMENTATION,
                collision_policy=ELEMENT_IDENTITY_COLLISION_POLICY,
                variants=variants,
            )
        )
        if len(self.rules) > MAX_FEATURE_SEQUENCE_RULES:
            raise TraceValidationError("sequence-rule count ceiling exceeded")

    def walk(
        self,
        source_schema: FeatureSourceSchema,
        expected: Any,
        path: str,
        *,
        stack: tuple[type[Any], ...] = (),
        depth: int = 0,
    ) -> None:
        if depth > MAX_FEATURE_SCHEMA_DEPTH:
            raise TraceValidationError("feature schema depth ceiling exceeded")
        origin = get_origin(expected)
        if origin is types.UnionType:
            arguments = get_args(expected)
            present = tuple(item for item in arguments if item is not type(None))
            has_absent = len(present) != len(arguments)
            scalar_kinds = tuple(
                kind for item in present if (kind := _scalar_kind(item)) is not None
            )
            if len(scalar_kinds) == len(present):
                kinds = (
                    *scalar_kinds,
                    *((FeatureValueKind.ABSENT,) if has_absent else ()),
                )
                self.add_path(source_schema, path, kinds)
                return
            if has_absent:
                self.add_path(source_schema, path, (FeatureValueKind.ABSENT,))
            for item in present:
                variant_path = path
                if len(present) > 1:
                    variant_name = getattr(item, "__name__", None)
                    if variant_name is None:
                        raise TraceValidationError(
                            "union variant lacks a stable type name"
                        )
                    variant_path = f"{path}.{variant_name}"
                self.walk(
                    source_schema,
                    item,
                    variant_path,
                    stack=stack,
                    depth=depth + 1,
                )
            return
        if origin is tuple:
            arguments = get_args(expected)
            if len(arguments) != 2 or arguments[1] is not Ellipsis:
                raise TraceValidationError("fixed-length tuple metadata is unsupported")
            element_type = arguments[0]
            self.add_sequence_rule(source_schema, path, element_type)
            element_path = f"{path}[*]"
            if isinstance(element_type, type) and is_dataclass(element_type):
                element_path = f"{element_path}.{element_type.__name__}"
            self.walk(
                source_schema,
                element_type,
                element_path,
                stack=stack,
                depth=depth + 1,
            )
            return
        kind = _scalar_kind(expected)
        if kind is not None:
            self.add_path(source_schema, path, (kind,))
            return
        if isinstance(expected, type) and is_dataclass(expected):
            if expected in stack:
                raise TraceValidationError("recursive feature schema is unsupported")
            hints = get_type_hints(expected)
            for field in fields(expected):
                self.walk(
                    source_schema,
                    hints[field.name],
                    f"{path}.{field.name}",
                    stack=(*stack, expected),
                    depth=depth + 1,
                )
            return
        raise TraceValidationError(f"unsupported feature schema metadata: {expected!r}")


def _union_variants(expected: Any) -> tuple[Any, ...]:
    if get_origin(expected) is types.UnionType:
        return tuple(item for item in get_args(expected) if item is not type(None))
    return (expected,)


def _element_identity_variants(expected: Any) -> tuple[ElementIdentityVariant, ...]:
    variants: list[ElementIdentityVariant] = []
    for candidate in _union_variants(expected):
        if _scalar_kind(candidate) is not None:
            variants.append(ElementIdentityVariant("SCALAR", ("$value",)))
            continue
        if not (isinstance(candidate, type) and is_dataclass(candidate)):
            raise TraceValidationError("sequence element type lacks an identity rule")
        identity_fields = _ELEMENT_IDENTITY_FIELDS.get(candidate)
        if identity_fields is None:
            raise TraceValidationError(
                f"sequence element {candidate.__name__} lacks an identity rule"
            )
        hints = get_type_hints(candidate)
        if any(field_name not in hints for field_name in identity_fields):
            raise TraceValidationError("element identity names an unknown field")
        variants.append(ElementIdentityVariant(candidate.__name__, identity_fields))
    return tuple(sorted(variants, key=lambda variant: variant.variant_type))


def _validate_element_identity_variants(
    variants: tuple[ElementIdentityVariant, ...],
) -> None:
    expected_by_name: dict[str, tuple[str, ...]] = {"SCALAR": ("$value",)}
    for expected_type, identity_fields in _ELEMENT_IDENTITY_FIELDS.items():
        existing = expected_by_name.setdefault(expected_type.__name__, identity_fields)
        if existing != identity_fields:
            raise TraceValidationError("element-identity type names are ambiguous")
    for variant in variants:
        if expected_by_name.get(variant.variant_type) != variant.identity_fields:
            raise TraceValidationError(
                "element-identity fields differ from the variant schema"
            )


def _source_descriptor(
    source_schema: FeatureSourceSchema,
    schema_version: str,
    roots: tuple[str, ...],
) -> SchemaRootBinding:
    descriptor = _SchemaDescriptor(
        source_schema=source_schema,
        schema_version=schema_version,
        generator_implementation=FIELD_PATH_GENERATOR_IMPLEMENTATION,
        safety_ceiling=_current_safety_ceiling(),
        roots=roots,
    )
    return SchemaRootBinding(
        source_schema=source_schema,
        schema_version=schema_version,
        root_type={
            FeatureSourceSchema.TRACE: "TraceRecord",
            FeatureSourceSchema.LIFECYCLE_SIDECAR: "ClosedLifecycleArtifact",
            FeatureSourceSchema.SCHEDULE_SIDECAR: "ClosedScheduleArtifact",
        }[source_schema],
        schema_descriptor_digest=canonical_digest(descriptor),
    )


def _current_catalog_components() -> tuple[
    tuple[SchemaRootBinding, ...],
    tuple[SequenceIdentityRule, ...],
    tuple[FieldPathEntry, ...],
]:
    builder = _CatalogBuilder(paths=[], rules=[])
    trace_descriptors: list[str] = []
    record_hints = get_type_hints(TraceRecord)
    for record_type, payload_types in trace_payload_schema_variants():
        for payload_type in payload_types:
            prefix = f"{record_type.value}.{payload_type.__name__}"
            trace_descriptors.append(_trace_root_fingerprint(record_type, payload_type))
            for field in fields(TraceRecord):
                expected = (
                    payload_type
                    if field.name == "payload"
                    else record_hints[field.name]
                )
                builder.walk(
                    FeatureSourceSchema.TRACE,
                    expected,
                    f"{prefix}.{field.name}",
                )

    lifecycle_prefix = "lifecycle_sidecar.ClosedLifecycleArtifact"
    builder.walk(
        FeatureSourceSchema.LIFECYCLE_SIDECAR,
        ClosedLifecycleArtifact,
        lifecycle_prefix,
    )
    schedule_prefix = "schedule_sidecar.ClosedScheduleArtifact"
    builder.walk(
        FeatureSourceSchema.SCHEDULE_SIDECAR,
        ClosedScheduleArtifact,
        schedule_prefix,
    )
    roots = (
        _source_descriptor(
            FeatureSourceSchema.LIFECYCLE_SIDECAR,
            LIFECYCLE_SIDECAR_SCHEMA_VERSION,
            (_type_fingerprint(ClosedLifecycleArtifact),),
        ),
        _source_descriptor(
            FeatureSourceSchema.SCHEDULE_SIDECAR,
            SCHEDULE_SIDECAR_SCHEMA_VERSION,
            (_type_fingerprint(ClosedScheduleArtifact),),
        ),
        _source_descriptor(
            FeatureSourceSchema.TRACE,
            TRACE_SCHEMA_VERSION,
            tuple(trace_descriptors),
        ),
    )
    return (
        tuple(
            sorted(roots, key=lambda root: (root.source_schema.value, root.root_type))
        ),
        tuple(
            sorted(
                builder.rules,
                key=lambda rule: (rule.source_schema.value, rule.collection_path),
            )
        ),
        tuple(
            sorted(
                builder.paths,
                key=lambda entry: (entry.source_schema.value, entry.field_path),
            )
        ),
    )


def build_field_path_catalog() -> FieldPathCatalog:
    """Generate the exact recursive catalog from the three frozen source roots."""

    roots, rules, paths = _current_catalog_components()
    return FieldPathCatalog(
        schema_version=FIELD_PATH_CATALOG_SCHEMA_VERSION,
        generator_implementation=FIELD_PATH_GENERATOR_IMPLEMENTATION,
        path_notation=FIELD_PATH_NOTATION,
        safety_ceiling=_current_safety_ceiling(),
        schema_roots=roots,
        sequence_identity_rules=rules,
        paths=paths,
    )


def validate_field_path_catalog(catalog: FieldPathCatalog) -> None:
    """Regenerate the current closed schemas and reject any catalog drift."""

    if type(catalog) is not FieldPathCatalog:
        raise TraceValidationError("field-path catalog has the wrong type")
    raw = canonical_json(catalog)
    replayed = parse_canonical_dataclass(
        raw,
        FieldPathCatalog,
        artifact_name="field-path-catalog dependency",
        max_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )
    if replayed != catalog:
        raise TraceValidationError("field-path catalog changes during replay")
    expected = build_field_path_catalog()
    if catalog != expected:
        raise TraceValidationError("field-path catalog differs from current schemas")


_ONLINE_PATH_ALLOWLIST: dict[FeatureSourceSchema, frozenset[str]] = {
    # The cutoff row is committed after feature construction and therefore cannot
    # serve as a source for its own feature-view digest.
    FeatureSourceSchema.TRACE: frozenset(),
    FeatureSourceSchema.LIFECYCLE_SIDECAR: frozenset(
        {
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.action",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.binding_state_after",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.binding_state_before",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.byte_count",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.lease_deadline_ns",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.observed_byte_count",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.payload_size",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.status",
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.timestamp_ns",
        }
    ),
    # A sealed schedule is predeclared, but its event and epoch contents are the
    # future demand label for the corresponding observation.
    FeatureSourceSchema.SCHEDULE_SIDECAR: frozenset(),
}


def _online_allowlist_keys() -> frozenset[tuple[FeatureSourceSchema, str]]:
    return frozenset(
        (source, path)
        for source in FeatureSourceSchema
        for path in _ONLINE_PATH_ALLOWLIST[source]
    )


def _online_allowlist_digest() -> str:
    values = tuple(
        f"{source.value}:{path}"
        for source in FeatureSourceSchema
        for path in sorted(_ONLINE_PATH_ALLOWLIST[source])
    )
    return canonical_digest(values)


_LABEL_ONLY_TRACE_PREFIXES = (
    "demand_intent.",
    "observation_terminal.",
    "reuse_epoch.",
    "schedule_watermark.",
)
_FORBIDDEN_PROXY_LEAVES = frozenset(
    {
        "batch_id",
        "batch_index",
        "batch_size",
        "event_id",
        "event_ordinal",
        "generation",
        "mapping_id",
        "operation_id",
        "parent_event_id",
        "parent_record_id",
        "record_id",
        "sequence",
        "transfer_id",
    }
)


def _classification_for(entry: FieldPathEntry) -> FeatureClassification:
    key = (entry.source_schema, entry.field_path)
    if key in _online_allowlist_keys():
        return FeatureClassification.ONLINE_ALLOWED
    # A physical or ordering shortcut stays forbidden even inside a label record.
    leaf = entry.field_path.rsplit(".", 1)[-1]
    if leaf in _FORBIDDEN_PROXY_LEAVES:
        return FeatureClassification.FORBIDDEN_PROXY
    if entry.source_schema == FeatureSourceSchema.SCHEDULE_SIDECAR:
        return FeatureClassification.LABEL_ONLY
    if entry.source_schema == FeatureSourceSchema.TRACE and entry.field_path.startswith(
        _LABEL_ONLY_TRACE_PREFIXES
    ):
        return FeatureClassification.LABEL_ONLY
    if (
        entry.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and ".closure." in entry.field_path
    ):
        return FeatureClassification.LABEL_ONLY
    return FeatureClassification.PROVENANCE_ONLY


def _classification_profile_digest(catalog: FieldPathCatalog) -> str:
    profile = _ClassificationProfile(
        schema_version=FEATURE_CLASSIFICATION_PROFILE_SCHEMA_VERSION,
        field_path_catalog_digest=canonical_digest(catalog),
        entries=tuple(
            _ClassificationProfileEntry(
                source_schema=entry.source_schema,
                field_path=entry.field_path,
                value_kinds=entry.value_kinds,
                classification=_classification_for(entry),
            )
            for entry in catalog.paths
        ),
    )
    return canonical_digest(profile)


def _source_root_digest(
    catalog: FieldPathCatalog,
    source_schema: FeatureSourceSchema,
) -> str:
    try:
        return next(
            root.schema_descriptor_digest
            for root in catalog.schema_roots
            if root.source_schema == source_schema
        )
    except StopIteration as exc:
        raise TraceValidationError(
            "availability rule source lacks a schema root"
        ) from exc


def _build_availability_rule(
    catalog: FieldPathCatalog,
    source_schema: FeatureSourceSchema,
    field_path: str,
) -> FeatureAvailabilityRule:
    if (source_schema, field_path) not in _online_allowlist_keys():
        raise TraceValidationError(
            "availability rule path is outside the frozen online allowlist"
        )
    if source_schema != FeatureSourceSchema.LIFECYCLE_SIDECAR:
        raise TraceValidationError(
            "online source lacks a pre-attempt availability rule"
        )
    identity = _AvailabilityRuleIdentity(
        schema_version=AVAILABILITY_RULE_IDENTITY_SCHEMA_VERSION,
        source_schema=source_schema,
        source_schema_descriptor_digest=_source_root_digest(
            catalog,
            source_schema,
        ),
        field_path=field_path,
        availability_kind=FeatureAvailabilityKind.WINDOWED_EVENT,
        value_extractor=FeatureValueExtractor.CLOSED_DATACLASS_FIELD_PATH,
        receipt_kind=FeatureAvailabilityReceipt.LIFECYCLE_CUTOFF_PREFIX,
        event_time_path=(
            "lifecycle_sidecar.ClosedLifecycleArtifact.events[*]."
            "LifecycleEvent.timestamp_ns"
        ),
        clock_domain=LIFECYCLE_CLOCK_DOMAIN,
    )
    return FeatureAvailabilityRule(
        schema_version=identity.schema_version,
        rule_id=canonical_digest(identity),
        source_schema=identity.source_schema,
        source_schema_descriptor_digest=identity.source_schema_descriptor_digest,
        field_path=identity.field_path,
        availability_kind=identity.availability_kind,
        value_extractor=identity.value_extractor,
        receipt_kind=identity.receipt_kind,
        event_time_path=identity.event_time_path,
        clock_domain=identity.clock_domain,
    )


def build_feature_availability_rule_catalog(
    catalog: FieldPathCatalog,
) -> FeatureAvailabilityRuleCatalog:
    """Build the exact content-addressed rules for every online field path."""

    validate_field_path_catalog(catalog)
    catalog_keys = {(entry.source_schema, entry.field_path) for entry in catalog.paths}
    missing = _online_allowlist_keys().difference(catalog_keys)
    if missing:
        raise TraceValidationError(
            f"online allowlist names unknown field paths: {sorted(missing)}"
        )
    rules = tuple(
        sorted(
            (
                _build_availability_rule(catalog, source, path)
                for source, path in _online_allowlist_keys()
            ),
            key=lambda rule: (rule.source_schema.value, rule.field_path),
        )
    )
    return FeatureAvailabilityRuleCatalog(
        schema_version=AVAILABILITY_RULE_CATALOG_SCHEMA_VERSION,
        field_path_catalog_digest=canonical_digest(catalog),
        field_path_catalog=catalog,
        online_allowlist_digest=_online_allowlist_digest(),
        rules=rules,
    )


def validate_feature_availability_rule_catalog(
    catalog: FeatureAvailabilityRuleCatalog,
) -> None:
    """Replay and regenerate a rule catalog, rejecting unused or rebound rules."""

    if type(catalog) is not FeatureAvailabilityRuleCatalog:
        raise TraceValidationError(
            "feature-availability rule catalog has the wrong type"
        )
    raw = canonical_json(catalog)
    replayed = parse_canonical_dataclass(
        raw,
        FeatureAvailabilityRuleCatalog,
        artifact_name="feature-availability-rule-catalog dependency",
        max_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )
    if replayed != catalog:
        raise TraceValidationError(
            "feature-availability rule catalog changes during replay"
        )
    expected = build_feature_availability_rule_catalog(catalog.field_path_catalog)
    if catalog != expected:
        raise TraceValidationError(
            "feature-availability rule catalog differs from current rules"
        )


def build_feature_classification_assignments(
    catalog: FieldPathCatalog,
    availability_rule_catalog: FeatureAvailabilityRuleCatalog,
) -> tuple[FeatureFieldAssignment, ...]:
    """Generate the frozen four-way classification for every catalog path."""

    validate_field_path_catalog(catalog)
    validate_feature_availability_rule_catalog(availability_rule_catalog)
    if availability_rule_catalog.field_path_catalog != catalog:
        raise TraceValidationError(
            "feature assignments and availability rules bind different catalogs"
        )
    rules = {
        (rule.source_schema, rule.field_path): rule
        for rule in availability_rule_catalog.rules
    }
    return tuple(
        FeatureFieldAssignment(
            source_schema=entry.source_schema,
            field_path=entry.field_path,
            classification=_classification_for(entry),
            availability_rule_id=(
                rules[(entry.source_schema, entry.field_path)].rule_id
                if _classification_for(entry) == FeatureClassification.ONLINE_ALLOWED
                else None
            ),
        )
        for entry in catalog.paths
    )


def build_feature_contract(
    catalog: FieldPathCatalog,
    assignments: tuple[FeatureFieldAssignment, ...],
    *,
    availability_rule_catalog: FeatureAvailabilityRuleCatalog,
) -> FeatureContract:
    """Bind one exhaustive classification without filling omitted paths."""

    validate_field_path_catalog(catalog)
    validate_feature_availability_rule_catalog(availability_rule_catalog)
    return FeatureContract(
        schema_version=FEATURE_CONTRACT_SCHEMA_VERSION,
        field_path_catalog_digest=canonical_digest(catalog),
        field_path_catalog=catalog,
        online_allowlist_digest=_online_allowlist_digest(),
        availability_rule_catalog_digest=canonical_digest(availability_rule_catalog),
        availability_rule_catalog=availability_rule_catalog,
        classification_profile_digest=_classification_profile_digest(catalog),
        assignments=assignments,
    )


def validate_feature_contract(contract: FeatureContract) -> None:
    """Replay a feature contract and recheck every embedded dependency."""

    if type(contract) is not FeatureContract:
        raise TraceValidationError("feature contract has the wrong type")
    raw = canonical_json(contract)
    replayed = parse_canonical_dataclass(
        raw,
        FeatureContract,
        artifact_name="feature-contract dependency",
        max_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )
    if replayed != contract:
        raise TraceValidationError("feature contract changes during replay")


def _output_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_parent(path: Path, artifact_name: str) -> int:
    if not path.is_absolute():
        raise TraceValidationError(f"{artifact_name} path must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise TraceValidationError(
            f"{artifact_name} parent must be a non-symlink directory"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(parent, flags)
        opened = os.fstat(descriptor)
        linked = parent.stat(follow_symlinks=False)
    except BaseException as exc:
        if descriptor is not None:
            with suppress(BaseException):
                os.close(descriptor)
        if isinstance(exc, Exception):
            raise TraceValidationError(
                f"cannot open {artifact_name} parent safely"
            ) from exc
        raise
    if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
        with suppress(BaseException):
            os.close(descriptor)
        raise TraceValidationError(f"{artifact_name} parent changed while opening")
    return descriptor


def _read_exact(descriptor: int, size: int) -> bytes:
    observed = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, size - offset, offset)
        if not chunk:
            raise OSError("feature artifact ended during stable readback")
        observed.extend(chunk)
        offset += len(chunk)
    return bytes(observed)


def _validate_parent_binding(
    path: Path,
    descriptor: int,
    artifact_name: str,
) -> None:
    try:
        opened = os.fstat(descriptor)
        linked = path.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError(f"{artifact_name} parent cannot be revalidated") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
    ):
        raise OSError(f"{artifact_name} parent path changed")


def _close_parent_and_validate(
    path: Path,
    descriptor: int,
    artifact_name: str,
) -> None:
    """Catch a parent-path replacement that races the descriptor close."""

    opened: os.stat_result | None = None
    failure: BaseException | None = None
    try:
        opened = os.fstat(descriptor)
    except BaseException as exc:
        failure = exc
    try:
        os.close(descriptor)
    except BaseException as exc:
        if failure is None:
            failure = exc
    if failure is None and opened is not None:
        try:
            linked = path.parent.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            ):
                raise OSError(f"{artifact_name} parent path changed during close")
        except BaseException as exc:
            failure = exc
    if failure is not None:
        raise failure


def _linked_state(
    descriptor: int,
    parent_descriptor: int,
    path: Path,
    expected_size: int,
) -> os.stat_result:
    opened = os.fstat(descriptor)
    linked = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or opened.st_nlink != 1
        or linked.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        or opened.st_size != expected_size
        or linked.st_size != expected_size
    ):
        raise OSError("feature artifact output identity changed")
    return opened


def _validate_artifact[T](artifact: T, expected: type[T]) -> None:
    if expected is FieldPathCatalog:
        validate_field_path_catalog(artifact)  # type: ignore[arg-type]
    elif expected is FeatureAvailabilityRuleCatalog:
        validate_feature_availability_rule_catalog(artifact)  # type: ignore[arg-type]
    elif expected is FeatureContract:
        validate_feature_contract(artifact)  # type: ignore[arg-type]
    else:
        raise TraceValidationError("unsupported feature artifact type")


def _write_artifact[T](
    path: Path,
    artifact: T,
    expected: type[T],
    artifact_name: str,
    *,
    validator: Callable[[T], None] | None = None,
) -> str:
    if type(artifact) is not expected:
        raise TraceValidationError(f"{artifact_name} has the wrong type")
    if validator is None:
        _validate_artifact(artifact, expected)
    else:
        validator(artifact)
    raw = canonical_json(artifact)
    if not raw or len(raw) > MAX_FEATURE_ARTIFACT_BYTES:
        raise TraceValidationError(f"{artifact_name} exceeds the size limit")
    parent_descriptor = _open_parent(path, artifact_name)
    descriptor: int | None = None
    created = False
    precreate_error: TraceValidationError | None = None
    commit_failure: BaseException | None = None
    published: os.stat_result | None = None
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(path.name, flags, 0o640, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            precreate_error = TraceValidationError(f"{artifact_name} is create-only")
            precreate_error.__cause__ = exc
        except OSError as exc:
            precreate_error = TraceValidationError(
                f"cannot create {artifact_name} safely"
            )
            precreate_error.__cause__ = exc
        else:
            created = True
            try:
                offset = 0
                while offset < len(raw):
                    count = os.write(descriptor, raw[offset:])
                    if count <= 0:
                        raise OSError("feature artifact write made no progress")
                    offset += count
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
                first_state = _linked_state(
                    descriptor, parent_descriptor, path, len(raw)
                )
                first = _read_exact(descriptor, len(raw))
                second_state = _linked_state(
                    descriptor, parent_descriptor, path, len(raw)
                )
                second = _read_exact(descriptor, len(raw))
                published = _linked_state(descriptor, parent_descriptor, path, len(raw))
                if (
                    first != raw
                    or second != raw
                    or _output_state(first_state) != _output_state(second_state)
                    or _output_state(second_state) != _output_state(published)
                ):
                    raise OSError("feature artifact changed during stable readback")
            except BaseException as exc:
                commit_failure = exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if commit_failure is None:
                    commit_failure = exc
        if published is not None:
            try:
                linked = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if _output_state(published) != _output_state(linked):
                    raise OSError("feature artifact changed after output close")
            except BaseException as exc:
                if commit_failure is None:
                    commit_failure = exc
        if created:
            try:
                _validate_parent_binding(path, parent_descriptor, artifact_name)
            except BaseException as exc:
                if commit_failure is None:
                    commit_failure = exc
        try:
            _close_parent_and_validate(path, parent_descriptor, artifact_name)
        except BaseException as exc:
            if created and commit_failure is None:
                commit_failure = exc
            elif not created and precreate_error is None:
                precreate_error = TraceValidationError(
                    f"cannot close {artifact_name} parent safely"
                )
                precreate_error.__cause__ = exc
    if commit_failure is not None:
        if isinstance(commit_failure, Exception):
            raise TraceCommitIndeterminateError(
                f"{artifact_name} durability is indeterminate"
            ) from commit_failure
        raise commit_failure
    if precreate_error is not None:
        raise precreate_error
    return sha256(raw).hexdigest()


def _read_stable(path: Path, artifact_name: str) -> bytes:
    if not path.is_absolute():
        raise TraceValidationError(f"{artifact_name} path must be absolute")
    parent_descriptor = _open_parent(path, artifact_name)
    try:
        before = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        with suppress(OSError):
            os.close(parent_descriptor)
        raise TraceValidationError(f"cannot inspect {artifact_name} safely") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > MAX_FEATURE_ARTIFACT_BYTES
    ):
        with suppress(OSError):
            os.close(parent_descriptor)
        raise TraceValidationError(f"{artifact_name} input identity is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        with suppress(OSError):
            os.close(parent_descriptor)
        raise TraceValidationError(f"cannot open {artifact_name} safely") from exc
    read_failure: BaseException | None = None
    first = b""
    after: os.stat_result | None = None
    try:
        opened = os.fstat(descriptor)
        if (
            _output_state(before) != _output_state(opened)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise TraceValidationError(f"{artifact_name} changed while opening")
        first = _read_exact(descriptor, opened.st_size)
        middle = os.fstat(descriptor)
        second = _read_exact(descriptor, opened.st_size)
        after = os.fstat(descriptor)
        if (
            first != second
            or _output_state(opened) != _output_state(middle)
            or _output_state(middle) != _output_state(after)
        ):
            raise TraceValidationError(f"{artifact_name} changed during read")
        linked = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _output_state(after) != _output_state(linked):
            raise TraceValidationError(f"{artifact_name} changed before close")
        _validate_parent_binding(path, parent_descriptor, artifact_name)
    except BaseException as exc:
        read_failure = exc
    finally:
        try:
            os.close(descriptor)
        except BaseException as exc:
            if read_failure is None:
                read_failure = exc
        if after is not None:
            try:
                linked = os.stat(
                    path.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if _output_state(after) != _output_state(linked):
                    raise OSError(f"{artifact_name} changed after file close")
                _validate_parent_binding(path, parent_descriptor, artifact_name)
            except BaseException as exc:
                if read_failure is None:
                    read_failure = exc
        try:
            _close_parent_and_validate(path, parent_descriptor, artifact_name)
        except BaseException as exc:
            if read_failure is None:
                read_failure = exc
    if read_failure is not None:
        if isinstance(read_failure, TraceValidationError):
            raise read_failure
        if isinstance(read_failure, Exception):
            raise TraceValidationError(
                f"cannot read {artifact_name} safely"
            ) from read_failure
        raise read_failure
    return first


def _load_artifact[T](
    path: Path,
    expected: type[T],
    artifact_name: str,
    *,
    validator: Callable[[T], None] | None = None,
) -> LoadedFeatureArtifact[T]:
    raw = _read_stable(path, artifact_name)
    artifact = parse_canonical_dataclass(
        raw,
        expected,
        artifact_name=artifact_name,
        max_bytes=MAX_FEATURE_ARTIFACT_BYTES,
    )
    if validator is None:
        _validate_artifact(artifact, expected)
    else:
        validator(artifact)
    return LoadedFeatureArtifact(
        artifact=artifact,
        digest=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def write_field_path_catalog(path: Path, artifact: FieldPathCatalog) -> str:
    return _write_artifact(
        path,
        artifact,
        FieldPathCatalog,
        "field-path-catalog artifact",
    )


def load_field_path_catalog(
    path: Path,
) -> LoadedFeatureArtifact[FieldPathCatalog]:
    return _load_artifact(
        path,
        FieldPathCatalog,
        "field-path-catalog artifact",
    )


def write_feature_availability_rule_catalog(
    path: Path,
    artifact: FeatureAvailabilityRuleCatalog,
) -> str:
    return _write_artifact(
        path,
        artifact,
        FeatureAvailabilityRuleCatalog,
        "feature-availability-rule-catalog artifact",
    )


def load_feature_availability_rule_catalog(
    path: Path,
) -> LoadedFeatureArtifact[FeatureAvailabilityRuleCatalog]:
    return _load_artifact(
        path,
        FeatureAvailabilityRuleCatalog,
        "feature-availability-rule-catalog artifact",
    )


def write_feature_contract(path: Path, artifact: FeatureContract) -> str:
    return _write_artifact(
        path,
        artifact,
        FeatureContract,
        "feature-contract artifact",
    )


def load_feature_contract(path: Path) -> LoadedFeatureArtifact[FeatureContract]:
    return _load_artifact(
        path,
        FeatureContract,
        "feature-contract artifact",
    )
