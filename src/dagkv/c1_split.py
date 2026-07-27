"""Canonical C1-B1 candidate universes and deterministic split evidence."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from dagkv.c1_trace import (
    TraceCommitIndeterminateError,
    TraceValidationError,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
)
from dagkv.domain import BlockKey, require_sha256, require_text

CUTOFF_PLAN_SCHEMA_VERSION = "dagkv.m3.c1_cutoff_plan.v1"
COHORT_TOKEN_CATALOG_SCHEMA_VERSION = "dagkv.m3.c1_cohort_token_catalog.v1"
CANDIDATE_UNIVERSE_SCHEMA_VERSION = "dagkv.m3.c1_candidate_universe.v1"
CANDIDATE_IDENTITY_SCHEMA_VERSION = "dagkv.m3.c1_candidate_identity.v1"
PREDECESSOR_EXCLUSION_SCHEMA_VERSION = "dagkv.m3.c1_predecessor_exclusion_catalog.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "dagkv.m3.c1_split_manifest.v1"
SPLIT_COMPONENT_SCHEMA_VERSION = "dagkv.m3.c1_split_component.v1"
PREDECESSOR_UNION_AUDIT_SCHEMA_VERSION = "dagkv.m3.c1_predecessor_union_audit.v1"
ROLE_ASSIGNMENT_ALGORITHM = "canonical_lineage_incidence_dsu_split_time_v1"
MAX_SPLIT_ARTIFACT_BYTES = 64 * 1024 * 1024


class UniversePurpose(StrEnum):
    """Closed pre-data lane for one candidate universe."""

    STRUCTURAL_FIXTURE = "STRUCTURAL_FIXTURE"
    EXCLUDED_PILOT = "EXCLUDED_PILOT"
    POST_PILOT_MAIN = "POST_PILOT_MAIN"


class SplitRole(StrEnum):
    """Chronological role assigned to one complete split component."""

    PILOT = "PILOT"
    TRAIN = "TRAIN"
    CAL_FIT = "CAL_FIT"
    CAL_RADIUS = "CAL_RADIUS"
    FORMAL = "FORMAL"


class SplitCohort(StrEnum):
    """Independent split cohort with one closed lineage-token policy."""

    PRIMARY_TEMPORAL = "PRIMARY_TEMPORAL"
    TEMPLATE_GENERALIZATION = "TEMPLATE_GENERALIZATION"
    CONTENT_ISOLATED = "CONTENT_ISOLATED"


class LineageFamily(StrEnum):
    """Pre-policy identities that create statistical dependence edges."""

    WORKFLOW_INSTANCE = "WORKFLOW_INSTANCE"
    SESSION = "SESSION"
    SOURCE_CASE = "SOURCE_CASE"
    SCHEDULED_TOOL_EXECUTION = "SCHEDULED_TOOL_EXECUTION"
    REFERENCE_EPOCH = "REFERENCE_EPOCH"
    SCHEDULE_COMPONENT = "SCHEDULE_COMPONENT"
    RANDOM_DRAW = "RANDOM_DRAW"
    DERIVED_EXAMPLE = "DERIVED_EXAMPLE"
    WORKFLOW_TEMPLATE = "WORKFLOW_TEMPLATE"
    CONTENT_LINEAGE = "CONTENT_LINEAGE"


class LineageApplicability(StrEnum):
    """Typed distinction between present and intentionally empty lineage."""

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CandidateDisposition(StrEnum):
    """Label-blind normalization result for one cutoff-plan slot."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


class IneligibilityReason(StrEnum):
    """Closed pre-observation reasons for excluding one planned slot."""

    ELIGIBILITY_RULE_REJECTED = "ELIGIBILITY_RULE_REJECTED"
    NORMALIZER_EXCLUDED = "NORMALIZER_EXCLUDED"
    UNSUPPORTED_CONTROL_FLOW = "UNSUPPORTED_CONTROL_FLOW"


_PURPOSE_ROLES = {
    UniversePurpose.STRUCTURAL_FIXTURE: tuple(SplitRole),
    UniversePurpose.EXCLUDED_PILOT: (SplitRole.PILOT,),
    UniversePurpose.POST_PILOT_MAIN: (
        SplitRole.TRAIN,
        SplitRole.CAL_FIT,
        SplitRole.CAL_RADIUS,
        SplitRole.FORMAL,
    ),
}

_BASE_LINEAGE_FAMILIES = frozenset(
    {
        LineageFamily.WORKFLOW_INSTANCE,
        LineageFamily.SESSION,
        LineageFamily.SOURCE_CASE,
        LineageFamily.SCHEDULED_TOOL_EXECUTION,
        LineageFamily.REFERENCE_EPOCH,
        LineageFamily.SCHEDULE_COMPONENT,
        LineageFamily.RANDOM_DRAW,
        LineageFamily.DERIVED_EXAMPLE,
    }
)
_COHORT_LINEAGE_FAMILIES = {
    SplitCohort.PRIMARY_TEMPORAL: _BASE_LINEAGE_FAMILIES,
    SplitCohort.TEMPLATE_GENERALIZATION: _BASE_LINEAGE_FAMILIES
    | {LineageFamily.WORKFLOW_TEMPLATE},
    SplitCohort.CONTENT_ISOLATED: _BASE_LINEAGE_FAMILIES
    | {LineageFamily.CONTENT_LINEAGE},
}
_COHORT_REQUIRED_FAMILY = {
    SplitCohort.PRIMARY_TEMPORAL: None,
    SplitCohort.TEMPLATE_GENERALIZATION: LineageFamily.WORKFLOW_TEMPLATE,
    SplitCohort.CONTENT_ISOLATED: LineageFamily.CONTENT_LINEAGE,
}


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TraceValidationError(f"{name} must be a tuple")


def _require_enum(name: str, value: object, expected: type[StrEnum]) -> None:
    if type(value) is not expected:
        raise TraceValidationError(f"{name} must be a {expected.__name__}")


def _require_sorted_unique_text(name: str, values: tuple[str, ...]) -> None:
    _require_tuple(name, values)
    for value in values:
        require_text(name, value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class LineageField:
    """One complete, typed, multi-value lineage family."""

    family: LineageFamily
    applicability: LineageApplicability
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("lineage family", self.family, LineageFamily)
        _require_enum(
            "lineage applicability",
            self.applicability,
            LineageApplicability,
        )
        _require_sorted_unique_text("lineage values", self.values)
        if self.applicability == LineageApplicability.PRESENT:
            if not self.values:
                raise TraceValidationError("present lineage requires values")
            if self.family in {
                LineageFamily.WORKFLOW_TEMPLATE,
                LineageFamily.CONTENT_LINEAGE,
            }:
                for value in self.values:
                    require_sha256("special-cohort lineage token", value)
        elif self.values:
            raise TraceValidationError("absent lineage cannot contain values")


def _validate_complete_lineage(lineage: tuple[LineageField, ...]) -> None:
    _require_tuple("typed lineage", lineage)
    expected = tuple(LineageFamily)
    observed: list[LineageFamily] = []
    for field in lineage:
        if type(field) is not LineageField:
            raise TraceValidationError("typed lineage contains an invalid field")
        observed.append(field.family)
    if tuple(observed) != expected:
        raise TraceValidationError(
            "typed lineage must contain every family once in canonical order"
        )


@dataclass(frozen=True, slots=True)
class CutoffPlanSlot:
    """One raw, label-blind opportunity to create a C1 observation."""

    candidate_slot_id: str
    schedule_case_id: str
    block_key: BlockKey
    cutoff_trigger_id: str
    split_time_ns: int
    primary_horizon_duration_ns: int
    feature_lookback_ns: int
    lineage: tuple[LineageField, ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_slot_id",
            "schedule_case_id",
            "cutoff_trigger_id",
        ):
            require_text(name, getattr(self, name))
        if type(self.block_key) is not BlockKey:
            raise TraceValidationError("cutoff-plan block key has the wrong type")
        _require_int("split_time_ns", self.split_time_ns)
        _require_int(
            "primary_horizon_duration_ns",
            self.primary_horizon_duration_ns,
            minimum=1,
        )
        _require_int("feature_lookback_ns", self.feature_lookback_ns)
        _validate_complete_lineage(self.lineage)
        source_case = next(
            field for field in self.lineage if field.family == LineageFamily.SOURCE_CASE
        )
        if source_case.applicability != LineageApplicability.PRESENT:
            raise TraceValidationError("cutoff-plan slot requires source-case lineage")


@dataclass(frozen=True, slots=True)
class CohortTokenCatalogEntry:
    """One frozen special-cohort token and its complete plan-slot incidence."""

    token_digest: str
    candidate_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256("cohort token digest", self.token_digest)
        _require_sorted_unique_text(
            "cohort token candidate-slot IDs",
            self.candidate_slot_ids,
        )
        if not self.candidate_slot_ids:
            raise TraceValidationError("cohort token must bind at least one plan slot")


@dataclass(frozen=True, slots=True)
class CohortTokenCatalog:
    """Canonical slot incidence; source membership is a bundle replay obligation."""

    schema_version: str
    lineage_family: LineageFamily
    source_catalog_digest: str
    entries: tuple[CohortTokenCatalogEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != COHORT_TOKEN_CATALOG_SCHEMA_VERSION:
            raise TraceValidationError("unsupported cohort-token catalog schema")
        _require_enum(
            "cohort-token lineage family",
            self.lineage_family,
            LineageFamily,
        )
        if self.lineage_family not in {
            LineageFamily.WORKFLOW_TEMPLATE,
            LineageFamily.CONTENT_LINEAGE,
        }:
            raise TraceValidationError("cohort-token catalog has an invalid family")
        require_sha256("cohort-token source-catalog digest", self.source_catalog_digest)
        _require_tuple("cohort-token catalog entries", self.entries)
        for entry in self.entries:
            if type(entry) is not CohortTokenCatalogEntry:
                raise TraceValidationError(
                    "cohort-token catalog contains an invalid entry"
                )
        tokens = tuple(entry.token_digest for entry in self.entries)
        if tokens != tuple(sorted(tokens)) or len(tokens) != len(set(tokens)):
            raise TraceValidationError(
                "cohort-token catalog entries must be token-sorted and unique"
            )


def build_cohort_token_catalog(
    *,
    cohort: SplitCohort,
    slots: tuple[CutoffPlanSlot, ...],
    workflow_template_digest: str,
    content_lineage_digest: str,
) -> CohortTokenCatalog | None:
    """Recompute the special-cohort token catalog from frozen plan slots."""

    _require_enum("cohort-token split cohort", cohort, SplitCohort)
    _require_tuple("cohort-token plan slots", slots)
    require_sha256("workflow-template catalog digest", workflow_template_digest)
    require_sha256("content-lineage catalog digest", content_lineage_digest)
    for slot in slots:
        if type(slot) is not CutoffPlanSlot:
            raise TraceValidationError("cohort-token catalog has an invalid plan slot")
    required_family = _COHORT_REQUIRED_FAMILY[cohort]
    if required_family is None:
        return None
    source_digest = (
        workflow_template_digest
        if required_family == LineageFamily.WORKFLOW_TEMPLATE
        else content_lineage_digest
    )
    members: dict[str, set[str]] = {}
    for slot in slots:
        field = next(item for item in slot.lineage if item.family == required_family)
        if field.applicability != LineageApplicability.PRESENT:
            continue
        for token in field.values:
            require_sha256("special-cohort lineage token", token)
            members.setdefault(token, set()).add(slot.candidate_slot_id)
    return CohortTokenCatalog(
        schema_version=COHORT_TOKEN_CATALOG_SCHEMA_VERSION,
        lineage_family=required_family,
        source_catalog_digest=source_digest,
        entries=tuple(
            CohortTokenCatalogEntry(
                token_digest=token,
                candidate_slot_ids=tuple(sorted(candidate_slot_ids)),
            )
            for token, candidate_slot_ids in sorted(members.items())
        ),
    )


@dataclass(frozen=True, slots=True)
class CutoffPlan:
    """Finite create-only plan from which all candidate slots are normalized."""

    schema_version: str
    purpose: UniversePurpose
    cohort: SplitCohort
    schedule_digest: str
    source_digest: str
    workflow_template_digest: str
    content_lineage_digest: str
    normalizer_digest: str
    eligibility_rule_digest: str
    temporal_axis_digest: str
    method_menu_digest: str
    cohort_token_catalog: CohortTokenCatalog | None
    slots: tuple[CutoffPlanSlot, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CUTOFF_PLAN_SCHEMA_VERSION:
            raise TraceValidationError("unsupported cutoff-plan schema")
        _require_enum("universe purpose", self.purpose, UniversePurpose)
        _require_enum("split cohort", self.cohort, SplitCohort)
        for name in (
            "schedule_digest",
            "source_digest",
            "workflow_template_digest",
            "content_lineage_digest",
            "normalizer_digest",
            "eligibility_rule_digest",
            "temporal_axis_digest",
            "method_menu_digest",
        ):
            require_sha256(name, getattr(self, name))
        _require_tuple("cutoff-plan slots", self.slots)
        slot_ids: list[str] = []
        opportunity_keys: set[tuple[str, str, str]] = set()
        source_cases_by_schedule: dict[str, tuple[str, ...]] = {}
        for slot in self.slots:
            if type(slot) is not CutoffPlanSlot:
                raise TraceValidationError("cutoff plan contains an invalid slot")
            slot_ids.append(slot.candidate_slot_id)
            opportunity_key = (
                slot.schedule_case_id,
                canonical_digest(slot.block_key),
                slot.cutoff_trigger_id,
            )
            if opportunity_key in opportunity_keys:
                raise TraceValidationError(
                    "cutoff plan contains a duplicate raw opportunity"
                )
            opportunity_keys.add(opportunity_key)
            source_cases = next(
                field.values
                for field in slot.lineage
                if field.family == LineageFamily.SOURCE_CASE
            )
            previous_source_cases = source_cases_by_schedule.setdefault(
                slot.schedule_case_id,
                source_cases,
            )
            if previous_source_cases != source_cases:
                raise TraceValidationError(
                    "one schedule case has inconsistent source-case lineage"
                )
        if tuple(slot_ids) != tuple(sorted(slot_ids)) or len(slot_ids) != len(
            set(slot_ids)
        ):
            raise TraceValidationError("cutoff-plan slots must be ID-sorted and unique")
        expected_catalog = build_cohort_token_catalog(
            cohort=self.cohort,
            slots=self.slots,
            workflow_template_digest=self.workflow_template_digest,
            content_lineage_digest=self.content_lineage_digest,
        )
        if self.cohort_token_catalog != expected_catalog:
            raise TraceValidationError(
                "cutoff-plan cohort-token catalog differs from slot incidence"
            )


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Exact pre-runtime material used to derive one candidate ID."""

    schema_version: str
    purpose: UniversePurpose
    cutoff_plan_digest: str
    candidate_slot_id: str
    schedule_case_id: str
    block_key: BlockKey
    cutoff_trigger_id: str
    split_time_ns: int
    primary_horizon_duration_ns: int
    feature_lookback_ns: int
    lineage: tuple[LineageField, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported candidate identity schema")
        _require_enum("candidate purpose", self.purpose, UniversePurpose)
        require_sha256("candidate cutoff-plan digest", self.cutoff_plan_digest)
        for name in (
            "candidate_slot_id",
            "schedule_case_id",
            "cutoff_trigger_id",
        ):
            require_text(name, getattr(self, name))
        if type(self.block_key) is not BlockKey:
            raise TraceValidationError("candidate block key has the wrong type")
        _require_int("candidate split_time_ns", self.split_time_ns)
        _require_int(
            "candidate primary_horizon_duration_ns",
            self.primary_horizon_duration_ns,
            minimum=1,
        )
        _require_int("candidate feature_lookback_ns", self.feature_lookback_ns)
        _validate_complete_lineage(self.lineage)


def candidate_identity(plan: CutoffPlan, slot: CutoffPlanSlot) -> CandidateIdentity:
    """Return the canonical pre-runtime identity for one slot in a plan."""

    if type(plan) is not CutoffPlan or type(slot) is not CutoffPlanSlot:
        raise TraceValidationError("candidate identity input has the wrong type")
    if slot not in plan.slots:
        raise TraceValidationError("candidate slot is absent from the cutoff plan")
    return CandidateIdentity(
        schema_version=CANDIDATE_IDENTITY_SCHEMA_VERSION,
        purpose=plan.purpose,
        cutoff_plan_digest=canonical_digest(plan),
        candidate_slot_id=slot.candidate_slot_id,
        schedule_case_id=slot.schedule_case_id,
        block_key=slot.block_key,
        cutoff_trigger_id=slot.cutoff_trigger_id,
        split_time_ns=slot.split_time_ns,
        primary_horizon_duration_ns=slot.primary_horizon_duration_ns,
        feature_lookback_ns=slot.feature_lookback_ns,
        lineage=slot.lineage,
    )


def derive_candidate_id(plan: CutoffPlan, slot: CutoffPlanSlot) -> str:
    """Derive one candidate ID without any runtime or outcome field."""

    return canonical_digest(candidate_identity(plan, slot))


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One exact eligible or typed-ineligible cutoff-plan disposition."""

    candidate_slot_id: str
    disposition: CandidateDisposition
    candidate_id: str | None
    ineligibility_reason: IneligibilityReason | None

    def __post_init__(self) -> None:
        require_text("candidate_slot_id", self.candidate_slot_id)
        _require_enum(
            "candidate disposition",
            self.disposition,
            CandidateDisposition,
        )
        if self.disposition == CandidateDisposition.ELIGIBLE:
            if self.candidate_id is None:
                raise TraceValidationError("eligible record lacks a candidate ID")
            require_sha256("candidate_id", self.candidate_id)
            if self.ineligibility_reason is not None:
                raise TraceValidationError("eligible record has an exclusion reason")
            return
        if self.candidate_id is not None:
            raise TraceValidationError("ineligible record has a candidate ID")
        _require_enum(
            "ineligibility reason",
            self.ineligibility_reason,
            IneligibilityReason,
        )


@dataclass(frozen=True, slots=True)
class PredecessorExclusionEntry:
    """Exact pilot candidate identity and lineage excluded from the main lane."""

    candidate_id: str
    lineage: tuple[LineageField, ...]

    def __post_init__(self) -> None:
        require_sha256("excluded candidate_id", self.candidate_id)
        _validate_complete_lineage(self.lineage)


@dataclass(frozen=True, slots=True)
class PredecessorExclusionCatalog:
    """Complete historical pilot exclusions carried into a main universe."""

    schema_version: str
    predecessor_universe_digest: str
    entries: tuple[PredecessorExclusionEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PREDECESSOR_EXCLUSION_SCHEMA_VERSION:
            raise TraceValidationError("unsupported predecessor exclusion schema")
        require_sha256(
            "predecessor universe digest",
            self.predecessor_universe_digest,
        )
        _require_tuple("predecessor exclusions", self.entries)
        candidate_ids: list[str] = []
        for entry in self.entries:
            if type(entry) is not PredecessorExclusionEntry:
                raise TraceValidationError(
                    "predecessor catalog contains an invalid entry"
                )
            candidate_ids.append(entry.candidate_id)
        if tuple(candidate_ids) != tuple(sorted(candidate_ids)) or len(
            candidate_ids
        ) != len(set(candidate_ids)):
            raise TraceValidationError(
                "predecessor exclusions must be candidate-ID sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class CandidateUniverse:
    """Closed normalization of every slot in one cutoff plan."""

    schema_version: str
    purpose: UniversePurpose
    cohort: SplitCohort
    cutoff_plan_digest: str
    cutoff_plan: CutoffPlan
    records: tuple[CandidateRecord, ...]
    predecessor_universe_digest: str | None
    predecessor_split_manifest_digest: str | None
    predecessor_exclusion_catalog_digest: str | None
    predecessor_exclusion_catalog: PredecessorExclusionCatalog | None

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_UNIVERSE_SCHEMA_VERSION:
            raise TraceValidationError("unsupported candidate-universe schema")
        _require_enum("candidate-universe purpose", self.purpose, UniversePurpose)
        _require_enum("candidate-universe cohort", self.cohort, SplitCohort)
        if type(self.cutoff_plan) is not CutoffPlan:
            raise TraceValidationError("candidate universe lacks a cutoff plan")
        if self.cutoff_plan.purpose != self.purpose:
            raise TraceValidationError("candidate universe purpose differs from plan")
        if self.cutoff_plan.cohort != self.cohort:
            raise TraceValidationError("candidate universe cohort differs from plan")
        expected_plan_digest = canonical_digest(self.cutoff_plan)
        require_sha256("cutoff-plan digest", self.cutoff_plan_digest)
        if self.cutoff_plan_digest != expected_plan_digest:
            raise TraceValidationError("candidate universe cutoff-plan digest differs")
        self._validate_predecessor_binding()
        self._validate_records()

    def _validate_predecessor_binding(self) -> None:
        predecessor_fields = (
            self.predecessor_universe_digest,
            self.predecessor_split_manifest_digest,
            self.predecessor_exclusion_catalog_digest,
            self.predecessor_exclusion_catalog,
        )
        if self.purpose != UniversePurpose.POST_PILOT_MAIN:
            if any(value is not None for value in predecessor_fields):
                raise TraceValidationError(
                    "non-main universe cannot bind predecessor evidence"
                )
            return
        if any(value is None for value in predecessor_fields):
            raise TraceValidationError("main universe lacks predecessor evidence")
        require_sha256(
            "predecessor universe digest",
            self.predecessor_universe_digest or "",
        )
        require_sha256(
            "predecessor split-manifest digest",
            self.predecessor_split_manifest_digest or "",
        )
        require_sha256(
            "predecessor exclusion-catalog digest",
            self.predecessor_exclusion_catalog_digest or "",
        )
        catalog = self.predecessor_exclusion_catalog
        if type(catalog) is not PredecessorExclusionCatalog:
            raise TraceValidationError("main predecessor catalog has the wrong type")
        if catalog.predecessor_universe_digest != self.predecessor_universe_digest:
            raise TraceValidationError("predecessor catalog binds another universe")
        if canonical_digest(catalog) != self.predecessor_exclusion_catalog_digest:
            raise TraceValidationError("predecessor exclusion-catalog digest differs")

    def _validate_records(self) -> None:
        _require_tuple("candidate records", self.records)
        slots = self.cutoff_plan.slots
        if len(self.records) != len(slots):
            raise TraceValidationError("candidate records do not conserve plan slots")
        candidate_ids: list[str] = []
        for slot, record in zip(slots, self.records, strict=True):
            if type(record) is not CandidateRecord:
                raise TraceValidationError("candidate universe has an invalid record")
            if record.candidate_slot_id != slot.candidate_slot_id:
                raise TraceValidationError("candidate record order differs from plan")
            if record.disposition == CandidateDisposition.ELIGIBLE:
                expected_id = derive_candidate_id(self.cutoff_plan, slot)
                if record.candidate_id != expected_id:
                    raise TraceValidationError("candidate ID differs from frozen slot")
                candidate_ids.append(expected_id)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise TraceValidationError("candidate universe contains a duplicate ID")

    @property
    def eligible_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            record.candidate_id
            for record in self.records
            if record.candidate_id is not None
        )


def build_predecessor_exclusion_catalog(
    predecessor: CandidateUniverse,
) -> PredecessorExclusionCatalog:
    """Derive exact candidate and lineage exclusions from a pilot universe."""

    if (
        type(predecessor) is not CandidateUniverse
        or predecessor.purpose != UniversePurpose.EXCLUDED_PILOT
    ):
        raise TraceValidationError("predecessor must be an excluded-pilot universe")
    slots = {slot.candidate_slot_id: slot for slot in predecessor.cutoff_plan.slots}
    entries = tuple(
        sorted(
            (
                PredecessorExclusionEntry(
                    candidate_id=record.candidate_id,
                    lineage=slots[record.candidate_slot_id].lineage,
                )
                for record in predecessor.records
                if record.candidate_id is not None
            ),
            key=lambda entry: entry.candidate_id,
        )
    )
    return PredecessorExclusionCatalog(
        schema_version=PREDECESSOR_EXCLUSION_SCHEMA_VERSION,
        predecessor_universe_digest=canonical_digest(predecessor),
        entries=entries,
    )


def build_candidate_universe(
    plan: CutoffPlan,
    *,
    ineligible: Mapping[str, IneligibilityReason] | None = None,
    predecessor_universe_digest: str | None = None,
    predecessor_split_manifest_digest: str | None = None,
    predecessor_exclusion_catalog: PredecessorExclusionCatalog | None = None,
) -> CandidateUniverse:
    """Normalize every slot using a complete label-blind decision mapping."""

    if type(plan) is not CutoffPlan:
        raise TraceValidationError("candidate-universe plan has the wrong type")
    try:
        ineligible = {} if ineligible is None else dict(ineligible)
    except (TypeError, ValueError) as exc:
        raise TraceValidationError("ineligibility mapping is invalid") from exc
    for slot_id, reason in ineligible.items():
        require_text("ineligible slot ID", slot_id)
        _require_enum("ineligibility reason", reason, IneligibilityReason)
    known_slot_ids = {slot.candidate_slot_id for slot in plan.slots}
    unknown = sorted(set(ineligible) - known_slot_ids)
    if unknown:
        raise TraceValidationError(
            f"ineligibility mapping has unknown slots: {unknown}"
        )
    records = tuple(
        CandidateRecord(
            candidate_slot_id=slot.candidate_slot_id,
            disposition=(
                CandidateDisposition.INELIGIBLE
                if slot.candidate_slot_id in ineligible
                else CandidateDisposition.ELIGIBLE
            ),
            candidate_id=(
                None
                if slot.candidate_slot_id in ineligible
                else derive_candidate_id(plan, slot)
            ),
            ineligibility_reason=ineligible.get(slot.candidate_slot_id),
        )
        for slot in plan.slots
    )
    catalog_digest = (
        canonical_digest(predecessor_exclusion_catalog)
        if predecessor_exclusion_catalog is not None
        else None
    )
    return CandidateUniverse(
        schema_version=CANDIDATE_UNIVERSE_SCHEMA_VERSION,
        purpose=plan.purpose,
        cohort=plan.cohort,
        cutoff_plan_digest=canonical_digest(plan),
        cutoff_plan=plan,
        records=records,
        predecessor_universe_digest=predecessor_universe_digest,
        predecessor_split_manifest_digest=predecessor_split_manifest_digest,
        predecessor_exclusion_catalog_digest=catalog_digest,
        predecessor_exclusion_catalog=predecessor_exclusion_catalog,
    )


@dataclass(frozen=True, slots=True)
class RoleInterval:
    """One half-open interval on the frozen campaign split-time axis."""

    role: SplitRole
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        _require_enum("split role", self.role, SplitRole)
        _require_int("role interval start_ns", self.start_ns)
        _require_int("role interval end_ns", self.end_ns, minimum=1)
        if self.end_ns <= self.start_ns:
            raise TraceValidationError("role interval must be nonempty")


@dataclass(frozen=True, slots=True)
class LineageIncidence:
    """Canonical token-to-candidate incidence row used to form DSU edges."""

    lineage_family: LineageFamily
    lineage_value: str
    member_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum("incidence lineage family", self.lineage_family, LineageFamily)
        require_text("incidence lineage value", self.lineage_value)
        _require_sorted_unique_text(
            "incidence member candidate IDs",
            self.member_candidate_ids,
        )
        if not self.member_candidate_ids:
            raise TraceValidationError("lineage incidence requires a candidate")
        for candidate_id in self.member_candidate_ids:
            require_sha256("incidence candidate ID", candidate_id)


@dataclass(frozen=True, slots=True)
class SplitComponentIdentity:
    """Exact material from which a component ID is derived."""

    schema_version: str
    candidate_ids: tuple[str, ...]
    incidence_rows: tuple[LineageIncidence, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_COMPONENT_SCHEMA_VERSION:
            raise TraceValidationError("unsupported split-component schema")
        _require_sorted_unique_text("component candidate IDs", self.candidate_ids)
        if not self.candidate_ids:
            raise TraceValidationError("split component cannot be empty")
        for candidate_id in self.candidate_ids:
            require_sha256("component candidate ID", candidate_id)
        _validate_incidence_order(self.incidence_rows)


@dataclass(frozen=True, slots=True)
class SplitComponent:
    """One complete connected component assigned wholly to one role."""

    component_id: str
    candidate_ids: tuple[str, ...]
    role: SplitRole

    def __post_init__(self) -> None:
        require_sha256("component_id", self.component_id)
        _require_sorted_unique_text("component candidate IDs", self.candidate_ids)
        if not self.candidate_ids:
            raise TraceValidationError("split component cannot be empty")
        for candidate_id in self.candidate_ids:
            require_sha256("component candidate ID", candidate_id)
        _require_enum("component role", self.role, SplitRole)


@dataclass(frozen=True, slots=True)
class CandidateRoleAssignment:
    """Derived binding from one candidate to its component and role."""

    candidate_id: str
    component_id: str
    role: SplitRole

    def __post_init__(self) -> None:
        require_sha256("assignment candidate_id", self.candidate_id)
        require_sha256("assignment component_id", self.component_id)
        _require_enum("assignment role", self.role, SplitRole)


def _validate_incidence_order(rows: tuple[LineageIncidence, ...]) -> None:
    _require_tuple("lineage incidence", rows)
    for row in rows:
        if type(row) is not LineageIncidence:
            raise TraceValidationError("lineage incidence contains an invalid row")
    keys = tuple((row.lineage_family.value, row.lineage_value) for row in rows)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise TraceValidationError("lineage incidence must be token-sorted and unique")


def _validate_role_intervals(
    purpose: UniversePurpose,
    intervals: tuple[RoleInterval, ...],
) -> None:
    _require_tuple("role intervals", intervals)
    for interval in intervals:
        if type(interval) is not RoleInterval:
            raise TraceValidationError("role intervals contain an invalid value")
    observed_roles = tuple(interval.role for interval in intervals)
    if observed_roles != _PURPOSE_ROLES[purpose]:
        raise TraceValidationError("role intervals differ from universe purpose")
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current.start_ns < previous.end_ns:
            raise TraceValidationError("role intervals overlap or regress")


class _DisjointSet:
    def __init__(self, values: tuple[str, ...]) -> None:
        self._parents = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parents[value]
        if parent != value:
            self._parents[value] = self.find(parent)
        return self._parents[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        self._parents[upper] = lower


@dataclass(frozen=True, slots=True)
class _CandidateProjection:
    candidate_id: str
    slot: CutoffPlanSlot
    role: SplitRole


def _eligible_projections(
    universe: CandidateUniverse,
    intervals: tuple[RoleInterval, ...],
) -> tuple[_CandidateProjection, ...]:
    slots = {slot.candidate_slot_id: slot for slot in universe.cutoff_plan.slots}
    projections: list[_CandidateProjection] = []
    for record in universe.records:
        if record.candidate_id is None:
            continue
        slot = slots[record.candidate_slot_id]
        matching = tuple(
            interval.role
            for interval in intervals
            if interval.start_ns <= slot.split_time_ns < interval.end_ns
        )
        if len(matching) != 1:
            raise TraceValidationError(
                "candidate split time does not resolve to exactly one role"
            )
        projections.append(
            _CandidateProjection(
                candidate_id=record.candidate_id,
                slot=slot,
                role=matching[0],
            )
        )
    return tuple(sorted(projections, key=lambda item: item.candidate_id))


def _derive_incidence(
    candidates: tuple[tuple[str, tuple[LineageField, ...]], ...],
    cohort: SplitCohort,
) -> tuple[LineageIncidence, ...]:
    _require_enum("incidence split cohort", cohort, SplitCohort)
    included_families = _COHORT_LINEAGE_FAMILIES[cohort]
    required_family = _COHORT_REQUIRED_FAMILY[cohort]
    members: dict[tuple[LineageFamily, str], set[str]] = {}
    for candidate_id, lineage in candidates:
        require_sha256("incidence source candidate ID", candidate_id)
        _validate_complete_lineage(lineage)
        lineage_by_family = {field.family: field for field in lineage}
        if (
            required_family is not None
            and lineage_by_family[required_family].applicability
            != LineageApplicability.PRESENT
        ):
            raise TraceValidationError(
                "eligible candidate lacks the cohort-specific lineage token"
            )
        for field in lineage:
            if (
                field.family not in included_families
                or field.applicability != LineageApplicability.PRESENT
            ):
                continue
            for value in field.values:
                members.setdefault((field.family, value), set()).add(candidate_id)
    return tuple(
        LineageIncidence(
            lineage_family=family,
            lineage_value=value,
            member_candidate_ids=tuple(sorted(candidate_ids)),
        )
        for (family, value), candidate_ids in sorted(
            members.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        )
    )


def _derive_components(
    candidate_roles: Mapping[str, SplitRole],
    incidence_rows: tuple[LineageIncidence, ...],
) -> tuple[SplitComponent, ...]:
    candidate_ids = tuple(sorted(candidate_roles))
    dsu = _DisjointSet(candidate_ids)
    for row in incidence_rows:
        unknown = sorted(set(row.member_candidate_ids) - set(candidate_ids))
        if unknown:
            raise TraceValidationError(
                f"lineage incidence references unknown candidates: {unknown}"
            )
        first = row.member_candidate_ids[0]
        for candidate_id in row.member_candidate_ids[1:]:
            dsu.union(first, candidate_id)
    grouped: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        grouped.setdefault(dsu.find(candidate_id), []).append(candidate_id)
    components: list[SplitComponent] = []
    for member_list in grouped.values():
        members = tuple(sorted(member_list))
        roles = {candidate_roles[candidate_id] for candidate_id in members}
        if len(roles) != 1:
            raise TraceValidationError("one lineage component crosses split roles")
        member_set = set(members)
        touching = tuple(
            row
            for row in incidence_rows
            if member_set.intersection(row.member_candidate_ids)
        )
        identity = SplitComponentIdentity(
            schema_version=SPLIT_COMPONENT_SCHEMA_VERSION,
            candidate_ids=members,
            incidence_rows=touching,
        )
        components.append(
            SplitComponent(
                component_id=canonical_digest(identity),
                candidate_ids=members,
                role=next(iter(roles)),
            )
        )
    return tuple(sorted(components, key=lambda component: component.candidate_ids))


@dataclass(frozen=True, slots=True)
class _DerivedSplit:
    max_primary_horizon_duration_ns: int
    max_feature_lookback_ns: int
    incidence_rows: tuple[LineageIncidence, ...]
    components: tuple[SplitComponent, ...]
    assignments: tuple[CandidateRoleAssignment, ...]


def _derive_split(
    universe: CandidateUniverse,
    intervals: tuple[RoleInterval, ...],
) -> _DerivedSplit:
    _validate_role_intervals(universe.purpose, intervals)
    projections = _eligible_projections(universe, intervals)
    horizons = tuple(
        projection.slot.primary_horizon_duration_ns for projection in projections
    )
    lookbacks = tuple(projection.slot.feature_lookback_ns for projection in projections)
    max_horizon = max(horizons, default=0)
    max_lookback = max(lookbacks, default=0)
    required_gap = max_horizon + max_lookback
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current.start_ns - previous.end_ns < required_gap:
            raise TraceValidationError("role interval guard gap is too small")
    incidence = _derive_incidence(
        tuple(
            (projection.candidate_id, projection.slot.lineage)
            for projection in projections
        ),
        universe.cohort,
    )
    roles = {projection.candidate_id: projection.role for projection in projections}
    components = _derive_components(roles, incidence)
    component_by_candidate = {
        candidate_id: component
        for component in components
        for candidate_id in component.candidate_ids
    }
    assignments = tuple(
        CandidateRoleAssignment(
            candidate_id=candidate_id,
            component_id=component_by_candidate[candidate_id].component_id,
            role=component_by_candidate[candidate_id].role,
        )
        for candidate_id in sorted(roles)
    )
    return _DerivedSplit(
        max_primary_horizon_duration_ns=max_horizon,
        max_feature_lookback_ns=max_lookback,
        incidence_rows=incidence,
        components=components,
        assignments=assignments,
    )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Self-contained deterministic split replay for one candidate universe."""

    schema_version: str
    purpose: UniversePurpose
    cohort: SplitCohort
    role_assignment_algorithm: str
    candidate_universe_digest: str
    candidate_universe: CandidateUniverse
    temporal_axis_digest: str
    intervals: tuple[RoleInterval, ...]
    max_primary_horizon_duration_ns: int
    max_feature_lookback_ns: int
    incidence_rows: tuple[LineageIncidence, ...]
    components: tuple[SplitComponent, ...]
    assignments: tuple[CandidateRoleAssignment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise TraceValidationError("unsupported split-manifest schema")
        _require_enum("split-manifest purpose", self.purpose, UniversePurpose)
        _require_enum("split-manifest cohort", self.cohort, SplitCohort)
        if self.role_assignment_algorithm != ROLE_ASSIGNMENT_ALGORITHM:
            raise TraceValidationError("unsupported role-assignment algorithm")
        if type(self.candidate_universe) is not CandidateUniverse:
            raise TraceValidationError("split manifest lacks a candidate universe")
        if self.candidate_universe.purpose != self.purpose:
            raise TraceValidationError("split purpose differs from universe")
        if self.candidate_universe.cohort != self.cohort:
            raise TraceValidationError("split cohort differs from universe")
        require_sha256("candidate-universe digest", self.candidate_universe_digest)
        if canonical_digest(self.candidate_universe) != self.candidate_universe_digest:
            raise TraceValidationError("split candidate-universe digest differs")
        require_sha256("split temporal-axis digest", self.temporal_axis_digest)
        if (
            self.temporal_axis_digest
            != self.candidate_universe.cutoff_plan.temporal_axis_digest
        ):
            raise TraceValidationError("split temporal axis differs from universe")
        expected = _derive_split(self.candidate_universe, self.intervals)
        observed = _DerivedSplit(
            max_primary_horizon_duration_ns=self.max_primary_horizon_duration_ns,
            max_feature_lookback_ns=self.max_feature_lookback_ns,
            incidence_rows=self.incidence_rows,
            components=self.components,
            assignments=self.assignments,
        )
        if observed != expected:
            raise TraceValidationError(
                "split manifest differs from deterministic replay"
            )


def build_split_manifest(
    universe: CandidateUniverse,
    intervals: tuple[RoleInterval, ...],
) -> SplitManifest:
    """Build and replay one purpose-specific deterministic split manifest."""

    if type(universe) is not CandidateUniverse:
        raise TraceValidationError("split universe has the wrong type")
    derived = _derive_split(universe, intervals)
    return SplitManifest(
        schema_version=SPLIT_MANIFEST_SCHEMA_VERSION,
        purpose=universe.purpose,
        cohort=universe.cohort,
        role_assignment_algorithm=ROLE_ASSIGNMENT_ALGORITHM,
        candidate_universe_digest=canonical_digest(universe),
        candidate_universe=universe,
        temporal_axis_digest=universe.cutoff_plan.temporal_axis_digest,
        intervals=intervals,
        max_primary_horizon_duration_ns=derived.max_primary_horizon_duration_ns,
        max_feature_lookback_ns=derived.max_feature_lookback_ns,
        incidence_rows=derived.incidence_rows,
        components=derived.components,
        assignments=derived.assignments,
    )


@dataclass(frozen=True, slots=True)
class PredecessorUnionAudit:
    """Fresh PILOT-to-main component and temporal-isolation replay."""

    schema_version: str
    pilot_manifest_digest: str
    pilot_manifest: SplitManifest
    main_manifest_digest: str
    main_manifest: SplitManifest
    predecessor_exclusion_catalog_digest: str
    temporal_axis_digest: str
    combined_max_primary_horizon_duration_ns: int
    combined_max_feature_lookback_ns: int
    required_boundary_gap_ns: int
    observed_boundary_gap_ns: int
    combined_incidence_rows: tuple[LineageIncidence, ...]
    combined_components: tuple[SplitComponent, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PREDECESSOR_UNION_AUDIT_SCHEMA_VERSION:
            raise TraceValidationError("unsupported predecessor-union audit schema")
        expected = _derive_predecessor_union_audit(
            self.pilot_manifest,
            self.main_manifest,
        )
        observed = _PredecessorAuditValues(
            pilot_manifest_digest=self.pilot_manifest_digest,
            main_manifest_digest=self.main_manifest_digest,
            predecessor_exclusion_catalog_digest=(
                self.predecessor_exclusion_catalog_digest
            ),
            temporal_axis_digest=self.temporal_axis_digest,
            combined_max_primary_horizon_duration_ns=(
                self.combined_max_primary_horizon_duration_ns
            ),
            combined_max_feature_lookback_ns=(self.combined_max_feature_lookback_ns),
            required_boundary_gap_ns=self.required_boundary_gap_ns,
            observed_boundary_gap_ns=self.observed_boundary_gap_ns,
            combined_incidence_rows=self.combined_incidence_rows,
            combined_components=self.combined_components,
        )
        if observed != expected:
            raise TraceValidationError(
                "predecessor-union audit differs from deterministic replay"
            )


@dataclass(frozen=True, slots=True)
class _PredecessorAuditValues:
    pilot_manifest_digest: str
    main_manifest_digest: str
    predecessor_exclusion_catalog_digest: str
    temporal_axis_digest: str
    combined_max_primary_horizon_duration_ns: int
    combined_max_feature_lookback_ns: int
    required_boundary_gap_ns: int
    observed_boundary_gap_ns: int
    combined_incidence_rows: tuple[LineageIncidence, ...]
    combined_components: tuple[SplitComponent, ...]


def _derive_predecessor_union_audit(
    pilot: SplitManifest,
    main: SplitManifest,
) -> _PredecessorAuditValues:
    if type(pilot) is not SplitManifest or type(main) is not SplitManifest:
        raise TraceValidationError("predecessor audit manifests have the wrong type")
    if pilot.purpose != UniversePurpose.EXCLUDED_PILOT:
        raise TraceValidationError("predecessor audit requires a pilot manifest")
    if main.purpose != UniversePurpose.POST_PILOT_MAIN:
        raise TraceValidationError("predecessor audit requires a main manifest")
    if pilot.cohort != main.cohort:
        raise TraceValidationError("pilot and main split cohorts differ")
    pilot_digest = canonical_digest(pilot)
    main_digest = canonical_digest(main)
    pilot_universe = pilot.candidate_universe
    main_universe = main.candidate_universe
    if pilot.temporal_axis_digest != main.temporal_axis_digest:
        raise TraceValidationError("pilot and main temporal axes differ")
    if (
        pilot_universe.cutoff_plan.method_menu_digest
        != main_universe.cutoff_plan.method_menu_digest
    ):
        raise TraceValidationError("pilot and main method menus differ")
    if (
        pilot_universe.cutoff_plan.normalizer_digest
        != main_universe.cutoff_plan.normalizer_digest
    ):
        raise TraceValidationError("pilot and main normalizers differ")
    if (
        pilot_universe.cutoff_plan.eligibility_rule_digest
        != main_universe.cutoff_plan.eligibility_rule_digest
    ):
        raise TraceValidationError("pilot and main eligibility rules differ")
    if main_universe.predecessor_universe_digest != canonical_digest(pilot_universe):
        raise TraceValidationError("main universe binds another pilot universe")
    if main_universe.predecessor_split_manifest_digest != pilot_digest:
        raise TraceValidationError("main universe binds another pilot split")
    expected_catalog = build_predecessor_exclusion_catalog(pilot_universe)
    catalog = main_universe.predecessor_exclusion_catalog
    if catalog != expected_catalog:
        raise TraceValidationError("main predecessor exclusions are incomplete")
    catalog_digest = canonical_digest(expected_catalog)
    if main_universe.predecessor_exclusion_catalog_digest != catalog_digest:
        raise TraceValidationError("main predecessor exclusion digest differs")

    pilot_ids = set(pilot_universe.eligible_candidate_ids)
    main_ids = set(main_universe.eligible_candidate_ids)
    if not pilot_ids:
        raise TraceValidationError(
            "predecessor audit cannot follow a zero-eligible pilot"
        )
    if not main_ids:
        raise TraceValidationError("predecessor audit requires an eligible main lane")
    reused = sorted(pilot_ids.intersection(main_ids))
    if reused:
        raise TraceValidationError(f"candidate IDs cross pilot and main: {reused}")
    pilot_slots = {
        record.candidate_id: slot
        for slot, record in zip(
            pilot_universe.cutoff_plan.slots,
            pilot_universe.records,
            strict=True,
        )
        if record.candidate_id is not None
    }
    main_slots = {
        record.candidate_id: slot
        for slot, record in zip(
            main_universe.cutoff_plan.slots,
            main_universe.records,
            strict=True,
        )
        if record.candidate_id is not None
    }
    pilot_schedule_cases = {slot.schedule_case_id for slot in pilot_slots.values()}
    main_schedule_cases = {slot.schedule_case_id for slot in main_slots.values()}
    reused_schedule_cases = sorted(
        pilot_schedule_cases.intersection(main_schedule_cases)
    )
    if reused_schedule_cases:
        raise TraceValidationError(
            f"schedule cases cross pilot and main: {reused_schedule_cases}"
        )
    incidence = _derive_incidence(
        tuple(
            sorted(
                (
                    (candidate_id, slot.lineage)
                    for candidate_id, slot in {**pilot_slots, **main_slots}.items()
                ),
                key=lambda item: item[0],
            )
        ),
        pilot.cohort,
    )
    role_by_candidate = {
        assignment.candidate_id: assignment.role
        for assignment in (*pilot.assignments, *main.assignments)
    }
    components = _derive_components(role_by_candidate, incidence)
    for component in components:
        roles = {
            role_by_candidate[candidate_id] for candidate_id in component.candidate_ids
        }
        if SplitRole.PILOT in roles and len(roles) > 1:
            raise TraceValidationError("one lineage component crosses pilot and main")

    max_horizon = max(
        pilot.max_primary_horizon_duration_ns,
        main.max_primary_horizon_duration_ns,
    )
    max_lookback = max(
        pilot.max_feature_lookback_ns,
        main.max_feature_lookback_ns,
    )
    required_gap = max_horizon + max_lookback
    pilot_interval = pilot.intervals[0]
    train_interval = main.intervals[0]
    observed_gap = train_interval.start_ns - pilot_interval.end_ns
    if observed_gap < required_gap:
        raise TraceValidationError("pilot-to-train guard gap is too small")
    return _PredecessorAuditValues(
        pilot_manifest_digest=pilot_digest,
        main_manifest_digest=main_digest,
        predecessor_exclusion_catalog_digest=catalog_digest,
        temporal_axis_digest=pilot.temporal_axis_digest,
        combined_max_primary_horizon_duration_ns=max_horizon,
        combined_max_feature_lookback_ns=max_lookback,
        required_boundary_gap_ns=required_gap,
        observed_boundary_gap_ns=observed_gap,
        combined_incidence_rows=incidence,
        combined_components=components,
    )


def build_predecessor_union_audit(
    pilot: SplitManifest,
    main: SplitManifest,
) -> PredecessorUnionAudit:
    """Build an exact predecessor-union replay or fail the main campaign."""

    values = _derive_predecessor_union_audit(pilot, main)
    return PredecessorUnionAudit(
        schema_version=PREDECESSOR_UNION_AUDIT_SCHEMA_VERSION,
        pilot_manifest_digest=values.pilot_manifest_digest,
        pilot_manifest=pilot,
        main_manifest_digest=values.main_manifest_digest,
        main_manifest=main,
        predecessor_exclusion_catalog_digest=(
            values.predecessor_exclusion_catalog_digest
        ),
        temporal_axis_digest=values.temporal_axis_digest,
        combined_max_primary_horizon_duration_ns=(
            values.combined_max_primary_horizon_duration_ns
        ),
        combined_max_feature_lookback_ns=(values.combined_max_feature_lookback_ns),
        required_boundary_gap_ns=values.required_boundary_gap_ns,
        observed_boundary_gap_ns=values.observed_boundary_gap_ns,
        combined_incidence_rows=values.combined_incidence_rows,
        combined_components=values.combined_components,
    )


@dataclass(frozen=True, slots=True)
class LoadedSplitArtifact[T]:
    """Stable file identity for one loaded canonical split artifact."""

    artifact: T
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256("loaded split artifact digest", self.digest)
        _require_int("loaded split artifact size", self.size_bytes, minimum=1)


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


def _validate_parent_binding(
    path: Path,
    parent_descriptor: int,
    artifact_name: str,
) -> None:
    """Reject publication through a parent path rebound after opening."""

    try:
        opened = os.fstat(parent_descriptor)
        linked = path.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise OSError(f"{artifact_name} parent cannot be revalidated") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
    ):
        raise OSError(f"{artifact_name} parent path changed during publication")


def _validate_output_binding(
    descriptor: int,
    parent_descriptor: int,
    path: Path,
    expected_size: int,
) -> os.stat_result:
    """Bind an open output descriptor to its single published directory entry."""

    opened = os.fstat(descriptor)
    linked = os.stat(
        path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or opened.st_nlink != 1
        or linked.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        or opened.st_size != expected_size
        or linked.st_size != expected_size
    ):
        raise OSError("split artifact output identity changed")
    return opened


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


def _read_exact_output(descriptor: int, expected_size: int) -> bytes:
    observed = bytearray()
    read_offset = 0
    while read_offset < expected_size:
        chunk = os.pread(descriptor, expected_size - read_offset, read_offset)
        if not chunk:
            raise OSError("split artifact ended during readback")
        observed.extend(chunk)
        read_offset += len(chunk)
    return bytes(observed)


def _validate_closed_output_binding(
    parent_descriptor: int,
    path: Path,
    expected: os.stat_result,
    expected_size: int,
) -> None:
    """Revalidate the published path after the output descriptor is closed."""

    linked = os.stat(
        path.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_nlink != 1
        or linked.st_size != expected_size
        or _output_state(expected) != _output_state(linked)
    ):
        raise OSError("split artifact output identity changed after close")


def _write_artifact[T](
    path: Path,
    artifact: T,
    expected: type[T],
    artifact_name: str,
) -> str:
    if type(artifact) is not expected:
        raise TraceValidationError(f"{artifact_name} has the wrong type")
    raw = canonical_json(artifact)
    if len(raw) > MAX_SPLIT_ARTIFACT_BYTES:
        raise TraceValidationError(f"{artifact_name} exceeds the size limit")
    parsed = parse_canonical_dataclass(
        raw,
        expected,
        artifact_name=artifact_name,
        max_bytes=MAX_SPLIT_ARTIFACT_BYTES,
    )
    if parsed != artifact:
        raise TraceValidationError(f"{artifact_name} changes during canonical replay")
    parent_descriptor = _open_parent(path, artifact_name)
    descriptor: int | None = None
    created = False
    precreate_error: TraceValidationError | None = None
    commit_failure: BaseException | None = None
    published_identity: os.stat_result | None = None
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
                        raise OSError("split artifact write made no progress")
                    offset += count
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
                pre_read_identity = _validate_output_binding(
                    descriptor,
                    parent_descriptor,
                    path,
                    len(raw),
                )
                first_observed = _read_exact_output(descriptor, len(raw))
                mid_read_identity = _validate_output_binding(
                    descriptor,
                    parent_descriptor,
                    path,
                    len(raw),
                )
                if first_observed != raw or _output_state(
                    pre_read_identity
                ) != _output_state(mid_read_identity):
                    raise OSError("split artifact differs from staged bytes")
                second_observed = _read_exact_output(descriptor, len(raw))
                published_identity = _validate_output_binding(
                    descriptor,
                    parent_descriptor,
                    path,
                    len(raw),
                )
                if second_observed != raw or _output_state(
                    mid_read_identity
                ) != _output_state(published_identity):
                    raise OSError("split artifact changed during stable readback")
            except BaseException as exc:
                commit_failure = exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if commit_failure is None:
                    commit_failure = exc
        if published_identity is not None:
            try:
                _validate_closed_output_binding(
                    parent_descriptor,
                    path,
                    published_identity,
                    len(raw),
                )
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
            os.close(parent_descriptor)
        except BaseException as exc:
            if created:
                if commit_failure is None:
                    commit_failure = exc
            elif precreate_error is None:
                precreate_error = TraceValidationError(
                    f"cannot close {artifact_name} safely"
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


def _read_stable_file(path: Path, artifact_name: str) -> bytes:
    if not path.is_absolute():
        raise TraceValidationError(f"{artifact_name} path must be absolute")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise TraceValidationError(f"{artifact_name} is missing") from exc
    except OSError as exc:
        raise TraceValidationError(f"cannot inspect {artifact_name} safely") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise TraceValidationError(
            f"{artifact_name} must be a singly linked regular non-symlink file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except BaseException as exc:
        if isinstance(exc, Exception):
            raise TraceValidationError(f"cannot open {artifact_name} safely") from exc
        raise
    read_failure: BaseException | None = None
    raw = b""
    linked: os.stat_result | None = None
    try:
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _output_state(before) != _output_state(opened)
            ):
                raise TraceValidationError(f"{artifact_name} changed while opening")
            if opened.st_size < 1 or opened.st_size > MAX_SPLIT_ARTIFACT_BYTES:
                raise TraceValidationError(f"{artifact_name} has an invalid size")
            first_raw = _read_exact_output(descriptor, opened.st_size)
            middle = os.fstat(descriptor)
            if _output_state(opened) != _output_state(middle):
                raise TraceValidationError(f"{artifact_name} changed while reading")
            second_raw = _read_exact_output(descriptor, opened.st_size)
            after = os.fstat(descriptor)
            if first_raw != second_raw or _output_state(middle) != _output_state(after):
                raise TraceValidationError(
                    f"{artifact_name} changed during stable readback"
                )
            linked = path.lstat()
            if (
                not stat.S_ISREG(linked.st_mode)
                or linked.st_nlink != 1
                or _output_state(after) != _output_state(linked)
            ):
                raise TraceValidationError(
                    f"{artifact_name} path changed while reading"
                )
            final_raw = _read_exact_output(descriptor, opened.st_size)
            final_opened = os.fstat(descriptor)
            if (
                second_raw != final_raw
                or _output_state(after) != _output_state(final_opened)
                or _output_state(linked) != _output_state(final_opened)
            ):
                raise TraceValidationError(
                    f"{artifact_name} changed after path revalidation"
                )
            raw = final_raw
        except BaseException as exc:
            read_failure = exc
    finally:
        try:
            os.close(descriptor)
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
    if linked is None:
        raise TraceValidationError(f"{artifact_name} lacks a stable path binding")
    try:
        final = path.lstat()
    except OSError as exc:
        raise TraceValidationError(f"cannot revalidate {artifact_name} safely") from exc
    if _output_state(linked) != _output_state(final):
        raise TraceValidationError(f"{artifact_name} changed after close")
    return raw


def _load_artifact[T](
    path: Path,
    expected: type[T],
    artifact_name: str,
) -> LoadedSplitArtifact[T]:
    raw = _read_stable_file(path, artifact_name)
    artifact = parse_canonical_dataclass(
        raw,
        expected,
        artifact_name=artifact_name,
        max_bytes=MAX_SPLIT_ARTIFACT_BYTES,
    )
    return LoadedSplitArtifact(
        artifact=artifact,
        digest=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def write_cutoff_plan(path: Path, artifact: CutoffPlan) -> str:
    """Create and durably publish one canonical cutoff plan."""

    return _write_artifact(path, artifact, CutoffPlan, "cutoff-plan artifact")


def load_cutoff_plan(path: Path) -> LoadedSplitArtifact[CutoffPlan]:
    """Load one stable canonical cutoff plan."""

    return _load_artifact(path, CutoffPlan, "cutoff-plan artifact")


def write_candidate_universe(path: Path, artifact: CandidateUniverse) -> str:
    """Create and durably publish one canonical candidate universe."""

    return _write_artifact(
        path,
        artifact,
        CandidateUniverse,
        "candidate-universe artifact",
    )


def load_candidate_universe(path: Path) -> LoadedSplitArtifact[CandidateUniverse]:
    """Load one stable canonical candidate universe."""

    return _load_artifact(path, CandidateUniverse, "candidate-universe artifact")


def write_predecessor_exclusion_catalog(
    path: Path,
    artifact: PredecessorExclusionCatalog,
) -> str:
    """Create and durably publish one canonical predecessor catalog."""

    return _write_artifact(
        path,
        artifact,
        PredecessorExclusionCatalog,
        "predecessor-exclusion catalog",
    )


def load_predecessor_exclusion_catalog(
    path: Path,
) -> LoadedSplitArtifact[PredecessorExclusionCatalog]:
    """Load one stable canonical predecessor catalog."""

    return _load_artifact(
        path,
        PredecessorExclusionCatalog,
        "predecessor-exclusion catalog",
    )


def write_split_manifest(path: Path, artifact: SplitManifest) -> str:
    """Create and durably publish one canonical split manifest."""

    return _write_artifact(path, artifact, SplitManifest, "split-manifest artifact")


def load_split_manifest(path: Path) -> LoadedSplitArtifact[SplitManifest]:
    """Load and deterministically replay one canonical split manifest."""

    return _load_artifact(path, SplitManifest, "split-manifest artifact")


def write_predecessor_union_audit(
    path: Path,
    artifact: PredecessorUnionAudit,
) -> str:
    """Create and durably publish one canonical predecessor-union audit."""

    return _write_artifact(
        path,
        artifact,
        PredecessorUnionAudit,
        "predecessor-union audit",
    )


def load_predecessor_union_audit(
    path: Path,
) -> LoadedSplitArtifact[PredecessorUnionAudit]:
    """Load and replay one stable canonical predecessor-union audit."""

    return _load_artifact(
        path,
        PredecessorUnionAudit,
        "predecessor-union audit",
    )
