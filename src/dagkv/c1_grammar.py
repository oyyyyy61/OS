"""Finite C1-B1 branch grammars, support catalogs, and instance bindings."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from itertools import product
from pathlib import Path

from dagkv.c1_schedule import ClosedScheduleArtifact
from dagkv.c1_split import CandidateUniverse
from dagkv.c1_trace import (
    TraceCommitIndeterminateError,
    TraceValidationError,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
)
from dagkv.domain import require_sha256, require_text

BRANCH_GRAMMAR_SCHEMA_VERSION = "dagkv.m3.branch_grammar.v1"
ASSIGNMENT_IDENTITY_SCHEMA_VERSION = "dagkv.m3.grammar_assignment_identity.v1"
OUTCOME_IDENTITY_SCHEMA_VERSION = "dagkv.m3.grammar_outcome_identity.v1"
FEASIBLE_SUPPORT_CATALOG_SCHEMA_VERSION = "dagkv.m3.feasible_support_catalog.v1"
SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION = (
    "dagkv.m3.feasible_support_catalog_identity.v1"
)
STRUCTURAL_ABSTENTION_SCHEMA_VERSION = "dagkv.m3.grammar_structural_abstention.v1"
GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION = "dagkv.m3.grammar_instance_binding.v1"
GRAMMAR_CASE_PROJECTION_SCHEMA_VERSION = "dagkv.m3.grammar_case_projection.v1"
GRAMMAR_BINDING_CONSERVATION_SCHEMA_VERSION = "dagkv.m3.grammar_binding_conservation.v1"
GRAMMAR_BINDING_INPUT_DIGESTS_SCHEMA_VERSION = (
    "dagkv.m3.grammar_binding_input_digests.v1"
)
SCHEDULE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION = (
    "dagkv.m3.schedule_case_catalog_identity.v1"
)
SOURCE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION = "dagkv.m3.source_case_catalog_identity.v1"
GRAMMAR_SAFETY_CEILING_SCHEMA_VERSION = "dagkv.m3.grammar_safety_ceiling.v1"
GRAMMAR_ENUMERATOR_IMPLEMENTATION = "finite_truth_table_65536_v1"
GRAMMAR_CANONICAL_ORDER = (
    "variables:variable_id;domains:text;rules:rule_id;clauses:clause_id;"
    "terms:variable_id;subsets:text;mappings:demand_site_id;"
    "assignments:cartesian_product;outcomes:outcome_id"
)
MAX_GRAMMAR_ASSIGNMENTS = 65_536
MAX_GRAMMAR_VARIABLES = 64
MAX_GRAMMAR_DOMAIN_VALUES = 256
MAX_GRAMMAR_RULES = 4_096
MAX_GRAMMAR_CLAUSES = 65_536
MAX_GRAMMAR_TERMS = 65_536
MAX_GRAMMAR_TERM_VALUES = 262_144
MAX_GRAMMAR_MATCH_WORK = 16_777_216
MAX_GRAMMAR_ASSIGNMENT_VALUE_CELLS = 262_144
MAX_GRAMMAR_IDENTIFIER_BYTES = 4_096
MAX_GRAMMAR_TOTAL_TEXT_BYTES = 16 * 1024 * 1024
MAX_GRAMMAR_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_GRAMMAR_DERIVED_CATALOG_BYTES = 64 * 1024 * 1024


class RuleDisposition(StrEnum):
    """Closed result of one DNF rule."""

    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"


class OutcomeKind(StrEnum):
    """Whether a feasible path contains a future demand."""

    USE = "USE"
    NO_USE = "NO_USE"


class InfeasibilityReason(StrEnum):
    """Typed, pre-observation reason for a truth-table row to be impossible."""

    XOR_CONSTRAINT = "XOR_CONSTRAINT"
    AND_CONSTRAINT = "AND_CONSTRAINT"
    PATH_CONSTRAINT = "PATH_CONSTRAINT"
    SHARED_LATENT_CONFLICT = "SHARED_LATENT_CONFLICT"
    TERMINAL_PATH = "TERMINAL_PATH"


class StructuralAbstentionReason(StrEnum):
    """Closed failures for which exhaustive support cannot be constructed."""

    SAFETY_CEILING_EXCEEDED = "SAFETY_CEILING_EXCEEDED"
    UNKNOWN_VARIABLE = "UNKNOWN_VARIABLE"
    UNBOUNDED_DOMAIN = "UNBOUNDED_DOMAIN"


def _require_enum(name: str, value: object, expected: type[StrEnum]) -> None:
    if type(value) is not expected:
        raise TraceValidationError(f"{name} must be a {expected.__name__}")


def _require_int(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_tuple(name: str, value: object) -> None:
    if not isinstance(value, tuple):
        raise TraceValidationError(f"{name} must be a tuple")


def _require_sorted_unique_text(name: str, values: tuple[str, ...]) -> None:
    _require_tuple(name, values)
    for value in values:
        require_text(name, value)
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise TraceValidationError(f"{name} must be sorted and unique")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is not None:
        require_text(name, value)


@dataclass(frozen=True, slots=True)
class GrammarSafetyCeiling:
    """Versioned parser and exhaustive-enumeration resource identity."""

    schema_version: str
    max_artifact_bytes: int
    max_variables: int
    max_domain_values_per_variable: int
    max_rules: int
    max_clauses: int
    max_terms: int
    max_term_values: int
    max_assignments: int
    max_match_work: int
    max_assignment_value_cells: int
    max_identifier_bytes: int
    max_total_text_bytes: int
    max_derived_catalog_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != GRAMMAR_SAFETY_CEILING_SCHEMA_VERSION:
            raise TraceValidationError("unsupported grammar-safety-ceiling schema")
        for name in (
            "max_artifact_bytes",
            "max_variables",
            "max_domain_values_per_variable",
            "max_rules",
            "max_clauses",
            "max_terms",
            "max_term_values",
            "max_assignments",
            "max_match_work",
            "max_assignment_value_cells",
            "max_identifier_bytes",
            "max_total_text_bytes",
            "max_derived_catalog_bytes",
        ):
            _require_int(f"grammar safety {name}", getattr(self, name))


def _current_safety_ceiling() -> GrammarSafetyCeiling:
    """Freeze every implementation resource bound into evidence identity."""

    return GrammarSafetyCeiling(
        schema_version=GRAMMAR_SAFETY_CEILING_SCHEMA_VERSION,
        max_artifact_bytes=MAX_GRAMMAR_ARTIFACT_BYTES,
        max_variables=MAX_GRAMMAR_VARIABLES,
        max_domain_values_per_variable=MAX_GRAMMAR_DOMAIN_VALUES,
        max_rules=MAX_GRAMMAR_RULES,
        max_clauses=MAX_GRAMMAR_CLAUSES,
        max_terms=MAX_GRAMMAR_TERMS,
        max_term_values=MAX_GRAMMAR_TERM_VALUES,
        max_assignments=MAX_GRAMMAR_ASSIGNMENTS,
        max_match_work=MAX_GRAMMAR_MATCH_WORK,
        max_assignment_value_cells=MAX_GRAMMAR_ASSIGNMENT_VALUE_CELLS,
        max_identifier_bytes=MAX_GRAMMAR_IDENTIFIER_BYTES,
        max_total_text_bytes=MAX_GRAMMAR_TOTAL_TEXT_BYTES,
        max_derived_catalog_bytes=MAX_GRAMMAR_DERIVED_CATALOG_BYTES,
    )


@dataclass(frozen=True, slots=True)
class BranchVariable:
    """One XOR variable and its finite domain, or an explicit unbounded fixture."""

    variable_id: str
    domain_values: tuple[str, ...] | None
    shared_random_source_id: str | None

    def __post_init__(self) -> None:
        require_text("branch variable ID", self.variable_id)
        _require_optional_text(
            "branch variable shared-random-source ID",
            self.shared_random_source_id,
        )
        if self.domain_values is not None:
            _require_sorted_unique_text(
                "branch variable domain",
                self.domain_values,
            )
            if not self.domain_values:
                raise TraceValidationError(
                    "finite branch variable domain cannot be empty; use null for "
                    "an unbounded negative fixture"
                )


@dataclass(frozen=True, slots=True)
class GrammarTerm:
    """One finite subset restriction in a conjunctive clause."""

    variable_id: str
    allowed_values: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text("grammar term variable ID", self.variable_id)
        _require_sorted_unique_text("grammar term values", self.allowed_values)
        if not self.allowed_values:
            raise TraceValidationError("grammar term values cannot be empty")


@dataclass(frozen=True, slots=True)
class GrammarClause:
    """One conjunction; omitted variables are all-domain wildcards."""

    clause_id: str
    terms: tuple[GrammarTerm, ...]

    def __post_init__(self) -> None:
        require_text("grammar clause ID", self.clause_id)
        _require_tuple("grammar clause terms", self.terms)
        if any(type(term) is not GrammarTerm for term in self.terms):
            raise TraceValidationError("grammar clause contains an invalid term")
        variable_ids = tuple(term.variable_id for term in self.terms)
        if variable_ids != tuple(sorted(variable_ids)):
            raise TraceValidationError("grammar clause terms must use canonical order")
        if len(variable_ids) != len(set(variable_ids)):
            raise TraceValidationError(
                "grammar clause contains more than one term for a variable"
            )


@dataclass(frozen=True, slots=True)
class SiteEpochMapping:
    """One member of the total many-to-one active-site mapping."""

    demand_site_id: str
    epoch_slot_id: str

    def __post_init__(self) -> None:
        require_text("outcome demand-site ID", self.demand_site_id)
        require_text("outcome epoch-slot ID", self.epoch_slot_id)


@dataclass(frozen=True, slots=True)
class FeasibleOutcome:
    """Frozen terminal, active-site, and reference-epoch semantics."""

    kind: OutcomeKind
    terminal_node_ids: tuple[str, ...]
    active_demand_site_ids: tuple[str, ...]
    site_epoch_mappings: tuple[SiteEpochMapping, ...]

    def __post_init__(self) -> None:
        _require_enum("outcome kind", self.kind, OutcomeKind)
        _require_sorted_unique_text(
            "outcome terminal-node IDs",
            self.terminal_node_ids,
        )
        if not self.terminal_node_ids:
            raise TraceValidationError("feasible outcome requires a terminal node")
        _require_sorted_unique_text(
            "outcome active demand-site IDs",
            self.active_demand_site_ids,
        )
        _require_tuple("outcome site-epoch mappings", self.site_epoch_mappings)
        if any(
            type(mapping) is not SiteEpochMapping
            for mapping in self.site_epoch_mappings
        ):
            raise TraceValidationError("outcome contains an invalid site-epoch mapping")
        mapping_order = tuple(
            (mapping.demand_site_id, mapping.epoch_slot_id)
            for mapping in self.site_epoch_mappings
        )
        if mapping_order != tuple(sorted(mapping_order)):
            raise TraceValidationError(
                "outcome site-epoch mappings must use canonical order"
            )
        mapped_sites = tuple(
            mapping.demand_site_id for mapping in self.site_epoch_mappings
        )
        if len(mapped_sites) != len(set(mapped_sites)):
            raise TraceValidationError(
                "an outcome demand site is mapped more than once"
            )
        if mapped_sites != self.active_demand_site_ids:
            raise TraceValidationError(
                "outcome mapping domain must equal its active demand sites"
            )
        if self.kind == OutcomeKind.NO_USE:
            if self.active_demand_site_ids or self.site_epoch_mappings:
                raise TraceValidationError(
                    "NO_USE outcome must have empty active domains"
                )
        elif not self.active_demand_site_ids:
            raise TraceValidationError("USE outcome requires an active demand site")

    @property
    def active_epoch_slot_ids(self) -> tuple[str, ...]:
        """Return the exact nonempty image of the active-site mapping."""

        return tuple(sorted({item.epoch_slot_id for item in self.site_epoch_mappings}))


@dataclass(frozen=True, slots=True)
class GrammarRule:
    """One disjunction of clauses with a single frozen semantic result."""

    rule_id: str
    clauses: tuple[GrammarClause, ...]
    disposition: RuleDisposition
    outcome: FeasibleOutcome | None
    infeasibility_reason: InfeasibilityReason | None

    def __post_init__(self) -> None:
        require_text("grammar rule ID", self.rule_id)
        _require_tuple("grammar rule clauses", self.clauses)
        if not self.clauses or any(
            type(clause) is not GrammarClause for clause in self.clauses
        ):
            raise TraceValidationError("grammar rule requires valid clauses")
        clause_ids = tuple(clause.clause_id for clause in self.clauses)
        if clause_ids != tuple(sorted(clause_ids)) or len(clause_ids) != len(
            set(clause_ids)
        ):
            raise TraceValidationError("grammar rule clauses must be sorted and unique")
        _require_enum("grammar rule disposition", self.disposition, RuleDisposition)
        if self.disposition == RuleDisposition.FEASIBLE:
            if type(self.outcome) is not FeasibleOutcome:
                raise TraceValidationError("feasible grammar rule requires one outcome")
            if self.infeasibility_reason is not None:
                raise TraceValidationError(
                    "feasible grammar rule cannot name an infeasibility reason"
                )
        else:
            if self.outcome is not None:
                raise TraceValidationError(
                    "infeasible grammar rule cannot contain an outcome"
                )
            _require_enum(
                "grammar infeasibility reason",
                self.infeasibility_reason,
                InfeasibilityReason,
            )


@dataclass(frozen=True, slots=True)
class BranchGrammar:
    """Create-only finite truth-table grammar for one workflow template."""

    schema_version: str
    canonical_order: str
    workflow_template_digest: str
    variables: tuple[BranchVariable, ...]
    demand_site_ids: tuple[str, ...]
    reference_epoch_slot_ids: tuple[str, ...]
    terminal_node_ids: tuple[str, ...]
    rules: tuple[GrammarRule, ...]

    def __post_init__(self) -> None:
        if self.schema_version != BRANCH_GRAMMAR_SCHEMA_VERSION:
            raise TraceValidationError("unsupported branch-grammar schema")
        if self.canonical_order != GRAMMAR_CANONICAL_ORDER:
            raise TraceValidationError("unsupported branch-grammar canonical order")
        require_sha256("workflow-template digest", self.workflow_template_digest)
        _require_tuple("branch grammar variables", self.variables)
        if any(type(variable) is not BranchVariable for variable in self.variables):
            raise TraceValidationError("branch grammar contains an invalid variable")
        variable_ids = tuple(variable.variable_id for variable in self.variables)
        if variable_ids != tuple(sorted(variable_ids)) or len(variable_ids) != len(
            set(variable_ids)
        ):
            raise TraceValidationError(
                "branch grammar variables must be sorted and unique"
            )
        _require_sorted_unique_text("grammar demand-site IDs", self.demand_site_ids)
        _require_sorted_unique_text(
            "grammar reference-epoch-slot IDs",
            self.reference_epoch_slot_ids,
        )
        _require_sorted_unique_text(
            "grammar terminal-node IDs",
            self.terminal_node_ids,
        )
        _require_tuple("branch grammar rules", self.rules)
        if not self.rules or any(type(rule) is not GrammarRule for rule in self.rules):
            raise TraceValidationError("branch grammar requires valid rules")
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if rule_ids != tuple(sorted(rule_ids)) or len(rule_ids) != len(set(rule_ids)):
            raise TraceValidationError("branch grammar rules must be sorted and unique")


@dataclass(frozen=True, slots=True)
class AssignmentValue:
    variable_id: str
    value: str

    def __post_init__(self) -> None:
        require_text("assignment variable ID", self.variable_id)
        require_text("assignment value", self.value)


@dataclass(frozen=True, slots=True)
class AssignmentIdentity:
    schema_version: str
    branch_grammar_digest: str
    values: tuple[AssignmentValue, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ASSIGNMENT_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported assignment-identity schema")
        require_sha256("assignment branch-grammar digest", self.branch_grammar_digest)
        _require_tuple("assignment values", self.values)
        if any(type(item) is not AssignmentValue for item in self.values):
            raise TraceValidationError("assignment contains an invalid value")
        ids = tuple(item.variable_id for item in self.values)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise TraceValidationError("assignment values must be sorted and unique")


@dataclass(frozen=True, slots=True)
class OutcomeIdentity:
    schema_version: str
    branch_grammar_digest: str
    outcome: FeasibleOutcome

    def __post_init__(self) -> None:
        if self.schema_version != OUTCOME_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported outcome-identity schema")
        require_sha256("outcome branch-grammar digest", self.branch_grammar_digest)
        if type(self.outcome) is not FeasibleOutcome:
            raise TraceValidationError("outcome identity has the wrong outcome type")


@dataclass(frozen=True, slots=True)
class SupportAssignment:
    assignment_id: str
    values: tuple[AssignmentValue, ...]
    rule_id: str
    clause_id: str
    disposition: RuleDisposition
    outcome_id: str | None
    infeasibility_reason: InfeasibilityReason | None

    def __post_init__(self) -> None:
        require_sha256("support assignment ID", self.assignment_id)
        _require_tuple("support assignment values", self.values)
        if any(type(item) is not AssignmentValue for item in self.values):
            raise TraceValidationError("support assignment contains an invalid value")
        require_text("support assignment rule ID", self.rule_id)
        require_text("support assignment clause ID", self.clause_id)
        _require_enum(
            "support assignment disposition", self.disposition, RuleDisposition
        )
        if self.disposition == RuleDisposition.FEASIBLE:
            if self.outcome_id is None:
                raise TraceValidationError("feasible assignment requires an outcome ID")
            require_sha256("support assignment outcome ID", self.outcome_id)
            if self.infeasibility_reason is not None:
                raise TraceValidationError(
                    "feasible assignment cannot contain an infeasibility reason"
                )
        else:
            if self.outcome_id is not None:
                raise TraceValidationError(
                    "infeasible assignment cannot contain an outcome ID"
                )
            _require_enum(
                "support assignment infeasibility reason",
                self.infeasibility_reason,
                InfeasibilityReason,
            )


@dataclass(frozen=True, slots=True)
class SupportOutcome:
    outcome_id: str
    outcome: FeasibleOutcome

    def __post_init__(self) -> None:
        require_sha256("support outcome ID", self.outcome_id)
        if type(self.outcome) is not FeasibleOutcome:
            raise TraceValidationError("support outcome has the wrong type")


@dataclass(frozen=True, slots=True)
class SupportCatalogIdentity:
    schema_version: str
    enumerator_implementation: str
    safety_ceiling: GrammarSafetyCeiling
    branch_grammar_digest: str
    assignments: tuple[SupportAssignment, ...]
    outcomes: tuple[SupportOutcome, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported support-catalog identity schema")
        if self.enumerator_implementation != GRAMMAR_ENUMERATOR_IMPLEMENTATION:
            raise TraceValidationError("unsupported grammar enumerator implementation")
        if self.safety_ceiling != _current_safety_ceiling():
            raise TraceValidationError("support identity safety ceiling differs")
        require_sha256("support identity grammar digest", self.branch_grammar_digest)


@dataclass(frozen=True, slots=True)
class _CompilationFailure:
    reason: StructuralAbstentionReason
    detail_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompilationValues:
    support_catalog_id: str
    assignments: tuple[SupportAssignment, ...]
    outcomes: tuple[SupportOutcome, ...]


@dataclass(frozen=True, slots=True)
class StructuralAbstention:
    """Self-contained replayable negative-fixture result with no catalog."""

    schema_version: str
    enumerator_implementation: str
    safety_ceiling: GrammarSafetyCeiling
    branch_grammar_digest: str
    branch_grammar: BranchGrammar
    reason: StructuralAbstentionReason
    detail_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != STRUCTURAL_ABSTENTION_SCHEMA_VERSION:
            raise TraceValidationError("unsupported structural-abstention schema")
        if self.enumerator_implementation != GRAMMAR_ENUMERATOR_IMPLEMENTATION:
            raise TraceValidationError("unsupported grammar enumerator implementation")
        if self.safety_ceiling != _current_safety_ceiling():
            raise TraceValidationError("abstention safety ceiling differs")
        require_sha256("abstention branch-grammar digest", self.branch_grammar_digest)
        if type(self.branch_grammar) is not BranchGrammar:
            raise TraceValidationError("abstention has the wrong grammar type")
        if self.branch_grammar_digest != canonical_digest(self.branch_grammar):
            raise TraceValidationError("abstention branch-grammar digest differs")
        _require_enum(
            "structural abstention reason",
            self.reason,
            StructuralAbstentionReason,
        )
        _require_sorted_unique_text("structural abstention details", self.detail_ids)
        replay = _derive_compilation(self.branch_grammar, self.safety_ceiling)
        if type(replay) is not _CompilationFailure:
            raise TraceValidationError("structural abstention grammar compiles fully")
        if replay.reason != self.reason or replay.detail_ids != self.detail_ids:
            raise TraceValidationError("structural abstention differs from replay")


@dataclass(frozen=True, slots=True)
class FeasibleSupportCatalog:
    """Exact exhaustive assignment table and its distinct feasible outcomes."""

    schema_version: str
    enumerator_implementation: str
    safety_ceiling: GrammarSafetyCeiling
    support_catalog_id: str
    branch_grammar_digest: str
    branch_grammar: BranchGrammar
    assignments: tuple[SupportAssignment, ...]
    outcomes: tuple[SupportOutcome, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FEASIBLE_SUPPORT_CATALOG_SCHEMA_VERSION:
            raise TraceValidationError("unsupported feasible-support-catalog schema")
        if self.enumerator_implementation != GRAMMAR_ENUMERATOR_IMPLEMENTATION:
            raise TraceValidationError("unsupported grammar enumerator implementation")
        if self.safety_ceiling != _current_safety_ceiling():
            raise TraceValidationError("support catalog safety ceiling differs")
        require_sha256("support catalog ID", self.support_catalog_id)
        require_sha256("support catalog grammar digest", self.branch_grammar_digest)
        if type(self.branch_grammar) is not BranchGrammar:
            raise TraceValidationError("support catalog has the wrong grammar type")
        derived = _derive_compilation(self.branch_grammar, self.safety_ceiling)
        if type(derived) is _CompilationFailure:
            raise TraceValidationError("support catalog grammar structurally abstains")
        if self.branch_grammar_digest != canonical_digest(self.branch_grammar):
            raise TraceValidationError("support catalog grammar digest differs")
        if (
            self.support_catalog_id != derived.support_catalog_id
            or self.assignments != derived.assignments
            or self.outcomes != derived.outcomes
        ):
            raise TraceValidationError("support catalog differs from exhaustive replay")


def _validate_outcome(grammar: BranchGrammar, outcome: FeasibleOutcome) -> None:
    unknown_terminals = set(outcome.terminal_node_ids).difference(
        grammar.terminal_node_ids
    )
    if unknown_terminals:
        raise TraceValidationError(
            f"outcome contains unknown terminal nodes: {sorted(unknown_terminals)}"
        )
    unknown_sites = set(outcome.active_demand_site_ids).difference(
        grammar.demand_site_ids
    )
    if unknown_sites:
        raise TraceValidationError(
            f"outcome contains unknown demand sites: {sorted(unknown_sites)}"
        )
    unknown_epochs = set(outcome.active_epoch_slot_ids).difference(
        grammar.reference_epoch_slot_ids
    )
    if unknown_epochs:
        raise TraceValidationError(
            f"outcome contains unknown active epoch slots: {sorted(unknown_epochs)}"
        )
    if outcome.kind == OutcomeKind.USE and not outcome.active_epoch_slot_ids:
        raise TraceValidationError("USE outcome has an empty active epoch domain")


def _grammar_text_values(grammar: BranchGrammar) -> Iterator[str]:
    """Yield every caller-controlled text cell used by compilation output."""

    for variable in grammar.variables:
        yield variable.variable_id
        if variable.shared_random_source_id is not None:
            yield variable.shared_random_source_id
        if variable.domain_values is not None:
            yield from variable.domain_values
    yield from grammar.demand_site_ids
    yield from grammar.reference_epoch_slot_ids
    yield from grammar.terminal_node_ids
    for rule in grammar.rules:
        yield rule.rule_id
        for clause in rule.clauses:
            yield clause.clause_id
            for term in clause.terms:
                yield term.variable_id
                yield from term.allowed_values
        if rule.outcome is not None:
            yield from rule.outcome.terminal_node_ids
            yield from rule.outcome.active_demand_site_ids
            for mapping in rule.outcome.site_epoch_mappings:
                yield mapping.demand_site_id
                yield mapping.epoch_slot_id


def _derive_compilation(
    grammar: BranchGrammar,
    safety_ceiling: GrammarSafetyCeiling,
) -> _CompilationValues | _CompilationFailure:
    grammar_raw = canonical_json(grammar)
    grammar_digest = sha256(grammar_raw).hexdigest()
    clause_count = sum(len(rule.clauses) for rule in grammar.rules)
    term_count = sum(
        len(clause.terms) for rule in grammar.rules for clause in rule.clauses
    )
    term_value_count = sum(
        len(term.allowed_values)
        for rule in grammar.rules
        for clause in rule.clauses
        for term in clause.terms
    )
    text_sizes = tuple(
        len(canonical_json(value)) for value in _grammar_text_values(grammar)
    )
    max_text_size = max(text_sizes, default=0)
    total_text_size = sum(text_sizes)
    ceiling_details: list[str] = []
    if len(grammar_raw) > safety_ceiling.max_artifact_bytes:
        ceiling_details.append(f"artifact_bytes:{len(grammar_raw)}")
    if max_text_size > safety_ceiling.max_identifier_bytes:
        ceiling_details.append(f"identifier_bytes:{max_text_size}")
    if total_text_size > safety_ceiling.max_total_text_bytes:
        ceiling_details.append(f"total_text_bytes:{total_text_size}")
    if len(grammar.variables) > safety_ceiling.max_variables:
        ceiling_details.append(f"variables:{len(grammar.variables)}")
    oversized_domains = tuple(
        variable.variable_id
        for variable in grammar.variables
        if variable.domain_values is not None
        and len(variable.domain_values) > safety_ceiling.max_domain_values_per_variable
    )
    ceiling_details.extend(f"domain:{item}" for item in oversized_domains)
    if len(grammar.rules) > safety_ceiling.max_rules:
        ceiling_details.append(f"rules:{len(grammar.rules)}")
    if clause_count > safety_ceiling.max_clauses:
        ceiling_details.append(f"clauses:{clause_count}")
    if term_count > safety_ceiling.max_terms:
        ceiling_details.append(f"terms:{term_count}")
    if term_value_count > safety_ceiling.max_term_values:
        ceiling_details.append(f"term_values:{term_value_count}")
    if ceiling_details:
        return _CompilationFailure(
            StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED,
            tuple(sorted(ceiling_details)),
        )
    unbounded = tuple(
        variable.variable_id
        for variable in grammar.variables
        if not variable.domain_values
    )
    if unbounded:
        return _CompilationFailure(
            StructuralAbstentionReason.UNBOUNDED_DOMAIN,
            unbounded,
        )
    variable_by_id = {variable.variable_id: variable for variable in grammar.variables}
    referenced = {
        term.variable_id
        for rule in grammar.rules
        for clause in rule.clauses
        for term in clause.terms
    }
    unknown_variables = tuple(sorted(referenced.difference(variable_by_id)))
    if unknown_variables:
        return _CompilationFailure(
            StructuralAbstentionReason.UNKNOWN_VARIABLE,
            unknown_variables,
        )
    assignment_count = 1
    for variable in grammar.variables:
        assert variable.domain_values is not None
        assignment_count *= len(variable.domain_values)
        if assignment_count > safety_ceiling.max_assignments:
            return _CompilationFailure(
                StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED,
                (f"assignments:{assignment_count}",),
            )
    match_work = assignment_count * (clause_count + term_count)
    if match_work > safety_ceiling.max_match_work:
        return _CompilationFailure(
            StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED,
            (f"match_work:{match_work}",),
        )
    assignment_value_cells = assignment_count * len(grammar.variables)
    if assignment_value_cells > safety_ceiling.max_assignment_value_cells:
        return _CompilationFailure(
            StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED,
            (f"assignment_value_cells:{assignment_value_cells}",),
        )
    value_row_upper_bound = sum(
        len(canonical_json(variable.variable_id))
        + max(len(canonical_json(value)) for value in variable.domain_values)
        + 64
        for variable in grammar.variables
        if variable.domain_values is not None
    )
    rule_clause_upper_bound = max(
        (
            len(canonical_json(rule.rule_id)) + len(canonical_json(clause.clause_id))
            for rule in grammar.rules
            for clause in rule.clauses
        ),
        default=0,
    )
    derived_catalog_upper_bound = 2 * len(grammar_raw) + assignment_count * (
        1_024 + value_row_upper_bound + rule_clause_upper_bound
    )
    if derived_catalog_upper_bound > safety_ceiling.max_derived_catalog_bytes:
        return _CompilationFailure(
            StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED,
            (f"derived_catalog_bytes:{derived_catalog_upper_bound}",),
        )
    for rule in grammar.rules:
        if rule.outcome is not None:
            _validate_outcome(grammar, rule.outcome)
        for clause in rule.clauses:
            for term in clause.terms:
                domain = variable_by_id[term.variable_id].domain_values
                assert domain is not None
                extra_values = set(term.allowed_values).difference(domain)
                if extra_values:
                    raise TraceValidationError(
                        "grammar term contains values outside its variable domain: "
                        f"{sorted(extra_values)}"
                    )

    support_outcome_by_rule: dict[str, SupportOutcome] = {}
    for rule in grammar.rules:
        if rule.outcome is None:
            continue
        outcome_id = canonical_digest(
            OutcomeIdentity(
                schema_version=OUTCOME_IDENTITY_SCHEMA_VERSION,
                branch_grammar_digest=grammar_digest,
                outcome=rule.outcome,
            )
        )
        support_outcome_by_rule[rule.rule_id] = SupportOutcome(
            outcome_id=outcome_id,
            outcome=rule.outcome,
        )

    domain_product = product(
        *(variable.domain_values for variable in grammar.variables)
    )
    assignments: list[SupportAssignment] = []
    outcomes: dict[str, SupportOutcome] = {}
    for raw_values in domain_product:
        values = tuple(
            AssignmentValue(variable_id=variable.variable_id, value=value)
            for variable, value in zip(grammar.variables, raw_values, strict=True)
        )
        value_by_id = {item.variable_id: item.value for item in values}
        matches: list[tuple[GrammarRule, GrammarClause]] = []
        for rule in grammar.rules:
            for clause in rule.clauses:
                if all(
                    value_by_id[term.variable_id] in term.allowed_values
                    for term in clause.terms
                ):
                    matches.append((rule, clause))
        if len(matches) != 1:
            rendered = ",".join(f"{item.variable_id}={item.value}" for item in values)
            if not matches:
                raise TraceValidationError(
                    f"grammar assignment has no matching clause: {rendered}"
                )
            pairs = sorted((rule.rule_id, clause.clause_id) for rule, clause in matches)
            raise TraceValidationError(
                f"grammar assignment matches multiple clauses {pairs}: {rendered}"
            )
        rule, clause = matches[0]
        assignment_id = canonical_digest(
            AssignmentIdentity(
                schema_version=ASSIGNMENT_IDENTITY_SCHEMA_VERSION,
                branch_grammar_digest=grammar_digest,
                values=values,
            )
        )
        outcome_id: str | None = None
        support_outcome = support_outcome_by_rule.get(rule.rule_id)
        if support_outcome is not None:
            outcome_id = support_outcome.outcome_id
            outcomes[outcome_id] = support_outcome
        assignments.append(
            SupportAssignment(
                assignment_id=assignment_id,
                values=values,
                rule_id=rule.rule_id,
                clause_id=clause.clause_id,
                disposition=rule.disposition,
                outcome_id=outcome_id,
                infeasibility_reason=rule.infeasibility_reason,
            )
        )
    assignment_tuple = tuple(assignments)
    outcome_tuple = tuple(outcomes[key] for key in sorted(outcomes))
    identity = SupportCatalogIdentity(
        schema_version=SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION,
        enumerator_implementation=GRAMMAR_ENUMERATOR_IMPLEMENTATION,
        safety_ceiling=safety_ceiling,
        branch_grammar_digest=grammar_digest,
        assignments=assignment_tuple,
        outcomes=outcome_tuple,
    )
    return _CompilationValues(
        support_catalog_id=canonical_digest(identity),
        assignments=assignment_tuple,
        outcomes=outcome_tuple,
    )


def compile_branch_grammar(
    grammar: BranchGrammar,
) -> FeasibleSupportCatalog | StructuralAbstention:
    """Exhaustively compile one grammar or return a typed structural abstention."""

    if type(grammar) is not BranchGrammar:
        raise TraceValidationError("branch grammar has the wrong type")
    safety_ceiling = _current_safety_ceiling()
    derived = _derive_compilation(grammar, safety_ceiling)
    if type(derived) is _CompilationFailure:
        return StructuralAbstention(
            schema_version=STRUCTURAL_ABSTENTION_SCHEMA_VERSION,
            enumerator_implementation=GRAMMAR_ENUMERATOR_IMPLEMENTATION,
            safety_ceiling=safety_ceiling,
            branch_grammar_digest=canonical_digest(grammar),
            branch_grammar=grammar,
            reason=derived.reason,
            detail_ids=derived.detail_ids,
        )
    return FeasibleSupportCatalog(
        schema_version=FEASIBLE_SUPPORT_CATALOG_SCHEMA_VERSION,
        enumerator_implementation=GRAMMAR_ENUMERATOR_IMPLEMENTATION,
        safety_ceiling=safety_ceiling,
        support_catalog_id=derived.support_catalog_id,
        branch_grammar_digest=canonical_digest(grammar),
        branch_grammar=grammar,
        assignments=derived.assignments,
        outcomes=derived.outcomes,
    )


@dataclass(frozen=True, slots=True)
class ActiveDemandSiteBinding:
    demand_site_id: str
    scheduled_demand_event_id: str

    def __post_init__(self) -> None:
        require_text("binding demand-site ID", self.demand_site_id)
        require_text(
            "binding scheduled-demand-event ID", self.scheduled_demand_event_id
        )


@dataclass(frozen=True, slots=True)
class ActiveEpochBinding:
    epoch_slot_id: str
    reference_epoch_id: str

    def __post_init__(self) -> None:
        require_text("binding epoch-slot ID", self.epoch_slot_id)
        require_text("binding reference-epoch ID", self.reference_epoch_id)


@dataclass(frozen=True, slots=True)
class GrammarCaseProjection:
    """Closed raw-source projection; bundle replay must reproduce it byte-exactly."""

    schema_version: str
    schedule_case_id: str
    source_artifact_digest: str
    source_case_digest: str
    source_schema_digest: str
    projection_rule_digest: str
    workflow_template_digest: str
    assignment_values: tuple[AssignmentValue, ...]
    active_site_bindings: tuple[ActiveDemandSiteBinding, ...]
    active_epoch_bindings: tuple[ActiveEpochBinding, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GRAMMAR_CASE_PROJECTION_SCHEMA_VERSION:
            raise TraceValidationError("unsupported grammar-case-projection schema")
        require_text("projection schedule-case ID", self.schedule_case_id)
        for name in (
            "source_artifact_digest",
            "source_case_digest",
            "source_schema_digest",
            "projection_rule_digest",
            "workflow_template_digest",
        ):
            require_sha256(f"projection {name}", getattr(self, name))
        _require_tuple("projection assignment values", self.assignment_values)
        if any(type(item) is not AssignmentValue for item in self.assignment_values):
            raise TraceValidationError(
                "projection contains an invalid assignment value"
            )
        assignment_ids = tuple(item.variable_id for item in self.assignment_values)
        if assignment_ids != tuple(sorted(assignment_ids)) or len(
            assignment_ids
        ) != len(set(assignment_ids)):
            raise TraceValidationError(
                "projection assignment values must be sorted and unique"
            )
        _require_tuple("projection active site bindings", self.active_site_bindings)
        if any(
            type(item) is not ActiveDemandSiteBinding
            for item in self.active_site_bindings
        ):
            raise TraceValidationError("projection contains an invalid active site")
        site_ids = tuple(item.demand_site_id for item in self.active_site_bindings)
        event_ids = tuple(
            item.scheduled_demand_event_id for item in self.active_site_bindings
        )
        if site_ids != tuple(sorted(site_ids)) or len(site_ids) != len(set(site_ids)):
            raise TraceValidationError(
                "projection active site bindings must be sorted and unique"
            )
        if len(event_ids) != len(set(event_ids)):
            raise TraceValidationError(
                "projection active sites must map bijectively to events"
            )
        _require_tuple("projection active epoch bindings", self.active_epoch_bindings)
        if any(
            type(item) is not ActiveEpochBinding for item in self.active_epoch_bindings
        ):
            raise TraceValidationError("projection contains an invalid active epoch")
        slot_ids = tuple(item.epoch_slot_id for item in self.active_epoch_bindings)
        epoch_ids = tuple(
            item.reference_epoch_id for item in self.active_epoch_bindings
        )
        if slot_ids != tuple(sorted(slot_ids)) or len(slot_ids) != len(set(slot_ids)):
            raise TraceValidationError(
                "projection active epoch bindings must be sorted and unique"
            )
        if len(epoch_ids) != len(set(epoch_ids)):
            raise TraceValidationError(
                "projection active epoch slots must map bijectively to epochs"
            )


@dataclass(frozen=True, slots=True)
class GrammarInstanceBinding:
    """Exact assignment/outcome realization for one frozen schedule case."""

    schema_version: str
    schedule_case_id: str
    schedule_digest: str
    source_artifact_digest: str
    source_case_digest: str
    source_schema_digest: str
    projection_rule_digest: str
    workflow_template_digest: str
    method_menu_digest: str
    closed_schedule: ClosedScheduleArtifact
    case_projection: GrammarCaseProjection
    branch_grammar_digest: str
    support_catalog_digest: str
    support_catalog: FeasibleSupportCatalog
    assignment_id: str
    assignment_values: tuple[AssignmentValue, ...]
    outcome_id: str
    active_site_bindings: tuple[ActiveDemandSiteBinding, ...]
    active_epoch_bindings: tuple[ActiveEpochBinding, ...]
    inactive_demand_site_ids: tuple[str, ...]
    inactive_epoch_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION:
            raise TraceValidationError("unsupported grammar-instance-binding schema")
        require_text("binding schedule-case ID", self.schedule_case_id)
        require_sha256("binding schedule digest", self.schedule_digest)
        require_sha256(
            "binding source-artifact digest",
            self.source_artifact_digest,
        )
        require_sha256("binding source-case digest", self.source_case_digest)
        require_sha256("binding source-schema digest", self.source_schema_digest)
        require_sha256("binding projection-rule digest", self.projection_rule_digest)
        require_sha256(
            "binding workflow-template digest",
            self.workflow_template_digest,
        )
        require_sha256("binding method-menu digest", self.method_menu_digest)
        if type(self.closed_schedule) is not ClosedScheduleArtifact:
            raise TraceValidationError("binding has the wrong closed-schedule type")
        if type(self.case_projection) is not GrammarCaseProjection:
            raise TraceValidationError("binding has the wrong case-projection type")
        projection = self.case_projection
        if (
            self.schedule_case_id != projection.schedule_case_id
            or self.source_artifact_digest != projection.source_artifact_digest
            or self.source_case_digest != projection.source_case_digest
            or self.source_schema_digest != projection.source_schema_digest
            or self.projection_rule_digest != projection.projection_rule_digest
            or self.workflow_template_digest != projection.workflow_template_digest
            or self.assignment_values != projection.assignment_values
            or self.active_site_bindings != projection.active_site_bindings
            or self.active_epoch_bindings != projection.active_epoch_bindings
        ):
            raise TraceValidationError("binding differs from its case projection")
        if self.closed_schedule.schedule_case_id != self.schedule_case_id:
            raise TraceValidationError("binding schedule-case identity differs")
        if canonical_digest(self.closed_schedule) != self.schedule_digest:
            raise TraceValidationError("binding closed-schedule digest differs")
        if self.closed_schedule.source_case_digest != self.source_case_digest:
            raise TraceValidationError("binding source-case digest differs")
        if self.closed_schedule.source_artifact_digest != self.source_artifact_digest:
            raise TraceValidationError("binding source-artifact digest differs")
        if self.closed_schedule.source_schema_digest != self.source_schema_digest:
            raise TraceValidationError("binding source-schema digest differs")
        require_sha256("binding branch-grammar digest", self.branch_grammar_digest)
        require_sha256("binding support-catalog digest", self.support_catalog_digest)
        require_sha256("binding assignment ID", self.assignment_id)
        require_sha256("binding outcome ID", self.outcome_id)
        if type(self.support_catalog) is not FeasibleSupportCatalog:
            raise TraceValidationError("binding has the wrong support-catalog type")
        grammar = self.support_catalog.branch_grammar
        if grammar.workflow_template_digest != self.workflow_template_digest:
            raise TraceValidationError("binding workflow-template digest differs")
        if self.branch_grammar_digest != canonical_digest(grammar):
            raise TraceValidationError("binding branch-grammar digest differs")
        if self.support_catalog_digest != canonical_digest(self.support_catalog):
            raise TraceValidationError("binding support-catalog digest differs")
        matching_assignments = tuple(
            item
            for item in self.support_catalog.assignments
            if item.assignment_id == self.assignment_id
        )
        if len(matching_assignments) != 1:
            raise TraceValidationError("binding assignment is absent or duplicated")
        assignment = matching_assignments[0]
        _require_tuple("binding assignment values", self.assignment_values)
        if self.assignment_values != assignment.values:
            raise TraceValidationError("binding assignment values differ")
        if assignment.disposition != RuleDisposition.FEASIBLE:
            raise TraceValidationError("binding assignment is infeasible")
        if assignment.outcome_id != self.outcome_id:
            raise TraceValidationError("binding outcome differs from its assignment")
        matching_outcomes = tuple(
            item
            for item in self.support_catalog.outcomes
            if item.outcome_id == self.outcome_id
        )
        if len(matching_outcomes) != 1:
            raise TraceValidationError("binding outcome is absent or duplicated")
        outcome = matching_outcomes[0].outcome

        _require_tuple("active demand-site bindings", self.active_site_bindings)
        if any(
            type(item) is not ActiveDemandSiteBinding
            for item in self.active_site_bindings
        ):
            raise TraceValidationError("binding contains an invalid active demand site")
        active_sites = tuple(item.demand_site_id for item in self.active_site_bindings)
        if active_sites != tuple(sorted(active_sites)) or len(active_sites) != len(
            set(active_sites)
        ):
            raise TraceValidationError(
                "active demand-site bindings must be sorted and unique"
            )
        concrete_demands = tuple(
            item.scheduled_demand_event_id for item in self.active_site_bindings
        )
        if len(concrete_demands) != len(set(concrete_demands)):
            raise TraceValidationError(
                "active demand sites must map bijectively to scheduled events"
            )
        if active_sites != outcome.active_demand_site_ids:
            raise TraceValidationError("binding active demand-site domain differs")

        _require_tuple("active epoch bindings", self.active_epoch_bindings)
        if any(
            type(item) is not ActiveEpochBinding for item in self.active_epoch_bindings
        ):
            raise TraceValidationError("binding contains an invalid active epoch")
        active_epochs = tuple(item.epoch_slot_id for item in self.active_epoch_bindings)
        if active_epochs != tuple(sorted(active_epochs)) or len(active_epochs) != len(
            set(active_epochs)
        ):
            raise TraceValidationError(
                "active epoch bindings must be sorted and unique"
            )
        concrete_epochs = tuple(
            item.reference_epoch_id for item in self.active_epoch_bindings
        )
        if len(concrete_epochs) != len(set(concrete_epochs)):
            raise TraceValidationError(
                "active epoch slots must map bijectively to reference epochs"
            )
        if active_epochs != outcome.active_epoch_slot_ids:
            raise TraceValidationError("binding active epoch-slot domain differs")

        event_by_id = {
            event.schedule_event_id: event for event in self.closed_schedule.events
        }
        if set(concrete_demands) != set(event_by_id):
            raise TraceValidationError(
                "binding scheduled-demand events do not conserve the closed schedule"
            )
        epoch_by_id = {
            epoch.reuse_epoch_id: epoch for epoch in self.closed_schedule.epochs
        }
        if set(concrete_epochs) != set(epoch_by_id):
            raise TraceValidationError(
                "binding reference epochs do not conserve the closed schedule"
            )
        epoch_id_by_slot = {
            item.epoch_slot_id: item.reference_epoch_id
            for item in self.active_epoch_bindings
        }
        slot_by_site = {
            item.demand_site_id: item.epoch_slot_id
            for item in outcome.site_epoch_mappings
        }
        event_ids_by_epoch: dict[str, set[str]] = {
            epoch_id: set() for epoch_id in epoch_by_id
        }
        for site_binding in self.active_site_bindings:
            event = event_by_id[site_binding.scheduled_demand_event_id]
            expected_epoch_id = epoch_id_by_slot[
                slot_by_site[site_binding.demand_site_id]
            ]
            if event.reuse_epoch_id != expected_epoch_id:
                raise TraceValidationError(
                    "binding event reuse epoch contradicts its template mapping"
                )
            event_ids_by_epoch[expected_epoch_id].add(event.schedule_event_id)
        for epoch_id, epoch in epoch_by_id.items():
            if event_ids_by_epoch[epoch_id] != set(epoch.schedule_event_ids):
                raise TraceValidationError(
                    "binding epoch member set differs from the closed schedule"
                )

        _require_sorted_unique_text(
            "inactive demand-site IDs",
            self.inactive_demand_site_ids,
        )
        _require_sorted_unique_text(
            "inactive epoch-slot IDs",
            self.inactive_epoch_slot_ids,
        )
        if set(active_sites).intersection(self.inactive_demand_site_ids):
            raise TraceValidationError("a demand site is both active and inactive")
        if tuple(sorted((*active_sites, *self.inactive_demand_site_ids))) != (
            grammar.demand_site_ids
        ):
            raise TraceValidationError("binding does not conserve every demand site")
        if set(active_epochs).intersection(self.inactive_epoch_slot_ids):
            raise TraceValidationError("an epoch slot is both active and inactive")
        if tuple(sorted((*active_epochs, *self.inactive_epoch_slot_ids))) != (
            grammar.reference_epoch_slot_ids
        ):
            raise TraceValidationError("binding does not conserve every epoch slot")
        if outcome.kind == OutcomeKind.NO_USE and (
            self.active_site_bindings or self.active_epoch_bindings
        ):
            raise TraceValidationError("NO_USE binding must have empty active maps")
        if outcome.kind == OutcomeKind.NO_USE and (
            self.closed_schedule.events or self.closed_schedule.epochs
        ):
            raise TraceValidationError(
                "NO_USE binding requires an empty closed schedule"
            )


@dataclass(frozen=True, slots=True)
class GrammarBindingInputDigests:
    """Canonical aggregate identities consumed by one cutoff plan."""

    schema_version: str
    schedule_digest: str
    source_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != GRAMMAR_BINDING_INPUT_DIGESTS_SCHEMA_VERSION:
            raise TraceValidationError(
                "unsupported grammar-binding-input-digests schema"
            )
        require_sha256("binding-input schedule digest", self.schedule_digest)
        require_sha256("binding-input source digest", self.source_digest)


@dataclass(frozen=True, slots=True)
class _ScheduleCaseCatalogEntry:
    schedule_case_id: str
    schedule_digest: str

    def __post_init__(self) -> None:
        require_text("schedule-case catalog ID", self.schedule_case_id)
        require_sha256("schedule-case catalog digest", self.schedule_digest)


@dataclass(frozen=True, slots=True)
class _SourceCaseCatalogEntry:
    schedule_case_id: str
    source_artifact_digest: str
    source_schema_digest: str
    source_case_digest: str

    def __post_init__(self) -> None:
        require_text("source-case catalog schedule-case ID", self.schedule_case_id)
        require_sha256(
            "source-case catalog artifact digest", self.source_artifact_digest
        )
        require_sha256("source-case catalog schema digest", self.source_schema_digest)
        require_sha256("source-case catalog case digest", self.source_case_digest)


@dataclass(frozen=True, slots=True)
class _ScheduleCaseCatalogIdentity:
    schema_version: str
    entries: tuple[_ScheduleCaseCatalogEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEDULE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported schedule-case catalog identity")
        _require_tuple("schedule-case catalog entries", self.entries)
        if any(type(entry) is not _ScheduleCaseCatalogEntry for entry in self.entries):
            raise TraceValidationError("schedule-case catalog has an invalid entry")
        case_ids = tuple(entry.schedule_case_id for entry in self.entries)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise TraceValidationError(
                "schedule-case catalog entries must be sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class _SourceCaseCatalogIdentity:
    schema_version: str
    entries: tuple[_SourceCaseCatalogEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION:
            raise TraceValidationError("unsupported source-case catalog identity")
        _require_tuple("source-case catalog entries", self.entries)
        if any(type(entry) is not _SourceCaseCatalogEntry for entry in self.entries):
            raise TraceValidationError("source-case catalog has an invalid entry")
        case_ids = tuple(entry.schedule_case_id for entry in self.entries)
        if case_ids != tuple(sorted(case_ids)) or len(case_ids) != len(set(case_ids)):
            raise TraceValidationError(
                "source-case catalog entries must be sorted and unique"
            )


@dataclass(frozen=True, slots=True)
class GrammarBindingReference:
    """Compact identity of one separately stored schedule-case binding."""

    schedule_case_id: str
    binding_digest: str
    schedule_digest: str
    source_artifact_digest: str
    source_case_digest: str
    source_schema_digest: str
    projection_rule_digest: str
    case_projection_digest: str
    workflow_template_digest: str
    method_menu_digest: str
    branch_grammar_digest: str
    support_catalog_digest: str
    support_catalog_id: str
    assignment_id: str
    outcome_id: str

    def __post_init__(self) -> None:
        require_text("binding-reference schedule-case ID", self.schedule_case_id)
        for name in (
            "binding_digest",
            "schedule_digest",
            "source_artifact_digest",
            "source_case_digest",
            "source_schema_digest",
            "projection_rule_digest",
            "case_projection_digest",
            "workflow_template_digest",
            "method_menu_digest",
            "branch_grammar_digest",
            "support_catalog_digest",
            "support_catalog_id",
            "assignment_id",
            "outcome_id",
        ):
            require_sha256(f"binding-reference {name}", getattr(self, name))


@dataclass(frozen=True, slots=True)
class GrammarBindingConservation:
    """Plan-level proof that every distinct schedule case has one binding."""

    schema_version: str
    candidate_universe_digest: str
    candidate_universe: CandidateUniverse
    input_digests: GrammarBindingInputDigests
    schedule_case_ids: tuple[str, ...]
    references: tuple[GrammarBindingReference, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GRAMMAR_BINDING_CONSERVATION_SCHEMA_VERSION:
            raise TraceValidationError(
                "unsupported grammar-binding-conservation schema"
            )
        require_sha256(
            "binding-conservation candidate-universe digest",
            self.candidate_universe_digest,
        )
        if type(self.candidate_universe) is not CandidateUniverse:
            raise TraceValidationError(
                "binding conservation has the wrong candidate-universe type"
            )
        if self.candidate_universe_digest != canonical_digest(self.candidate_universe):
            raise TraceValidationError(
                "binding-conservation candidate-universe digest differs"
            )
        if type(self.input_digests) is not GrammarBindingInputDigests:
            raise TraceValidationError(
                "binding conservation has the wrong input-digests type"
            )
        _require_sorted_unique_text(
            "binding-conservation schedule-case IDs",
            self.schedule_case_ids,
        )
        expected_case_ids = tuple(
            sorted(
                {
                    slot.schedule_case_id
                    for slot in self.candidate_universe.cutoff_plan.slots
                }
            )
        )
        if self.schedule_case_ids != expected_case_ids:
            raise TraceValidationError(
                "binding conservation does not cover the cutoff-plan schedule cases"
            )
        _require_tuple("binding-conservation references", self.references)
        if any(
            type(reference) is not GrammarBindingReference
            for reference in self.references
        ):
            raise TraceValidationError(
                "binding conservation contains an invalid reference"
            )
        reference_case_ids = tuple(
            reference.schedule_case_id for reference in self.references
        )
        if reference_case_ids != self.schedule_case_ids:
            raise TraceValidationError(
                "binding references must cover schedule cases in canonical order"
            )
        observed_input_digests = _binding_input_digests_from_references(self.references)
        if self.input_digests != observed_input_digests:
            raise TraceValidationError(
                "binding-conservation input digests differ from its references"
            )
        plan = self.candidate_universe.cutoff_plan
        if self.input_digests.schedule_digest != plan.schedule_digest:
            raise TraceValidationError(
                "binding schedule digest differs from the cutoff plan"
            )
        if self.input_digests.source_digest != plan.source_digest:
            raise TraceValidationError(
                "binding source digest differs from the cutoff plan"
            )
        template_digest = plan.workflow_template_digest
        if any(
            reference.workflow_template_digest != template_digest
            for reference in self.references
        ):
            raise TraceValidationError(
                "binding reference uses another workflow template"
            )
        if any(
            reference.method_menu_digest != plan.method_menu_digest
            for reference in self.references
        ):
            raise TraceValidationError(
                "binding method menu differs from the cutoff plan"
            )
        grammar_identities = {
            (
                reference.branch_grammar_digest,
                reference.support_catalog_digest,
                reference.support_catalog_id,
            )
            for reference in self.references
        }
        if self.references and len(grammar_identities) != 1:
            raise TraceValidationError(
                "one workflow template has multiple grammar/support identities"
            )
        projection_rules: dict[tuple[str, str], str] = {}
        for reference in self.references:
            key = (
                reference.workflow_template_digest,
                reference.source_schema_digest,
            )
            previous = projection_rules.setdefault(
                key,
                reference.projection_rule_digest,
            )
            if previous != reference.projection_rule_digest:
                raise TraceValidationError(
                    "one template/source schema has multiple projection rules"
                )


def _binding_reference(binding: GrammarInstanceBinding) -> GrammarBindingReference:
    return GrammarBindingReference(
        schedule_case_id=binding.schedule_case_id,
        binding_digest=canonical_digest(binding),
        schedule_digest=binding.schedule_digest,
        source_artifact_digest=binding.source_artifact_digest,
        source_case_digest=binding.source_case_digest,
        source_schema_digest=binding.source_schema_digest,
        projection_rule_digest=binding.projection_rule_digest,
        case_projection_digest=canonical_digest(binding.case_projection),
        workflow_template_digest=binding.workflow_template_digest,
        method_menu_digest=binding.method_menu_digest,
        branch_grammar_digest=binding.branch_grammar_digest,
        support_catalog_digest=binding.support_catalog_digest,
        support_catalog_id=binding.support_catalog.support_catalog_id,
        assignment_id=binding.assignment_id,
        outcome_id=binding.outcome_id,
    )


def _binding_input_digests_from_references(
    references: tuple[GrammarBindingReference, ...],
) -> GrammarBindingInputDigests:
    schedule_identity = _ScheduleCaseCatalogIdentity(
        schema_version=SCHEDULE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION,
        entries=tuple(
            _ScheduleCaseCatalogEntry(
                schedule_case_id=reference.schedule_case_id,
                schedule_digest=reference.schedule_digest,
            )
            for reference in references
        ),
    )
    source_identity = _SourceCaseCatalogIdentity(
        schema_version=SOURCE_CASE_CATALOG_IDENTITY_SCHEMA_VERSION,
        entries=tuple(
            _SourceCaseCatalogEntry(
                schedule_case_id=reference.schedule_case_id,
                source_artifact_digest=reference.source_artifact_digest,
                source_schema_digest=reference.source_schema_digest,
                source_case_digest=reference.source_case_digest,
            )
            for reference in references
        ),
    )
    return GrammarBindingInputDigests(
        schema_version=GRAMMAR_BINDING_INPUT_DIGESTS_SCHEMA_VERSION,
        schedule_digest=canonical_digest(schedule_identity),
        source_digest=canonical_digest(source_identity),
    )


def _replay_grammar_instance_binding(
    binding: GrammarInstanceBinding,
) -> GrammarInstanceBinding:
    if type(binding) is not GrammarInstanceBinding:
        raise TraceValidationError("binding conservation has an invalid binding")
    raw = canonical_json(binding)
    if len(raw) > MAX_GRAMMAR_ARTIFACT_BYTES:
        raise TraceValidationError(
            "grammar-instance-binding dependency exceeds the size limit"
        )
    replayed = parse_canonical_dataclass(
        raw,
        GrammarInstanceBinding,
        artifact_name="grammar-instance-binding dependency",
        max_bytes=MAX_GRAMMAR_ARTIFACT_BYTES,
    )
    if replayed != binding:
        raise TraceValidationError(
            "grammar-instance-binding dependency changes during canonical replay"
        )
    return replayed


def derive_grammar_binding_input_digests(
    bindings: tuple[GrammarInstanceBinding, ...],
) -> GrammarBindingInputDigests:
    """Derive the schedule/source catalog identities consumed by a cutoff plan."""

    _require_tuple("grammar instance bindings", bindings)
    binding_by_case: dict[str, GrammarInstanceBinding] = {}
    for candidate in bindings:
        binding = _replay_grammar_instance_binding(candidate)
        if binding.schedule_case_id in binding_by_case:
            raise TraceValidationError(
                "binding input digest inventory contains a duplicate schedule case"
            )
        binding_by_case[binding.schedule_case_id] = binding
    references = tuple(
        _binding_reference(binding_by_case[case_id])
        for case_id in sorted(binding_by_case)
    )
    return _binding_input_digests_from_references(references)


def build_grammar_binding_conservation(
    candidate_universe: CandidateUniverse,
    bindings: tuple[GrammarInstanceBinding, ...],
) -> GrammarBindingConservation:
    """Build exact schedule-case coverage without reading service outcomes."""

    if type(candidate_universe) is not CandidateUniverse:
        raise TraceValidationError(
            "binding-conservation candidate universe has the wrong type"
        )
    _require_tuple("grammar instance bindings", bindings)
    binding_by_case: dict[str, GrammarInstanceBinding] = {}
    for candidate in bindings:
        binding = _replay_grammar_instance_binding(candidate)
        if binding.schedule_case_id in binding_by_case:
            raise TraceValidationError(
                "binding conservation contains a duplicate schedule case"
            )
        binding_by_case[binding.schedule_case_id] = binding
    expected_case_ids = tuple(
        sorted({slot.schedule_case_id for slot in candidate_universe.cutoff_plan.slots})
    )
    observed_case_ids = tuple(sorted(binding_by_case))
    if observed_case_ids != expected_case_ids:
        missing = sorted(set(expected_case_ids).difference(observed_case_ids))
        extra = sorted(set(observed_case_ids).difference(expected_case_ids))
        raise TraceValidationError(
            "binding schedule-case set differs from the cutoff plan: "
            f"missing={missing}, extra={extra}"
        )
    references = tuple(
        _binding_reference(binding_by_case[case_id]) for case_id in expected_case_ids
    )
    return GrammarBindingConservation(
        schema_version=GRAMMAR_BINDING_CONSERVATION_SCHEMA_VERSION,
        candidate_universe_digest=canonical_digest(candidate_universe),
        candidate_universe=candidate_universe,
        input_digests=_binding_input_digests_from_references(references),
        schedule_case_ids=expected_case_ids,
        references=references,
    )


def validate_grammar_binding_conservation(
    artifact: GrammarBindingConservation,
    bindings: tuple[GrammarInstanceBinding, ...],
) -> None:
    """Replay one compact conservation artifact against its binding inventory."""

    if type(artifact) is not GrammarBindingConservation:
        raise TraceValidationError("binding-conservation artifact has the wrong type")
    expected = build_grammar_binding_conservation(
        artifact.candidate_universe,
        bindings,
    )
    if artifact != expected:
        raise TraceValidationError(
            "binding-conservation artifact differs from binding replay"
        )


@dataclass(frozen=True, slots=True)
class LoadedGrammarArtifact[T]:
    artifact: T
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256("loaded grammar artifact digest", self.digest)
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise TraceValidationError("loaded grammar artifact size is invalid")


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


def _validate_parent_binding(path: Path, descriptor: int, artifact_name: str) -> None:
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
        raise OSError(f"{artifact_name} parent path changed during publication")


def _validate_output_binding(
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
        raise OSError("grammar artifact output identity changed")
    return opened


def _read_exact(descriptor: int, expected_size: int) -> bytes:
    observed = bytearray()
    offset = 0
    while offset < expected_size:
        chunk = os.pread(descriptor, expected_size - offset, offset)
        if not chunk:
            raise OSError("grammar artifact ended during stable readback")
        observed.extend(chunk)
        offset += len(chunk)
    return bytes(observed)


def _write_artifact[T](
    path: Path,
    artifact: T,
    expected: type[T],
    artifact_name: str,
) -> str:
    if type(artifact) is not expected:
        raise TraceValidationError(f"{artifact_name} has the wrong type")
    raw = canonical_json(artifact)
    if len(raw) > MAX_GRAMMAR_ARTIFACT_BYTES:
        raise TraceValidationError(f"{artifact_name} exceeds the size limit")
    if (
        parse_canonical_dataclass(
            raw,
            expected,
            artifact_name=artifact_name,
            max_bytes=MAX_GRAMMAR_ARTIFACT_BYTES,
        )
        != artifact
    ):
        raise TraceValidationError(f"{artifact_name} changes during canonical replay")
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
                        raise OSError("grammar artifact write made no progress")
                    offset += count
                os.fsync(descriptor)
                os.fsync(parent_descriptor)
                first_state = _validate_output_binding(
                    descriptor, parent_descriptor, path, len(raw)
                )
                first = _read_exact(descriptor, len(raw))
                second_state = _validate_output_binding(
                    descriptor, parent_descriptor, path, len(raw)
                )
                second = _read_exact(descriptor, len(raw))
                published = _validate_output_binding(
                    descriptor, parent_descriptor, path, len(raw)
                )
                if (
                    first != raw
                    or second != raw
                    or _output_state(first_state) != _output_state(second_state)
                    or _output_state(second_state) != _output_state(published)
                ):
                    raise OSError("grammar artifact changed during stable readback")
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
                    raise OSError("grammar artifact changed after output close")
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
            if created and commit_failure is None:
                commit_failure = exc
            elif not created and precreate_error is None:
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
    failure: BaseException | None = None
    raw = b""
    linked: os.stat_result | None = None
    try:
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or _output_state(before) != _output_state(opened)
                or opened.st_size < 1
                or opened.st_size > MAX_GRAMMAR_ARTIFACT_BYTES
            ):
                raise TraceValidationError(f"{artifact_name} changed while opening")
            first = _read_exact(descriptor, opened.st_size)
            middle = os.fstat(descriptor)
            second = _read_exact(descriptor, opened.st_size)
            after = os.fstat(descriptor)
            linked = path.lstat()
            if (
                first != second
                or _output_state(opened) != _output_state(middle)
                or _output_state(middle) != _output_state(after)
                or _output_state(after) != _output_state(linked)
            ):
                raise TraceValidationError(
                    f"{artifact_name} changed during stable readback"
                )
            third = _read_exact(descriptor, opened.st_size)
            final_opened = os.fstat(descriptor)
            if (
                second != third
                or _output_state(after) != _output_state(final_opened)
                or _output_state(linked) != _output_state(final_opened)
            ):
                raise TraceValidationError(
                    f"{artifact_name} changed after path revalidation"
                )
            raw = third
        except BaseException as exc:
            failure = exc
    finally:
        try:
            os.close(descriptor)
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        if isinstance(failure, TraceValidationError):
            raise failure
        if isinstance(failure, Exception):
            raise TraceValidationError(
                f"cannot read {artifact_name} safely"
            ) from failure
        raise failure
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
) -> LoadedGrammarArtifact[T]:
    raw = _read_stable_file(path, artifact_name)
    artifact = parse_canonical_dataclass(
        raw,
        expected,
        artifact_name=artifact_name,
        max_bytes=MAX_GRAMMAR_ARTIFACT_BYTES,
    )
    return LoadedGrammarArtifact(
        artifact=artifact,
        digest=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def write_branch_grammar(path: Path, artifact: BranchGrammar) -> str:
    return _write_artifact(path, artifact, BranchGrammar, "branch-grammar artifact")


def load_branch_grammar(path: Path) -> LoadedGrammarArtifact[BranchGrammar]:
    return _load_artifact(path, BranchGrammar, "branch-grammar artifact")


def write_feasible_support_catalog(
    path: Path,
    artifact: FeasibleSupportCatalog,
) -> str:
    return _write_artifact(
        path,
        artifact,
        FeasibleSupportCatalog,
        "feasible-support-catalog artifact",
    )


def load_feasible_support_catalog(
    path: Path,
) -> LoadedGrammarArtifact[FeasibleSupportCatalog]:
    return _load_artifact(
        path,
        FeasibleSupportCatalog,
        "feasible-support-catalog artifact",
    )


def write_structural_abstention(path: Path, artifact: StructuralAbstention) -> str:
    return _write_artifact(
        path,
        artifact,
        StructuralAbstention,
        "grammar-structural-abstention artifact",
    )


def load_structural_abstention(
    path: Path,
) -> LoadedGrammarArtifact[StructuralAbstention]:
    return _load_artifact(
        path,
        StructuralAbstention,
        "grammar-structural-abstention artifact",
    )


def write_grammar_instance_binding(
    path: Path,
    artifact: GrammarInstanceBinding,
) -> str:
    return _write_artifact(
        path,
        artifact,
        GrammarInstanceBinding,
        "grammar-instance-binding artifact",
    )


def load_grammar_instance_binding(
    path: Path,
) -> LoadedGrammarArtifact[GrammarInstanceBinding]:
    return _load_artifact(
        path,
        GrammarInstanceBinding,
        "grammar-instance-binding artifact",
    )


def write_grammar_binding_conservation(
    path: Path,
    artifact: GrammarBindingConservation,
    bindings: tuple[GrammarInstanceBinding, ...],
) -> str:
    """Create one compact audit after replaying all referenced bindings."""

    validate_grammar_binding_conservation(artifact, bindings)
    return _write_artifact(
        path,
        artifact,
        GrammarBindingConservation,
        "grammar-binding-conservation artifact",
    )


def load_grammar_binding_conservation(
    path: Path,
    bindings: tuple[GrammarInstanceBinding, ...],
) -> LoadedGrammarArtifact[GrammarBindingConservation]:
    """Load one compact audit and replay its separate binding inventory."""

    loaded = _load_artifact(
        path,
        GrammarBindingConservation,
        "grammar-binding-conservation artifact",
    )
    validate_grammar_binding_conservation(loaded.artifact, bindings)
    return loaded
