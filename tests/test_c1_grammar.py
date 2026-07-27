"""Focused tests for finite C1-B1 branch grammar support."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv
import dagkv.c1_grammar as grammar_module
from dagkv.c1_grammar import (
    ASSIGNMENT_IDENTITY_SCHEMA_VERSION,
    BRANCH_GRAMMAR_SCHEMA_VERSION,
    FEASIBLE_SUPPORT_CATALOG_SCHEMA_VERSION,
    GRAMMAR_BINDING_CONSERVATION_SCHEMA_VERSION,
    GRAMMAR_BINDING_INPUT_DIGESTS_SCHEMA_VERSION,
    GRAMMAR_CANONICAL_ORDER,
    GRAMMAR_CASE_PROJECTION_SCHEMA_VERSION,
    GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION,
    OUTCOME_IDENTITY_SCHEMA_VERSION,
    STRUCTURAL_ABSTENTION_SCHEMA_VERSION,
    SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION,
    ActiveDemandSiteBinding,
    ActiveEpochBinding,
    AssignmentIdentity,
    BranchGrammar,
    BranchVariable,
    FeasibleOutcome,
    FeasibleSupportCatalog,
    GrammarBindingConservation,
    GrammarBindingInputDigests,
    GrammarCaseProjection,
    GrammarClause,
    GrammarInstanceBinding,
    GrammarRule,
    GrammarTerm,
    InfeasibilityReason,
    OutcomeIdentity,
    OutcomeKind,
    RuleDisposition,
    SiteEpochMapping,
    StructuralAbstention,
    StructuralAbstentionReason,
    build_grammar_binding_conservation,
    compile_branch_grammar,
    derive_grammar_binding_input_digests,
    load_branch_grammar,
    load_feasible_support_catalog,
    load_grammar_binding_conservation,
    load_grammar_instance_binding,
    load_structural_abstention,
    validate_grammar_binding_conservation,
    write_branch_grammar,
    write_feasible_support_catalog,
    write_grammar_binding_conservation,
    write_grammar_instance_binding,
    write_structural_abstention,
)
from dagkv.c1_schedule import ScheduleDemandEvent, ScheduleEpoch
from dagkv.c1_split import UniversePurpose, build_candidate_universe
from dagkv.c1_trace import (
    ScheduleProducerKind,
    TraceCommitIndeterminateError,
    TraceValidationError,
    canonical_digest,
    canonical_json,
)
from dagkv.domain import BlockKey, ExecutionRef, WorkflowKey
from tests.test_c1_schedule import _artifact as _schedule_artifact
from tests.test_c1_split import _plan as _cutoff_plan
from tests.test_c1_split import _slot as _cutoff_slot


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _term(variable: str, *values: str) -> GrammarTerm:
    return GrammarTerm(variable_id=variable, allowed_values=tuple(sorted(values)))


def _clause(clause_id: str, *terms: GrammarTerm) -> GrammarClause:
    return GrammarClause(
        clause_id=clause_id,
        terms=tuple(sorted(terms, key=lambda item: item.variable_id)),
    )


def _use_outcome() -> FeasibleOutcome:
    return FeasibleOutcome(
        kind=OutcomeKind.USE,
        terminal_node_ids=("terminal-success",),
        active_demand_site_ids=("site-a", "site-b"),
        site_epoch_mappings=(
            SiteEpochMapping("site-a", "epoch-shared"),
            SiteEpochMapping("site-b", "epoch-shared"),
        ),
    )


def _no_use_outcome() -> FeasibleOutcome:
    return FeasibleOutcome(
        kind=OutcomeKind.NO_USE,
        terminal_node_ids=("terminal-skip",),
        active_demand_site_ids=(),
        site_epoch_mappings=(),
    )


def _grammar(
    *,
    workflow_template_digest: str | None = None,
    shared_random_source_id: str = "draw-branch",
) -> BranchGrammar:
    return BranchGrammar(
        schema_version=BRANCH_GRAMMAR_SCHEMA_VERSION,
        canonical_order=GRAMMAR_CANONICAL_ORDER,
        workflow_template_digest=(
            _digest("template")
            if workflow_template_digest is None
            else workflow_template_digest
        ),
        variables=(
            BranchVariable("branch", ("left", "right"), shared_random_source_id),
            BranchVariable("gate", ("closed", "open"), shared_random_source_id),
        ),
        demand_site_ids=("site-a", "site-b", "site-unused"),
        reference_epoch_slot_ids=("epoch-shared", "epoch-unused"),
        terminal_node_ids=(
            "terminal-blocked",
            "terminal-skip",
            "terminal-success",
        ),
        rules=(
            GrammarRule(
                rule_id="blocked",
                clauses=(
                    _clause(
                        "right-closed",
                        _term("branch", "right"),
                        _term("gate", "closed"),
                    ),
                ),
                disposition=RuleDisposition.INFEASIBLE,
                outcome=None,
                infeasibility_reason=InfeasibilityReason.PATH_CONSTRAINT,
            ),
            GrammarRule(
                rule_id="no-use",
                clauses=(
                    _clause(
                        "right-open", _term("branch", "right"), _term("gate", "open")
                    ),
                ),
                disposition=RuleDisposition.FEASIBLE,
                outcome=_no_use_outcome(),
                infeasibility_reason=None,
            ),
            GrammarRule(
                rule_id="use",
                clauses=(_clause("left", _term("branch", "left")),),
                disposition=RuleDisposition.FEASIBLE,
                outcome=_use_outcome(),
                infeasibility_reason=None,
            ),
        ),
    )


def _catalog(
    *,
    workflow_template_digest: str | None = None,
    shared_random_source_id: str = "draw-branch",
) -> FeasibleSupportCatalog:
    catalog = compile_branch_grammar(
        _grammar(
            workflow_template_digest=workflow_template_digest,
            shared_random_source_id=shared_random_source_id,
        )
    )
    assert isinstance(catalog, FeasibleSupportCatalog)
    return catalog


def _assignment(
    catalog: FeasibleSupportCatalog,
    **values: str,
):
    for assignment in catalog.assignments:
        observed = {item.variable_id: item.value for item in assignment.values}
        if observed == values:
            return assignment
    raise AssertionError(f"assignment absent: {values}")


def _binding(
    *,
    no_use: bool = False,
    schedule_case_id: str = "schedule-case-1",
    workflow_template_digest: str | None = None,
    method_menu_digest: str | None = None,
    shared_random_source_id: str = "draw-branch",
) -> GrammarInstanceBinding:
    catalog = _catalog(
        workflow_template_digest=workflow_template_digest,
        shared_random_source_id=shared_random_source_id,
    )
    assignment = _assignment(
        catalog,
        branch="right" if no_use else "left",
        gate="open",
    )
    assert assignment.outcome_id is not None
    schedule = _closed_schedule(
        empty=no_use,
        schedule_case_id=schedule_case_id,
    )
    active_sites = (
        ()
        if no_use
        else (
            ActiveDemandSiteBinding("site-a", "demand-1"),
            ActiveDemandSiteBinding("site-b", "demand-2"),
        )
    )
    active_epochs = (
        () if no_use else (ActiveEpochBinding("epoch-shared", "reference-1"),)
    )
    projection_rule_digest = _digest("projection-rule")
    projection = GrammarCaseProjection(
        schema_version=GRAMMAR_CASE_PROJECTION_SCHEMA_VERSION,
        schedule_case_id=schedule.schedule_case_id,
        source_artifact_digest=schedule.source_artifact_digest,
        source_case_digest=schedule.source_case_digest,
        source_schema_digest=schedule.source_schema_digest,
        projection_rule_digest=projection_rule_digest,
        workflow_template_digest=catalog.branch_grammar.workflow_template_digest,
        assignment_values=assignment.values,
        active_site_bindings=active_sites,
        active_epoch_bindings=active_epochs,
    )
    return GrammarInstanceBinding(
        schema_version=GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION,
        schedule_case_id=schedule.schedule_case_id,
        schedule_digest=canonical_digest(schedule),
        source_artifact_digest=schedule.source_artifact_digest,
        source_case_digest=schedule.source_case_digest,
        source_schema_digest=schedule.source_schema_digest,
        projection_rule_digest=projection_rule_digest,
        workflow_template_digest=catalog.branch_grammar.workflow_template_digest,
        method_menu_digest=(
            _digest("menu") if method_menu_digest is None else method_menu_digest
        ),
        closed_schedule=schedule,
        case_projection=projection,
        branch_grammar_digest=catalog.branch_grammar_digest,
        support_catalog_digest=canonical_digest(catalog),
        support_catalog=catalog,
        assignment_id=assignment.assignment_id,
        assignment_values=assignment.values,
        outcome_id=assignment.outcome_id,
        active_site_bindings=active_sites,
        active_epoch_bindings=active_epochs,
        inactive_demand_site_ids=(
            ("site-a", "site-b", "site-unused") if no_use else ("site-unused",)
        ),
        inactive_epoch_slot_ids=(
            ("epoch-shared", "epoch-unused") if no_use else ("epoch-unused",)
        ),
    )


def _closed_schedule(
    *,
    empty: bool = False,
    cross_epoch: bool = False,
    schedule_case_id: str = "schedule-case-1",
):
    if empty:
        return _schedule_artifact(
            (),
            (),
            schedule_case_id=schedule_case_id,
            source_case_digest=_digest(f"source-case-{schedule_case_id}"),
        )
    block = BlockKey(
        content_digest=_digest("block"),
        parent_digest=None,
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        adapter_fingerprint=None,
        block_size_tokens=16,
        kv_dtype="bf16",
    )
    workflow = WorkflowKey("workflow", 0)
    events = tuple(
        ScheduleDemandEvent(
            event_ordinal=ordinal,
            schedule_event_id=f"demand-{ordinal + 1}",
            scheduled_access_ns=(10 + 10 * ordinal if cross_epoch else 10),
            claim_id=f"claim-{ordinal + 1}",
            retention_binding_id=f"retention-{ordinal + 1}",
            request_binding_id=f"request-binding-{ordinal + 1}",
            workflow=workflow,
            node_id=f"node-{ordinal + 1}",
            execution_ref=ExecutionRef(
                workflow=workflow,
                request_id=f"request-{ordinal + 1}",
                sequence_id=f"sequence-{ordinal + 1}",
                logical_block_index=0,
            ),
            block_key=block,
            reuse_epoch_id=(
                f"reference-{ordinal + 1}" if cross_epoch else "reference-1"
            ),
            source_record_id=f"source-record-{ordinal + 1}",
            source_record_digest=_digest(f"source-record-{ordinal + 1}"),
        )
        for ordinal in range(2)
    )
    epochs = (
        tuple(
            ScheduleEpoch(
                reuse_epoch_id=f"reference-{ordinal + 1}",
                access_ns=10 + 10 * ordinal,
                block_key=block,
                schedule_event_ids=(f"demand-{ordinal + 1}",),
            )
            for ordinal in range(2)
        )
        if cross_epoch
        else (
            ScheduleEpoch(
                reuse_epoch_id="reference-1",
                access_ns=10,
                block_key=block,
                schedule_event_ids=("demand-1", "demand-2"),
            ),
        )
    )
    return _schedule_artifact(
        events,
        epochs,
        schedule_case_id=schedule_case_id,
        source_case_digest=_digest(f"source-case-{schedule_case_id}"),
        producer_kind=ScheduleProducerKind.REPLAY,
    )


def _cross_epoch_binding() -> GrammarInstanceBinding:
    grammar = _grammar()
    outcome = replace(
        _use_outcome(),
        site_epoch_mappings=(
            SiteEpochMapping("site-a", "epoch-a"),
            SiteEpochMapping("site-b", "epoch-b"),
        ),
    )
    grammar = replace(
        grammar,
        reference_epoch_slot_ids=("epoch-a", "epoch-b", "epoch-unused"),
        rules=(*grammar.rules[:-1], replace(grammar.rules[-1], outcome=outcome)),
    )
    catalog = compile_branch_grammar(grammar)
    assert isinstance(catalog, FeasibleSupportCatalog)
    assignment = _assignment(catalog, branch="left", gate="open")
    assert assignment.outcome_id is not None
    schedule = _closed_schedule(cross_epoch=True)
    active_sites = (
        ActiveDemandSiteBinding("site-a", "demand-1"),
        ActiveDemandSiteBinding("site-b", "demand-2"),
    )
    active_epochs = (
        ActiveEpochBinding("epoch-a", "reference-1"),
        ActiveEpochBinding("epoch-b", "reference-2"),
    )
    projection_rule_digest = _digest("cross-epoch-projection-rule")
    projection = GrammarCaseProjection(
        schema_version=GRAMMAR_CASE_PROJECTION_SCHEMA_VERSION,
        schedule_case_id=schedule.schedule_case_id,
        source_artifact_digest=schedule.source_artifact_digest,
        source_case_digest=schedule.source_case_digest,
        source_schema_digest=schedule.source_schema_digest,
        projection_rule_digest=projection_rule_digest,
        workflow_template_digest=grammar.workflow_template_digest,
        assignment_values=assignment.values,
        active_site_bindings=active_sites,
        active_epoch_bindings=active_epochs,
    )
    return GrammarInstanceBinding(
        schema_version=GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION,
        schedule_case_id=schedule.schedule_case_id,
        schedule_digest=canonical_digest(schedule),
        source_artifact_digest=schedule.source_artifact_digest,
        source_case_digest=schedule.source_case_digest,
        source_schema_digest=schedule.source_schema_digest,
        projection_rule_digest=projection_rule_digest,
        workflow_template_digest=grammar.workflow_template_digest,
        method_menu_digest=_digest("menu"),
        closed_schedule=schedule,
        case_projection=projection,
        branch_grammar_digest=catalog.branch_grammar_digest,
        support_catalog_digest=canonical_digest(catalog),
        support_catalog=catalog,
        assignment_id=assignment.assignment_id,
        assignment_values=assignment.values,
        outcome_id=assignment.outcome_id,
        active_site_bindings=active_sites,
        active_epoch_bindings=active_epochs,
        inactive_demand_site_ids=("site-unused",),
        inactive_epoch_slot_ids=("epoch-unused",),
    )


def _binding_conservation(
    schedule_case_ids: tuple[str, ...] = ("schedule-case-1", "schedule-case-2"),
) -> tuple[
    GrammarBindingConservation,
    tuple[GrammarInstanceBinding, ...],
]:
    plan = _cutoff_plan(
        UniversePurpose.EXCLUDED_PILOT,
        tuple(
            _cutoff_slot(
                f"slot-{index}",
                10 + 20 * index,
                schedule_case_id=schedule_case_id,
            )
            for index, schedule_case_id in enumerate(schedule_case_ids)
        ),
    )
    bindings = tuple(
        _binding(
            schedule_case_id=schedule_case_id,
            workflow_template_digest=plan.workflow_template_digest,
            method_menu_digest=plan.method_menu_digest,
        )
        for schedule_case_id in sorted(set(schedule_case_ids))
    )
    input_digests = derive_grammar_binding_input_digests(bindings)
    plan = replace(
        plan,
        schedule_digest=input_digests.schedule_digest,
        source_digest=input_digests.source_digest,
    )
    universe = build_candidate_universe(plan)
    return build_grammar_binding_conservation(universe, bindings), bindings


def test_exhaustive_canonical_enumeration_and_content_ids() -> None:
    grammar = _grammar()
    catalog = compile_branch_grammar(grammar)

    assert isinstance(catalog, FeasibleSupportCatalog)
    assert len(catalog.assignments) == 4
    assert [
        tuple((item.variable_id, item.value) for item in assignment.values)
        for assignment in catalog.assignments
    ] == [
        (("branch", "left"), ("gate", "closed")),
        (("branch", "left"), ("gate", "open")),
        (("branch", "right"), ("gate", "closed")),
        (("branch", "right"), ("gate", "open")),
    ]
    first = catalog.assignments[0]
    assert first.assignment_id == canonical_digest(
        AssignmentIdentity(
            schema_version=ASSIGNMENT_IDENTITY_SCHEMA_VERSION,
            branch_grammar_digest=canonical_digest(grammar),
            values=first.values,
        )
    )
    assert len(catalog.outcomes) == 2
    assert catalog.support_catalog_id != canonical_digest(grammar)


def test_wildcard_clause_supplies_and_semantics() -> None:
    catalog = _catalog()
    left = [item for item in catalog.assignments if item.rule_id == "use"]

    assert len(left) == 2
    assert {item.clause_id for item in left} == {"left"}
    assert len({item.outcome_id for item in left}) == 1


def test_frozen_rule_outcome_is_hashed_once_per_verifier_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grammar = _grammar()
    real_digest = grammar_module.canonical_digest
    outcome_hashes = 0

    def count_outcome_hashes(value: object) -> str:
        nonlocal outcome_hashes
        if isinstance(value, OutcomeIdentity):
            outcome_hashes += 1
        return real_digest(value)

    monkeypatch.setattr(grammar_module, "canonical_digest", count_outcome_hashes)

    catalog = compile_branch_grammar(grammar)

    assert isinstance(catalog, FeasibleSupportCatalog)
    assert outcome_hashes == 4


def test_shared_latent_identity_is_frozen_in_grammar_digest() -> None:
    grammar = _grammar()
    assert {item.shared_random_source_id for item in grammar.variables} == {
        "draw-branch"
    }
    changed = replace(
        grammar,
        variables=(
            replace(grammar.variables[0], shared_random_source_id="another-draw"),
            grammar.variables[1],
        ),
    )

    assert canonical_digest(grammar) != canonical_digest(changed)
    assert isinstance(compile_branch_grammar(changed), FeasibleSupportCatalog)


def test_zero_variable_grammar_has_one_canonical_assignment() -> None:
    grammar = BranchGrammar(
        schema_version=BRANCH_GRAMMAR_SCHEMA_VERSION,
        canonical_order=GRAMMAR_CANONICAL_ORDER,
        workflow_template_digest=_digest("constant-template"),
        variables=(),
        demand_site_ids=(),
        reference_epoch_slot_ids=(),
        terminal_node_ids=("terminal",),
        rules=(
            GrammarRule(
                rule_id="constant",
                clauses=(GrammarClause("wildcard", ()),),
                disposition=RuleDisposition.FEASIBLE,
                outcome=FeasibleOutcome(
                    kind=OutcomeKind.NO_USE,
                    terminal_node_ids=("terminal",),
                    active_demand_site_ids=(),
                    site_epoch_mappings=(),
                ),
                infeasibility_reason=None,
            ),
        ),
    )

    catalog = compile_branch_grammar(grammar)

    assert isinstance(catalog, FeasibleSupportCatalog)
    assert len(catalog.assignments) == 1
    assert catalog.assignments[0].values == ()


def test_shared_latent_diagonal_is_feasible_and_off_diagonal_is_typed() -> None:
    no_use = FeasibleOutcome(
        kind=OutcomeKind.NO_USE,
        terminal_node_ids=("terminal",),
        active_demand_site_ids=(),
        site_epoch_mappings=(),
    )
    grammar = BranchGrammar(
        schema_version=BRANCH_GRAMMAR_SCHEMA_VERSION,
        canonical_order=GRAMMAR_CANONICAL_ORDER,
        workflow_template_digest=_digest("shared-latent-template"),
        variables=(
            BranchVariable("first", ("0", "1"), "shared-draw"),
            BranchVariable("second", ("0", "1"), "shared-draw"),
        ),
        demand_site_ids=(),
        reference_epoch_slot_ids=(),
        terminal_node_ids=("terminal",),
        rules=(
            GrammarRule(
                rule_id="diagonal",
                clauses=(
                    _clause("one", _term("first", "1"), _term("second", "1")),
                    _clause("zero", _term("first", "0"), _term("second", "0")),
                ),
                disposition=RuleDisposition.FEASIBLE,
                outcome=no_use,
                infeasibility_reason=None,
            ),
            GrammarRule(
                rule_id="off-diagonal",
                clauses=(
                    _clause("one-zero", _term("first", "1"), _term("second", "0")),
                    _clause("zero-one", _term("first", "0"), _term("second", "1")),
                ),
                disposition=RuleDisposition.INFEASIBLE,
                outcome=None,
                infeasibility_reason=InfeasibilityReason.SHARED_LATENT_CONFLICT,
            ),
        ),
    )

    catalog = compile_branch_grammar(grammar)

    assert isinstance(catalog, FeasibleSupportCatalog)
    for assignment in catalog.assignments:
        values = {item.variable_id: item.value for item in assignment.values}
        expected = (
            RuleDisposition.FEASIBLE
            if values["first"] == values["second"]
            else RuleDisposition.INFEASIBLE
        )
        assert assignment.disposition == expected


def test_no_use_and_many_to_one_epoch_outcomes_are_explicit() -> None:
    catalog = _catalog()
    no_use = next(
        item.outcome
        for item in catalog.outcomes
        if item.outcome.kind == OutcomeKind.NO_USE
    )
    use = next(
        item.outcome
        for item in catalog.outcomes
        if item.outcome.kind == OutcomeKind.USE
    )

    assert no_use.active_demand_site_ids == ()
    assert no_use.active_epoch_slot_ids == ()
    assert use.active_epoch_slot_ids == ("epoch-shared",)
    assert len(use.site_epoch_mappings) == 2


def test_same_rule_clause_overlap_fails_closed() -> None:
    grammar = _grammar()
    overlapping_rule = replace(
        grammar.rules[-1],
        clauses=(
            _clause("left", _term("branch", "left")),
            _clause("left-open", _term("branch", "left"), _term("gate", "open")),
        ),
    )
    grammar = replace(grammar, rules=(*grammar.rules[:-1], overlapping_rule))

    with pytest.raises(TraceValidationError, match="multiple clauses"):
        compile_branch_grammar(grammar)


def test_cross_rule_overlap_fails_closed() -> None:
    grammar = _grammar()
    no_use = replace(
        grammar.rules[1],
        clauses=(_clause("all-open", _term("gate", "open")),),
    )
    grammar = replace(grammar, rules=(grammar.rules[0], no_use, grammar.rules[2]))

    with pytest.raises(TraceValidationError, match="multiple clauses"):
        compile_branch_grammar(grammar)


def test_missing_assignment_fails_closed() -> None:
    grammar = _grammar()
    grammar = replace(grammar, rules=(grammar.rules[0], grammar.rules[2]))

    with pytest.raises(TraceValidationError, match="no matching clause"):
        compile_branch_grammar(grammar)


def test_duplicate_term_and_noncanonical_order_fail_closed() -> None:
    with pytest.raises(TraceValidationError, match="more than one term"):
        GrammarClause(
            "bad",
            (_term("branch", "left"), _term("branch", "right")),
        )

    with pytest.raises(TraceValidationError, match="canonical order"):
        GrammarClause(
            "bad",
            (_term("gate", "open"), _term("branch", "left")),
        )


def test_outcome_mapping_must_be_total_and_single_valued() -> None:
    with pytest.raises(TraceValidationError, match="domain must equal"):
        FeasibleOutcome(
            kind=OutcomeKind.USE,
            terminal_node_ids=("terminal-success",),
            active_demand_site_ids=("site-a", "site-b"),
            site_epoch_mappings=(SiteEpochMapping("site-a", "epoch-shared"),),
        )

    with pytest.raises(TraceValidationError, match="more than once"):
        FeasibleOutcome(
            kind=OutcomeKind.USE,
            terminal_node_ids=("terminal-success",),
            active_demand_site_ids=("site-a",),
            site_epoch_mappings=(
                SiteEpochMapping("site-a", "epoch-shared"),
                SiteEpochMapping("site-a", "epoch-unused"),
            ),
        )


def test_feasible_outcome_requires_a_terminal_node() -> None:
    with pytest.raises(TraceValidationError, match="requires a terminal node"):
        replace(_no_use_outcome(), terminal_node_ids=())


def test_unknown_variable_returns_typed_abstention_without_catalog() -> None:
    grammar = _grammar()
    bad_rule = replace(
        grammar.rules[2],
        clauses=(_clause("unknown", _term("unfrozen-variable", "x")),),
    )
    grammar = replace(grammar, rules=(*grammar.rules[:-1], bad_rule))

    result = compile_branch_grammar(grammar)

    assert isinstance(result, StructuralAbstention)
    assert result.reason == StructuralAbstentionReason.UNKNOWN_VARIABLE
    assert result.detail_ids == ("unfrozen-variable",)
    assert not hasattr(result, "support_catalog_id")


def test_unbounded_domain_returns_typed_abstention() -> None:
    grammar = _grammar()
    grammar = replace(
        grammar,
        variables=(
            replace(grammar.variables[0], domain_values=None),
            grammar.variables[1],
        ),
    )

    result = compile_branch_grammar(grammar)

    assert isinstance(result, StructuralAbstention)
    assert result.reason == StructuralAbstentionReason.UNBOUNDED_DOMAIN
    assert result.detail_ids == ("branch",)


def test_empty_finite_domain_is_not_a_second_unbounded_encoding() -> None:
    with pytest.raises(TraceValidationError, match="cannot be empty"):
        BranchVariable("branch", (), None)


def test_enumeration_ceiling_returns_typed_abstention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_ASSIGNMENTS", 3)

    result = compile_branch_grammar(_grammar())

    assert isinstance(result, StructuralAbstention)
    assert result.reason == StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED


def test_all_resource_ceilings_are_inclusive_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_ASSIGNMENTS", 4)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_VARIABLES", 2)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_DOMAIN_VALUES", 2)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_RULES", 3)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_CLAUSES", 3)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_TERMS", 5)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_TERM_VALUES", 5)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_MATCH_WORK", 32)
    monkeypatch.setattr(grammar_module, "MAX_GRAMMAR_ASSIGNMENT_VALUE_CELLS", 8)

    assert isinstance(compile_branch_grammar(_grammar()), FeasibleSupportCatalog)


@pytest.mark.parametrize(
    ("constant", "limit", "detail"),
    (
        ("MAX_GRAMMAR_VARIABLES", 1, "variables:"),
        ("MAX_GRAMMAR_DOMAIN_VALUES", 1, "domain:"),
        ("MAX_GRAMMAR_RULES", 2, "rules:"),
        ("MAX_GRAMMAR_CLAUSES", 2, "clauses:"),
        ("MAX_GRAMMAR_TERMS", 1, "terms:"),
        ("MAX_GRAMMAR_TERM_VALUES", 4, "term_values:"),
        ("MAX_GRAMMAR_MATCH_WORK", 31, "match_work:"),
        (
            "MAX_GRAMMAR_ASSIGNMENT_VALUE_CELLS",
            7,
            "assignment_value_cells:",
        ),
        ("MAX_GRAMMAR_IDENTIFIER_BYTES", 3, "identifier_bytes:"),
        ("MAX_GRAMMAR_TOTAL_TEXT_BYTES", 1, "total_text_bytes:"),
        (
            "MAX_GRAMMAR_DERIVED_CATALOG_BYTES",
            1,
            "derived_catalog_bytes:",
        ),
    ),
)
def test_parser_structure_ceilings_abstain_without_partial_catalog(
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    detail: str,
) -> None:
    monkeypatch.setattr(grammar_module, constant, limit)

    result = compile_branch_grammar(_grammar())

    assert isinstance(result, StructuralAbstention)
    assert result.reason == StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED
    assert any(item.startswith(detail) for item in result.detail_ids)
    assert not hasattr(result, "assignments")


def test_canonical_grammar_byte_ceiling_abstains_before_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grammar = _grammar()
    monkeypatch.setattr(
        grammar_module,
        "MAX_GRAMMAR_ARTIFACT_BYTES",
        len(canonical_json(grammar)) - 1,
    )

    result = compile_branch_grammar(grammar)

    assert isinstance(result, StructuralAbstention)
    assert result.detail_ids == (f"artifact_bytes:{len(canonical_json(grammar))}",)


def test_long_clause_label_amplification_hits_catalog_ceiling_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = tuple(f"v{index:03d}" for index in range(100))
    grammar = BranchGrammar(
        schema_version=BRANCH_GRAMMAR_SCHEMA_VERSION,
        canonical_order=GRAMMAR_CANONICAL_ORDER,
        workflow_template_digest=_digest("amplification-template"),
        variables=(BranchVariable("choice", domain, None),),
        demand_site_ids=(),
        reference_epoch_slot_ids=(),
        terminal_node_ids=("terminal",),
        rules=(
            GrammarRule(
                rule_id="rule",
                clauses=(GrammarClause("x" * 4_000, ()),),
                disposition=RuleDisposition.FEASIBLE,
                outcome=FeasibleOutcome(
                    kind=OutcomeKind.NO_USE,
                    terminal_node_ids=("terminal",),
                    active_demand_site_ids=(),
                    site_epoch_mappings=(),
                ),
                infeasibility_reason=None,
            ),
        ),
    )
    monkeypatch.setattr(
        grammar_module,
        "MAX_GRAMMAR_DERIVED_CATALOG_BYTES",
        200_000,
    )

    result = compile_branch_grammar(grammar)

    assert isinstance(result, StructuralAbstention)
    assert result.reason == StructuralAbstentionReason.SAFETY_CEILING_EXCEEDED
    assert result.detail_ids[0].startswith("derived_catalog_bytes:")


def test_outcome_unknown_site_epoch_and_terminal_fail_closed() -> None:
    grammar = _grammar()
    outcome = _use_outcome()
    cases = (
        replace(
            outcome,
            active_demand_site_ids=("site-a", "site-unknown"),
            site_epoch_mappings=(
                SiteEpochMapping("site-a", "epoch-shared"),
                SiteEpochMapping("site-unknown", "epoch-shared"),
            ),
        ),
        replace(
            outcome,
            site_epoch_mappings=(
                SiteEpochMapping("site-a", "epoch-shared"),
                SiteEpochMapping("site-b", "epoch-unknown"),
            ),
        ),
        replace(outcome, terminal_node_ids=("terminal-unknown",)),
    )
    messages = ("unknown demand sites", "unknown active epoch", "unknown terminal")

    for candidate, message in zip(cases, messages, strict=True):
        changed = replace(grammar.rules[-1], outcome=candidate)
        with pytest.raises(TraceValidationError, match=message):
            compile_branch_grammar(
                replace(grammar, rules=(*grammar.rules[:-1], changed))
            )


def test_term_value_outside_domain_fails_closed() -> None:
    grammar = _grammar()
    changed = replace(
        grammar.rules[-1],
        clauses=(_clause("bad-value", _term("branch", "middle")),),
    )

    with pytest.raises(TraceValidationError, match="outside"):
        compile_branch_grammar(replace(grammar, rules=(*grammar.rules[:-1], changed)))


def test_support_catalog_rejects_assignment_outcome_and_id_tamper() -> None:
    catalog = _catalog()
    first = catalog.assignments[0]
    tampered_assignment = replace(first, clause_id="changed")

    with pytest.raises(TraceValidationError, match="exhaustive replay"):
        replace(catalog, assignments=(tampered_assignment, *catalog.assignments[1:]))
    with pytest.raises(TraceValidationError, match="exhaustive replay"):
        replace(catalog, outcomes=catalog.outcomes[:-1])
    with pytest.raises(TraceValidationError, match="exhaustive replay"):
        replace(catalog, support_catalog_id="0" * 64)


def test_use_instance_binding_conserves_exact_active_domains() -> None:
    binding = _binding()

    assert tuple(item.demand_site_id for item in binding.active_site_bindings) == (
        "site-a",
        "site-b",
    )
    assert tuple(item.epoch_slot_id for item in binding.active_epoch_bindings) == (
        "epoch-shared",
    )
    assert binding.inactive_demand_site_ids == ("site-unused",)
    assert binding.inactive_epoch_slot_ids == ("epoch-unused",)


def test_no_use_binding_has_complete_inactive_domains() -> None:
    binding = _binding(no_use=True)

    assert binding.active_site_bindings == ()
    assert binding.active_epoch_bindings == ()
    assert (
        binding.inactive_demand_site_ids
        == binding.support_catalog.branch_grammar.demand_site_ids
    )
    assert (
        binding.inactive_epoch_slot_ids
        == binding.support_catalog.branch_grammar.reference_epoch_slot_ids
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "active_site_bindings",
            (ActiveDemandSiteBinding("site-a", "demand-1"),),
            "case projection",
        ),
        (
            "active_epoch_bindings",
            (),
            "case projection",
        ),
        (
            "inactive_demand_site_ids",
            (),
            "conserve every demand site",
        ),
        (
            "inactive_epoch_slot_ids",
            (),
            "conserve every epoch slot",
        ),
        ("assignment_id", "0" * 64, "assignment is absent"),
        ("assignment_values", (), "case projection"),
        ("outcome_id", "0" * 64, "outcome differs"),
        ("source_case_digest", "0" * 64, "case projection"),
    ),
)
def test_instance_binding_rejects_domain_and_identity_tamper(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(TraceValidationError, match=message):
        replace(_binding(), **{field: value})


def test_instance_binding_requires_bijective_concrete_ids() -> None:
    binding = _binding()
    duplicate_demands = (
        ActiveDemandSiteBinding("site-a", "same-demand"),
        ActiveDemandSiteBinding("site-b", "same-demand"),
    )

    with pytest.raises(TraceValidationError, match="bijectively"):
        replace(binding.case_projection, active_site_bindings=duplicate_demands)


def test_projection_rejects_same_outcome_assignment_rebind() -> None:
    binding = _binding()
    same_outcome = _assignment(
        binding.support_catalog,
        branch="left",
        gate="closed",
    )
    assert same_outcome.outcome_id == binding.outcome_id

    with pytest.raises(TraceValidationError, match="case projection"):
        replace(
            binding,
            assignment_id=same_outcome.assignment_id,
            assignment_values=same_outcome.values,
        )


def test_projection_rejects_same_epoch_site_event_swap() -> None:
    binding = _binding()
    swapped = (
        ActiveDemandSiteBinding("site-a", "demand-2"),
        ActiveDemandSiteBinding("site-b", "demand-1"),
    )

    with pytest.raises(TraceValidationError, match="case projection"):
        replace(binding, active_site_bindings=swapped)


def test_cross_epoch_swap_contradicts_schedule_even_if_projection_is_changed() -> None:
    binding = _cross_epoch_binding()
    swapped = (
        ActiveDemandSiteBinding("site-a", "demand-2"),
        ActiveDemandSiteBinding("site-b", "demand-1"),
    )
    projection = replace(binding.case_projection, active_site_bindings=swapped)

    with pytest.raises(TraceValidationError, match="reuse epoch contradicts"):
        replace(
            binding,
            active_site_bindings=swapped,
            case_projection=projection,
        )


def test_projection_and_grammar_reject_template_digest_replacement() -> None:
    binding = _binding()
    changed_digest = _digest("another-template")

    with pytest.raises(TraceValidationError, match="case projection"):
        replace(binding, workflow_template_digest=changed_digest)

    projection = replace(
        binding.case_projection,
        workflow_template_digest=changed_digest,
    )
    with pytest.raises(TraceValidationError, match="workflow-template digest differs"):
        replace(
            binding,
            workflow_template_digest=changed_digest,
            case_projection=projection,
        )


def test_instance_binding_consumes_every_schedule_event_and_epoch() -> None:
    binding = _binding()
    omitted_event = (
        ActiveDemandSiteBinding("site-a", "demand-1"),
        ActiveDemandSiteBinding("site-b", "unknown-demand"),
    )

    omitted_projection = replace(
        binding.case_projection,
        active_site_bindings=omitted_event,
    )
    with pytest.raises(TraceValidationError, match="conserve the closed schedule"):
        replace(
            binding,
            active_site_bindings=omitted_event,
            case_projection=omitted_projection,
        )

    unknown_epoch = (ActiveEpochBinding("epoch-shared", "unknown-epoch"),)
    epoch_projection = replace(
        binding.case_projection,
        active_epoch_bindings=unknown_epoch,
    )
    with pytest.raises(TraceValidationError, match="conserve the closed schedule"):
        replace(
            binding,
            active_epoch_bindings=unknown_epoch,
            case_projection=epoch_projection,
        )


def test_no_use_binding_rejects_nonempty_schedule() -> None:
    binding = _binding(no_use=True)
    schedule = _closed_schedule()

    with pytest.raises(TraceValidationError, match="conserve the closed schedule"):
        replace(
            binding,
            closed_schedule=schedule,
            schedule_digest=canonical_digest(schedule),
            source_case_digest=schedule.source_case_digest,
        )


def test_instance_binding_rejects_infeasible_assignment() -> None:
    binding = _binding()
    assignment = _assignment(binding.support_catalog, branch="right", gate="closed")
    projection = replace(
        binding.case_projection,
        assignment_values=assignment.values,
    )

    with pytest.raises(TraceValidationError, match="infeasible"):
        replace(
            binding,
            assignment_id=assignment.assignment_id,
            assignment_values=assignment.values,
            case_projection=projection,
        )


def test_binding_conservation_covers_distinct_plan_cases_once() -> None:
    artifact, bindings = _binding_conservation(
        ("schedule-case-1", "schedule-case-1", "schedule-case-2")
    )

    assert artifact.schedule_case_ids == ("schedule-case-1", "schedule-case-2")
    assert tuple(item.schedule_case_id for item in artifact.references) == (
        "schedule-case-1",
        "schedule-case-2",
    )
    assert artifact.input_digests == derive_grammar_binding_input_digests(
        tuple(reversed(bindings))
    )
    assert (
        artifact.input_digests.schedule_digest
        == artifact.candidate_universe.cutoff_plan.schedule_digest
    )
    assert (
        artifact.input_digests.source_digest
        == artifact.candidate_universe.cutoff_plan.source_digest
    )
    validate_grammar_binding_conservation(artifact, bindings)


def test_binding_conservation_rejects_missing_extra_and_duplicate_cases() -> None:
    artifact, bindings = _binding_conservation()
    template_digest = artifact.candidate_universe.cutoff_plan.workflow_template_digest

    with pytest.raises(TraceValidationError, match="missing=.*schedule-case-2"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            bindings[:-1],
        )
    extra = _binding(
        schedule_case_id="schedule-case-extra",
        workflow_template_digest=template_digest,
    )
    with pytest.raises(TraceValidationError, match="extra=.*schedule-case-extra"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (*bindings, extra),
        )
    with pytest.raises(TraceValidationError, match="duplicate schedule case"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (bindings[0], bindings[0], bindings[1]),
        )


def test_binding_conservation_rejects_plan_template_drift() -> None:
    artifact, bindings = _binding_conservation()
    changed = _binding(
        schedule_case_id="schedule-case-1",
        workflow_template_digest=_digest("different-plan-template"),
    )

    with pytest.raises(TraceValidationError, match="another workflow template"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (changed, bindings[1]),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("schedule_digest", "binding schedule digest differs"),
        ("source_digest", "binding source digest differs"),
    ),
)
def test_binding_conservation_rejects_plan_input_digest_drift(
    field: str,
    message: str,
) -> None:
    artifact, bindings = _binding_conservation()
    plan = replace(
        artifact.candidate_universe.cutoff_plan,
        **{field: _digest(f"different-{field}")},
    )
    universe = build_candidate_universe(plan)

    with pytest.raises(TraceValidationError, match=message):
        build_grammar_binding_conservation(universe, bindings)


def test_binding_conservation_rejects_method_menu_drift() -> None:
    artifact, bindings = _binding_conservation()
    changed = replace(bindings[1], method_menu_digest=_digest("different-menu"))

    with pytest.raises(TraceValidationError, match="method menu differs"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (bindings[0], changed),
        )


def test_binding_conservation_rejects_per_case_grammar_drift() -> None:
    artifact, bindings = _binding_conservation()
    plan = artifact.candidate_universe.cutoff_plan
    changed = _binding(
        schedule_case_id="schedule-case-2",
        workflow_template_digest=plan.workflow_template_digest,
        method_menu_digest=plan.method_menu_digest,
        shared_random_source_id="different-draw",
    )

    with pytest.raises(TraceValidationError, match="multiple grammar/support"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (bindings[0], changed),
        )


def test_binding_conservation_rejects_projection_rule_drift() -> None:
    artifact, bindings = _binding_conservation()
    changed_digest = _digest("different-projection-rule")
    projection = replace(
        bindings[1].case_projection,
        projection_rule_digest=changed_digest,
    )
    changed = replace(
        bindings[1],
        projection_rule_digest=changed_digest,
        case_projection=projection,
    )

    with pytest.raises(TraceValidationError, match="multiple projection rules"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (bindings[0], changed),
        )


def test_binding_conservation_replays_binding_dependencies() -> None:
    artifact, bindings = _binding_conservation()
    forged = object.__new__(GrammarInstanceBinding)
    for field in fields(GrammarInstanceBinding):
        object.__setattr__(forged, field.name, getattr(bindings[0], field.name))
    object.__setattr__(forged, "assignment_id", "0" * 64)

    with pytest.raises(TraceValidationError, match="assignment is absent"):
        build_grammar_binding_conservation(
            artifact.candidate_universe,
            (forged, bindings[1]),
        )


def test_binding_conservation_replay_rejects_reference_tamper() -> None:
    artifact, bindings = _binding_conservation()
    changed_reference = replace(artifact.references[0], binding_digest="0" * 64)
    tampered = replace(
        artifact,
        references=(changed_reference, *artifact.references[1:]),
    )

    with pytest.raises(TraceValidationError, match="differs from binding replay"):
        validate_grammar_binding_conservation(tampered, bindings)


def test_binding_conservation_create_only_round_trip(tmp_path: Path) -> None:
    artifact, bindings = _binding_conservation()
    path = tmp_path / "binding-conservation.json"

    digest = write_grammar_binding_conservation(path, artifact, bindings)
    loaded = load_grammar_binding_conservation(path, bindings)

    assert loaded.artifact == artifact
    assert loaded.digest == digest == canonical_digest(artifact)
    with pytest.raises(TraceValidationError, match="create-only"):
        write_grammar_binding_conservation(path, artifact, bindings)
    with pytest.raises(TraceValidationError, match="schedule-case set differs"):
        load_grammar_binding_conservation(path, bindings[:-1])


def _artifacts() -> tuple[tuple[str, object, object, object], ...]:
    grammar = _grammar()
    catalog = _catalog()
    abstention_grammar = replace(
        grammar,
        variables=(
            replace(grammar.variables[0], domain_values=None),
            grammar.variables[1],
        ),
    )
    abstention = compile_branch_grammar(abstention_grammar)
    assert isinstance(abstention, StructuralAbstention)
    return (
        ("grammar", grammar, write_branch_grammar, load_branch_grammar),
        (
            "catalog",
            catalog,
            write_feasible_support_catalog,
            load_feasible_support_catalog,
        ),
        (
            "abstention",
            abstention,
            write_structural_abstention,
            load_structural_abstention,
        ),
        (
            "binding",
            _binding(),
            write_grammar_instance_binding,
            load_grammar_instance_binding,
        ),
    )


@pytest.mark.parametrize("name,artifact,writer,loader", _artifacts())
def test_create_only_artifact_round_trip(
    tmp_path: Path,
    name: str,
    artifact: object,
    writer: object,
    loader: object,
) -> None:
    path = tmp_path / f"{name}.json"
    digest = writer(path, artifact)  # type: ignore[operator]
    loaded = loader(path)  # type: ignore[operator]

    assert digest == canonical_digest(artifact)
    assert loaded.artifact == artifact
    assert loaded.digest == digest
    assert loaded.size_bytes == len(canonical_json(artifact))


def test_artifact_write_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "grammar.json"
    write_branch_grammar(path, _grammar())

    with pytest.raises(TraceValidationError, match="create-only"):
        write_branch_grammar(path, _grammar())


def test_racing_create_only_writers_publish_exactly_one_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "grammar.json"
    grammar = _grammar()

    def write() -> str:
        return write_branch_grammar(path, grammar)

    successes: list[str] = []
    failures: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(write), executor.submit(write))
        for future in futures:
            try:
                successes.append(future.result())
            except BaseException as exc:
                failures.append(exc)

    assert successes == [canonical_digest(grammar)]
    assert len(failures) == 1
    assert isinstance(failures[0], TraceValidationError)
    assert "create-only" in str(failures[0])
    assert load_branch_grammar(path).artifact == grammar


def test_equal_length_writer_tamper_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "grammar.json"
    real_pread = grammar_module.os.pread
    tampered = False

    def tamper(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal tampered
        observed = real_pread(descriptor, size, offset)
        if not tampered:
            os.pwrite(descriptor, b"X" * len(observed), offset)
            tampered = True
        return observed

    monkeypatch.setattr(grammar_module.os, "pread", tamper)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_branch_grammar(path, _grammar())

    assert tampered


def test_loader_rejects_equal_length_tamper_during_third_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "grammar.json"
    grammar = _grammar()
    write_branch_grammar(path, grammar)
    raw = path.read_bytes()
    real_pread = grammar_module.os.pread
    calls = 0

    def tamper_on_third_read(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 3:
            path.write_bytes(b"X" * len(raw))
        return real_pread(descriptor, size, offset)

    monkeypatch.setattr(grammar_module.os, "pread", tamper_on_third_read)

    with pytest.raises(TraceValidationError, match="after path revalidation"):
        load_branch_grammar(path)

    assert calls == 3
    assert path.read_bytes() != raw


def test_semantic_catalog_tamper_fails_during_load(tmp_path: Path) -> None:
    catalog = _catalog()
    raw = json.loads(canonical_json(catalog))
    raw["assignments"][0]["clause_id"] = "tampered"
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(raw, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(TraceValidationError, match="exhaustive replay"):
        load_feasible_support_catalog(path)


def test_forged_structural_abstention_rejects_valid_grammar_on_write_and_load(
    tmp_path: Path,
) -> None:
    valid_grammar = _grammar()
    negative_grammar = replace(
        valid_grammar,
        variables=(
            replace(valid_grammar.variables[0], domain_values=None),
            valid_grammar.variables[1],
        ),
    )
    abstention = compile_branch_grammar(negative_grammar)
    assert isinstance(abstention, StructuralAbstention)

    with pytest.raises(TraceValidationError, match="compiles fully"):
        replace(
            abstention,
            branch_grammar_digest=canonical_digest(valid_grammar),
            branch_grammar=valid_grammar,
        )

    forged = object.__new__(StructuralAbstention)
    for field in fields(StructuralAbstention):
        object.__setattr__(forged, field.name, getattr(abstention, field.name))
    object.__setattr__(forged, "branch_grammar", valid_grammar)
    object.__setattr__(
        forged,
        "branch_grammar_digest",
        canonical_digest(valid_grammar),
    )
    write_path = tmp_path / "forged-write.json"
    with pytest.raises(TraceValidationError, match="compiles fully"):
        write_structural_abstention(write_path, forged)
    assert not write_path.exists()

    raw = json.loads(canonical_json(abstention))
    raw["branch_grammar"] = json.loads(canonical_json(valid_grammar))
    raw["branch_grammar_digest"] = canonical_digest(valid_grammar)
    load_path = tmp_path / "forged-load.json"
    load_path.write_text(
        json.dumps(raw, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )
    with pytest.raises(TraceValidationError, match="compiles fully"):
        load_structural_abstention(load_path)


def test_loader_rejects_noncanonical_framing_symlink_and_hardlink(
    tmp_path: Path,
) -> None:
    grammar = _grammar()
    newline = tmp_path / "newline.json"
    newline.write_bytes(canonical_json(grammar) + b"\n")
    with pytest.raises(TraceValidationError, match="framing is not canonical"):
        load_branch_grammar(newline)

    original = tmp_path / "original.json"
    write_branch_grammar(original, grammar)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(original)
    with pytest.raises(TraceValidationError, match="non-symlink"):
        load_branch_grammar(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(TraceValidationError, match="singly linked"):
        load_branch_grammar(original)


def test_closed_schema_versions() -> None:
    grammar = _grammar()
    catalog = _catalog()
    binding = _binding()

    with pytest.raises(TraceValidationError, match="branch-grammar schema"):
        replace(grammar, schema_version="dagkv.m3.branch_grammar.v2")
    with pytest.raises(TraceValidationError, match="support-catalog schema"):
        replace(catalog, schema_version="dagkv.m3.feasible_support_catalog.v2")
    with pytest.raises(TraceValidationError, match="instance-binding schema"):
        replace(binding, schema_version="dagkv.m3.grammar_instance_binding.v2")


def test_public_schema_constants_match_protocol() -> None:
    assert BRANCH_GRAMMAR_SCHEMA_VERSION == "dagkv.m3.branch_grammar.v1"
    assert FEASIBLE_SUPPORT_CATALOG_SCHEMA_VERSION == (
        "dagkv.m3.feasible_support_catalog.v1"
    )
    assert STRUCTURAL_ABSTENTION_SCHEMA_VERSION == (
        "dagkv.m3.grammar_structural_abstention.v1"
    )
    assert GRAMMAR_INSTANCE_BINDING_SCHEMA_VERSION == (
        "dagkv.m3.grammar_instance_binding.v1"
    )
    assert GRAMMAR_BINDING_CONSERVATION_SCHEMA_VERSION == (
        "dagkv.m3.grammar_binding_conservation.v1"
    )
    assert GRAMMAR_BINDING_INPUT_DIGESTS_SCHEMA_VERSION == (
        "dagkv.m3.grammar_binding_input_digests.v1"
    )


def test_top_level_package_exports_grammar_contract() -> None:
    assert dagkv.BranchGrammar is BranchGrammar
    assert dagkv.GrammarInstanceBinding is GrammarInstanceBinding
    assert dagkv.GrammarBindingConservation is GrammarBindingConservation
    assert dagkv.GrammarBindingInputDigests is GrammarBindingInputDigests
    assert dagkv.compile_branch_grammar is compile_branch_grammar
    assert (
        dagkv.build_grammar_binding_conservation is build_grammar_binding_conservation
    )
    assert (
        dagkv.derive_grammar_binding_input_digests
        is derive_grammar_binding_input_digests
    )
    assert (
        dagkv.ASSIGNMENT_IDENTITY_SCHEMA_VERSION == ASSIGNMENT_IDENTITY_SCHEMA_VERSION
    )
    assert dagkv.OUTCOME_IDENTITY_SCHEMA_VERSION == OUTCOME_IDENTITY_SCHEMA_VERSION
    assert (
        dagkv.SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION
        == SUPPORT_CATALOG_IDENTITY_SCHEMA_VERSION
    )
