"""Focused tests for C1-B1 candidate universes and deterministic splits."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv
import dagkv.c1_split as split_module
from dagkv.c1_split import (
    CANDIDATE_UNIVERSE_SCHEMA_VERSION,
    COHORT_TOKEN_CATALOG_SCHEMA_VERSION,
    CUTOFF_PLAN_SCHEMA_VERSION,
    PREDECESSOR_EXCLUSION_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    CandidateDisposition,
    CohortTokenCatalog,
    CohortTokenCatalogEntry,
    CutoffPlan,
    CutoffPlanSlot,
    IneligibilityReason,
    LineageApplicability,
    LineageFamily,
    LineageField,
    PredecessorExclusionCatalog,
    RoleInterval,
    SplitCohort,
    SplitManifest,
    SplitRole,
    UniversePurpose,
    build_candidate_universe,
    build_cohort_token_catalog,
    build_predecessor_exclusion_catalog,
    build_predecessor_union_audit,
    build_split_manifest,
    derive_candidate_id,
    load_candidate_universe,
    load_cutoff_plan,
    load_predecessor_exclusion_catalog,
    load_predecessor_union_audit,
    load_split_manifest,
    write_candidate_universe,
    write_cutoff_plan,
    write_predecessor_exclusion_catalog,
    write_predecessor_union_audit,
    write_split_manifest,
)
from dagkv.c1_trace import (
    TraceCommitIndeterminateError,
    TraceValidationError,
    canonical_digest,
    canonical_json,
)
from dagkv.domain import BlockKey


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _block(label: str = "shared") -> BlockKey:
    return BlockKey(
        content_digest=_digest(f"content-{label}"),
        parent_digest=None,
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        adapter_fingerprint=None,
        block_size_tokens=16,
        kv_dtype="bf16",
    )


def _lineage(
    present: Mapping[LineageFamily, tuple[str, ...]] | None = None,
    *,
    absent: tuple[LineageFamily, ...] = (),
) -> tuple[LineageField, ...]:
    present = {} if present is None else present
    overlap = set(present).intersection(absent)
    assert not overlap
    return tuple(
        LineageField(
            family=family,
            applicability=(
                LineageApplicability.PRESENT
                if family in present
                else (
                    LineageApplicability.ABSENT
                    if family in absent
                    else LineageApplicability.NOT_APPLICABLE
                )
            ),
            values=present.get(family, ()),
        )
        for family in LineageFamily
    )


def _slot(
    slot_id: str,
    split_time_ns: int,
    *,
    lineage: tuple[LineageField, ...] | None = None,
    horizon_ns: int = 10,
    lookback_ns: int = 5,
    schedule_case_id: str | None = None,
) -> CutoffPlanSlot:
    resolved_schedule_case_id = schedule_case_id or f"case-{slot_id}"
    resolved_lineage = _lineage() if lineage is None else lineage
    source_case = next(
        field for field in resolved_lineage if field.family == LineageFamily.SOURCE_CASE
    )
    if source_case.applicability != LineageApplicability.PRESENT:
        resolved_lineage = tuple(
            (
                LineageField(
                    family=LineageFamily.SOURCE_CASE,
                    applicability=LineageApplicability.PRESENT,
                    values=(resolved_schedule_case_id,),
                )
                if field.family == LineageFamily.SOURCE_CASE
                else field
            )
            for field in resolved_lineage
        )
    return CutoffPlanSlot(
        candidate_slot_id=slot_id,
        schedule_case_id=resolved_schedule_case_id,
        block_key=_block(),
        cutoff_trigger_id=f"trigger-{slot_id}",
        split_time_ns=split_time_ns,
        primary_horizon_duration_ns=horizon_ns,
        feature_lookback_ns=lookback_ns,
        lineage=resolved_lineage,
    )


def _plan(
    purpose: UniversePurpose,
    slots: tuple[CutoffPlanSlot, ...],
    *,
    cohort: SplitCohort = SplitCohort.PRIMARY_TEMPORAL,
    temporal_axis: str = "axis",
    method_menu: str = "menu",
    normalizer: str = "normalizer",
    eligibility_rule: str = "eligibility",
) -> CutoffPlan:
    slots = tuple(sorted(slots, key=lambda slot: slot.candidate_slot_id))
    workflow_template_digest = _digest("workflow-template")
    content_lineage_digest = _digest("content-lineage")
    return CutoffPlan(
        schema_version=CUTOFF_PLAN_SCHEMA_VERSION,
        purpose=purpose,
        cohort=cohort,
        schedule_digest=_digest(f"schedule-{purpose.value}"),
        source_digest=_digest(f"source-{purpose.value}"),
        workflow_template_digest=workflow_template_digest,
        content_lineage_digest=content_lineage_digest,
        normalizer_digest=_digest(normalizer),
        eligibility_rule_digest=_digest(eligibility_rule),
        temporal_axis_digest=_digest(temporal_axis),
        method_menu_digest=_digest(method_menu),
        cohort_token_catalog=build_cohort_token_catalog(
            cohort=cohort,
            slots=slots,
            workflow_template_digest=workflow_template_digest,
            content_lineage_digest=content_lineage_digest,
        ),
        slots=slots,
    )


def _fixture_intervals(*, gap: int = 15) -> tuple[RoleInterval, ...]:
    width = 100
    start = 0
    intervals = []
    for role in SplitRole:
        intervals.append(RoleInterval(role=role, start_ns=start, end_ns=start + width))
        start += width + gap
    return tuple(intervals)


def _pilot_intervals() -> tuple[RoleInterval, ...]:
    return (RoleInterval(role=SplitRole.PILOT, start_ns=0, end_ns=100),)


def _main_intervals(
    *,
    train_start: int = 115,
    gap: int = 15,
) -> tuple[RoleInterval, ...]:
    width = 100
    start = train_start
    intervals = []
    for role in (
        SplitRole.TRAIN,
        SplitRole.CAL_FIT,
        SplitRole.CAL_RADIUS,
        SplitRole.FORMAL,
    ):
        intervals.append(RoleInterval(role=role, start_ns=start, end_ns=start + width))
        start += width + gap
    return tuple(intervals)


def _pilot_and_main(
    *,
    pilot_lineage: tuple[LineageField, ...] | None = None,
    main_slots: tuple[CutoffPlanSlot, ...] | None = None,
    main_axis: str = "axis",
    pilot_menu: str = "menu",
    main_menu: str = "menu",
    main_normalizer: str = "normalizer",
    main_eligibility_rule: str = "eligibility",
    pilot_cohort: SplitCohort = SplitCohort.PRIMARY_TEMPORAL,
    main_cohort: SplitCohort | None = None,
    train_start: int = 115,
    catalog_transform: Callable[
        [PredecessorExclusionCatalog], PredecessorExclusionCatalog
    ]
    | None = None,
    predecessor_split_digest: str | None = None,
) -> tuple[SplitManifest, SplitManifest]:
    pilot_plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (_slot("pilot", 10, lineage=pilot_lineage),),
        cohort=pilot_cohort,
        method_menu=pilot_menu,
    )
    pilot_universe = build_candidate_universe(pilot_plan)
    pilot_manifest = build_split_manifest(pilot_universe, _pilot_intervals())
    catalog = build_predecessor_exclusion_catalog(pilot_universe)
    if catalog_transform is not None:
        catalog = catalog_transform(catalog)
    if main_slots is None:
        main_slots = (_slot("main", train_start + 5),)
    main_plan = _plan(
        UniversePurpose.POST_PILOT_MAIN,
        main_slots,
        cohort=main_cohort or pilot_cohort,
        temporal_axis=main_axis,
        method_menu=main_menu,
        normalizer=main_normalizer,
        eligibility_rule=main_eligibility_rule,
    )
    main_universe = build_candidate_universe(
        main_plan,
        predecessor_universe_digest=canonical_digest(pilot_universe),
        predecessor_split_manifest_digest=(
            predecessor_split_digest or canonical_digest(pilot_manifest)
        ),
        predecessor_exclusion_catalog=catalog,
    )
    main_manifest = build_split_manifest(
        main_universe,
        _main_intervals(train_start=train_start),
    )
    return pilot_manifest, main_manifest


def test_lineage_accepts_multiple_values_and_typed_empty_families() -> None:
    lineage = _lineage(
        {
            LineageFamily.REFERENCE_EPOCH: ("epoch-1", "epoch-2"),
            LineageFamily.SCHEDULED_TOOL_EXECUTION: ("tool-1", "tool-2"),
        },
        absent=(LineageFamily.SESSION,),
    )

    assert lineage[1].applicability == LineageApplicability.ABSENT
    epoch = next(
        field for field in lineage if field.family == LineageFamily.REFERENCE_EPOCH
    )
    assert epoch.values == ("epoch-1", "epoch-2")


@pytest.mark.parametrize(
    ("applicability", "values"),
    [
        (LineageApplicability.PRESENT, ()),
        (LineageApplicability.ABSENT, ("unexpected",)),
        (LineageApplicability.NOT_APPLICABLE, ("unexpected",)),
        (LineageApplicability.PRESENT, ("z", "a")),
        (LineageApplicability.PRESENT, ("a", "a")),
    ],
)
def test_lineage_rejects_invalid_applicability_and_order(
    applicability: LineageApplicability,
    values: tuple[str, ...],
) -> None:
    with pytest.raises(TraceValidationError):
        LineageField(
            family=LineageFamily.SESSION,
            applicability=applicability,
            values=values,
        )


def test_cutoff_plan_requires_complete_canonical_lineage() -> None:
    with pytest.raises(TraceValidationError, match="every family"):
        _slot("slot", 10, lineage=_lineage()[:-1])


def test_cutoff_plan_slot_requires_source_case_lineage() -> None:
    slot = _slot("slot", 10)

    with pytest.raises(TraceValidationError, match="requires source-case"):
        replace(slot, lineage=_lineage())


def test_cutoff_plan_rejects_duplicate_raw_opportunity() -> None:
    first = _slot("a", 10, schedule_case_id="shared-case")
    duplicate = replace(first, candidate_slot_id="b")

    with pytest.raises(TraceValidationError, match="duplicate raw opportunity"):
        _plan(UniversePurpose.EXCLUDED_PILOT, (first, duplicate))


def test_cutoff_plan_rejects_inconsistent_source_mapping_for_schedule_case() -> None:
    first = _slot(
        "a",
        10,
        schedule_case_id="shared-case",
        lineage=_lineage({LineageFamily.SOURCE_CASE: ("source-a",)}),
    )
    second = _slot(
        "b",
        20,
        schedule_case_id="shared-case",
        lineage=_lineage({LineageFamily.SOURCE_CASE: ("source-b",)}),
    )

    with pytest.raises(TraceValidationError, match="inconsistent source-case"):
        _plan(UniversePurpose.EXCLUDED_PILOT, (first, second))


def test_candidate_universe_conserves_eligible_and_ineligible_slots() -> None:
    plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (_slot("a", 10), _slot("b", 20)),
    )
    universe = build_candidate_universe(
        plan,
        ineligible={"b": IneligibilityReason.UNSUPPORTED_CONTROL_FLOW},
    )

    assert len(universe.records) == len(plan.slots)
    assert universe.records[0].disposition == CandidateDisposition.ELIGIBLE
    assert universe.records[0].candidate_id == derive_candidate_id(plan, plan.slots[0])
    assert universe.records[1].disposition == CandidateDisposition.INELIGIBLE
    assert universe.records[1].candidate_id is None
    assert (
        universe.records[1].ineligibility_reason
        == IneligibilityReason.UNSUPPORTED_CONTROL_FLOW
    )


def test_candidate_universe_rejects_unknown_normalizer_slot() -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))

    with pytest.raises(TraceValidationError, match="unknown slots"):
        build_candidate_universe(
            plan,
            ineligible={"foreign": IneligibilityReason.NORMALIZER_EXCLUDED},
        )


def test_candidate_universe_rejects_malformed_normalizer_mapping() -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))

    with pytest.raises(ValueError, match="ineligible slot ID"):
        build_candidate_universe(
            plan,
            ineligible={
                1: IneligibilityReason.NORMALIZER_EXCLUDED,
                "foreign": IneligibilityReason.NORMALIZER_EXCLUDED,
            },
        )


def test_candidate_id_changes_with_purpose_and_pre_runtime_lineage() -> None:
    slot = _slot(
        "a",
        10,
        lineage=_lineage({LineageFamily.SOURCE_CASE: ("case-a",)}),
    )
    pilot = _plan(UniversePurpose.EXCLUDED_PILOT, (slot,))
    fixture = _plan(UniversePurpose.STRUCTURAL_FIXTURE, (slot,))
    changed = replace(
        slot,
        lineage=_lineage({LineageFamily.SOURCE_CASE: ("case-b",)}),
    )
    changed_plan = _plan(UniversePurpose.EXCLUDED_PILOT, (changed,))

    assert derive_candidate_id(pilot, slot) != derive_candidate_id(fixture, slot)
    assert derive_candidate_id(pilot, slot) != derive_candidate_id(
        changed_plan, changed
    )


def test_non_main_universe_rejects_predecessor_evidence() -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))

    with pytest.raises(TraceValidationError, match="non-main"):
        build_candidate_universe(
            plan,
            predecessor_universe_digest=_digest("predecessor"),
        )


def test_main_universe_requires_complete_predecessor_evidence() -> None:
    plan = _plan(UniversePurpose.POST_PILOT_MAIN, (_slot("a", 120),))

    with pytest.raises(TraceValidationError, match="lacks predecessor"):
        build_candidate_universe(plan)


def test_transitive_lineage_and_multiple_epochs_form_one_component() -> None:
    plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (
            _slot(
                "a",
                10,
                lineage=_lineage({LineageFamily.SESSION: ("session",)}),
            ),
            _slot(
                "b",
                20,
                lineage=_lineage(
                    {
                        LineageFamily.SESSION: ("session",),
                        LineageFamily.REFERENCE_EPOCH: ("epoch-1",),
                    }
                ),
            ),
            _slot(
                "c",
                30,
                lineage=_lineage(
                    {
                        LineageFamily.REFERENCE_EPOCH: (
                            "epoch-1",
                            "epoch-2",
                        )
                    }
                ),
            ),
        ),
    )
    manifest = build_split_manifest(
        build_candidate_universe(plan),
        _pilot_intervals(),
    )

    assert len(manifest.components) == 1
    assert len(manifest.components[0].candidate_ids) == 3
    epoch_rows = tuple(
        row
        for row in manifest.incidence_rows
        if row.lineage_family == LineageFamily.REFERENCE_EPOCH
    )
    assert tuple(row.lineage_value for row in epoch_rows) == ("epoch-1", "epoch-2")
    assert len(epoch_rows[0].member_candidate_ids) == 2
    assert len(epoch_rows[1].member_candidate_ids) == 1


def test_absent_lineage_does_not_create_component_edges() -> None:
    plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (
            _slot(
                "a",
                10,
                lineage=_lineage(absent=(LineageFamily.SESSION,)),
            ),
            _slot(
                "b",
                20,
                lineage=_lineage(absent=(LineageFamily.SESSION,)),
            ),
        ),
    )
    manifest = build_split_manifest(
        build_candidate_universe(plan),
        _pilot_intervals(),
    )

    assert len(manifest.components) == 2
    assert all(
        row.lineage_family != LineageFamily.SESSION for row in manifest.incidence_rows
    )


@pytest.mark.parametrize(
    ("cohort", "expected_components", "included_family"),
    [
        (SplitCohort.PRIMARY_TEMPORAL, 2, None),
        (
            SplitCohort.TEMPLATE_GENERALIZATION,
            1,
            LineageFamily.WORKFLOW_TEMPLATE,
        ),
        (SplitCohort.CONTENT_ISOLATED, 1, LineageFamily.CONTENT_LINEAGE),
    ],
)
def test_cohort_adds_only_its_declared_lineage_token(
    cohort: SplitCohort,
    expected_components: int,
    included_family: LineageFamily | None,
) -> None:
    shared_special = {
        LineageFamily.WORKFLOW_TEMPLATE: (_digest("template-shared"),),
        LineageFamily.CONTENT_LINEAGE: (_digest("content-shared"),),
    }
    universe = build_candidate_universe(
        _plan(
            UniversePurpose.EXCLUDED_PILOT,
            (
                _slot("a", 10, lineage=_lineage(shared_special)),
                _slot("b", 20, lineage=_lineage(shared_special)),
            ),
            cohort=cohort,
        )
    )

    manifest = build_split_manifest(universe, _pilot_intervals())

    assert manifest.cohort == cohort
    assert len(manifest.components) == expected_components
    special_rows = tuple(
        row
        for row in manifest.incidence_rows
        if row.lineage_family
        in {LineageFamily.WORKFLOW_TEMPLATE, LineageFamily.CONTENT_LINEAGE}
    )
    if included_family is None:
        assert special_rows == ()
    else:
        assert tuple(row.lineage_family for row in special_rows) == (included_family,)


def test_special_cohort_catalog_preserves_distinct_components() -> None:
    plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (
            _slot(
                "a",
                10,
                lineage=_lineage(
                    {LineageFamily.WORKFLOW_TEMPLATE: (_digest("template-a"),)}
                ),
            ),
            _slot(
                "b",
                20,
                lineage=_lineage(
                    {LineageFamily.WORKFLOW_TEMPLATE: (_digest("template-b"),)}
                ),
            ),
        ),
        cohort=SplitCohort.TEMPLATE_GENERALIZATION,
    )

    assert plan.cohort_token_catalog is not None
    assert type(plan.cohort_token_catalog) is CohortTokenCatalog
    assert all(
        type(entry) is CohortTokenCatalogEntry
        for entry in plan.cohort_token_catalog.entries
    )
    assert len(plan.cohort_token_catalog.entries) == 2
    manifest = build_split_manifest(
        build_candidate_universe(plan),
        _pilot_intervals(),
    )
    assert len(manifest.components) == 2


@pytest.mark.parametrize("tamper", ("source", "token", "membership"))
def test_cutoff_plan_rejects_cohort_catalog_tamper(tamper: str) -> None:
    plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (
            _slot(
                "a",
                10,
                lineage=_lineage(
                    {LineageFamily.CONTENT_LINEAGE: (_digest("content-a"),)}
                ),
            ),
        ),
        cohort=SplitCohort.CONTENT_ISOLATED,
    )
    catalog = plan.cohort_token_catalog
    assert catalog is not None
    if tamper == "source":
        changed = replace(catalog, source_catalog_digest=_digest("other-source"))
    else:
        entry = catalog.entries[0]
        changed_entry = (
            replace(entry, token_digest=_digest("other-token"))
            if tamper == "token"
            else replace(entry, candidate_slot_ids=("foreign",))
        )
        changed = replace(catalog, entries=(changed_entry,))

    with pytest.raises(TraceValidationError, match="catalog differs"):
        replace(plan, cohort_token_catalog=changed)


def test_special_cohort_rejects_non_digest_token() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _plan(
            UniversePurpose.EXCLUDED_PILOT,
            (
                _slot(
                    "a",
                    10,
                    lineage=_lineage(
                        {LineageFamily.WORKFLOW_TEMPLATE: ("not-a-digest",)}
                    ),
                ),
            ),
            cohort=SplitCohort.TEMPLATE_GENERALIZATION,
        )


def test_primary_cohort_also_rejects_non_digest_special_lineage() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _plan(
            UniversePurpose.EXCLUDED_PILOT,
            (
                _slot(
                    "a",
                    10,
                    lineage=_lineage(
                        {LineageFamily.CONTENT_LINEAGE: ("not-a-digest",)}
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    "cohort",
    [SplitCohort.TEMPLATE_GENERALIZATION, SplitCohort.CONTENT_ISOLATED],
)
def test_non_primary_cohort_requires_its_candidate_token(
    cohort: SplitCohort,
) -> None:
    universe = build_candidate_universe(
        _plan(
            UniversePurpose.EXCLUDED_PILOT,
            (_slot("a", 10),),
            cohort=cohort,
        )
    )

    with pytest.raises(TraceValidationError, match="cohort-specific"):
        build_split_manifest(universe, _pilot_intervals())


def test_structural_fixture_requires_all_five_roles() -> None:
    universe = build_candidate_universe(
        _plan(UniversePurpose.STRUCTURAL_FIXTURE, (_slot("a", 10),))
    )

    with pytest.raises(TraceValidationError, match="universe purpose"):
        build_split_manifest(universe, _pilot_intervals())


def test_half_open_interval_boundary_and_unassigned_gap_fail_closed() -> None:
    accepted_plan = _plan(
        UniversePurpose.STRUCTURAL_FIXTURE,
        (_slot("train-start", 115),),
    )
    manifest = build_split_manifest(
        build_candidate_universe(accepted_plan),
        _fixture_intervals(),
    )
    assert manifest.assignments[0].role == SplitRole.TRAIN

    gap_plan = _plan(
        UniversePurpose.STRUCTURAL_FIXTURE,
        (_slot("pilot-end", 100),),
    )
    with pytest.raises(TraceValidationError, match="exactly one role"):
        build_split_manifest(
            build_candidate_universe(gap_plan),
            _fixture_intervals(),
        )


def test_guard_gap_accepts_equality_and_rejects_one_ns_deficit() -> None:
    universe = build_candidate_universe(
        _plan(UniversePurpose.STRUCTURAL_FIXTURE, (_slot("a", 10),))
    )

    accepted = build_split_manifest(universe, _fixture_intervals(gap=15))
    assert accepted.max_primary_horizon_duration_ns == 10
    assert accepted.max_feature_lookback_ns == 5
    with pytest.raises(TraceValidationError, match="guard gap"):
        build_split_manifest(universe, _fixture_intervals(gap=14))


def test_lineage_component_cannot_cross_roles() -> None:
    shared = _lineage({LineageFamily.SESSION: ("shared-session",)})
    universe = build_candidate_universe(
        _plan(
            UniversePurpose.STRUCTURAL_FIXTURE,
            (_slot("pilot", 10, lineage=shared), _slot("train", 120, lineage=shared)),
        )
    )

    with pytest.raises(TraceValidationError, match="crosses split roles"):
        build_split_manifest(universe, _fixture_intervals())


def test_split_manifest_replays_all_caller_supplied_derived_fields() -> None:
    universe = build_candidate_universe(
        _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    )
    manifest = build_split_manifest(universe, _pilot_intervals())

    with pytest.raises(TraceValidationError, match="deterministic replay"):
        replace(manifest, max_feature_lookback_ns=6)


def test_split_manifest_rejects_component_and_incidence_tamper() -> None:
    universe = build_candidate_universe(
        _plan(
            UniversePurpose.EXCLUDED_PILOT,
            (
                _slot(
                    "a",
                    10,
                    lineage=_lineage({LineageFamily.SESSION: ("session",)}),
                ),
            ),
        )
    )
    manifest = build_split_manifest(universe, _pilot_intervals())

    with pytest.raises(TraceValidationError, match="deterministic replay"):
        replace(manifest, incidence_rows=())
    changed_component = replace(
        manifest.components[0],
        component_id=_digest("changed-component"),
    )
    with pytest.raises(TraceValidationError, match="deterministic replay"):
        replace(manifest, components=(changed_component,))


def test_predecessor_union_audit_accepts_exact_boundary_gap() -> None:
    pilot, main = _pilot_and_main()

    audit = build_predecessor_union_audit(pilot, main)

    assert audit.observed_boundary_gap_ns == 15
    assert audit.required_boundary_gap_ns == 15
    assert audit.temporal_axis_digest == _digest("axis")


def test_predecessor_union_rejects_direct_source_case_overlap() -> None:
    shared = _lineage({LineageFamily.SOURCE_CASE: ("shared-case",)})
    pilot, main = _pilot_and_main(
        pilot_lineage=shared,
        main_slots=(_slot("main", 120, lineage=shared),),
    )

    with pytest.raises(TraceValidationError, match="crosses split roles"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_reused_schedule_case() -> None:
    pilot, main = _pilot_and_main(
        main_slots=(
            _slot(
                "main",
                120,
                schedule_case_id="case-pilot",
            ),
        ),
    )

    with pytest.raises(TraceValidationError, match="schedule cases cross"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_reused_schedule_with_forged_source_lineage() -> None:
    pilot, main = _pilot_and_main(
        pilot_lineage=_lineage({LineageFamily.SOURCE_CASE: ("pilot-source",)}),
        main_slots=(
            _slot(
                "main",
                120,
                schedule_case_id="case-pilot",
                lineage=_lineage({LineageFamily.SOURCE_CASE: ("forged-main-source",)}),
            ),
        ),
    )

    with pytest.raises(TraceValidationError, match="schedule cases cross"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_transitive_cross_universe_lineage() -> None:
    pilot_lineage = _lineage({LineageFamily.SESSION: ("shared-session",)})
    main_slots = (
        _slot(
            "main-a",
            120,
            lineage=_lineage(
                {
                    LineageFamily.SESSION: ("shared-session",),
                    LineageFamily.DERIVED_EXAMPLE: ("derived-link",),
                }
            ),
        ),
        _slot(
            "main-b",
            130,
            lineage=_lineage({LineageFamily.DERIVED_EXAMPLE: ("derived-link",)}),
        ),
    )
    pilot, main = _pilot_and_main(
        pilot_lineage=pilot_lineage,
        main_slots=main_slots,
    )

    with pytest.raises(TraceValidationError, match="crosses split roles"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_one_ns_boundary_deficit() -> None:
    pilot, main = _pilot_and_main(train_start=114)

    with pytest.raises(TraceValidationError, match="pilot-to-train guard"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_temporal_axis_drift() -> None:
    pilot, main = _pilot_and_main(main_axis="other-axis")

    with pytest.raises(TraceValidationError, match="temporal axes differ"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_method_menu_drift() -> None:
    pilot, main = _pilot_and_main(main_menu="other-menu")

    with pytest.raises(TraceValidationError, match="method menus differ"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_normalizer_drift() -> None:
    pilot, main = _pilot_and_main(main_normalizer="other-normalizer")

    with pytest.raises(TraceValidationError, match="normalizers differ"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_eligibility_rule_drift() -> None:
    pilot, main = _pilot_and_main(main_eligibility_rule="other-eligibility")

    with pytest.raises(TraceValidationError, match="eligibility rules differ"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_cohort_drift() -> None:
    pilot, main = _pilot_and_main(
        pilot_cohort=SplitCohort.PRIMARY_TEMPORAL,
        main_cohort=SplitCohort.CONTENT_ISOLATED,
        main_slots=(
            _slot(
                "main",
                120,
                lineage=_lineage(
                    {LineageFamily.CONTENT_LINEAGE: (_digest("main-content"),)}
                ),
            ),
        ),
    )

    with pytest.raises(TraceValidationError, match="split cohorts differ"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_catalog_omission() -> None:
    def empty_catalog(
        catalog: PredecessorExclusionCatalog,
    ) -> PredecessorExclusionCatalog:
        return PredecessorExclusionCatalog(
            schema_version=PREDECESSOR_EXCLUSION_SCHEMA_VERSION,
            predecessor_universe_digest=catalog.predecessor_universe_digest,
            entries=(),
        )

    pilot, main = _pilot_and_main(catalog_transform=empty_catalog)

    with pytest.raises(TraceValidationError, match="exclusions are incomplete"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_wrong_pilot_split_binding() -> None:
    pilot, main = _pilot_and_main(predecessor_split_digest=_digest("wrong-split"))

    with pytest.raises(TraceValidationError, match="another pilot split"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_zero_eligible_pilot() -> None:
    pilot_plan = _plan(
        UniversePurpose.EXCLUDED_PILOT,
        (_slot("pilot", 10),),
    )
    pilot_universe = build_candidate_universe(
        pilot_plan,
        ineligible={"pilot": IneligibilityReason.ELIGIBILITY_RULE_REJECTED},
    )
    pilot = build_split_manifest(pilot_universe, _pilot_intervals())
    catalog = build_predecessor_exclusion_catalog(pilot_universe)
    main_plan = _plan(
        UniversePurpose.POST_PILOT_MAIN,
        (_slot("main", 120),),
    )
    main_universe = build_candidate_universe(
        main_plan,
        predecessor_universe_digest=canonical_digest(pilot_universe),
        predecessor_split_manifest_digest=canonical_digest(pilot),
        predecessor_exclusion_catalog=catalog,
    )
    main = build_split_manifest(main_universe, _main_intervals())

    with pytest.raises(TraceValidationError, match="zero-eligible pilot"):
        build_predecessor_union_audit(pilot, main)


def test_predecessor_union_rejects_zero_eligible_main_lane() -> None:
    pilot, populated_main = _pilot_and_main()
    populated_universe = populated_main.candidate_universe
    main_universe = build_candidate_universe(
        populated_universe.cutoff_plan,
        ineligible={"main": IneligibilityReason.ELIGIBILITY_RULE_REJECTED},
        predecessor_universe_digest=populated_universe.predecessor_universe_digest,
        predecessor_split_manifest_digest=(
            populated_universe.predecessor_split_manifest_digest
        ),
        predecessor_exclusion_catalog=(
            populated_universe.predecessor_exclusion_catalog
        ),
    )
    main = build_split_manifest(main_universe, _main_intervals())

    with pytest.raises(TraceValidationError, match="eligible main lane"):
        build_predecessor_union_audit(pilot, main)


def _all_artifacts() -> tuple[
    tuple[
        str,
        object,
        Callable[[Path, object], str],
        Callable[[Path], object],
    ],
    ...,
]:
    pilot, main = _pilot_and_main()
    audit = build_predecessor_union_audit(pilot, main)
    plan = pilot.candidate_universe.cutoff_plan
    universe = pilot.candidate_universe
    catalog = build_predecessor_exclusion_catalog(universe)
    return (
        ("plan", plan, write_cutoff_plan, load_cutoff_plan),
        ("universe", universe, write_candidate_universe, load_candidate_universe),
        (
            "predecessor-catalog",
            catalog,
            write_predecessor_exclusion_catalog,
            load_predecessor_exclusion_catalog,
        ),
        ("split", pilot, write_split_manifest, load_split_manifest),
        (
            "audit",
            audit,
            write_predecessor_union_audit,
            load_predecessor_union_audit,
        ),
    )


@pytest.mark.parametrize("name,artifact,writer,loader", _all_artifacts())
def test_canonical_artifact_round_trip(
    tmp_path: Path,
    name: str,
    artifact: object,
    writer: Callable[[Path, object], str],
    loader: Callable[[Path], object],
) -> None:
    path = tmp_path / f"{name}.json"

    digest = writer(path, artifact)
    loaded = loader(path)

    assert digest == canonical_digest(artifact)
    assert loaded.artifact == artifact
    assert loaded.digest == digest
    assert loaded.size_bytes == len(canonical_json(artifact))


def test_artifact_write_is_create_only(tmp_path: Path) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    write_cutoff_plan(path, plan)

    with pytest.raises(TraceValidationError, match="create-only"):
        write_cutoff_plan(path, plan)


def test_artifact_precreate_os_error_is_validation_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "precreate-failure.json"
    real_open = split_module.os.open

    def fail_artifact_open(*args: object, **kwargs: object) -> int:
        if kwargs.get("dir_fd") is not None:
            raise PermissionError("injected pre-create failure")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(split_module.os, "open", fail_artifact_open)

    with pytest.raises(TraceValidationError, match="cannot create"):
        write_cutoff_plan(path, plan)

    assert not path.exists()


def test_artifact_writer_retries_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "short-write.json"
    real_write = split_module.os.write

    def short_write(descriptor: int, data: bytes) -> int:
        return real_write(descriptor, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(split_module.os, "write", short_write)

    digest = write_cutoff_plan(path, plan)

    assert load_cutoff_plan(path).digest == digest


@pytest.mark.parametrize(
    "failure",
    (
        "write",
        "runtime",
        "postopen_fileexists",
        "fsync",
        "fstat",
        "stat",
        "readback",
    ),
)
def test_artifact_postcreate_io_failure_is_indeterminate_and_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / f"failed-{failure}.json"
    if failure == "write":
        monkeypatch.setattr(split_module.os, "write", lambda *_: 0)
    elif failure == "runtime":

        def fail_with_runtime(*_: object) -> int:
            raise RuntimeError("injected post-open runtime failure")

        monkeypatch.setattr(split_module.os, "write", fail_with_runtime)
    elif failure == "postopen_fileexists":

        def fail_with_file_exists(*_: object) -> int:
            raise FileExistsError("injected post-open conflict")

        monkeypatch.setattr(split_module.os, "write", fail_with_file_exists)
    elif failure == "fsync":

        def fail_fsync(_: int) -> None:
            raise OSError("injected fsync failure")

        monkeypatch.setattr(split_module.os, "fsync", fail_fsync)
    elif failure == "fstat":
        real_fstat = split_module.os.fstat
        calls = 0

        def fail_artifact_fstat(descriptor: int) -> os.stat_result:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected artifact fstat failure")
            return real_fstat(descriptor)

        monkeypatch.setattr(split_module.os, "fstat", fail_artifact_fstat)
    elif failure == "stat":
        real_stat = split_module.os.stat

        def fail_stat(*args: object, **kwargs: object) -> os.stat_result:
            if kwargs.get("dir_fd") is not None:
                raise OSError("injected artifact stat failure")
            return real_stat(*args, **kwargs)

        monkeypatch.setattr(split_module.os, "stat", fail_stat)
    else:
        monkeypatch.setattr(split_module.os, "pread", lambda *_: b"")

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    monkeypatch.undo()
    assert path.exists()
    with pytest.raises(TraceValidationError, match="create-only"):
        write_cutoff_plan(path, plan)


def test_artifact_close_failure_is_indeterminate_and_closes_both_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "close-failure.json"
    real_close = split_module.os.close
    closed: list[int] = []

    def fail_first_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 1:
            raise OSError("injected file close failure")

    monkeypatch.setattr(split_module.os, "close", fail_first_close)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert len(closed) == 2


def test_artifact_parent_close_failure_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "parent-close-failure.json"
    real_close = split_module.os.close
    closed: list[int] = []

    def fail_parent_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 2:
            raise OSError("injected parent close failure")

    monkeypatch.setattr(split_module.os, "close", fail_parent_close)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert len(closed) == 2


def test_precreate_conflict_is_not_masked_by_parent_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "existing.json"
    write_cutoff_plan(path, plan)
    real_close = split_module.os.close

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("injected parent close failure")

    monkeypatch.setattr(split_module.os, "close", fail_close)

    with pytest.raises(TraceValidationError, match="create-only"):
        write_cutoff_plan(path, plan)


def test_artifact_primary_failure_is_not_masked_by_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "combined-failure.json"
    real_close = split_module.os.close
    closed: list[int] = []
    monkeypatch.setattr(split_module.os, "write", lambda *_: 0)

    def fail_first_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if len(closed) == 1:
            raise OSError("injected close failure")

    monkeypatch.setattr(split_module.os, "close", fail_first_close)

    with pytest.raises(TraceCommitIndeterminateError) as captured:
        write_cutoff_plan(path, plan)

    assert len(closed) == 2
    assert captured.value.__cause__ is not None
    assert "made no progress" in str(captured.value.__cause__)


def test_artifact_cancellation_closes_both_descriptors_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedAbort(BaseException):
        pass

    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "abort.json"
    real_close = split_module.os.close
    closed: list[int] = []

    def abort_write(*_: object) -> int:
        raise InjectedAbort("injected cancellation")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(split_module.os, "write", abort_write)
    monkeypatch.setattr(split_module.os, "close", record_close)

    with pytest.raises(InjectedAbort, match="cancellation"):
        write_cutoff_plan(path, plan)

    assert len(closed) == 2


def test_open_parent_runtime_failure_closes_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "parent-runtime.json"
    real_close = split_module.os.close
    closed: list[int] = []

    def fail_fstat(_: int) -> os.stat_result:
        raise RuntimeError("injected parent fstat failure")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(split_module.os, "fstat", fail_fstat)
    monkeypatch.setattr(split_module.os, "close", record_close)

    with pytest.raises(TraceValidationError, match="parent safely"):
        write_cutoff_plan(path, plan)

    assert len(closed) == 1


def test_artifact_parent_rebinding_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    path = parent / "plan.json"
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    real_pread = split_module.os.pread
    rebound = False

    def rebind_parent(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal rebound
        chunk = real_pread(descriptor, size, offset)
        if not rebound:
            parent.rename(moved_parent)
            parent.mkdir()
            rebound = True
        return chunk

    monkeypatch.setattr(split_module.os, "pread", rebind_parent)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert rebound
    assert not path.exists()
    assert (moved_parent / path.name).exists()


def test_artifact_late_hardlink_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    alias = tmp_path / "late-hardlink.json"
    real_pread = split_module.os.pread
    linked = False

    def add_hardlink(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal linked
        chunk = real_pread(descriptor, size, offset)
        if not linked:
            os.link(path, alias)
            linked = True
        return chunk

    monkeypatch.setattr(split_module.os, "pread", add_hardlink)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert linked
    assert path.stat().st_nlink == 2


def test_artifact_equal_length_overwrite_during_readback_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    real_pread = split_module.os.pread
    tampered = False

    def overwrite_after_read(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal tampered
        chunk = real_pread(descriptor, size, offset)
        if not tampered:
            os.pwrite(descriptor, b"X" * len(chunk), offset)
            tampered = True
        return chunk

    monkeypatch.setattr(split_module.os, "pread", overwrite_after_read)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert tampered
    assert path.read_bytes() != canonical_json(plan)


def test_artifact_path_replacement_during_readback_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    displaced = tmp_path / "displaced.json"
    raw = canonical_json(plan)
    real_pread = split_module.os.pread
    replaced = False

    def replace_after_read(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal replaced
        chunk = real_pread(descriptor, size, offset)
        if not replaced:
            path.rename(displaced)
            path.write_bytes(b"Y" * len(raw))
            replaced = True
        return chunk

    monkeypatch.setattr(split_module.os, "pread", replace_after_read)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_cutoff_plan(path, plan)

    assert replaced
    assert displaced.read_bytes() == raw
    assert path.read_bytes() != raw


def test_artifact_writer_rejects_relative_and_symlink_parent_paths(
    tmp_path: Path,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))

    with pytest.raises(TraceValidationError, match="absolute"):
        write_cutoff_plan(Path("relative-plan.json"), plan)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(TraceValidationError, match="non-symlink directory"):
        write_cutoff_plan(alias_parent / "plan.json", plan)


def test_candidate_universe_tamper_fails_during_load(tmp_path: Path) -> None:
    universe = build_candidate_universe(
        _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    )
    path = tmp_path / "universe.json"
    raw = json.loads(canonical_json(universe))
    raw["records"][0]["candidate_id"] = "0" * 64
    path.write_text(
        json.dumps(raw, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(TraceValidationError, match="candidate ID differs"):
        load_candidate_universe(path)


def test_noncanonical_artifact_framing_fails(tmp_path: Path) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    path.write_bytes(canonical_json(plan) + b"\n")

    with pytest.raises(TraceValidationError, match="framing is not canonical"):
        load_cutoff_plan(path)


def test_artifact_loader_rejects_equal_length_overwrite_after_stale_lstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    write_cutoff_plan(path, plan)
    raw = path.read_bytes()
    real_lstat = Path.lstat
    calls = 0

    def overwrite_after_stale_lstat(candidate: Path) -> os.stat_result:
        nonlocal calls
        observed = real_lstat(candidate)
        if candidate == path:
            calls += 1
            if calls == 2:
                path.write_bytes(b"X" * len(raw))
        return observed

    monkeypatch.setattr(Path, "lstat", overwrite_after_stale_lstat)

    with pytest.raises(TraceValidationError, match="changed"):
        load_cutoff_plan(path)

    assert calls >= 2
    assert path.read_bytes() != raw


def test_artifact_loader_close_failure_is_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    path = tmp_path / "plan.json"
    write_cutoff_plan(path, plan)
    real_close = split_module.os.close

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("injected loader close failure")

    monkeypatch.setattr(split_module.os, "close", fail_close)

    with pytest.raises(TraceValidationError, match="cannot read"):
        load_cutoff_plan(path)


def test_artifact_loader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    original = tmp_path / "plan.json"
    write_cutoff_plan(original, plan)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(original)

    original_bytes = original.read_bytes()
    with pytest.raises(TraceValidationError, match="create-only"):
        write_cutoff_plan(symlink, plan)
    assert original.read_bytes() == original_bytes
    with pytest.raises(TraceValidationError, match="non-symlink"):
        load_cutoff_plan(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(TraceValidationError, match="singly linked"):
        load_cutoff_plan(original)


def test_split_schema_versions_are_closed() -> None:
    plan = _plan(UniversePurpose.EXCLUDED_PILOT, (_slot("a", 10),))
    universe = build_candidate_universe(plan)
    manifest = build_split_manifest(universe, _pilot_intervals())

    with pytest.raises(TraceValidationError, match="cutoff-plan schema"):
        replace(plan, schema_version="dagkv.m3.c1_cutoff_plan.v2")
    with pytest.raises(TraceValidationError, match="candidate-universe schema"):
        replace(universe, schema_version="dagkv.m3.c1_candidate_universe.v2")
    with pytest.raises(TraceValidationError, match="split-manifest schema"):
        replace(manifest, schema_version="dagkv.m3.c1_split_manifest.v2")


def test_public_schema_constants_match_artifacts() -> None:
    assert CUTOFF_PLAN_SCHEMA_VERSION == "dagkv.m3.c1_cutoff_plan.v1"
    assert COHORT_TOKEN_CATALOG_SCHEMA_VERSION == (
        "dagkv.m3.c1_cohort_token_catalog.v1"
    )
    assert CANDIDATE_UNIVERSE_SCHEMA_VERSION == ("dagkv.m3.c1_candidate_universe.v1")
    assert SPLIT_MANIFEST_SCHEMA_VERSION == "dagkv.m3.c1_split_manifest.v1"


def test_package_exports_stable_split_api() -> None:
    public_names = (
        "CANDIDATE_UNIVERSE_SCHEMA_VERSION",
        "COHORT_TOKEN_CATALOG_SCHEMA_VERSION",
        "CUTOFF_PLAN_SCHEMA_VERSION",
        "PREDECESSOR_EXCLUSION_SCHEMA_VERSION",
        "PREDECESSOR_UNION_AUDIT_SCHEMA_VERSION",
        "ROLE_ASSIGNMENT_ALGORITHM",
        "SPLIT_MANIFEST_SCHEMA_VERSION",
        "CandidateDisposition",
        "CandidateUniverse",
        "CohortTokenCatalog",
        "CutoffPlan",
        "CutoffPlanSlot",
        "IneligibilityReason",
        "LineageApplicability",
        "LineageFamily",
        "LineageField",
        "LoadedSplitArtifact",
        "PredecessorExclusionCatalog",
        "PredecessorUnionAudit",
        "RoleInterval",
        "SplitCohort",
        "SplitManifest",
        "SplitRole",
        "UniversePurpose",
        "build_candidate_universe",
        "build_cohort_token_catalog",
        "build_predecessor_exclusion_catalog",
        "build_predecessor_union_audit",
        "build_split_manifest",
        "derive_candidate_id",
        "load_candidate_universe",
        "load_cutoff_plan",
        "load_predecessor_exclusion_catalog",
        "load_predecessor_union_audit",
        "load_split_manifest",
        "write_candidate_universe",
        "write_cutoff_plan",
        "write_predecessor_exclusion_catalog",
        "write_predecessor_union_audit",
        "write_split_manifest",
    )

    for name in public_names:
        assert getattr(dagkv, name) is not None
        assert name in dagkv.__all__
