"""Focused tests for exhaustive C1-B1 feature schema classification."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv
import dagkv.c1_features as feature_module
from dagkv.c1_features import (
    AVAILABILITY_RULE_CATALOG_SCHEMA_VERSION,
    FEATURE_CLASSIFICATION_PROFILE_SCHEMA_VERSION,
    FEATURE_CONTRACT_SCHEMA_VERSION,
    FIELD_PATH_CATALOG_SCHEMA_VERSION,
    FROZEN_FEATURE_CLASSIFICATION_PROFILE_DIGEST,
    FeatureAvailabilityKind,
    FeatureAvailabilityReceipt,
    FeatureAvailabilityRuleCatalog,
    FeatureClassification,
    FeatureContract,
    FeatureFieldAssignment,
    FeatureSourceSchema,
    FeatureValueKind,
    FieldPathCatalog,
    build_feature_availability_rule_catalog,
    build_feature_classification_assignments,
    build_feature_contract,
    build_field_path_catalog,
    load_feature_availability_rule_catalog,
    load_feature_contract,
    load_field_path_catalog,
    validate_feature_availability_rule_catalog,
    validate_field_path_catalog,
    write_feature_availability_rule_catalog,
    write_feature_contract,
    write_field_path_catalog,
)
from dagkv.c1_trace import (
    TraceCommitIndeterminateError,
    TraceValidationError,
    canonical_digest,
    canonical_json,
)


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _contract() -> FeatureContract:
    catalog = build_field_path_catalog()
    rules = build_feature_availability_rule_catalog(catalog)
    assignments = build_feature_classification_assignments(
        catalog,
        rules,
    )
    return build_feature_contract(
        catalog,
        assignments,
        availability_rule_catalog=rules,
    )


def _path(
    catalog: FieldPathCatalog,
    source: FeatureSourceSchema,
    value: str,
):
    return next(
        entry
        for entry in catalog.paths
        if entry.source_schema == source and entry.field_path == value
    )


def test_catalog_is_deterministic_and_binds_all_closed_roots() -> None:
    first = build_field_path_catalog()
    second = build_field_path_catalog()

    assert first == second
    assert canonical_digest(first) == canonical_digest(second)
    assert len(first.paths) == 432
    assert len(first.sequence_identity_rules) == 21
    assert tuple(root.source_schema for root in first.schema_roots) == (
        FeatureSourceSchema.LIFECYCLE_SIDECAR,
        FeatureSourceSchema.SCHEDULE_SIDECAR,
        FeatureSourceSchema.TRACE,
    )
    assert len({root.schema_descriptor_digest for root in first.schema_roots}) == 3
    validate_field_path_catalog(first)


def test_trace_catalog_expands_only_legal_payload_variants() -> None:
    catalog = build_field_path_catalog()
    trace_paths = {
        entry.field_path
        for entry in catalog.paths
        if entry.source_schema == FeatureSourceSchema.TRACE
    }
    required_prefixes = {
        "trace_header.TraceHeaderPayload",
        "workflow_topology.WorkflowTopologyPayload",
        "cutoff.CutoffPayload",
        "forecast_attempt.PredictedAttemptPayload",
        "forecast_attempt.AbstainedAttemptPayload",
        "demand_intent.DemandIntentPayload",
        "reuse_epoch.ReuseEpochPayload",
        "schedule_watermark.ReplayScheduleWatermarkPayload",
        "schedule_watermark.NaturalTraceWatermarkPayload",
        "observation_terminal.ObservationTerminalPayload",
    }

    assert {".".join(path.split(".")[:2]) for path in trace_paths} == required_prefixes
    assert not any(
        path.startswith("trace_header.PredictedAttemptPayload") for path in trace_paths
    )


def test_catalog_expands_nested_and_schedule_union_variants() -> None:
    catalog = build_field_path_catalog()
    paths = {entry.field_path for entry in catalog.paths}

    for variant in (
        "ResidentExecMapService",
        "H2DExecMapService",
        "H2DFailedService",
        "RequestCancelledService",
    ):
        assert any(f"service_terminals[*].{variant}." in path for path in paths)
    assert any("closure.ReplayScheduleClosure." in path for path in paths)
    assert any("closure.NaturalScheduleClosure." in path for path in paths)

    service_rule = next(
        rule
        for rule in catalog.sequence_identity_rules
        if rule.collection_path.endswith("payload.service_terminals")
    )
    assert tuple(variant.variant_type for variant in service_rule.variants) == (
        "H2DExecMapService",
        "H2DFailedService",
        "RequestCancelledService",
        "ResidentExecMapService",
    )
    assert all(
        variant.identity_fields == ("intent_record_id",)
        for variant in service_rule.variants
    )


def test_catalog_represents_optional_absence_and_sequence_wildcards() -> None:
    catalog = build_field_path_catalog()
    optional_scalar = _path(
        catalog,
        FeatureSourceSchema.TRACE,
        "cutoff.CutoffPayload.payload.last_event_id",
    )
    optional_dataclass = _path(
        catalog,
        FeatureSourceSchema.LIFECYCLE_SIDECAR,
        "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent.workflow",
    )

    assert optional_scalar.value_kinds == (
        FeatureValueKind.ABSENT,
        FeatureValueKind.TEXT,
    )
    assert optional_dataclass.value_kinds == (FeatureValueKind.ABSENT,)
    assert any(
        entry.field_path.endswith("events[*].LifecycleEvent.workflow.workflow_id")
        for entry in catalog.paths
    )
    assert all(
        "[0]" not in entry.field_path and "[1]" not in entry.field_path
        for entry in catalog.paths
    )
    assert all(
        rule.collection_path + "[*]" in entry.field_path
        for rule in catalog.sequence_identity_rules
        for entry in (
            next(
                item
                for item in catalog.paths
                if item.source_schema == rule.source_schema
                and (
                    item.field_path == rule.collection_path + "[*]"
                    or item.field_path.startswith(rule.collection_path + "[*].")
                )
            ),
        )
    )


def test_catalog_rejects_omission_even_after_consistent_reseal() -> None:
    catalog = build_field_path_catalog()
    omitted = replace(catalog, paths=catalog.paths[:-1])

    with pytest.raises(TraceValidationError, match="differs from current schemas"):
        validate_field_path_catalog(omitted)


def test_catalog_rejects_duplicate_unsorted_and_unbound_wildcard() -> None:
    catalog = build_field_path_catalog()

    with pytest.raises(TraceValidationError, match="sorted and unique"):
        replace(catalog, paths=(*catalog.paths, catalog.paths[-1]))
    with pytest.raises(TraceValidationError, match="sorted and unique"):
        replace(catalog, paths=tuple(reversed(catalog.paths)))
    with pytest.raises(TraceValidationError, match="differ from current schemas"):
        replace(
            catalog,
            sequence_identity_rules=catalog.sequence_identity_rules[1:],
        )
    with pytest.raises(TraceValidationError, match="runtime sequence index"):
        replace(
            catalog.paths[0],
            field_path=f"{catalog.paths[0].field_path}[0]",
        )


def test_sequence_rules_bind_canonical_ids_and_variant_identity_fields() -> None:
    catalog = build_field_path_catalog()
    rule = catalog.sequence_identity_rules[0]

    with pytest.raises(TraceValidationError, match="rule ID differs"):
        replace(rule, rule_id="0" * 64)

    invalid_variant = replace(
        rule.variants[0],
        identity_fields=("nonexistent_field",),
    )
    invalid_identity = feature_module._SequenceRuleIdentity(
        source_schema=rule.source_schema,
        collection_path=rule.collection_path,
        implementation=rule.implementation,
        collision_policy=rule.collision_policy,
        variants=(invalid_variant,),
    )
    with pytest.raises(TraceValidationError, match="variant schema"):
        replace(
            rule,
            variants=(invalid_variant,),
            rule_id=canonical_digest(invalid_identity),
        )


def test_catalog_rejects_forged_duplicate_sequence_rule_ids() -> None:
    catalog = build_field_path_catalog()
    original = catalog.sequence_identity_rules[1]
    forged = object.__new__(type(original))
    for field in fields(type(original)):
        object.__setattr__(forged, field.name, getattr(original, field.name))
    object.__setattr__(
        forged,
        "rule_id",
        catalog.sequence_identity_rules[0].rule_id,
    )
    rules = list(catalog.sequence_identity_rules)
    rules[1] = forged

    with pytest.raises(TraceValidationError, match="rule IDs are duplicated"):
        replace(catalog, sequence_identity_rules=tuple(rules))


def test_catalog_rejects_a_known_variant_rebound_to_another_path() -> None:
    catalog = build_field_path_catalog()
    lifecycle_index, lifecycle_rule = next(
        (index, rule)
        for index, rule in enumerate(catalog.sequence_identity_rules)
        if rule.collection_path == "lifecycle_sidecar.ClosedLifecycleArtifact.events"
    )
    workflow_variant = next(
        rule.variants[0]
        for rule in catalog.sequence_identity_rules
        if rule.collection_path.endswith("payload.workflow_spec.nodes")
    )
    rebound_identity = feature_module._SequenceRuleIdentity(
        source_schema=lifecycle_rule.source_schema,
        collection_path=lifecycle_rule.collection_path,
        implementation=lifecycle_rule.implementation,
        collision_policy=lifecycle_rule.collision_policy,
        variants=(workflow_variant,),
    )
    rebound = replace(
        lifecycle_rule,
        rule_id=canonical_digest(rebound_identity),
        variants=(workflow_variant,),
    )
    rules = list(catalog.sequence_identity_rules)
    rules[lifecycle_index] = rebound

    with pytest.raises(TraceValidationError, match="differ from current schemas"):
        replace(catalog, sequence_identity_rules=tuple(rules))


def test_catalog_rejects_descriptor_and_generator_drift() -> None:
    catalog = build_field_path_catalog()
    changed_root = replace(
        catalog.schema_roots[0],
        schema_descriptor_digest="0" * 64,
    )
    changed = replace(
        catalog,
        schema_roots=(changed_root, *catalog.schema_roots[1:]),
    )

    with pytest.raises(TraceValidationError, match="differs from current schemas"):
        validate_field_path_catalog(changed)
    with pytest.raises(TraceValidationError, match="field-path generator"):
        replace(catalog, generator_implementation="caller_schema_walk_v1")


def test_catalog_rejects_forged_nested_runtime_types() -> None:
    catalog = build_field_path_catalog()
    original = catalog.paths[-1]
    forged = object.__new__(type(original))
    for field in fields(type(original)):
        object.__setattr__(forged, field.name, getattr(original, field.name))
    object.__setattr__(forged, "source_schema", original.source_schema.value)

    with pytest.raises(TraceValidationError, match="wrong enum type"):
        replace(catalog, paths=(*catalog.paths[:-1], forged))


def test_feature_contract_is_an_exact_catalog_bijection() -> None:
    contract = _contract()

    assert len(contract.assignments) == len(contract.field_path_catalog.paths)
    assert tuple(
        (assignment.source_schema.value, assignment.field_path)
        for assignment in contract.assignments
    ) == tuple(
        (entry.source_schema.value, entry.field_path)
        for entry in contract.field_path_catalog.paths
    )
    online = tuple(
        assignment
        for assignment in contract.assignments
        if assignment.classification == FeatureClassification.ONLINE_ALLOWED
    )
    assert len(online) == 9
    assert all(
        assignment.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and assignment.availability_rule_id is not None
        for assignment in online
    )
    assert {assignment.availability_rule_id for assignment in online} == {
        rule.rule_id for rule in contract.availability_rule_catalog.rules
    }


def test_frozen_profile_has_all_four_semantic_classes() -> None:
    contract = _contract()
    counts = {
        classification: sum(
            assignment.classification == classification
            for assignment in contract.assignments
        )
        for classification in FeatureClassification
    }

    assert counts == {
        FeatureClassification.ONLINE_ALLOWED: 9,
        FeatureClassification.LABEL_ONLY: 201,
        FeatureClassification.PROVENANCE_ONLY: 178,
        FeatureClassification.FORBIDDEN_PROXY: 44,
    }
    assert contract.classification_profile_digest == (
        FROZEN_FEATURE_CLASSIFICATION_PROFILE_DIGEST
    )
    by_key = {
        (assignment.source_schema, assignment.field_path): assignment.classification
        for assignment in contract.assignments
    }
    assert (
        by_key[
            (
                FeatureSourceSchema.LIFECYCLE_SIDECAR,
                "lifecycle_sidecar.ClosedLifecycleArtifact.events[*]."
                "LifecycleEvent.action",
            )
        ]
        == FeatureClassification.ONLINE_ALLOWED
    )
    assert (
        by_key[
            (
                FeatureSourceSchema.SCHEDULE_SIDECAR,
                "schedule_sidecar.ClosedScheduleArtifact.events[*]."
                "ScheduleDemandEvent.scheduled_access_ns",
            )
        ]
        == FeatureClassification.LABEL_ONLY
    )
    assert (
        by_key[
            (
                FeatureSourceSchema.SCHEDULE_SIDECAR,
                "schedule_sidecar.ClosedScheduleArtifact.events[*]."
                "ScheduleDemandEvent.event_ordinal",
            )
        ]
        == FeatureClassification.FORBIDDEN_PROXY
    )
    assert (
        by_key[
            (
                FeatureSourceSchema.TRACE,
                "cutoff.CutoffPayload.payload.cutoff_ns",
            )
        ]
        == FeatureClassification.PROVENANCE_ONLY
    )


def test_feature_contract_rejects_missing_extra_and_duplicate_assignments() -> None:
    contract = _contract()

    with pytest.raises(TraceValidationError, match="missing="):
        replace(contract, assignments=contract.assignments[:-1])
    extra = replace(
        contract.assignments[-1],
        field_path="schedule_sidecar.UnknownRoot.value",
    )
    with pytest.raises(TraceValidationError, match="extra="):
        replace(
            contract,
            assignments=tuple(
                sorted(
                    (*contract.assignments, extra),
                    key=lambda item: (item.source_schema.value, item.field_path),
                )
            ),
        )
    with pytest.raises(TraceValidationError, match="sorted and unique"):
        replace(
            contract,
            assignments=(*contract.assignments, contract.assignments[-1]),
        )


def test_online_assignment_rejects_zero_online_and_future_schedule() -> None:
    contract = _contract()
    zero_online = tuple(
        replace(
            assignment,
            classification=FeatureClassification.PROVENANCE_ONLY,
            availability_rule_id=None,
        )
        if assignment.classification == FeatureClassification.ONLINE_ALLOWED
        else assignment
        for assignment in contract.assignments
    )

    with pytest.raises(TraceValidationError, match="frozen allowlist"):
        replace(contract, assignments=zero_online)

    schedule_index = next(
        index
        for index, assignment in enumerate(contract.assignments)
        if assignment.source_schema == FeatureSourceSchema.SCHEDULE_SIDECAR
        and assignment.field_path.endswith("scheduled_access_ns")
    )
    changed = list(contract.assignments)
    changed[schedule_index] = replace(
        changed[schedule_index],
        classification=FeatureClassification.ONLINE_ALLOWED,
        availability_rule_id=contract.availability_rule_catalog.rules[0].rule_id,
    )

    with pytest.raises(TraceValidationError, match="frozen allowlist"):
        replace(contract, assignments=tuple(changed))


def test_online_assignment_requires_its_exact_path_rule() -> None:
    contract = _contract()
    online_indices = tuple(
        index
        for index, assignment in enumerate(contract.assignments)
        if assignment.classification == FeatureClassification.ONLINE_ALLOWED
    )
    assert len(online_indices) > 1
    changed = list(contract.assignments)
    changed[online_indices[0]] = replace(
        changed[online_indices[0]],
        availability_rule_id=changed[online_indices[1]].availability_rule_id,
    )

    with pytest.raises(TraceValidationError, match="exact availability rule"):
        replace(contract, assignments=tuple(changed))


def test_assignment_rejects_missing_or_illegal_rule_binding() -> None:
    contract = _contract()
    online = next(
        assignment
        for assignment in contract.assignments
        if assignment.classification == FeatureClassification.ONLINE_ALLOWED
    )
    offline = next(
        assignment
        for assignment in contract.assignments
        if assignment.classification != FeatureClassification.ONLINE_ALLOWED
    )

    with pytest.raises(TraceValidationError, match="online feature lacks"):
        replace(online, availability_rule_id=None)
    with pytest.raises(TraceValidationError, match="non-online feature"):
        replace(offline, availability_rule_id=_digest("illegal-rule"))


def test_feature_contract_rejects_forged_assignment_runtime_types() -> None:
    contract = _contract()
    index = next(
        index
        for index, assignment in enumerate(contract.assignments)
        if assignment.classification == FeatureClassification.ONLINE_ALLOWED
    )
    original = contract.assignments[index]
    forged = object.__new__(FeatureFieldAssignment)
    for field in fields(FeatureFieldAssignment):
        object.__setattr__(forged, field.name, getattr(original, field.name))
    object.__setattr__(forged, "classification", original.classification.value)
    changed = list(contract.assignments)
    changed[index] = forged

    with pytest.raises(TraceValidationError, match="wrong enum type"):
        replace(contract, assignments=tuple(changed))


def test_availability_rule_catalog_is_closed_and_content_addressed() -> None:
    catalog = build_field_path_catalog()
    rules = build_feature_availability_rule_catalog(catalog)

    assert len(rules.rules) == 9
    assert all(
        rule.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and rule.availability_kind == FeatureAvailabilityKind.WINDOWED_EVENT
        and rule.receipt_kind == FeatureAvailabilityReceipt.LIFECYCLE_CUTOFF_PREFIX
        for rule in rules.rules
    )
    validate_feature_availability_rule_catalog(rules)
    with pytest.raises(TraceValidationError, match="frozen online allowlist"):
        replace(rules, rules=rules.rules[:-1])
    with pytest.raises(TraceValidationError, match="rule ID differs"):
        replace(rules.rules[0], rule_id="0" * 64)

    original = rules.rules[0]
    changed_identity = replace(
        feature_module._availability_rule_identity(original),
        source_schema_descriptor_digest="0" * 64,
    )
    rebound = replace(
        original,
        source_schema_descriptor_digest="0" * 64,
        rule_id=canonical_digest(changed_identity),
    )
    with pytest.raises(TraceValidationError, match="another source schema"):
        replace(rules, rules=(rebound, *rules.rules[1:]))


def test_feature_contract_rejects_catalog_allowlist_and_profile_drift() -> None:
    contract = _contract()

    with pytest.raises(TraceValidationError, match="online allowlist differs"):
        replace(contract, online_allowlist_digest="0" * 64)
    with pytest.raises(TraceValidationError, match="catalog digest differs"):
        replace(contract, availability_rule_catalog_digest="0" * 64)
    with pytest.raises(TraceValidationError, match="classification profile differs"):
        replace(contract, classification_profile_digest="0" * 64)


def test_feature_contract_rejects_nononline_classification_swaps() -> None:
    contract = _contract()
    changed = list(contract.assignments)
    selected = {
        classification: next(
            index
            for index, assignment in enumerate(changed)
            if assignment.classification == classification
        )
        for classification in (
            FeatureClassification.LABEL_ONLY,
            FeatureClassification.PROVENANCE_ONLY,
            FeatureClassification.FORBIDDEN_PROXY,
        )
    }
    changed[selected[FeatureClassification.LABEL_ONLY]] = replace(
        changed[selected[FeatureClassification.LABEL_ONLY]],
        classification=FeatureClassification.PROVENANCE_ONLY,
    )
    changed[selected[FeatureClassification.PROVENANCE_ONLY]] = replace(
        changed[selected[FeatureClassification.PROVENANCE_ONLY]],
        classification=FeatureClassification.FORBIDDEN_PROXY,
    )
    changed[selected[FeatureClassification.FORBIDDEN_PROXY]] = replace(
        changed[selected[FeatureClassification.FORBIDDEN_PROXY]],
        classification=FeatureClassification.LABEL_ONLY,
    )

    with pytest.raises(TraceValidationError, match="classification profile"):
        replace(contract, assignments=tuple(changed))


def test_feature_contract_rejects_embedded_catalog_omission() -> None:
    contract = _contract()
    catalog = contract.field_path_catalog
    omitted = replace(catalog, paths=catalog.paths[:-1])

    with pytest.raises(TraceValidationError, match="differs from current schemas"):
        replace(
            contract,
            field_path_catalog=omitted,
            field_path_catalog_digest=canonical_digest(omitted),
            assignments=contract.assignments[:-1],
        )


@pytest.mark.parametrize("kind", ("catalog", "rules", "contract"))
def test_feature_artifact_create_only_round_trip(tmp_path: Path, kind: str) -> None:
    if kind == "catalog":
        artifact = build_field_path_catalog()
        writer = write_field_path_catalog
        loader = load_field_path_catalog
    elif kind == "rules":
        artifact = build_feature_availability_rule_catalog(build_field_path_catalog())
        writer = write_feature_availability_rule_catalog
        loader = load_feature_availability_rule_catalog
    else:
        artifact = _contract()
        writer = write_feature_contract
        loader = load_feature_contract
    path = tmp_path / f"{kind}.json"

    digest = writer(path, artifact)
    loaded = loader(path)

    assert loaded.artifact == artifact
    assert loaded.digest == digest == canonical_digest(artifact)
    assert loaded.size_bytes == len(canonical_json(artifact))
    with pytest.raises(TraceValidationError, match="create-only"):
        writer(path, artifact)


def test_racing_feature_writers_publish_one_exact_artifact(tmp_path: Path) -> None:
    catalog = build_field_path_catalog()
    path = tmp_path / "catalog.json"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(write_field_path_catalog, path, catalog) for _ in range(2)
        )
    results: list[str] = []
    failures: list[BaseException] = []
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as exc:
            failures.append(exc)

    assert results == [canonical_digest(catalog)]
    assert len(failures) == 1
    assert isinstance(failures[0], TraceValidationError)
    assert "create-only" in str(failures[0])
    assert load_field_path_catalog(path).artifact == catalog


def test_forged_omitted_catalog_rejected_before_write(tmp_path: Path) -> None:
    catalog = build_field_path_catalog()
    forged = object.__new__(FieldPathCatalog)
    for field in fields(FieldPathCatalog):
        object.__setattr__(forged, field.name, getattr(catalog, field.name))
    object.__setattr__(forged, "paths", catalog.paths[:-1])
    path = tmp_path / "forged.json"

    with pytest.raises(TraceValidationError, match="differs from current schemas"):
        write_field_path_catalog(path, forged)
    assert not path.exists()


def test_semantic_catalog_omission_rejected_during_load(tmp_path: Path) -> None:
    catalog = build_field_path_catalog()
    payload = json.loads(canonical_json(catalog))
    payload["paths"] = payload["paths"][:-1]
    path = tmp_path / "omitted.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(TraceValidationError, match="differs from current schemas"):
        load_field_path_catalog(path)


def test_feature_writer_detects_equal_length_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.json"
    catalog = build_field_path_catalog()
    real_pread = feature_module.os.pread
    tampered = False

    def tamper(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal tampered
        observed = real_pread(descriptor, size, offset)
        if not tampered:
            os.pwrite(descriptor, b"X" * len(observed), offset)
            tampered = True
        return observed

    monkeypatch.setattr(feature_module.os, "pread", tamper)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_field_path_catalog(path, catalog)
    assert tampered


def test_feature_writer_rejects_parent_rebind_during_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "writer-parent"
    parent.mkdir()
    detached = tmp_path / "writer-parent-detached"
    target = parent.stat()
    real_close = feature_module.os.close
    rebound = False

    def close_and_rebind(descriptor: int) -> None:
        nonlocal rebound
        opened = os.fstat(descriptor)
        target_parent = (opened.st_dev, opened.st_ino) == (
            target.st_dev,
            target.st_ino,
        )
        real_close(descriptor)
        if target_parent and not rebound:
            parent.rename(detached)
            parent.mkdir()
            rebound = True

    monkeypatch.setattr(feature_module.os, "close", close_and_rebind)

    with pytest.raises(TraceCommitIndeterminateError, match="indeterminate"):
        write_field_path_catalog(parent / "catalog.json", build_field_path_catalog())
    assert rebound


def test_feature_loader_rejects_parent_rebind_during_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "loader-parent"
    parent.mkdir()
    path = parent / "catalog.json"
    write_field_path_catalog(path, build_field_path_catalog())
    detached = tmp_path / "loader-parent-detached"
    target = parent.stat()
    real_close = feature_module.os.close
    rebound = False

    def close_and_rebind(descriptor: int) -> None:
        nonlocal rebound
        opened = os.fstat(descriptor)
        target_parent = (opened.st_dev, opened.st_ino) == (
            target.st_dev,
            target.st_ino,
        )
        real_close(descriptor)
        if target_parent and not rebound:
            parent.rename(detached)
            parent.mkdir()
            rebound = True

    monkeypatch.setattr(feature_module.os, "close", close_and_rebind)

    with pytest.raises(TraceValidationError, match="cannot read"):
        load_field_path_catalog(path)
    assert rebound


def test_loader_rejects_noncanonical_symlink_and_hardlink(tmp_path: Path) -> None:
    catalog = build_field_path_catalog()
    newline = tmp_path / "newline.json"
    newline.write_bytes(canonical_json(catalog) + b"\n")
    with pytest.raises(TraceValidationError, match="framing is not canonical"):
        load_field_path_catalog(newline)

    original = tmp_path / "original.json"
    write_field_path_catalog(original, catalog)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(original)
    with pytest.raises(TraceValidationError, match="input identity is invalid"):
        load_field_path_catalog(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(TraceValidationError, match="input identity is invalid"):
        load_field_path_catalog(original)


def test_feature_paths_must_be_absolute(tmp_path: Path) -> None:
    catalog = build_field_path_catalog()

    with pytest.raises(TraceValidationError, match="path must be absolute"):
        write_field_path_catalog(Path("relative.json"), catalog)
    with pytest.raises(TraceValidationError, match="path must be absolute"):
        load_field_path_catalog(Path("relative.json"))


def test_public_feature_schema_versions() -> None:
    assert AVAILABILITY_RULE_CATALOG_SCHEMA_VERSION == (
        "dagkv.m3.feature_availability_rule_catalog.v1"
    )
    assert FIELD_PATH_CATALOG_SCHEMA_VERSION == "dagkv.m3.field_path_catalog.v1"
    assert FEATURE_CLASSIFICATION_PROFILE_SCHEMA_VERSION == (
        "dagkv.m3.feature_classification_profile.v1"
    )
    assert FEATURE_CONTRACT_SCHEMA_VERSION == "dagkv.m3.feature_contract.v1"


def test_top_level_package_exports_feature_contract() -> None:
    assert dagkv.FieldPathCatalog is FieldPathCatalog
    assert dagkv.FeatureAvailabilityRuleCatalog is FeatureAvailabilityRuleCatalog
    assert dagkv.FeatureContract is FeatureContract
    assert dagkv.build_field_path_catalog is build_field_path_catalog
    assert dagkv.build_feature_availability_rule_catalog is (
        build_feature_availability_rule_catalog
    )
    assert dagkv.build_feature_classification_assignments is (
        build_feature_classification_assignments
    )
    assert dagkv.build_feature_contract is build_feature_contract
    assert dagkv.validate_feature_availability_rule_catalog is (
        validate_feature_availability_rule_catalog
    )
    assert dagkv.validate_field_path_catalog is validate_field_path_catalog
