"""Focused tests for the closed C1-B1 derivation language."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from functools import cache
from hashlib import sha256
from pathlib import Path

import pytest

import dagkv
import dagkv.c1_derivation as derivation_module
from dagkv.c1_derivation import (
    DERIVATION_REGISTRY_SCHEMA_VERSION,
    CanonicalFeatureValue,
    DependencySelectorKind,
    DerivationDependencySelector,
    DerivationDependencySlot,
    DerivationEntry,
    DerivationOpcode,
    DerivationParameters,
    DerivationRegistry,
    DerivationSafetyCeiling,
    DerivedFeatureValue,
    SourceFeatureValue,
    absent_feature_value,
    bool_feature_value,
    build_derivation_entry,
    build_derivation_registry,
    build_derived_feature_graph,
    build_source_feature_value,
    derivation_exact_one_selector,
    enum_feature_value,
    evaluate_derivation,
    int_feature_value,
    load_derivation_registry,
    source_exact_all_selector,
    text_feature_value,
    validate_derivation_registry,
    validate_derived_feature_graph,
    write_derivation_registry,
)
from dagkv.c1_features import (
    FeatureAvailabilityKind,
    FeatureContract,
    FeatureSourceSchema,
    FeatureValueKind,
    build_feature_availability_rule_catalog,
    build_feature_classification_assignments,
    build_feature_contract,
    build_field_path_catalog,
)
from dagkv.c1_trace import TraceValidationError, canonical_digest, canonical_json
from dagkv.domain import IdentityError

_LIFECYCLE_PREFIX = (
    "lifecycle_sidecar.ClosedLifecycleArtifact.events[*].LifecycleEvent."
)


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _path(leaf: str) -> str:
    return f"{_LIFECYCLE_PREFIX}{leaf}"


@cache
def _contract() -> FeatureContract:
    catalog = build_field_path_catalog()
    rules = build_feature_availability_rule_catalog(catalog)
    assignments = build_feature_classification_assignments(catalog, rules)
    return build_feature_contract(
        catalog,
        assignments,
        availability_rule_catalog=rules,
    )


def _rule_id(path: str) -> str:
    return next(
        rule.rule_id
        for rule in _contract().availability_rule_catalog.rules
        if rule.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and rule.field_path == path
    )


def _path_kinds(path: str) -> tuple[FeatureValueKind, ...]:
    return next(
        entry.value_kinds
        for entry in _contract().field_path_catalog.paths
        if entry.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and entry.field_path == path
    )


def _source_slot(slot_id: str, path: str) -> DerivationDependencySlot:
    return DerivationDependencySlot(
        slot_id=slot_id,
        selector=source_exact_all_selector(
            FeatureSourceSchema.LIFECYCLE_SIDECAR,
            path,
            _rule_id(path),
        ),
        allowed_kinds=_path_kinds(path),
    )


def _derived_slot(
    slot_id: str,
    entry: DerivationEntry,
) -> DerivationDependencySlot:
    return DerivationDependencySlot(
        slot_id=slot_id,
        selector=derivation_exact_one_selector(entry.derivation_id),
        allowed_kinds=(entry.output_kind,),
    )


@cache
def _entries() -> tuple[DerivationEntry, ...]:
    count = build_derivation_entry(
        derivation_key="01_count_actions",
        opcode=DerivationOpcode.COUNT,
        dependency_slots=(_source_slot("actions", _path("action")),),
        output_kind=FeatureValueKind.INT,
    )
    summed = build_derivation_entry(
        derivation_key="02_sum_byte_count",
        opcode=DerivationOpcode.SUM_INT,
        dependency_slots=(_source_slot("bytes", _path("byte_count")),),
        output_kind=FeatureValueKind.INT,
    )
    minimum = build_derivation_entry(
        derivation_key="03_min_observed",
        opcode=DerivationOpcode.MIN_INT,
        dependency_slots=(_source_slot("observed", _path("observed_byte_count")),),
        output_kind=FeatureValueKind.INT,
    )
    maximum = build_derivation_entry(
        derivation_key="04_max_payload",
        opcode=DerivationOpcode.MAX_INT,
        dependency_slots=(_source_slot("payloads", _path("payload_size")),),
        output_kind=FeatureValueKind.INT,
    )
    subtracted = build_derivation_entry(
        derivation_key="05_sub_span",
        opcode=DerivationOpcode.SUB_INT,
        dependency_slots=(
            _derived_slot("maximum", maximum),
            _derived_slot("minimum", minimum),
        ),
        output_kind=FeatureValueKind.INT,
    )
    equal = build_derivation_entry(
        derivation_key="06_equal_count_sum",
        opcode=DerivationOpcode.EQUAL,
        dependency_slots=(
            _derived_slot("count", count),
            _derived_slot("sum", summed),
        ),
        output_kind=FeatureValueKind.BOOL,
    )
    all_true = build_derivation_entry(
        derivation_key="07_bool_all",
        opcode=DerivationOpcode.BOOL_ALL,
        dependency_slots=(_derived_slot("equality", equal),),
        output_kind=FeatureValueKind.BOOL,
    )
    any_true = build_derivation_entry(
        derivation_key="08_bool_any",
        opcode=DerivationOpcode.BOOL_ANY,
        dependency_slots=(_derived_slot("all", all_true),),
        output_kind=FeatureValueKind.BOOL,
    )
    clamped = build_derivation_entry(
        derivation_key="09_clamp_span",
        opcode=DerivationOpcode.CLAMP_INT,
        dependency_slots=(_derived_slot("span", subtracted),),
        output_kind=FeatureValueKind.INT,
        parameters=DerivationParameters(clamp_low=0, clamp_high=10),
    )
    bucketed = build_derivation_entry(
        derivation_key="10_bucket_span",
        opcode=DerivationOpcode.RIGHT_CLOSED_BUCKET_INT,
        dependency_slots=(_derived_slot("clamped", clamped),),
        output_kind=FeatureValueKind.INT,
        parameters=DerivationParameters(bucket_edges=(0, 10)),
    )
    identity = build_derivation_entry(
        derivation_key="11_identity_bucket",
        opcode=DerivationOpcode.IDENTITY,
        dependency_slots=(_derived_slot("bucket", bucketed),),
        output_kind=FeatureValueKind.INT,
    )
    return (
        count,
        summed,
        minimum,
        maximum,
        subtracted,
        equal,
        all_true,
        any_true,
        clamped,
        bucketed,
        identity,
    )


@cache
def _registry() -> DerivationRegistry:
    return build_derivation_registry(_contract(), _entries())


def _entry(opcode: DerivationOpcode) -> DerivationEntry:
    return next(entry for entry in _entries() if entry.opcode == opcode)


def _typed_equal_entry(kind: FeatureValueKind) -> DerivationEntry:
    return build_derivation_entry(
        derivation_key=f"equal_{kind.value.lower()}",
        opcode=DerivationOpcode.EQUAL,
        dependency_slots=(
            DerivationDependencySlot(
                slot_id="left",
                selector=derivation_exact_one_selector(_digest("equal-left")),
                allowed_kinds=(kind,),
            ),
            DerivationDependencySlot(
                slot_id="right",
                selector=derivation_exact_one_selector(_digest("equal-right")),
                allowed_kinds=(kind,),
            ),
        ),
        output_kind=FeatureValueKind.BOOL,
    )


def _sequence_rule_id() -> str:
    return next(
        rule.rule_id
        for rule in _contract().field_path_catalog.sequence_identity_rules
        if rule.source_schema == FeatureSourceSchema.LIFECYCLE_SIDECAR
        and rule.collection_path == "lifecycle_sidecar.ClosedLifecycleArtifact.events"
    )


def _source_value(
    *,
    event_label: str,
    path: str,
    value: CanonicalFeatureValue,
    event_ns: int,
    source_artifact_id: str | None = None,
    feature_contract_digest: str | None = None,
    availability_rule_id: str | None = None,
) -> SourceFeatureValue:
    return build_source_feature_value(
        feature_contract_digest=(
            feature_contract_digest or canonical_digest(_contract())
        ),
        source_schema=FeatureSourceSchema.LIFECYCLE_SIDECAR,
        field_path=path,
        source_artifact_id=source_artifact_id or _digest("prefix-content"),
        source_record_ref=f"event-{event_label}",
        source_record_digest=_digest(f"event-record-{event_label}"),
        element_identity=_digest(f"element-{event_label}"),
        availability_kind=FeatureAvailabilityKind.WINDOWED_EVENT,
        availability_rule_id=availability_rule_id or _rule_id(path),
        sequence_rule_id=_sequence_rule_id(),
        committed_ns=100,
        event_ns=event_ns,
        lookback_start_ns=80,
        value=value,
    )


@cache
def _source_values() -> tuple[SourceFeatureValue, ...]:
    values = (
        _source_value(
            event_label="a",
            path=_path("action"),
            value=enum_feature_value("dagkv.domain.LedgerAction", "ALLOCATE"),
            event_ns=90,
        ),
        _source_value(
            event_label="a",
            path=_path("byte_count"),
            value=int_feature_value(3),
            event_ns=90,
        ),
        _source_value(
            event_label="a",
            path=_path("observed_byte_count"),
            value=int_feature_value(2),
            event_ns=90,
        ),
        _source_value(
            event_label="a",
            path=_path("payload_size"),
            value=int_feature_value(8),
            event_ns=90,
        ),
        _source_value(
            event_label="b",
            path=_path("action"),
            value=enum_feature_value("dagkv.domain.LedgerAction", "LOAD"),
            event_ns=95,
        ),
        _source_value(
            event_label="b",
            path=_path("byte_count"),
            value=int_feature_value(9),
            event_ns=95,
        ),
        _source_value(
            event_label="b",
            path=_path("observed_byte_count"),
            value=int_feature_value(4),
            event_ns=95,
        ),
        _source_value(
            event_label="b",
            path=_path("payload_size"),
            value=int_feature_value(20),
            event_ns=95,
        ),
    )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.source_schema.value,
                item.field_path,
                item.element_identity,
                item.value_id,
            ),
        )
    )


def test_registry_uses_the_exact_closed_opcode_and_selector_sets() -> None:
    registry = _registry()

    assert tuple(DerivationOpcode) == (
        DerivationOpcode.IDENTITY,
        DerivationOpcode.COUNT,
        DerivationOpcode.SUM_INT,
        DerivationOpcode.MIN_INT,
        DerivationOpcode.MAX_INT,
        DerivationOpcode.SUB_INT,
        DerivationOpcode.BOOL_ALL,
        DerivationOpcode.BOOL_ANY,
        DerivationOpcode.EQUAL,
        DerivationOpcode.CLAMP_INT,
        DerivationOpcode.RIGHT_CLOSED_BUCKET_INT,
    )
    assert tuple(DependencySelectorKind) == (
        DependencySelectorKind.SOURCE_EXACT_ALL,
        DependencySelectorKind.DERIVATION_EXACT_ONE,
    )
    assert {entry.opcode for entry in registry.entries} == set(DerivationOpcode)
    assert len({entry.derivation_id for entry in registry.entries}) == 11
    assert registry.feature_contract == _contract()
    assert registry.safety_ceiling.max_expanded_edges == 65_536
    validate_derivation_registry(registry)
    with pytest.raises(TraceValidationError, match="safety ceiling differs"):
        replace(
            registry,
            safety_ceiling=replace(
                registry.safety_ceiling,
                max_expanded_edges=registry.safety_ceiling.max_expanded_edges + 1,
            ),
        )


def test_canonical_values_preserve_exact_scalar_types() -> None:
    values = (
        absent_feature_value(),
        bool_feature_value(False),
        enum_feature_value("dagkv.domain.LedgerStatus", "SUCCEEDED"),
        int_feature_value(0),
        text_feature_value(""),
    )

    assert tuple(value.kind for value in values) == tuple(FeatureValueKind)
    assert len({canonical_digest(value) for value in values}) == len(values)
    with pytest.raises(TraceValidationError, match="wrong type"):
        int_feature_value(True)  # type: ignore[arg-type]
    with pytest.raises(TraceValidationError, match="inactive payload"):
        replace(int_feature_value(1), text_value="1")
    with pytest.raises(IdentityError, match="enum type"):
        enum_feature_value("", "SUCCEEDED")


def test_integer_and_count_opcodes_use_exact_unbounded_arithmetic() -> None:
    huge = 10**200

    assert evaluate_derivation(
        _entry(DerivationOpcode.IDENTITY),
        (int_feature_value(7),),
    ) == int_feature_value(7)
    assert evaluate_derivation(
        _entry(DerivationOpcode.COUNT),
        (),
    ) == int_feature_value(0)
    assert evaluate_derivation(
        _entry(DerivationOpcode.COUNT),
        (absent_feature_value(), bool_feature_value(True)),
    ) == int_feature_value(2)
    assert evaluate_derivation(
        _entry(DerivationOpcode.SUM_INT),
        (int_feature_value(huge), int_feature_value(huge)),
    ) == int_feature_value(2 * huge)
    assert evaluate_derivation(
        _entry(DerivationOpcode.MIN_INT),
        (int_feature_value(3), int_feature_value(-4), int_feature_value(8)),
    ) == int_feature_value(-4)
    assert evaluate_derivation(
        _entry(DerivationOpcode.MAX_INT),
        (int_feature_value(3), int_feature_value(-4), int_feature_value(8)),
    ) == int_feature_value(8)
    assert evaluate_derivation(
        _entry(DerivationOpcode.SUB_INT),
        (int_feature_value(-3), int_feature_value(8)),
    ) == int_feature_value(-11)


def test_boolean_equality_clamp_and_bucket_semantics() -> None:
    assert evaluate_derivation(
        _entry(DerivationOpcode.BOOL_ALL),
        (bool_feature_value(True), bool_feature_value(False)),
    ) == bool_feature_value(False)
    assert evaluate_derivation(
        _entry(DerivationOpcode.BOOL_ANY),
        (bool_feature_value(False), bool_feature_value(True)),
    ) == bool_feature_value(True)
    assert evaluate_derivation(
        _typed_equal_entry(FeatureValueKind.TEXT),
        (text_feature_value("x"), text_feature_value("x")),
    ) == bool_feature_value(True)
    with pytest.raises(TraceValidationError, match="kind differs"):
        evaluate_derivation(
            _entry(DerivationOpcode.EQUAL),
            (int_feature_value(1), bool_feature_value(True)),
        )
    with pytest.raises(TraceValidationError, match="enum.*different types"):
        evaluate_derivation(
            _typed_equal_entry(FeatureValueKind.ENUM),
            (enum_feature_value("A", "X"), enum_feature_value("B", "X")),
        )

    clamp = _entry(DerivationOpcode.CLAMP_INT)
    assert evaluate_derivation(clamp, (int_feature_value(-1),)) == int_feature_value(0)
    assert evaluate_derivation(clamp, (int_feature_value(7),)) == int_feature_value(7)
    assert evaluate_derivation(clamp, (int_feature_value(12),)) == int_feature_value(10)
    bucket = _entry(DerivationOpcode.RIGHT_CLOSED_BUCKET_INT)
    expected = ((-1, 0), (0, 0), (1, 1), (10, 1), (11, 2))
    assert (
        tuple(
            (value, evaluate_derivation(bucket, (int_feature_value(value),)).int_value)
            for value, _ in expected
        )
        == expected
    )


def test_derivations_reject_wrong_arity_kind_and_parameters() -> None:
    with pytest.raises(TraceValidationError, match="count differs"):
        evaluate_derivation(_entry(DerivationOpcode.MIN_INT), ())
    with pytest.raises(TraceValidationError, match="count differs"):
        evaluate_derivation(
            _entry(DerivationOpcode.SUB_INT),
            (int_feature_value(1),),
        )
    with pytest.raises(TraceValidationError, match="kind differs"):
        evaluate_derivation(
            _entry(DerivationOpcode.SUM_INT),
            (bool_feature_value(True),),
        )
    with pytest.raises(TraceValidationError, match="parameters are invalid"):
        build_derivation_entry(
            derivation_key="invalid_clamp",
            opcode=DerivationOpcode.CLAMP_INT,
            dependency_slots=(
                _derived_slot("value", _entry(DerivationOpcode.SUM_INT)),
            ),
            output_kind=FeatureValueKind.INT,
            parameters=DerivationParameters(clamp_low=2, clamp_high=1),
        )
    with pytest.raises(TraceValidationError, match="sorted and unique"):
        DerivationParameters(bucket_edges=(1, 1))
    with pytest.raises(TraceValidationError, match="IDENTITY signature differs"):
        build_derivation_entry(
            derivation_key="raw_identity",
            opcode=DerivationOpcode.IDENTITY,
            dependency_slots=(_source_slot("raw", _path("byte_count")),),
            output_kind=FeatureValueKind.INT,
        )
    with pytest.raises(TraceValidationError, match="invalid parameters"):
        build_derivation_entry(
            derivation_key="invalid_parameter_type",
            opcode=DerivationOpcode.COUNT,
            dependency_slots=(),
            output_kind=FeatureValueKind.INT,
            parameters=0,  # type: ignore[arg-type]
        )
    with pytest.raises(TraceValidationError, match="byte limit"):
        build_derivation_entry(
            derivation_key="x" * 4_097,
            opcode=DerivationOpcode.COUNT,
            dependency_slots=(),
            output_kind=FeatureValueKind.INT,
        )
    with pytest.raises(TraceValidationError, match="bucket-edge ceiling"):
        DerivationParameters(bucket_edges=tuple(range(4_097)))


def test_registry_rejects_source_selector_contract_and_type_drift() -> None:
    action_path = _path("action")
    wrong_rule_slot = DerivationDependencySlot(
        slot_id="actions",
        selector=source_exact_all_selector(
            FeatureSourceSchema.LIFECYCLE_SIDECAR,
            action_path,
            _digest("wrong-rule"),
        ),
        allowed_kinds=_path_kinds(action_path),
    )
    wrong_rule = build_derivation_entry(
        derivation_key="wrong_rule",
        opcode=DerivationOpcode.COUNT,
        dependency_slots=(wrong_rule_slot,),
        output_kind=FeatureValueKind.INT,
    )
    with pytest.raises(TraceValidationError, match="exact online contract"):
        build_derivation_registry(_contract(), (wrong_rule,))

    wrong_type_slot = replace(
        _source_slot("actions", action_path),
        allowed_kinds=(FeatureValueKind.INT,),
    )
    wrong_type = build_derivation_entry(
        derivation_key="wrong_type",
        opcode=DerivationOpcode.COUNT,
        dependency_slots=(wrong_type_slot,),
        output_kind=FeatureValueKind.INT,
    )
    with pytest.raises(TraceValidationError, match="expected types differ"):
        build_derivation_registry(_contract(), (wrong_type,))


def test_registry_rejects_unknown_and_duplicate_derived_selectors() -> None:
    unknown_slot = DerivationDependencySlot(
        slot_id="unknown",
        selector=derivation_exact_one_selector(_digest("unknown-derivation")),
        allowed_kinds=(FeatureValueKind.INT,),
    )
    unknown = build_derivation_entry(
        derivation_key="unknown",
        opcode=DerivationOpcode.IDENTITY,
        dependency_slots=(unknown_slot,),
        output_kind=FeatureValueKind.INT,
    )
    with pytest.raises(TraceValidationError, match="unknown entry"):
        build_derivation_registry(_contract(), (unknown,))

    shared = _derived_slot("left", _entry(DerivationOpcode.SUM_INT))
    with pytest.raises(TraceValidationError, match="selectors are duplicated"):
        build_derivation_entry(
            derivation_key="duplicate",
            opcode=DerivationOpcode.SUB_INT,
            dependency_slots=(shared, replace(shared, slot_id="right")),
            output_kind=FeatureValueKind.INT,
        )
    with pytest.raises(TraceValidationError, match="invalid entry"):
        build_derivation_registry(_contract(), (object(),))  # type: ignore[arg-type]


def test_registry_cycle_detection_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_id = _digest("cycle-first")
    second_id = _digest("cycle-second")
    original_digest = derivation_module.canonical_digest

    def cycle_digest(value):
        if isinstance(value, derivation_module._DerivationIdentity):
            if value.derivation_key == "cycle-a":
                return first_id
            if value.derivation_key == "cycle-b":
                return second_id
        return original_digest(value)

    monkeypatch.setattr(derivation_module, "canonical_digest", cycle_digest)
    first = build_derivation_entry(
        derivation_key="cycle-a",
        opcode=DerivationOpcode.IDENTITY,
        dependency_slots=(
            DerivationDependencySlot(
                slot_id="second",
                selector=derivation_exact_one_selector(second_id),
                allowed_kinds=(FeatureValueKind.INT,),
            ),
        ),
        output_kind=FeatureValueKind.INT,
    )
    second = build_derivation_entry(
        derivation_key="cycle-b",
        opcode=DerivationOpcode.IDENTITY,
        dependency_slots=(
            DerivationDependencySlot(
                slot_id="first",
                selector=derivation_exact_one_selector(first_id),
                allowed_kinds=(FeatureValueKind.INT,),
            ),
        ),
        output_kind=FeatureValueKind.INT,
    )

    with pytest.raises(TraceValidationError, match="contains a cycle"):
        build_derivation_registry(_contract(), (first, second))


def test_deep_registry_and_graph_use_iterative_topological_replay() -> None:
    seed = build_derivation_entry(
        derivation_key="0000_seed",
        opcode=DerivationOpcode.COUNT,
        dependency_slots=(),
        output_kind=FeatureValueKind.INT,
    )
    entries = [seed]
    for index in range(1, 1_102):
        entries.append(
            build_derivation_entry(
                derivation_key=f"{index:04d}_identity",
                opcode=DerivationOpcode.IDENTITY,
                dependency_slots=(_derived_slot("prior", entries[-1]),),
                output_kind=FeatureValueKind.INT,
            )
        )
    registry = build_derivation_registry(_contract(), tuple(entries))

    nodes = build_derived_feature_graph(registry, ())

    assert len(nodes) == 1_102
    assert nodes[-1].output == int_feature_value(0)


def test_expanded_edge_ceiling_is_checked_before_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(derivation_module, "MAX_DERIVATION_EXPANDED_EDGES", 1)
    count = _entry(DerivationOpcode.COUNT)
    registry = build_derivation_registry(_contract(), (count,))
    source_values = tuple(
        value for value in _source_values() if value.field_path == _path("action")
    )

    with pytest.raises(TraceValidationError, match="expanded-edge ceiling"):
        build_derived_feature_graph(registry, source_values)


def test_registry_rejects_identity_contract_and_nested_type_drift() -> None:
    registry = _registry()
    entry = registry.entries[0]

    with pytest.raises(TraceValidationError, match="ID differs"):
        replace(entry, derivation_id="0" * 64)
    with pytest.raises(TraceValidationError, match="contract digest differs"):
        replace(registry, feature_contract_digest="0" * 64)
    with pytest.raises(TraceValidationError, match="key-sorted and unique"):
        replace(registry, entries=tuple(reversed(registry.entries)))

    original_slot = entry.dependency_slots[0]
    forged_slot = object.__new__(DerivationDependencySlot)
    for field in fields(DerivationDependencySlot):
        object.__setattr__(forged_slot, field.name, getattr(original_slot, field.name))
    object.__setattr__(forged_slot, "allowed_kinds", (FeatureValueKind.ENUM.value,))
    forged_entry = object.__new__(type(entry))
    for field in fields(type(entry)):
        object.__setattr__(forged_entry, field.name, getattr(entry, field.name))
    object.__setattr__(forged_entry, "dependency_slots", (forged_slot,))

    with pytest.raises(TraceValidationError, match="invalid value kind"):
        replace(registry, entries=(forged_entry, *registry.entries[1:]))


def test_source_value_identity_uses_the_frozen_protocol_fields() -> None:
    value = _source_values()[0]
    identity = derivation_module._SourceValueIdentity(
        schema_version=value.schema_version,
        feature_contract_digest=value.feature_contract_digest,
        field_path=value.field_path,
        source_artifact_id=value.source_artifact_id,
        source_record_ref=value.source_record_ref,
        element_identity=value.element_identity,
        typed_value_digest=value.value_digest,
        availability_kind=value.availability_kind,
    )

    assert value.value_id == canonical_digest(identity)
    with pytest.raises(TraceValidationError, match="value digest differs"):
        replace(value, value_digest=_digest("forged-value"))
    with pytest.raises(TraceValidationError, match="lacks event_ns"):
        replace(value, event_ns=None)
    with pytest.raises(TraceValidationError, match="predates its lookback"):
        replace(value, event_ns=value.lookback_start_ns - 1)
    with pytest.raises(TraceValidationError, match="follows its commit"):
        replace(value, event_ns=value.committed_ns + 1)


def test_derived_graph_expands_all_raw_matches_and_replays_every_entry() -> None:
    registry = _registry()
    source_values = _source_values()

    nodes = build_derived_feature_graph(registry, source_values)
    replayed = validate_derived_feature_graph(registry, source_values, nodes)

    assert replayed == nodes
    assert tuple(node.derivation_id for node in nodes) == tuple(
        entry.derivation_id for entry in registry.entries
    )
    expected_outputs = (2, 12, 2, 20, 18, False, False, False, 10, 1, 1)
    assert (
        tuple(
            node.output.bool_value
            if node.output.kind == FeatureValueKind.BOOL
            else node.output.int_value
            for node in nodes
        )
        == expected_outputs
    )

    sum_node = nodes[1]
    byte_leaves = tuple(
        sorted(
            (
                value
                for value in source_values
                if value.field_path == _path("byte_count")
            ),
            key=lambda value: (value.element_identity, value.value_id),
        )
    )
    assert sum_node.dependency_value_ids == tuple(
        value.value_id for value in byte_leaves
    )


def test_derived_graph_rejects_subset_reorder_and_resealed_output() -> None:
    registry = _registry()
    source_values = _source_values()
    nodes = build_derived_feature_graph(registry, source_values)

    with pytest.raises(TraceValidationError, match="cover each registry entry"):
        validate_derived_feature_graph(registry, source_values, nodes[:-1])
    with pytest.raises(TraceValidationError, match="cover each registry entry"):
        validate_derived_feature_graph(
            registry,
            source_values,
            (nodes[1], nodes[0], *nodes[2:]),
        )

    sum_node = nodes[1]
    reordered_identity = replace(
        derivation_module._derived_value_identity(sum_node),
        dependency_value_ids=tuple(reversed(sum_node.dependency_value_ids)),
    )
    reordered = replace(
        sum_node,
        dependency_value_ids=reordered_identity.dependency_value_ids,
        value_id=canonical_digest(reordered_identity),
    )
    reordered_nodes = (nodes[0], reordered, *nodes[2:])
    with pytest.raises(TraceValidationError, match="differs during replay"):
        validate_derived_feature_graph(registry, source_values, reordered_nodes)

    last = nodes[-1]
    forged_output = int_feature_value(99)
    forged_identity = replace(
        derivation_module._derived_value_identity(last),
        output=forged_output,
        output_digest=canonical_digest(forged_output),
    )
    resealed = replace(
        last,
        output=forged_output,
        output_digest=forged_identity.output_digest,
        value_id=canonical_digest(forged_identity),
    )
    with pytest.raises(TraceValidationError, match="differs during replay"):
        validate_derived_feature_graph(
            registry,
            source_values,
            (*nodes[:-1], resealed),
        )


def test_derived_graph_rejects_ambiguous_or_noncontract_source_values() -> None:
    registry = _registry()
    source_values = _source_values()

    with pytest.raises(TraceValidationError, match="canonically sorted"):
        build_derived_feature_graph(registry, tuple(reversed(source_values)))

    original = source_values[0]
    duplicate = _source_value(
        event_label="a",
        path=original.field_path,
        value=enum_feature_value("dagkv.domain.LedgerAction", "LOAD"),
        event_ns=90,
    )
    duplicated = tuple(
        sorted(
            (*source_values, duplicate),
            key=lambda item: (
                item.source_schema.value,
                item.field_path,
                item.element_identity,
                item.value_id,
            ),
        )
    )
    with pytest.raises(TraceValidationError, match="path elements are duplicated"):
        build_derived_feature_graph(registry, duplicated)

    wrong_contract = _source_value(
        event_label="x",
        path=_path("byte_count"),
        value=int_feature_value(1),
        event_ns=90,
        feature_contract_digest=_digest("another-contract"),
    )
    with pytest.raises(TraceValidationError, match="binds another contract"):
        build_derived_feature_graph(registry, (wrong_contract,))

    wrong_rule = _source_value(
        event_label="x",
        path=_path("byte_count"),
        value=int_feature_value(1),
        event_ns=90,
        availability_rule_id=_digest("another-rule"),
    )
    with pytest.raises(TraceValidationError, match="outside the online contract"):
        build_derived_feature_graph(registry, (wrong_rule,))

    wrong_sequence = replace(original, sequence_rule_id=_digest("another-sequence"))
    with pytest.raises(TraceValidationError, match="sequence-rule ID differs"):
        build_derived_feature_graph(registry, (wrong_sequence,))


def test_derived_graph_rejects_cross_cutoff_source_envelopes() -> None:
    first = _source_value(
        event_label="a",
        path=_path("byte_count"),
        value=int_feature_value(1),
        event_ns=90,
    )
    second = _source_value(
        event_label="b",
        path=_path("byte_count"),
        value=int_feature_value(2),
        event_ns=95,
        source_artifact_id=_digest("another-prefix"),
    )
    values = tuple(
        sorted(
            (first, second),
            key=lambda item: (
                item.source_schema.value,
                item.field_path,
                item.element_identity,
                item.value_id,
            ),
        )
    )

    with pytest.raises(TraceValidationError, match="cross cutoff envelopes"):
        build_derived_feature_graph(_registry(), values)


def test_empty_count_and_sum_are_explicit_registry_outputs() -> None:
    count = build_derivation_entry(
        derivation_key="count_empty",
        opcode=DerivationOpcode.COUNT,
        dependency_slots=(),
        output_kind=FeatureValueKind.INT,
    )
    summed = build_derivation_entry(
        derivation_key="sum_empty",
        opcode=DerivationOpcode.SUM_INT,
        dependency_slots=(),
        output_kind=FeatureValueKind.INT,
    )
    registry = build_derivation_registry(_contract(), (count, summed))

    nodes = build_derived_feature_graph(registry, ())

    assert tuple(node.output for node in nodes) == (
        int_feature_value(0),
        int_feature_value(0),
    )


def test_nonempty_variadic_opcode_fails_on_an_empty_exact_all_expansion() -> None:
    minimum = build_derivation_entry(
        derivation_key="minimum",
        opcode=DerivationOpcode.MIN_INT,
        dependency_slots=(_source_slot("values", _path("byte_count")),),
        output_kind=FeatureValueKind.INT,
    )
    registry = build_derivation_registry(_contract(), (minimum,))

    with pytest.raises(TraceValidationError, match="count differs"):
        build_derived_feature_graph(registry, ())


def test_derivation_registry_create_only_round_trip(tmp_path: Path) -> None:
    registry = _registry()
    path = (tmp_path / "registry.json").resolve()

    digest = write_derivation_registry(path, registry)
    loaded = load_derivation_registry(path)

    assert loaded.artifact == registry
    assert loaded.digest == digest == canonical_digest(registry)
    assert loaded.size_bytes == len(canonical_json(registry))
    with pytest.raises(TraceValidationError, match="create-only"):
        write_derivation_registry(path, registry)


def test_derivation_registry_load_rejects_nested_selector_tamper(
    tmp_path: Path,
) -> None:
    registry = _registry()
    payload = json.loads(canonical_json(registry))
    payload["entries"][0]["dependency_slots"][0]["selector"]["availability_rule_id"] = (
        "0" * 64
    )
    path = (tmp_path / "tampered.json").resolve()
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="ascii",
    )

    with pytest.raises(TraceValidationError, match="ID differs"):
        load_derivation_registry(path)


def test_public_derivation_exports() -> None:
    assert DERIVATION_REGISTRY_SCHEMA_VERSION == "dagkv.m3.derivation_registry.v1"
    expected = {
        "CanonicalFeatureValue": CanonicalFeatureValue,
        "DependencySelectorKind": DependencySelectorKind,
        "DerivationDependencySelector": DerivationDependencySelector,
        "DerivationDependencySlot": DerivationDependencySlot,
        "DerivationEntry": DerivationEntry,
        "DerivationOpcode": DerivationOpcode,
        "DerivationParameters": DerivationParameters,
        "DerivationRegistry": DerivationRegistry,
        "DerivationSafetyCeiling": DerivationSafetyCeiling,
        "DerivedFeatureValue": DerivedFeatureValue,
        "SourceFeatureValue": SourceFeatureValue,
        "absent_feature_value": absent_feature_value,
        "bool_feature_value": bool_feature_value,
        "build_derivation_entry": build_derivation_entry,
        "build_derivation_registry": build_derivation_registry,
        "build_derived_feature_graph": build_derived_feature_graph,
        "build_source_feature_value": build_source_feature_value,
        "derivation_exact_one_selector": derivation_exact_one_selector,
        "enum_feature_value": enum_feature_value,
        "evaluate_derivation": evaluate_derivation,
        "int_feature_value": int_feature_value,
        "load_derivation_registry": load_derivation_registry,
        "source_exact_all_selector": source_exact_all_selector,
        "text_feature_value": text_feature_value,
        "validate_derivation_registry": validate_derivation_registry,
        "validate_derived_feature_graph": validate_derived_feature_graph,
        "write_derivation_registry": write_derivation_registry,
    }
    assert all(getattr(dagkv, name) is value for name, value in expected.items())
    assert dagkv.validate_feature_contract(_contract()) is None
