"""M3/C1 dependence-aware shared-lease aggregation tests."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256

import pytest

from dagkv.c1_leases import (
    DependenceGroup,
    ForecastSource,
    JointOutcome,
    LeaseOwnerSnapshot,
    LeasePriorityMode,
    ReuseClaim,
    SharedLeaseForecast,
    SharedLeasePolicySnapshot,
    aggregate_shared_lease,
)
from dagkv.domain import (
    BindingHandle,
    BindingKind,
    BindingState,
    BlockKey,
    ExecutionRef,
    IdentityError,
    ReplicaId,
    ResidencyState,
    Tier,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)
from dagkv.orchestrator import LifecycleOrchestrator


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _owner(
    binding_id: str,
    workflow_id: str,
    *nodes: str,
    created_ns: int = 0,
) -> LeaseOwnerSnapshot:
    return LeaseOwnerSnapshot(
        binding_id=binding_id,
        workflow=WorkflowKey(workflow_id, 0),
        created_ns=created_ns,
        eligible_node_ids=tuple(nodes),
    )


def _snapshot(
    block_key: BlockKey,
    *owners: LeaseOwnerSnapshot,
    runtime_event_count: int = 7,
) -> SharedLeasePolicySnapshot:
    return SharedLeasePolicySnapshot(
        block_key=block_key,
        runtime_event_count=runtime_event_count,
        location_version=1,
        residency=ResidencyState.GPU_ONLY,
        owners=tuple(owners),
    )


def _claim(
    claim_id: str,
    owner: LeaseOwnerSnapshot,
    node_id: str,
    epoch_id: str,
    access_ns: int,
) -> ReuseClaim:
    return ReuseClaim(
        claim_id=claim_id,
        binding_id=owner.binding_id,
        workflow=owner.workflow,
        node_id=node_id,
        reuse_epoch_id=epoch_id,
        access_ns=access_ns,
    )


def _group(
    group_id: str,
    *outcomes: JointOutcome,
    radius_ppm: int = 0,
) -> DependenceGroup:
    return DependenceGroup(
        group_id=group_id,
        outcomes=tuple(outcomes),
        total_variation_radius_ppm=radius_ppm,
    )


def _forecast(
    block_key: BlockKey,
    *groups: DependenceGroup,
    runtime_event_count: int = 7,
    generated_ns: int = 0,
    horizon_ns: int = 30,
    source: ForecastSource = ForecastSource.PREDICTED,
) -> SharedLeaseForecast:
    return SharedLeaseForecast(
        forecast_id="forecast-1",
        block_key=block_key,
        runtime_event_count=runtime_event_count,
        generated_ns=generated_ns,
        horizon_ns=horizon_ns,
        source=source,
        predictor_digest=_digest("predictor"),
        dependence_digest=_digest("dependence-model"),
        independence_basis="groups are independent workflow arrivals",
        groups=tuple(groups),
    )


def test_mutually_exclusive_branches_preserve_joint_union(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "branch-a", "branch-b")
    branch_a = _claim("claim-a", owner, "branch-a", "epoch-a", 10)
    branch_b = _claim("claim-b", owner, "branch-b", "epoch-b", 10)
    group = _group(
        "exclusive",
        JointOutcome("a", 400_000, (branch_a,)),
        JointOutcome("b", 400_000, (branch_b,)),
        JointOutcome("neither", 200_000),
    )

    point = aggregate_shared_lease(
        _snapshot(block_key, owner),
        _forecast(block_key, group),
    ).at_deadline(30)

    assert point.first_physical_readmission.nominal == Fraction(4, 5)
    assert point.expected_unique_reuse_epochs.nominal == Fraction(4, 5)
    assert point.expected_repeated_reuse_epochs.nominal == 0
    assert point.additive_claim_score == Fraction(4, 5)
    assert point.independent_marginal_union_score == Fraction(16, 25)


def test_correlated_fanout_counts_one_physical_epoch(
    block_key: BlockKey,
) -> None:
    owner_a = _owner("retention-a", "workflow-a", "fanout-a")
    owner_b = _owner("retention-b", "workflow-b", "fanout-b")
    claims = (
        _claim("claim-a", owner_a, "fanout-a", "shared-epoch", 10),
        _claim("claim-b", owner_b, "fanout-b", "shared-epoch", 10),
    )
    group = _group(
        "correlated-fanout",
        JointOutcome("both", 500_000, claims),
        JointOutcome("neither", 500_000),
    )

    point = aggregate_shared_lease(
        _snapshot(block_key, owner_a, owner_b),
        _forecast(block_key, group),
    ).at_deadline(30)

    assert point.first_physical_readmission.nominal == Fraction(1, 2)
    assert point.expected_unique_reuse_epochs.nominal == Fraction(1, 2)
    assert point.expected_repeated_reuse_epochs.nominal == 0
    assert point.additive_claim_score == 1
    assert point.independent_marginal_union_score == Fraction(3, 4)


def test_repeated_reuses_are_separate_from_first_readmission(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "first", "repeat")
    claims = (
        _claim("claim-first", owner, "first", "epoch-first", 10),
        _claim("claim-repeat", owner, "repeat", "epoch-repeat", 20),
    )
    group = _group(
        "correlated-repeat",
        JointOutcome("both", 500_000, claims),
        JointOutcome("neither", 500_000),
    )

    profile = aggregate_shared_lease(
        _snapshot(block_key, owner),
        _forecast(block_key, group),
    )

    first = profile.at_deadline(10)
    assert first.first_physical_readmission.nominal == Fraction(1, 2)
    assert first.expected_unique_reuse_epochs.nominal == Fraction(1, 2)
    assert first.expected_repeated_reuse_epochs.nominal == 0
    horizon = profile.at_deadline(30)
    assert horizon.first_physical_readmission.nominal == Fraction(1, 2)
    assert horizon.expected_unique_reuse_epochs.nominal == 1
    assert horizon.expected_repeated_reuse_epochs.nominal == Fraction(1, 2)


def test_independent_workflows_use_product_only_between_groups(
    block_key: BlockKey,
) -> None:
    owner_a = _owner("retention-a", "workflow-a", "next-a")
    owner_b = _owner("retention-b", "workflow-b", "next-b")
    group_a = _group(
        "workflow-a",
        JointOutcome(
            "hit-a",
            500_000,
            (_claim("claim-a", owner_a, "next-a", "epoch-a", 10),),
        ),
        JointOutcome("miss-a", 500_000),
    )
    group_b = _group(
        "workflow-b",
        JointOutcome(
            "hit-b",
            500_000,
            (_claim("claim-b", owner_b, "next-b", "epoch-b", 20),),
        ),
        JointOutcome("miss-b", 500_000),
    )

    point = aggregate_shared_lease(
        _snapshot(block_key, owner_a, owner_b),
        _forecast(block_key, group_a, group_b),
    ).at_deadline(30)

    assert point.first_physical_readmission.nominal == Fraction(3, 4)
    assert point.expected_unique_reuse_epochs.nominal == 1
    assert point.expected_repeated_reuse_epochs.nominal == Fraction(1, 4)
    assert point.additive_claim_score == 1
    assert point.independent_marginal_union_score == Fraction(3, 4)


def test_total_variation_radius_produces_exact_sound_bounds(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "next")
    group = _group(
        "drift",
        JointOutcome(
            "hit",
            200_000,
            (_claim("claim", owner, "next", "epoch", 10),),
        ),
        JointOutcome("miss", 800_000),
        radius_ppm=100_000,
    )

    point = aggregate_shared_lease(
        _snapshot(block_key, owner),
        _forecast(block_key, group),
    ).at_deadline(30)

    assert point.first_physical_readmission.lower == Fraction(1, 10)
    assert point.first_physical_readmission.nominal == Fraction(1, 5)
    assert point.first_physical_readmission.upper == Fraction(3, 10)
    assert point.expected_unique_reuse_epochs == point.first_physical_readmission
    assert point.expected_repeated_reuse_epochs.lower == 0
    assert point.expected_repeated_reuse_epochs.upper == Fraction(1, 5)


def test_total_variation_bounds_match_exhaustive_mass_reallocation(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "one", "three-a", "three-b")
    one = (_claim("one", owner, "one", "epoch-one", 10),)
    three = (
        _claim("three-a", owner, "three-a", "epoch-a", 10),
        _claim("three-b", owner, "three-b", "epoch-b", 10),
        _claim("three-c", owner, "three-b", "epoch-c", 10),
    )
    group = _group(
        "enumerated-drift",
        JointOutcome("zero", 300_000),
        JointOutcome("one", 400_000, one),
        JointOutcome("three", 300_000, three),
        radius_ppm=200_000,
    )
    point = aggregate_shared_lease(
        _snapshot(block_key, owner),
        _forecast(block_key, group),
    ).at_deadline(30)

    first_values: list[Fraction] = []
    epoch_values: list[Fraction] = []
    repeat_values: list[Fraction] = []
    nominal = (300_000, 400_000, 300_000)
    for zero_mass in range(0, 1_000_001, 100_000):
        for one_mass in range(0, 1_000_001 - zero_mass, 100_000):
            three_mass = 1_000_000 - zero_mass - one_mass
            candidate = (zero_mass, one_mass, three_mass)
            tv_distance = (
                sum(
                    abs(actual - expected)
                    for actual, expected in zip(candidate, nominal, strict=True)
                )
                // 2
            )
            if tv_distance > 200_000:
                continue
            first = Fraction(one_mass + three_mass, 1_000_000)
            epochs = Fraction(one_mass + 3 * three_mass, 1_000_000)
            first_values.append(first)
            epoch_values.append(epochs)
            repeat_values.append(epochs - first)

    assert point.first_physical_readmission.lower == min(first_values)
    assert point.first_physical_readmission.upper == max(first_values)
    assert point.expected_unique_reuse_epochs.lower == min(epoch_values)
    assert point.expected_unique_reuse_epochs.upper == max(epoch_values)
    assert point.expected_repeated_reuse_epochs.lower <= min(repeat_values)
    assert point.expected_repeated_reuse_epochs.upper >= max(repeat_values)


def test_independent_group_robust_bounds_cover_exhaustive_products(
    block_key: BlockKey,
) -> None:
    owner_a = _owner("retention-a", "workflow-a", "next-a")
    owner_b = _owner("retention-b", "workflow-b", "next-b")
    claim_a = _claim("claim-a", owner_a, "next-a", "epoch-a", 10)
    claim_b = _claim("claim-b", owner_b, "next-b", "epoch-b", 10)
    group_a = _group(
        "a",
        JointOutcome("a-hit", 300_000, (claim_a,)),
        JointOutcome("a-miss", 700_000),
        radius_ppm=100_000,
    )
    group_b = _group(
        "b",
        JointOutcome("b-hit", 600_000, (claim_b,)),
        JointOutcome("b-miss", 400_000),
        radius_ppm=200_000,
    )
    point = aggregate_shared_lease(
        _snapshot(block_key, owner_a, owner_b),
        _forecast(block_key, group_a, group_b),
    ).at_deadline(30)

    first_values: list[Fraction] = []
    epoch_values: list[Fraction] = []
    repeat_values: list[Fraction] = []
    for hit_a in range(200_000, 400_001, 100_000):
        for hit_b in range(400_000, 800_001, 100_000):
            first = 1 - Fraction(1_000_000 - hit_a, 1_000_000) * Fraction(
                1_000_000 - hit_b,
                1_000_000,
            )
            epochs = Fraction(hit_a + hit_b, 1_000_000)
            first_values.append(first)
            epoch_values.append(epochs)
            repeat_values.append(epochs - first)

    assert point.first_physical_readmission.lower == min(first_values)
    assert point.first_physical_readmission.upper == max(first_values)
    assert point.expected_unique_reuse_epochs.lower == min(epoch_values)
    assert point.expected_unique_reuse_epochs.upper == max(epoch_values)
    assert point.expected_repeated_reuse_epochs.lower <= min(repeat_values)
    assert point.expected_repeated_reuse_epochs.upper >= max(repeat_values)


def test_priority_modes_are_independently_selectable(block_key: BlockKey) -> None:
    owner_a = _owner("retention-a", "workflow-a", "fanout-a")
    owner_b = _owner("retention-b", "workflow-b", "fanout-b")
    claims = (
        _claim("claim-a", owner_a, "fanout-a", "shared", 10),
        _claim("claim-b", owner_b, "fanout-b", "shared", 10),
    )
    group = _group(
        "fanout",
        JointOutcome("both", 500_000, claims),
        JointOutcome("none", 500_000),
        radius_ppm=100_000,
    )
    point = aggregate_shared_lease(
        _snapshot(block_key, owner_a, owner_b),
        _forecast(block_key, group),
    ).at_deadline(30)

    assert point.priority(LeasePriorityMode.C1_NOMINAL) == Fraction(1, 2)
    assert point.priority(LeasePriorityMode.C1_ROBUST_LOWER) == Fraction(2, 5)
    assert point.priority(LeasePriorityMode.PBKV_STYLE_ADDITIVE) == 1
    assert point.priority(LeasePriorityMode.INDEPENDENT_MARGINAL_UNION) == Fraction(
        3, 4
    )


def test_oracle_forecast_is_excluded_from_online_scoring(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "next")
    group = _group(
        "oracle",
        JointOutcome(
            "hit",
            1_000_000,
            (_claim("claim", owner, "next", "epoch", 10),),
        ),
    )
    snapshot = _snapshot(block_key, owner)
    forecast = _forecast(block_key, group, source=ForecastSource.ORACLE)

    with pytest.raises(IdentityError, match="oracle forecast"):
        aggregate_shared_lease(snapshot, forecast)
    profile = aggregate_shared_lease(snapshot, forecast, allow_oracle=True)
    assert profile.at_deadline(30).first_physical_readmission.nominal == 1


def test_forecast_rejects_stale_or_cross_owner_scope(block_key: BlockKey) -> None:
    owner = _owner("retention-a", "workflow-a", "next")
    group = _group(
        "scope",
        JointOutcome(
            "hit",
            1_000_000,
            (_claim("claim", owner, "next", "epoch", 10),),
        ),
    )
    forecast = _forecast(block_key, group)

    with pytest.raises(IdentityError, match="stale"):
        aggregate_shared_lease(
            _snapshot(block_key, owner, runtime_event_count=8),
            forecast,
        )
    with pytest.raises(IdentityError, match="active retention owner"):
        aggregate_shared_lease(_snapshot(block_key), forecast)

    wrong_owner = replace(owner, workflow=WorkflowKey("workflow-b", 0))
    with pytest.raises(IdentityError, match="workflow ownership"):
        aggregate_shared_lease(_snapshot(block_key, wrong_owner), forecast)


def test_forecast_rejects_ineligible_node_and_predated_owner(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "other", created_ns=5)
    claim = ReuseClaim(
        claim_id="claim",
        binding_id=owner.binding_id,
        workflow=owner.workflow,
        node_id="next",
        reuse_epoch_id="epoch",
        access_ns=10,
    )
    group = _group("scope", JointOutcome("hit", 1_000_000, (claim,)))

    with pytest.raises(IdentityError, match="ineligible DAG node"):
        aggregate_shared_lease(
            _snapshot(block_key, owner),
            _forecast(block_key, group, generated_ns=5),
        )
    with pytest.raises(IdentityError, match="predates its retention owner"):
        aggregate_shared_lease(
            _snapshot(block_key, replace(owner, eligible_node_ids=("next",))),
            _forecast(block_key, group, generated_ns=4),
        )


def test_probability_and_dependence_identities_fail_closed(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "next", "other")
    claim = _claim("claim", owner, "next", "epoch", 10)

    with pytest.raises(IdentityError, match="sum to 1,000,000"):
        _group(
            "bad-mass",
            JointOutcome("hit", 500_000, (claim,)),
            JointOutcome("miss", 499_999),
        )

    changed = replace(claim, node_id="other")
    with pytest.raises(IdentityError, match="identity changes"):
        _group(
            "identity-drift",
            JointOutcome("first", 500_000, (claim,)),
            JointOutcome("second", 500_000, (changed,)),
        )

    conflicting_time = replace(claim, claim_id="claim-2", access_ns=11)
    with pytest.raises(IdentityError, match="conflicting access timestamps"):
        JointOutcome("bad-epoch", 1_000_000, (claim, conflicting_time))


def test_claim_or_epoch_cannot_cross_independent_groups(
    block_key: BlockKey,
) -> None:
    owner = _owner("retention-a", "workflow-a", "next")
    claim = _claim("claim", owner, "next", "epoch", 10)
    first = _group("first", JointOutcome("hit-a", 1_000_000, (claim,)))
    second = _group("second", JointOutcome("hit-b", 1_000_000, (claim,)))

    with pytest.raises(IdentityError, match="claim cannot span"):
        _forecast(block_key, first, second)

    changed_id = replace(claim, claim_id="claim-2")
    second = _group("second", JointOutcome("hit-b", 1_000_000, (changed_id,)))
    with pytest.raises(IdentityError, match="reuse epoch cannot span"):
        _forecast(block_key, first, second)


def test_empty_outcomes_form_a_zero_profile(block_key: BlockKey) -> None:
    group = _group("zero", JointOutcome("none", 1_000_000))
    point = aggregate_shared_lease(
        _snapshot(block_key),
        _forecast(block_key, group),
    ).at_deadline(30)

    assert point.first_physical_readmission.nominal == 0
    assert point.expected_unique_reuse_epochs.nominal == 0
    assert point.expected_repeated_reuse_epochs.nominal == 0
    assert point.additive_claim_score == 0
    assert point.independent_marginal_union_score == 0


def test_orchestrator_snapshot_is_detached_scoped_and_state_bound(
    block_key: BlockKey,
) -> None:
    runtime = LifecycleOrchestrator(run_id="c1-snapshot", phase="m3_c1_component")
    workflow = WorkflowKey("workflow-a", 0)
    runtime.register_workflow(
        WorkflowSpec(
            key=workflow,
            nodes=(WorkflowNode("root"), WorkflowNode("next", ("root",))),
        )
    )
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "gpu0", "slot0", 1),
        byte_capacity=4096,
        payload_size=2048,
        payload_digest=_digest("payload"),
        timestamp_ns=1,
    )
    retention = BindingHandle(workflow, "retention", "retention-binding")
    runtime.bind_owner(
        retention,
        node_id="root",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=2,
    )
    runtime.bind_owner(
        BindingHandle(workflow, "request", "request-binding"),
        node_id="root",
        block_key=block_key,
        kind=BindingKind.REQUEST,
        state=BindingState.RETAINED,
        execution_ref=ExecutionRef(workflow, "request", "sequence", 0),
        timestamp_ns=2,
    )

    snapshot = runtime.shared_lease_policy_snapshot(block_key)
    assert [owner.binding_id for owner in snapshot.owners] == ["retention-binding"]
    assert snapshot.owners[0].eligible_node_ids == ("next", "root")
    claim = ReuseClaim(
        claim_id="future-next",
        binding_id="retention-binding",
        workflow=workflow,
        node_id="next",
        reuse_epoch_id="next-epoch",
        access_ns=10,
    )
    group = _group("runtime", JointOutcome("next", 1_000_000, (claim,)))
    forecast = _forecast(
        block_key,
        group,
        runtime_event_count=snapshot.runtime_event_count,
        generated_ns=2,
        horizon_ns=10,
    )
    before = runtime.events
    profile = aggregate_shared_lease(snapshot, forecast)
    assert profile.at_deadline(10).first_physical_readmission.nominal == 1
    assert runtime.events == before

    runtime.open_lease(
        retention,
        "lease-after-snapshot",
        registered_ns=3,
        deadline_ns=10,
        reason="c1 state advance",
    )
    advanced = runtime.shared_lease_policy_snapshot(block_key)
    assert advanced.runtime_event_count == snapshot.runtime_event_count + 1
    with pytest.raises(IdentityError, match="stale"):
        aggregate_shared_lease(advanced, forecast)


def test_released_retention_owner_disappears_from_policy_snapshot(
    block_key: BlockKey,
) -> None:
    runtime = LifecycleOrchestrator(run_id="c1-release", phase="m3_c1_component")
    workflow = WorkflowKey("workflow-a", 0)
    runtime.register_workflow(WorkflowSpec(key=workflow, nodes=(WorkflowNode("root"),)))
    runtime.register_gpu_block(
        block_key,
        ReplicaId(Tier.GPU, "gpu0", "slot0", 1),
        byte_capacity=4096,
        payload_size=2048,
        payload_digest=_digest("payload"),
        timestamp_ns=1,
    )
    handle = BindingHandle(workflow, "retention", "retention-binding")
    runtime.bind_owner(
        handle,
        node_id="root",
        block_key=block_key,
        kind=BindingKind.WORKFLOW_RETENTION,
        state=BindingState.RETAINED,
        execution_ref=None,
        timestamp_ns=2,
    )
    assert runtime.shared_lease_policy_snapshot(block_key).owners

    runtime.release_binding(handle, timestamp_ns=3)
    assert runtime.shared_lease_policy_snapshot(block_key).owners == ()
