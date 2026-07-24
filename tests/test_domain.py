"""Canonical schema validation tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from dagkv.domain import (
    BlockKey,
    IdentityError,
    Lease,
    LeaseState,
    LedgerAction,
    ReplicaId,
    StateTransitionError,
    Tier,
    Transfer,
    TransferDirection,
    TransferState,
    WorkflowKey,
    WorkflowNode,
    WorkflowSpec,
)


def test_digest_identity_is_required_and_canonical(
    digest: Callable[[str], str],
) -> None:
    """Missing and uppercase digests cannot create alias identities."""

    values = {
        "content_digest": digest("content"),
        "parent_digest": None,
        "model_fingerprint": "model",
        "tokenizer_fingerprint": "tokenizer",
        "adapter_fingerprint": None,
        "block_size_tokens": 16,
        "kv_dtype": "bfloat16",
    }
    with pytest.raises(IdentityError, match="content_digest"):
        BlockKey(**(values | {"content_digest": None}))  # type: ignore[arg-type]
    with pytest.raises(IdentityError, match="lowercase"):
        BlockKey(**(values | {"content_digest": digest("content").upper()}))


def test_workflow_rejects_unknown_edges_and_cycles() -> None:
    """Only a closed acyclic topology reaches runtime state."""

    key = WorkflowKey("dag", 1)
    with pytest.raises(IdentityError, match="unknown workflow predecessors"):
        WorkflowSpec(key, (WorkflowNode("a", ("missing",)),))
    with pytest.raises(IdentityError, match="acyclic"):
        WorkflowSpec(
            key,
            (
                WorkflowNode("a", ("b",)),
                WorkflowNode("b", ("a",)),
            ),
        )


def test_transfer_validates_direction_and_exact_terminal_replay(
    block_key: BlockKey,
    digest: Callable[[str], str],
) -> None:
    """Transfer geometry and duplicate terminal inputs are fail-closed."""

    gpu = ReplicaId(Tier.GPU, "cuda:0", "7", 1)
    cpu = ReplicaId(Tier.CPU, "numa:0", "7", 1)
    payload_digest = digest("payload")
    transfer = Transfer(
        transfer_id="save-1",
        direction=TransferDirection.D2H,
        block_key=block_key,
        source_replica=gpu,
        target_replica=cpu,
        declared_bytes=128,
        payload_digest=payload_digest,
        started_ns=10,
        scheduled_event_id="evt-1",
        ledger_action=LedgerAction.SAVE,
    )
    assert transfer.terminate(
        TransferState.COMPLETED,
        20,
        observed_bytes=128,
        observed_digest=payload_digest,
    )
    assert not transfer.terminate(
        TransferState.COMPLETED,
        20,
        observed_bytes=128,
        observed_digest=payload_digest,
    )
    with pytest.raises(StateTransitionError, match="conflicting transfer"):
        transfer.terminate(
            TransferState.COMPLETED,
            20,
            observed_bytes=64,
            observed_digest=payload_digest,
        )
    with pytest.raises(IdentityError, match="direction, tiers, and action"):
        Transfer(
            transfer_id="bad-save",
            direction=TransferDirection.D2H,
            block_key=block_key,
            source_replica=cpu,
            target_replica=gpu,
            declared_bytes=128,
            payload_digest=payload_digest,
            started_ns=10,
            scheduled_event_id="evt-2",
            ledger_action=LedgerAction.SAVE,
        )


def test_lease_allows_only_exact_terminal_replay(block_key: BlockKey) -> None:
    """A conflicting duplicate lease terminal is observable."""

    lease = Lease(
        lease_id="lease-1",
        binding_id="retention-1",
        block_key=block_key,
        registered_ns=10,
        deadline_ns=20,
        reason="future fanout",
    )
    assert lease.terminate(LeaseState.EXPIRED, 20)
    assert not lease.terminate(LeaseState.EXPIRED, 20)
    with pytest.raises(StateTransitionError, match="conflicting lease"):
        lease.terminate(LeaseState.EXPIRED, 21)
