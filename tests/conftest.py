"""Shared deterministic fixtures for DAGKV component tests."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

import pytest

from dagkv.domain import BlockKey, WorkflowKey, WorkflowNode, WorkflowSpec


@pytest.fixture
def digest() -> Callable[[str], str]:
    """Return a stable SHA-256 helper."""

    def make_digest(value: str) -> str:
        return sha256(value.encode()).hexdigest()

    return make_digest


@pytest.fixture
def block_key(digest: Callable[[str], str]) -> BlockKey:
    """Return one complete canonical block identity."""

    return BlockKey(
        content_digest=digest("content"),
        parent_digest=digest("parent"),
        model_fingerprint="qwen3-8b-test",
        tokenizer_fingerprint="qwen3-tokenizer-test",
        adapter_fingerprint=None,
        block_size_tokens=16,
        kv_dtype="bfloat16",
        cache_salt="component-test",
    )


@pytest.fixture
def workflow_spec() -> WorkflowSpec:
    """Return a single-node workflow for lifecycle-focused tests."""

    return WorkflowSpec(
        key=WorkflowKey("workflow-a", 0),
        nodes=(WorkflowNode("agent-a"),),
    )
