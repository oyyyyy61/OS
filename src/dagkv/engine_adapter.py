"""Fail-closed projection of engine DMA observations into DAGKV state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from dagkv.domain import (
    DAGKVError,
    IdentityError,
    ReplicaId,
    Tier,
    TransferCommand,
    TransferDirection,
    TransferIntegrityError,
    require_optional_sha256,
    require_sha256,
    require_text,
)
from dagkv.orchestrator import LifecycleOrchestrator


@dataclass(frozen=True, slots=True)
class EngineFingerprint:
    """Immutable provenance required for one engine observation stream."""

    engine: str
    version: str
    source_revision: str
    source_diff_digest: str
    model_fingerprint: str

    def __post_init__(self) -> None:
        require_text("engine", self.engine)
        require_text("engine version", self.version)
        require_text("source revision", self.source_revision)
        require_sha256("source_diff_digest", self.source_diff_digest)
        require_text("model fingerprint", self.model_fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "version": self.version,
            "source_revision": self.source_revision,
            "source_diff_digest": self.source_diff_digest,
            "model_fingerprint": self.model_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: object) -> EngineFingerprint:
        row = _object(value, "engine fingerprint")
        _exact_keys(
            row,
            {
                "engine",
                "version",
                "source_revision",
                "source_diff_digest",
                "model_fingerprint",
            },
            "engine fingerprint",
        )
        return cls(**row)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class PhysicalEndpoint:
    """One allocator-authoritative physical KV location."""

    tier: Tier
    device_id: str
    slot_id: str
    generation: int
    generation_source: str

    def __post_init__(self) -> None:
        require_text("endpoint device_id", self.device_id)
        require_text("endpoint slot_id", self.slot_id)
        require_text("endpoint generation_source", self.generation_source)
        if type(self.generation) is not int or self.generation < 1:
            raise IdentityError("endpoint generation must be a positive integer")

    @property
    def replica_id(self) -> ReplicaId:
        return ReplicaId(
            tier=self.tier,
            device_id=self.device_id,
            slot_id=self.slot_id,
            generation=self.generation,
        )

    @classmethod
    def from_replica(
        cls,
        replica: ReplicaId,
        *,
        generation_source: str,
    ) -> PhysicalEndpoint:
        return cls(
            tier=replica.tier,
            device_id=replica.device_id,
            slot_id=replica.slot_id,
            generation=replica.generation,
            generation_source=generation_source,
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "tier": self.tier.value,
            "device_id": self.device_id,
            "slot_id": self.slot_id,
            "generation": self.generation,
            "generation_source": self.generation_source,
        }

    @classmethod
    def from_dict(cls, value: object) -> PhysicalEndpoint:
        row = _object(value, "physical endpoint")
        _exact_keys(
            row,
            {
                "tier",
                "device_id",
                "slot_id",
                "generation",
                "generation_source",
            },
            "physical endpoint",
        )
        try:
            tier = Tier(row["tier"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IdentityError("physical endpoint has an invalid tier") from exc
        return cls(
            tier=tier,
            device_id=row["device_id"],  # type: ignore[arg-type]
            slot_id=row["slot_id"],  # type: ignore[arg-type]
            generation=row["generation"],  # type: ignore[arg-type]
            generation_source=row["generation_source"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DMATerminalObservation:
    """Complete terminal report from a generation-aware DMA adapter."""

    SCHEMA_VERSION: ClassVar[str] = "dagkv_dma_terminal_observation_v1"

    engine_job_id: str
    transfer_id: str
    direction: TransferDirection
    source: PhysicalEndpoint
    target: PhysicalEndpoint
    payload_bytes: int
    reported_bytes: int
    source_digest: str
    target_digest: str | None
    submitted_ns: int
    terminal_ns: int
    engine_success: bool
    error: str | None
    fingerprint: EngineFingerprint

    def __post_init__(self) -> None:
        require_text("engine_job_id", self.engine_job_id)
        require_text("transfer_id", self.transfer_id)
        if type(self.payload_bytes) is not int or self.payload_bytes <= 0:
            raise IdentityError("payload_bytes must be a positive integer")
        if type(self.reported_bytes) is not int or self.reported_bytes < 0:
            raise IdentityError("reported_bytes must be a non-negative integer")
        if type(self.submitted_ns) is not int or self.submitted_ns < 0:
            raise IdentityError("submitted_ns must be a non-negative integer")
        if type(self.terminal_ns) is not int or self.terminal_ns < self.submitted_ns:
            raise IdentityError("terminal_ns must not predate submission")
        if type(self.engine_success) is not bool:
            raise IdentityError("engine_success must be a bool")
        require_sha256("source_digest", self.source_digest)
        require_optional_sha256("target_digest", self.target_digest)
        if self.engine_success:
            if self.target_digest is None:
                raise IdentityError("successful DMA requires a target digest")
            if self.error is not None:
                raise IdentityError("successful DMA cannot carry an error")
        else:
            if not isinstance(self.error, str) or not self.error:
                raise IdentityError("failed DMA requires a non-empty error")
        expected = (
            (Tier.GPU, Tier.CPU)
            if self.direction == TransferDirection.D2H
            else (Tier.CPU, Tier.GPU)
        )
        if (self.source.tier, self.target.tier) != expected:
            raise IdentityError("DMA direction disagrees with endpoint tiers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "engine_job_id": self.engine_job_id,
            "transfer_id": self.transfer_id,
            "direction": self.direction.value,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "payload_bytes": self.payload_bytes,
            "reported_bytes": self.reported_bytes,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "submitted_ns": self.submitted_ns,
            "terminal_ns": self.terminal_ns,
            "engine_success": self.engine_success,
            "error": self.error,
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> DMATerminalObservation:
        row = _object(value, "DMA terminal observation")
        keys = {
            "schema_version",
            "engine_job_id",
            "transfer_id",
            "direction",
            "source",
            "target",
            "payload_bytes",
            "reported_bytes",
            "source_digest",
            "target_digest",
            "submitted_ns",
            "terminal_ns",
            "engine_success",
            "error",
            "fingerprint",
        }
        _exact_keys(row, keys, "DMA terminal observation")
        if row["schema_version"] != cls.SCHEMA_VERSION:
            raise IdentityError("unsupported DMA observation schema")
        try:
            direction = TransferDirection(row["direction"])
        except (TypeError, ValueError) as exc:
            raise IdentityError("DMA observation has an invalid direction") from exc
        return cls(
            engine_job_id=row["engine_job_id"],  # type: ignore[arg-type]
            transfer_id=row["transfer_id"],  # type: ignore[arg-type]
            direction=direction,
            source=PhysicalEndpoint.from_dict(row["source"]),
            target=PhysicalEndpoint.from_dict(row["target"]),
            payload_bytes=row["payload_bytes"],  # type: ignore[arg-type]
            reported_bytes=row["reported_bytes"],  # type: ignore[arg-type]
            source_digest=row["source_digest"],  # type: ignore[arg-type]
            target_digest=row["target_digest"],  # type: ignore[arg-type]
            submitted_ns=row["submitted_ns"],  # type: ignore[arg-type]
            terminal_ns=row["terminal_ns"],  # type: ignore[arg-type]
            engine_success=row["engine_success"],  # type: ignore[arg-type]
            error=row["error"],  # type: ignore[arg-type]
            fingerprint=EngineFingerprint.from_dict(row["fingerprint"]),
        )


def commit_transfer_observation(
    orchestrator: LifecycleOrchestrator,
    command: TransferCommand,
    observation: DMATerminalObservation,
) -> bool:
    """Close one canonical transfer using only public orchestrator methods."""

    mismatches: list[str] = []
    transfer = orchestrator.transfer_snapshot(command.transfer_id)
    if observation.transfer_id != command.transfer_id:
        mismatches.append("transfer identity")
    if observation.direction != command.direction:
        mismatches.append("direction")
    if observation.source.replica_id != command.source_replica:
        mismatches.append("source allocation generation")
    if observation.target.replica_id != command.target_replica:
        mismatches.append("target allocation generation")
    if observation.payload_bytes != command.byte_count:
        mismatches.append("declared payload bytes")
    if (
        transfer.direction != command.direction
        or transfer.source_replica != command.source_replica
        or transfer.target_replica != command.target_replica
        or transfer.declared_bytes != command.byte_count
        or transfer.payload_digest != command.payload_digest
        or transfer.block_key != command.block_key
        or transfer.ledger_action != command.action
    ):
        mismatches.append("command snapshot")

    if observation.engine_success:
        if observation.reported_bytes != command.byte_count:
            mismatches.append("reported bytes")
        if observation.source_digest != command.payload_digest:
            mismatches.append("source payload digest")
        if observation.target_digest != command.payload_digest:
            mismatches.append("target payload digest")

    if mismatches:
        error = "DMA observation mismatch: " + ", ".join(mismatches)
        _fail_expected_transfer(orchestrator, command, observation, error)
        raise TransferIntegrityError(error)

    if not observation.engine_success:
        assert observation.error is not None
        return orchestrator.fail_transfer(
            command.transfer_id,
            timestamp_ns=observation.terminal_ns,
            observed_bytes=observation.reported_bytes,
            observed_digest=observation.target_digest,
            error=observation.error,
        )

    assert observation.target_digest is not None
    return orchestrator.complete_transfer(
        command.transfer_id,
        timestamp_ns=observation.terminal_ns,
        observed_bytes=observation.reported_bytes,
        observed_digest=observation.target_digest,
    )


def _fail_expected_transfer(
    orchestrator: LifecycleOrchestrator,
    command: TransferCommand,
    observation: DMATerminalObservation,
    error: str,
) -> None:
    try:
        orchestrator.fail_transfer(
            command.transfer_id,
            timestamp_ns=observation.terminal_ns,
            observed_bytes=observation.reported_bytes,
            observed_digest=observation.target_digest,
            error=error,
        )
    except DAGKVError:
        # Preserve the original mismatch when a conflicting duplicate terminal
        # has already closed the transfer.
        raise


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise IdentityError(f"{name} must be a JSON object with string keys")
    return value


def _exact_keys(row: dict[str, object], expected: set[str], name: str) -> None:
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise IdentityError(f"{name} fields differ: missing={missing}, extra={extra}")
