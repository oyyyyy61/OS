"""Create-only C1-B0 evidence bundle and fresh independent verifier."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from dagkv.c1_commit import (
    CanonicalTraceCommitter,
    CutoffCommitRequest,
    DemandCommitRequest,
    ObservationCloseRequest,
    ObservationTerminalSpec,
    SealedTraceReceipt,
    TraceOperationCommit,
    TraceOperationKind,
    TracePreambleRequest,
)
from dagkv.c1_lifecycle import (
    CanonicalLifecycleEvidenceGate,
    ClosedLifecycleArtifact,
)
from dagkv.c1_schedule import (
    CanonicalScheduleEvidenceGate,
    ClosedScheduleArtifact,
    ReplayScheduleClosure,
)
from dagkv.c1_trace import (
    MAX_TRACE_BYTES,
    CutoffPayload,
    DemandIntentPayload,
    DemandLabel,
    EvidenceRole,
    ObservationTerminalPayload,
    ReuseEpochPayload,
    ScheduleWatermarkPayload,
    TraceCommitIndeterminateError,
    TraceHeaderPayload,
    TraceRecord,
    TraceRecordType,
    TraceValidationError,
    TraceValidationReceipt,
    WorkflowTopologyPayload,
    canonical_digest,
    canonical_json,
    parse_canonical_dataclass,
    parse_trace_record,
    reconstruct_demand_labels,
    trace_stream_digest,
    validate_trace,
    validate_trace_for_labels,
)
from dagkv.domain import require_sha256, require_text

C1_B0_SEGMENT_SCHEMA_VERSION = "dagkv.m3.c1_b0_segment_commit.v1"
C1_B0_ATTEMPT_SCHEMA_VERSION = "dagkv.m3.c1_b0_finalization_attempt.v1"
C1_B0_MANIFEST_SCHEMA_VERSION = "dagkv.m3.c1_b0_bundle_manifest.v1"
C1_B0_FINAL_SEAL_SCHEMA_VERSION = "dagkv.m3.c1_b0_final_seal.v1"
C1_B0_CLAIM_SCOPE = "C1_B0_SCHEMA_RECONSTRUCTION_COMPONENT_ONLY"
C1_B0_STATUS = "C1_B0_SCHEMA_RECONSTRUCTION_COMPONENT_VERIFIED"
C1_B0_LIFECYCLE_PHASE = "m3_c1b"
C1_B0_LIFECYCLE_SOURCE = "dagkv.orchestrator"
C1_B0_OPEN_GATES = (
    "C1_B1_SPLIT_LEAKAGE",
    "C1_B2_EXCLUDED_PILOT",
    "C1_B3_CALIBRATION_FREEZE",
    "C1_B4_FORMAL_COVERAGE",
    "C1_C_POLICY_EFFECT",
    "C1_D_REAL_GPU",
)
EMPTY_SHA256 = sha256(b"").hexdigest()
MAX_BUNDLE_JSON_BYTES = 32 * 1024 * 1024
MAX_SIDECAR_BYTES = 64 * 1024 * 1024

SCHEDULE_PAYLOAD = "000000.schedule.json"
SCHEDULE_COMMIT = "000000.schedule.commit.json"
LIFECYCLE_PAYLOAD = "000001.lifecycle.json"
LIFECYCLE_COMMIT = "000001.lifecycle.commit.json"
TRACE_PAYLOAD = "000002.trace.jsonl"
TRACE_COMMIT = "000002.trace.commit.json"
ATTEMPT = "C1_B0_ATTEMPT.json"
MANIFEST = "C1_B0_MANIFEST.json"
FINAL_SEAL = "C1_B0_FINAL_SEAL.json"

PAYLOAD_NAMES = (SCHEDULE_PAYLOAD, LIFECYCLE_PAYLOAD, TRACE_PAYLOAD)
COMMIT_NAMES = (SCHEDULE_COMMIT, LIFECYCLE_COMMIT, TRACE_COMMIT)
PRESEAL_NAMES = (
    ATTEMPT,
    SCHEDULE_PAYLOAD,
    SCHEDULE_COMMIT,
    LIFECYCLE_PAYLOAD,
    LIFECYCLE_COMMIT,
    TRACE_PAYLOAD,
    TRACE_COMMIT,
    MANIFEST,
)
FINAL_INVENTORY = (*PRESEAL_NAMES, FINAL_SEAL)


class C1BundleSegmentRole(StrEnum):
    SCHEDULE = "SCHEDULE"
    LIFECYCLE = "LIFECYCLE"
    TRACE = "TRACE"


SEGMENT_LAYOUT = (
    (C1BundleSegmentRole.SCHEDULE, SCHEDULE_PAYLOAD, SCHEDULE_COMMIT),
    (C1BundleSegmentRole.LIFECYCLE, LIFECYCLE_PAYLOAD, LIFECYCLE_COMMIT),
    (C1BundleSegmentRole.TRACE, TRACE_PAYLOAD, TRACE_COMMIT),
)


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise TraceValidationError(f"{name} must be an integer >= {minimum}")


def _require_basename(name: str, value: str) -> None:
    require_text(name, value)
    if "\x00" in value or value in {".", ".."} or Path(value).name != value:
        raise TraceValidationError(f"{name} must be a plain basename")


@dataclass(frozen=True, slots=True)
class C1B0FinalizationAttempt:
    schema_version: str
    bundle_id: str
    claim_scope: str
    protocol_digest: str
    verifier_digest: str
    implementation_digest: str
    environment_digest: str
    payload_basenames: tuple[str, ...]
    payload_sizes_bytes: tuple[int, ...]
    payload_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != C1_B0_ATTEMPT_SCHEMA_VERSION:
            raise TraceValidationError("unsupported C1-B0 attempt schema")
        require_text("attempt bundle_id", self.bundle_id)
        if self.claim_scope != C1_B0_CLAIM_SCOPE:
            raise TraceValidationError("C1-B0 attempt broadens its claim scope")
        require_sha256("attempt protocol_digest", self.protocol_digest)
        require_sha256("attempt verifier_digest", self.verifier_digest)
        require_sha256("attempt implementation_digest", self.implementation_digest)
        require_sha256("attempt environment_digest", self.environment_digest)
        if self.payload_basenames != PAYLOAD_NAMES:
            raise TraceValidationError(
                "C1-B0 attempt payload inventory is not canonical"
            )
        if (
            not isinstance(self.payload_sizes_bytes, tuple)
            or len(self.payload_sizes_bytes) != len(PAYLOAD_NAMES)
            or not isinstance(self.payload_sha256s, tuple)
            or len(self.payload_sha256s) != len(PAYLOAD_NAMES)
        ):
            raise TraceValidationError(
                "C1-B0 attempt payload identities are incomplete"
            )
        for size_bytes in self.payload_sizes_bytes:
            _require_int("attempt payload size", size_bytes, minimum=1)
        for digest in self.payload_sha256s:
            require_sha256("attempt payload sha256", digest)


@dataclass(frozen=True, slots=True)
class C1BundleSegmentCommit:
    schema_version: str
    bundle_id: str
    trace_pair_id: str
    segment_index: int
    role: C1BundleSegmentRole
    payload_basename: str
    payload_size_bytes: int
    payload_record_count: int
    payload_sha256: str
    previous_segment_commit_sha256: str
    sealed_trace: SealedTraceReceipt | None

    def __post_init__(self) -> None:
        if self.schema_version != C1_B0_SEGMENT_SCHEMA_VERSION:
            raise TraceValidationError("unsupported C1-B0 segment commit schema")
        require_text("bundle_id", self.bundle_id)
        require_text("segment trace_pair_id", self.trace_pair_id)
        _require_int("segment_index", self.segment_index)
        if self.segment_index >= len(SEGMENT_LAYOUT):
            raise TraceValidationError("segment index is outside the fixed layout")
        expected_role, expected_payload, _ = SEGMENT_LAYOUT[self.segment_index]
        if self.role != expected_role or self.payload_basename != expected_payload:
            raise TraceValidationError(
                "segment role or payload path differs from layout"
            )
        _require_basename("segment payload_basename", self.payload_basename)
        _require_int("segment payload_size_bytes", self.payload_size_bytes, minimum=1)
        _require_int(
            "segment payload_record_count", self.payload_record_count, minimum=1
        )
        require_sha256("segment payload_sha256", self.payload_sha256)
        require_sha256(
            "previous_segment_commit_sha256",
            self.previous_segment_commit_sha256,
        )
        if self.segment_index == 0 and (
            self.previous_segment_commit_sha256 != EMPTY_SHA256
        ):
            raise TraceValidationError(
                "first segment does not start at the empty digest"
            )
        if self.role == C1BundleSegmentRole.TRACE:
            if type(self.sealed_trace) is not SealedTraceReceipt:
                raise TraceValidationError("trace segment lacks its typed closure")
            if (
                self.sealed_trace.trace_pair_id != self.trace_pair_id
                or self.sealed_trace.trace_basename != self.payload_basename
                or self.sealed_trace.closure.size_bytes != self.payload_size_bytes
                or self.sealed_trace.closure.record_count != self.payload_record_count
                or self.sealed_trace.closure.stream_digest != self.payload_sha256
            ):
                raise TraceValidationError(
                    "trace segment differs from its typed closure"
                )
        elif self.sealed_trace is not None:
            raise TraceValidationError("non-trace segment contains a trace closure")


@dataclass(frozen=True, slots=True)
class C1BundleSegmentRef:
    segment_index: int
    role: C1BundleSegmentRole
    commit_basename: str
    commit_sha256: str

    def __post_init__(self) -> None:
        _require_int("segment ref index", self.segment_index)
        if self.segment_index >= len(SEGMENT_LAYOUT):
            raise TraceValidationError("segment ref index is outside the fixed layout")
        expected_role, _, expected_commit = SEGMENT_LAYOUT[self.segment_index]
        if self.role != expected_role or self.commit_basename != expected_commit:
            raise TraceValidationError("segment reference differs from fixed layout")
        _require_basename("segment commit_basename", self.commit_basename)
        require_sha256("segment commit_sha256", self.commit_sha256)


@dataclass(frozen=True, slots=True)
class C1B0BundleManifest:
    schema_version: str
    bundle_id: str
    status: str
    claim_scope: str
    open_gates: tuple[str, ...]
    trace_pair_id: str
    run_id: str
    schedule_id: str
    schedule_case_id: str
    protocol_digest: str
    verifier_digest: str
    implementation_digest: str
    environment_digest: str
    segments: tuple[C1BundleSegmentRef, ...]
    lifecycle_validation: TraceValidationReceipt
    schedule_validation: TraceValidationReceipt
    demand_labels: tuple[DemandLabel, ...]

    def __post_init__(self) -> None:
        if self.schema_version != C1_B0_MANIFEST_SCHEMA_VERSION:
            raise TraceValidationError("unsupported C1-B0 manifest schema")
        for name in (
            "bundle_id",
            "trace_pair_id",
            "run_id",
            "schedule_id",
            "schedule_case_id",
        ):
            require_text(name, getattr(self, name))
        if self.status != C1_B0_STATUS:
            raise TraceValidationError("C1-B0 manifest changes its component status")
        if self.claim_scope != C1_B0_CLAIM_SCOPE:
            raise TraceValidationError("C1-B0 manifest broadens its claim scope")
        if self.open_gates != C1_B0_OPEN_GATES:
            raise TraceValidationError("C1-B0 manifest changes the declared open gates")
        require_sha256("protocol_digest", self.protocol_digest)
        require_sha256("verifier_digest", self.verifier_digest)
        require_sha256("implementation_digest", self.implementation_digest)
        require_sha256("environment_digest", self.environment_digest)
        if not isinstance(self.segments, tuple) or len(self.segments) != len(
            SEGMENT_LAYOUT
        ):
            raise TraceValidationError("manifest segment references are not canonical")
        for index, ((role, _, commit_name), reference) in enumerate(
            zip(SEGMENT_LAYOUT, self.segments, strict=True)
        ):
            if (
                reference.segment_index != index
                or reference.role != role
                or reference.commit_basename != commit_name
            ):
                raise TraceValidationError(
                    "manifest segment references are not canonical"
                )
        lifecycle = self.lifecycle_validation
        schedule = self.schedule_validation
        if (
            lifecycle.role != EvidenceRole.LIFECYCLE
            or schedule.role != EvidenceRole.SCHEDULE
            or lifecycle.trace_pair_id != self.trace_pair_id
            or schedule.trace_pair_id != self.trace_pair_id
            or lifecycle.trace_digest != schedule.trace_digest
            or lifecycle.verifier_digest != self.verifier_digest
            or schedule.verifier_digest != self.verifier_digest
            or lifecycle.verified_observation_ids != schedule.verified_observation_ids
        ):
            raise TraceValidationError("manifest evidence receipts are inconsistent")
        if not isinstance(self.demand_labels, tuple):
            raise TraceValidationError("manifest demand labels must be a tuple")
        label_ids = tuple(label.observation_id for label in self.demand_labels)
        if (
            label_ids != tuple(sorted(label_ids))
            or len(label_ids) != len(set(label_ids))
            or label_ids != lifecycle.verified_observation_ids
        ):
            raise TraceValidationError(
                "manifest labels do not cover verified observations"
            )


@dataclass(frozen=True, slots=True)
class C1BundleFileIdentity:
    basename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_basename("bundle file basename", self.basename)
        _require_int("bundle file size_bytes", self.size_bytes, minimum=1)
        require_sha256("bundle file sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class C1B0FinalSeal:
    schema_version: str
    bundle_id: str
    status: str
    claim_scope: str
    manifest_basename: str
    manifest_size_bytes: int
    manifest_sha256: str
    segment_count: int
    last_segment_commit_sha256: str
    preseal_files: tuple[C1BundleFileIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != C1_B0_FINAL_SEAL_SCHEMA_VERSION:
            raise TraceValidationError("unsupported C1-B0 final seal schema")
        require_text("final seal bundle_id", self.bundle_id)
        if self.status != C1_B0_STATUS:
            raise TraceValidationError("C1-B0 final seal changes component status")
        if self.claim_scope != C1_B0_CLAIM_SCOPE:
            raise TraceValidationError("C1-B0 final seal broadens its claim scope")
        if self.manifest_basename != MANIFEST:
            raise TraceValidationError("final seal names another manifest")
        _require_int("manifest_size_bytes", self.manifest_size_bytes, minimum=1)
        require_sha256("manifest_sha256", self.manifest_sha256)
        if self.segment_count != len(SEGMENT_LAYOUT):
            raise TraceValidationError("final seal has the wrong segment count")
        require_sha256(
            "last_segment_commit_sha256",
            self.last_segment_commit_sha256,
        )
        if not isinstance(self.preseal_files, tuple):
            raise TraceValidationError("final seal preseal inventory must be a tuple")
        if tuple(item.basename for item in self.preseal_files) != PRESEAL_NAMES:
            raise TraceValidationError("final seal preseal inventory is not canonical")
        manifest_identity = self.preseal_files[-1]
        if (
            manifest_identity.size_bytes != self.manifest_size_bytes
            or manifest_identity.sha256 != self.manifest_sha256
        ):
            raise TraceValidationError("final seal manifest identity differs")


@dataclass(frozen=True, slots=True)
class ValidatedC1B0Bundle:
    bundle_id: str
    status: str
    claim_scope: str
    open_gates: tuple[str, ...]
    trace_pair_id: str
    manifest_sha256: str
    final_seal_sha256: str
    schedule_sha256: str
    lifecycle_sha256: str
    trace_sha256: str
    verified_observation_ids: tuple[str, ...]
    demand_labels: tuple[DemandLabel, ...]

    def __post_init__(self) -> None:
        require_text("validated bundle_id", self.bundle_id)
        require_text("validated trace_pair_id", self.trace_pair_id)
        if self.status != C1_B0_STATUS:
            raise TraceValidationError("validated C1-B0 status changed")
        if self.claim_scope != C1_B0_CLAIM_SCOPE:
            raise TraceValidationError("validated C1-B0 scope changed")
        if self.open_gates != C1_B0_OPEN_GATES:
            raise TraceValidationError("validated C1-B0 open gates changed")
        for name in (
            "manifest_sha256",
            "final_seal_sha256",
            "schedule_sha256",
            "lifecycle_sha256",
            "trace_sha256",
        ):
            require_sha256(name, getattr(self, name))
        if tuple(label.observation_id for label in self.demand_labels) != (
            self.verified_observation_ids
        ):
            raise TraceValidationError(
                "validated labels differ from verified observations"
            )


@dataclass(frozen=True, slots=True)
class _StableFileSnapshot:
    raw: bytes
    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


def _max_file_bytes(basename: str) -> int:
    if basename == TRACE_PAYLOAD:
        return MAX_TRACE_BYTES
    if basename in {SCHEDULE_PAYLOAD, LIFECYCLE_PAYLOAD}:
        return MAX_SIDECAR_BYTES
    return MAX_BUNDLE_JSON_BYTES


def _open_directory(
    path: Path, *, require_read_only: bool
) -> tuple[int, os.stat_result]:
    if not path.is_absolute():
        raise TraceValidationError("C1-B0 bundle root must be absolute")
    try:
        linked = path.lstat()
    except FileNotFoundError as exc:
        raise TraceValidationError("C1-B0 bundle root is missing") from exc
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise TraceValidationError("C1-B0 bundle root must be a non-symlink directory")
    if require_read_only and linked.st_mode & 0o222:
        raise TraceValidationError("sealed C1-B0 bundle root must be read-only")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise TraceValidationError("cannot open C1-B0 bundle root safely") from exc
    if (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino) or (
        require_read_only and opened.st_mode & 0o222
    ):
        os.close(descriptor)
        raise TraceValidationError("C1-B0 bundle root changed while opening")
    return descriptor, opened


def _open_parent_directory(root: Path, opened_root: os.stat_result) -> int:
    parent = root.parent
    linked = parent.lstat()
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISDIR(linked.st_mode):
        raise TraceValidationError("C1-B0 bundle parent must be a direct directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(parent, flags)
        opened_parent = os.fstat(descriptor)
        linked_root = os.stat(
            root.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise TraceValidationError("cannot open C1-B0 bundle parent safely") from exc
    if (linked.st_dev, linked.st_ino) != (
        opened_parent.st_dev,
        opened_parent.st_ino,
    ) or (linked_root.st_dev, linked_root.st_ino) != (
        opened_root.st_dev,
        opened_root.st_ino,
    ):
        os.close(descriptor)
        raise TraceValidationError("C1-B0 bundle parent or root entry changed")
    return descriptor


def _assert_directory_identity(
    root: Path,
    descriptor: int,
    opened: os.stat_result,
) -> None:
    linked = root.lstat()
    current = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(linked.st_mode)
        or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        or linked.st_mode & 0o222
        or current.st_mode & 0o222
    ):
        raise TraceValidationError("C1-B0 bundle root identity changed")


def _require_inventory(descriptor: int, expected: tuple[str, ...]) -> None:
    observed = tuple(sorted(os.listdir(descriptor)))
    if observed != tuple(sorted(expected)):
        raise TraceValidationError(
            f"C1-B0 bundle inventory differs: expected {expected}, observed {observed}"
        )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_snapshot(
    root_descriptor: int,
    basename: str,
    *,
    max_bytes: int,
    require_read_only: bool,
) -> _StableFileSnapshot:
    _require_basename("bundle child basename", basename)
    try:
        linked = os.stat(
            basename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise TraceValidationError(f"bundle child is missing: {basename}") from exc
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_nlink != 1
        or (require_read_only and linked.st_mode & 0o222)
    ):
        raise TraceValidationError(
            f"bundle child must be singly linked, regular, and read-only: {basename}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(basename, flags, dir_fd=root_descriptor)
    except OSError as exc:
        raise TraceValidationError(
            f"cannot open bundle child safely: {basename}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (require_read_only and opened.st_mode & 0o222)
            or _stat_identity(opened) != _stat_identity(linked)
        ):
            raise TraceValidationError(
                f"bundle child changed while opening: {basename}"
            )
        raw = bytearray()
        while len(raw) <= max_bytes:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, max_bytes + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        relinked = os.stat(
            basename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(relinked.st_mode)
            or after.st_nlink != 1
            or relinked.st_nlink != 1
            or (
                require_read_only
                and (after.st_mode & 0o222 or relinked.st_mode & 0o222)
            )
            or _stat_identity(after) != _stat_identity(opened)
            or _stat_identity(relinked) != _stat_identity(opened)
        ):
            raise TraceValidationError(
                f"bundle child changed while reading: {basename}"
            )
    except OSError as exc:
        raise TraceValidationError(
            f"cannot read bundle child safely: {basename}"
        ) from exc
    finally:
        os.close(descriptor)
    if not raw or len(raw) > max_bytes:
        raise TraceValidationError(f"bundle child has an invalid size: {basename}")
    return _StableFileSnapshot(
        raw=bytes(raw),
        device=after.st_dev,
        inode=after.st_ino,
        mode=stat.S_IMODE(after.st_mode),
        link_count=after.st_nlink,
        size_bytes=after.st_size,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
    )


def _read_stable_file(
    root_descriptor: int,
    basename: str,
    *,
    max_bytes: int,
    require_read_only: bool,
) -> bytes:
    return _read_stable_snapshot(
        root_descriptor,
        basename,
        max_bytes=max_bytes,
        require_read_only=require_read_only,
    ).raw


def _write_create_only(
    root_descriptor: int,
    basename: str,
    raw: bytes,
) -> str:
    _require_basename("bundle output basename", basename)
    if not raw or len(raw) > MAX_BUNDLE_JSON_BYTES:
        raise TraceValidationError("bundle output has an invalid size")
    descriptor: int | None = None
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            basename,
            flags,
            0o440,
            dir_fd=root_descriptor,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise OSError("bundle write made no progress")
            offset += count
        os.fsync(descriptor)
        os.fsync(root_descriptor)
        opened = os.fstat(descriptor)
        linked = os.stat(
            basename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or opened.st_size != len(raw)
        ):
            raise OSError("bundle output identity changed")
        observed = bytearray()
        offset = 0
        while offset < len(raw):
            chunk = os.pread(descriptor, len(raw) - offset, offset)
            if not chunk:
                raise OSError("bundle output ended during readback")
            observed.extend(chunk)
            offset += len(chunk)
        if bytes(observed) != raw:
            raise OSError("bundle output differs from staged bytes")
    except FileExistsError as exc:
        raise TraceValidationError(f"bundle output is create-only: {basename}") from exc
    except BaseException as exc:
        if not isinstance(exc, Exception):
            raise
        raise TraceCommitIndeterminateError(
            f"bundle output durability is indeterminate: {basename}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return sha256(raw).hexdigest()


def _seal_child_read_only(root_descriptor: int, basename: str) -> None:
    _require_basename("bundle sealed child basename", basename)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(basename, flags, dir_fd=root_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise OSError("bundle child identity changed before sealing")
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        linked = os.stat(
            basename,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(sealed.st_mode)
            or sealed.st_nlink != 1
            or sealed.st_mode & 0o222
            or _stat_identity(sealed) != _stat_identity(linked)
            or (sealed.st_dev, sealed.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise OSError("bundle child identity changed while sealing")
    finally:
        os.close(descriptor)


def _parse_trace(raw: bytes) -> tuple[TraceRecord, ...]:
    if not raw.endswith(b"\n") or b"\r" in raw or len(raw) > MAX_TRACE_BYTES:
        raise TraceValidationError("bundle trace JSONL framing is invalid")
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        raise TraceValidationError("bundle trace JSONL contains a blank line")
    records = tuple(parse_trace_record(line) for line in lines)
    validate_trace(records)
    return records


def _operation_types_are_valid(
    kind: TraceOperationKind,
    records: tuple[TraceRecord, ...],
) -> bool:
    types = tuple(record.record_type for record in records)
    if kind == TraceOperationKind.PREAMBLE:
        return (
            len(types) >= 2
            and types[0] == TraceRecordType.TRACE_HEADER
            and all(item == TraceRecordType.WORKFLOW_TOPOLOGY for item in types[1:])
        )
    if kind == TraceOperationKind.CUTOFF_ATTEMPT:
        return types == (
            TraceRecordType.CUTOFF,
            TraceRecordType.FORECAST_ATTEMPT,
        )
    if kind == TraceOperationKind.DEMAND_INTENT:
        return bool(types) and all(
            item == TraceRecordType.DEMAND_INTENT for item in types
        )
    if kind == TraceOperationKind.OBSERVATION_CLOSE:
        if not types or types[-1] != TraceRecordType.OBSERVATION_TERMINAL:
            return False
        body = types[:-1]
        watermark_count = body.count(TraceRecordType.SCHEDULE_WATERMARK)
        if watermark_count > 1:
            return False
        if watermark_count == 1:
            if body[-1] != TraceRecordType.SCHEDULE_WATERMARK:
                return False
            body = body[:-1]
        return all(item == TraceRecordType.REUSE_EPOCH for item in body)
    return False


def _operation_request_digest(request: object, view_digest: str | None = None) -> str:
    hasher = sha256(canonical_json(request))
    if view_digest is not None:
        hasher.update(b"\n")
        hasher.update(view_digest.encode("ascii"))
    return hasher.hexdigest()


def _operation_record_id(
    operation_id: str,
    record: TraceRecord,
    local_index: int,
) -> str:
    hasher = sha256()
    for value in (
        record.trace_id,
        operation_id,
        record.record_type.value,
        str(local_index),
    ):
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\x00")
    hasher.update(canonical_json(record.payload))
    return f"c1-{hasher.hexdigest()}"


def _verify_operation_metadata(
    operation: TraceOperationCommit,
    records: tuple[TraceRecord, ...],
) -> None:
    if operation.record_ids != tuple(
        _operation_record_id(operation.operation_id, record, index)
        for index, record in enumerate(records)
    ):
        raise TraceValidationError("typed operation ID differs from its raw records")

    first = records[0]
    observation_ids = {record.observation_id for record in records}
    expected_view_digest: str | None
    if operation.kind == TraceOperationKind.PREAMBLE:
        header = first.payload
        topologies = tuple(record.payload for record in records[1:])
        if not isinstance(header, TraceHeaderPayload) or not all(
            isinstance(item, WorkflowTopologyPayload) for item in topologies
        ):
            raise TraceValidationError("typed preamble payloads changed type")
        request: object = TracePreambleRequest(
            operation_id=operation.operation_id,
            header=header,
            topologies=topologies,  # type: ignore[arg-type]
        )
        expected_event_count = 0
        expected_view_digest = canonical_digest(request)
        expected_request_digest = _operation_request_digest(request)
    elif operation.kind == TraceOperationKind.CUTOFF_ATTEMPT:
        cutoff = first.payload
        if (
            not isinstance(cutoff, CutoffPayload)
            or len(records) != 2
            or records[0].observation_id is None
        ):
            raise TraceValidationError("typed cutoff payloads changed type")
        request = CutoffCommitRequest(
            operation_id=operation.operation_id,
            observation_id=records[0].observation_id,
            attempt=records[1].payload,  # type: ignore[arg-type]
        )
        expected_event_count = cutoff.lifecycle_event_count
        expected_view_digest = cutoff.atomic_cutoff_view_digest
        expected_request_digest = _operation_request_digest(
            request,
            expected_view_digest,
        )
    elif operation.kind == TraceOperationKind.DEMAND_INTENT:
        payloads = tuple(record.payload for record in records)
        if (
            len(observation_ids) != 1
            or first.observation_id is None
            or not all(isinstance(item, DemandIntentPayload) for item in payloads)
        ):
            raise TraceValidationError("typed demand payloads changed type")
        event_counts = {item.pre_service_event_count for item in payloads}  # type: ignore[union-attr]
        if len(event_counts) != 1:
            raise TraceValidationError("typed demand spans runtime prefixes")
        request = DemandCommitRequest(
            operation_id=operation.operation_id,
            observation_id=first.observation_id,
            schedule_event_ids=tuple(
                item.schedule_event_id
                for item in payloads  # type: ignore[union-attr]
            ),
        )
        expected_event_count = next(iter(event_counts))
        # The full pre-service view is not duplicated in the v3 trace. Its digest is
        # writer-attested here and protected by the externally anchored final seal.
        expected_view_digest = None
        expected_request_digest = _operation_request_digest(
            request,
            operation.runtime_view_digest,
        )
    elif operation.kind == TraceOperationKind.OBSERVATION_CLOSE:
        terminal = records[-1].payload
        if (
            len(observation_ids) != 1
            or first.observation_id is None
            or not isinstance(terminal, ObservationTerminalPayload)
        ):
            raise TraceValidationError("typed close payloads changed type")
        epochs = tuple(
            record.payload
            for record in records[:-1]
            if isinstance(record.payload, ReuseEpochPayload)
        )
        services = tuple(
            sorted(
                (service for epoch in epochs for service in epoch.service_terminals),
                key=lambda item: item.intent_record_id,
            )
        )
        watermarks = tuple(
            record.payload
            for record in records[:-1]
            if isinstance(record.payload, ScheduleWatermarkPayload)
        )
        request = ObservationCloseRequest(
            operation_id=operation.operation_id,
            observation_id=first.observation_id,
            services=services,
            watermark=watermarks[0] if watermarks else None,
            terminal=ObservationTerminalSpec(
                status=terminal.status,
                reason=terminal.reason,
                label_available_ns=terminal.label_available_ns,
                last_verified_event_count=terminal.last_verified_event_count,
                last_verified_event_id=terminal.last_verified_event_id,
                last_verified_event_timestamp_ns=(
                    terminal.last_verified_event_timestamp_ns
                ),
            ),
        )
        expected_event_count = terminal.last_verified_event_count
        expected_view_digest = canonical_digest(request)
        expected_request_digest = _operation_request_digest(request)
    else:  # pragma: no cover - the operation enum is closed
        raise TraceValidationError("typed operation kind is unknown")

    if (
        operation.runtime_event_count != expected_event_count
        or operation.request_digest != expected_request_digest
        or (
            expected_view_digest is not None
            and operation.runtime_view_digest != expected_view_digest
        )
    ):
        raise TraceValidationError("typed operation metadata differs from raw records")


def _verify_typed_trace(
    raw: bytes,
    sealed: SealedTraceReceipt,
) -> tuple[TraceRecord, ...]:
    records = _parse_trace(raw)
    closure = sealed.closure
    if (
        sealed.trace_basename != TRACE_PAYLOAD
        or closure.record_count != len(records)
        or closure.size_bytes != len(raw)
        or closure.first_record_id != records[0].record_id
        or closure.last_record_id != records[-1].record_id
        or closure.stream_digest != sha256(raw).hexdigest()
    ):
        raise TraceValidationError("typed trace closure differs from raw bytes")
    encoded_lines = tuple(line + b"\n" for line in raw[:-1].split(b"\n"))
    offsets = [0]
    for line in encoded_lines:
        offsets.append(offsets[-1] + len(line))
    committed_stream = sha256()
    for operation in sealed.operations:
        start = operation.sequence_start
        end = operation.sequence_end
        if (
            end > len(records)
            or operation.byte_start != offsets[start]
            or operation.byte_end != offsets[end]
        ):
            raise TraceValidationError(
                "typed operation byte and sequence ranges differ"
            )
        batch_records = records[start:end]
        batch = raw[operation.byte_start : operation.byte_end]
        prior_stream_digest = committed_stream.hexdigest()
        committed_stream.update(batch)
        if (
            operation.record_ids != tuple(record.record_id for record in batch_records)
            or operation.batch_digest != sha256(batch).hexdigest()
            or operation.prior_stream_digest != prior_stream_digest
            or operation.committed_stream_digest != committed_stream.hexdigest()
            or not _operation_types_are_valid(operation.kind, batch_records)
        ):
            raise TraceValidationError("typed operation chain differs from trace bytes")
        _verify_operation_metadata(operation, batch_records)
    if sealed.operations[0].kind != TraceOperationKind.PREAMBLE:
        raise TraceValidationError(
            "typed trace does not start with a preamble operation"
        )
    return records


def _evidence_replay(
    root: Path,
    records: tuple[TraceRecord, ...],
    *,
    lifecycle_digest: str,
    verifier_digest: str,
) -> tuple[TraceValidationReceipt, TraceValidationReceipt, tuple[DemandLabel, ...]]:
    authorized = validate_trace_for_labels(
        records,
        lifecycle_gate=CanonicalLifecycleEvidenceGate(
            root / LIFECYCLE_PAYLOAD,
            lifecycle_digest,
            verifier_digest,
        ),
        schedule_gate=CanonicalScheduleEvidenceGate(
            root / SCHEDULE_PAYLOAD,
            verifier_digest,
        ),
    )
    lifecycle = authorized.lifecycle_validation
    schedule = authorized.schedule_validation
    if lifecycle is None or schedule is None:
        raise TraceValidationError("concrete evidence replay returned no receipts")
    labels = reconstruct_demand_labels(authorized)
    if tuple(label.observation_id for label in labels) != (
        lifecycle.verified_observation_ids
    ):
        raise TraceValidationError(
            "batch label reconstruction changed observation order"
        )
    return lifecycle, schedule, labels


def _identity(basename: str, raw: bytes) -> C1BundleFileIdentity:
    return C1BundleFileIdentity(
        basename=basename,
        size_bytes=len(raw),
        sha256=sha256(raw).hexdigest(),
    )


def finalize_c1_b0_bundle(
    root: Path,
    *,
    bundle_id: str,
    trace_committer: CanonicalTraceCommitter,
    protocol_digest: str,
    verifier_digest: str,
    expected_implementation_digest: str,
    expected_environment_digest: str,
) -> ValidatedC1B0Bundle:
    """Publish descriptors, manifest, and final seal for three closed payloads."""

    require_text("bundle_id", bundle_id)
    require_sha256("protocol_digest", protocol_digest)
    require_sha256("verifier_digest", verifier_digest)
    require_sha256(
        "expected_implementation_digest",
        expected_implementation_digest,
    )
    require_sha256("expected_environment_digest", expected_environment_digest)
    if type(trace_committer) is not CanonicalTraceCommitter:
        raise TraceValidationError(
            "C1-B0 finalization requires the concrete canonical trace committer"
        )
    if trace_committer.path != root / TRACE_PAYLOAD:
        raise TraceValidationError(
            "typed trace committer does not own the fixed bundle trace path"
        )
    root_descriptor, opened_root = _open_directory(root, require_read_only=False)
    parent_descriptor: int | None = None
    final_seal_digest: str | None = None
    try:
        parent_descriptor = _open_parent_directory(root, opened_root)
        _require_inventory(root_descriptor, PAYLOAD_NAMES)
        initial_payload_raws = tuple(
            _read_stable_file(
                root_descriptor,
                name,
                max_bytes=_max_file_bytes(name),
                require_read_only=False,
            )
            for name in PAYLOAD_NAMES
        )
        attempt = C1B0FinalizationAttempt(
            schema_version=C1_B0_ATTEMPT_SCHEMA_VERSION,
            bundle_id=bundle_id,
            claim_scope=C1_B0_CLAIM_SCOPE,
            protocol_digest=protocol_digest,
            verifier_digest=verifier_digest,
            implementation_digest=expected_implementation_digest,
            environment_digest=expected_environment_digest,
            payload_basenames=PAYLOAD_NAMES,
            payload_sizes_bytes=tuple(len(raw) for raw in initial_payload_raws),
            payload_sha256s=tuple(
                sha256(raw).hexdigest() for raw in initial_payload_raws
            ),
        )
        _write_create_only(root_descriptor, ATTEMPT, canonical_json(attempt))
        sealed = trace_committer.seal_trace()
        sealed = trace_committer.snapshot_sealed_receipt(sealed)
        current_payload_raws = tuple(
            _read_stable_file(
                root_descriptor,
                name,
                max_bytes=_max_file_bytes(name),
                require_read_only=False,
            )
            for name in PAYLOAD_NAMES
        )
        if current_payload_raws != initial_payload_raws:
            raise TraceValidationError(
                "C1-B0 payload changed after finalization attempt started"
            )
        schedule_raw, lifecycle_raw, trace_raw = initial_payload_raws
        schedule_artifact = parse_canonical_dataclass(
            schedule_raw,
            ClosedScheduleArtifact,
            artifact_name="bundle schedule artifact",
            max_bytes=MAX_SIDECAR_BYTES,
        )
        lifecycle_artifact = parse_canonical_dataclass(
            lifecycle_raw,
            ClosedLifecycleArtifact,
            artifact_name="bundle lifecycle artifact",
            max_bytes=MAX_SIDECAR_BYTES,
        )
        if type(schedule_artifact.closure) is not ReplayScheduleClosure:
            raise TraceValidationError(
                "C1-B0 bundle accepts only a controlled replay schedule"
            )
        schedule_digest = sha256(schedule_raw).hexdigest()
        if (
            schedule_artifact != trace_committer.schedule
            or schedule_digest != trace_committer.schedule_artifact_digest
        ):
            raise TraceValidationError("bundle schedule differs from typed committer")
        normalized_sealed = SealedTraceReceipt(
            trace_pair_id=sealed.trace_pair_id,
            trace_basename=TRACE_PAYLOAD,
            closure=sealed.closure,
            operations=sealed.operations,
        )
        records = _verify_typed_trace(trace_raw, normalized_sealed)
        lifecycle_digest = sha256(lifecycle_raw).hexdigest()
        lifecycle_receipt, schedule_receipt, labels = _evidence_replay(
            root,
            records,
            lifecycle_digest=lifecycle_digest,
            verifier_digest=verifier_digest,
        )
        header = records[0].payload
        if not isinstance(header, TraceHeaderPayload):
            raise TraceValidationError("bundle trace header changed type")
        if (
            header.implementation_digest != expected_implementation_digest
            or header.environment_digest != expected_environment_digest
            or lifecycle_artifact.phase != C1_B0_LIFECYCLE_PHASE
            or lifecycle_artifact.source != C1_B0_LIFECYCLE_SOURCE
            or lifecycle_artifact.implementation_digest
            != expected_implementation_digest
            or lifecycle_artifact.environment_digest != expected_environment_digest
        ):
            raise TraceValidationError(
                "C1-B0 lifecycle provenance differs from external anchors"
            )

        payload_raws = (schedule_raw, lifecycle_raw, trace_raw)
        payload_counts = (1, 1, len(records))
        previous_digest = EMPTY_SHA256
        refs: list[C1BundleSegmentRef] = []
        for index, ((role, payload_name, commit_name), raw, count) in enumerate(
            zip(SEGMENT_LAYOUT, payload_raws, payload_counts, strict=True)
        ):
            current_raw = _read_stable_file(
                root_descriptor,
                payload_name,
                max_bytes=(
                    MAX_TRACE_BYTES
                    if role == C1BundleSegmentRole.TRACE
                    else MAX_SIDECAR_BYTES
                ),
                require_read_only=False,
            )
            if current_raw != raw:
                raise TraceValidationError(
                    "bundle payload changed before segment commit"
                )
            segment = C1BundleSegmentCommit(
                schema_version=C1_B0_SEGMENT_SCHEMA_VERSION,
                bundle_id=bundle_id,
                trace_pair_id=header.trace_pair_id,
                segment_index=index,
                role=role,
                payload_basename=payload_name,
                payload_size_bytes=len(raw),
                payload_record_count=count,
                payload_sha256=sha256(raw).hexdigest(),
                previous_segment_commit_sha256=previous_digest,
                sealed_trace=(
                    normalized_sealed if role == C1BundleSegmentRole.TRACE else None
                ),
            )
            segment_raw = canonical_json(segment)
            segment_digest = _write_create_only(
                root_descriptor,
                commit_name,
                segment_raw,
            )
            refs.append(
                C1BundleSegmentRef(
                    segment_index=index,
                    role=role,
                    commit_basename=commit_name,
                    commit_sha256=segment_digest,
                )
            )
            previous_digest = segment_digest

        manifest = C1B0BundleManifest(
            schema_version=C1_B0_MANIFEST_SCHEMA_VERSION,
            bundle_id=bundle_id,
            status=C1_B0_STATUS,
            claim_scope=C1_B0_CLAIM_SCOPE,
            open_gates=C1_B0_OPEN_GATES,
            trace_pair_id=header.trace_pair_id,
            run_id=records[0].run_id,
            schedule_id=records[0].schedule_id,
            schedule_case_id=records[0].schedule_case_id,
            protocol_digest=protocol_digest,
            verifier_digest=verifier_digest,
            implementation_digest=expected_implementation_digest,
            environment_digest=expected_environment_digest,
            segments=tuple(refs),
            lifecycle_validation=lifecycle_receipt,
            schedule_validation=schedule_receipt,
            demand_labels=labels,
        )
        manifest_raw = canonical_json(manifest)
        manifest_digest = _write_create_only(
            root_descriptor,
            MANIFEST,
            manifest_raw,
        )
        preseal_raws = {
            name: _read_stable_file(
                root_descriptor,
                name,
                max_bytes=_max_file_bytes(name),
                require_read_only=False,
            )
            for name in PRESEAL_NAMES
        }
        final_seal = C1B0FinalSeal(
            schema_version=C1_B0_FINAL_SEAL_SCHEMA_VERSION,
            bundle_id=bundle_id,
            status=C1_B0_STATUS,
            claim_scope=C1_B0_CLAIM_SCOPE,
            manifest_basename=MANIFEST,
            manifest_size_bytes=len(manifest_raw),
            manifest_sha256=manifest_digest,
            segment_count=len(SEGMENT_LAYOUT),
            last_segment_commit_sha256=previous_digest,
            preseal_files=tuple(
                _identity(name, preseal_raws[name]) for name in PRESEAL_NAMES
            ),
        )
        final_raw = canonical_json(final_seal)
        final_seal_digest = _write_create_only(
            root_descriptor,
            FINAL_SEAL,
            final_raw,
        )
        for name in FINAL_INVENTORY:
            _seal_child_read_only(root_descriptor, name)
        os.fchmod(root_descriptor, 0o550)
        os.fsync(root_descriptor)
        os.fsync(parent_descriptor)
        _assert_directory_identity(root, root_descriptor, opened_root)
    except BaseException as exc:
        if isinstance(exc, OSError):
            raise TraceCommitIndeterminateError(
                "C1-B0 bundle finalization durability is indeterminate"
            ) from exc
        raise
    finally:
        try:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
        finally:
            os.close(root_descriptor)
    if final_seal_digest is None:  # pragma: no cover - protected by exceptions
        raise TraceCommitIndeterminateError("C1-B0 final seal was not published")
    return validate_c1_b0_bundle(
        root,
        expected_final_seal_sha256=final_seal_digest,
        expected_protocol_digest=protocol_digest,
        expected_verifier_digest=verifier_digest,
        expected_implementation_digest=expected_implementation_digest,
        expected_environment_digest=expected_environment_digest,
    )


def validate_c1_b0_bundle(
    root: Path,
    *,
    expected_final_seal_sha256: str,
    expected_protocol_digest: str,
    expected_verifier_digest: str,
    expected_implementation_digest: str,
    expected_environment_digest: str,
) -> ValidatedC1B0Bundle:
    """Freshly replay a sealed bundle from raw bytes and external anchors."""

    require_sha256("expected_final_seal_sha256", expected_final_seal_sha256)
    require_sha256("expected_protocol_digest", expected_protocol_digest)
    require_sha256("expected_verifier_digest", expected_verifier_digest)
    require_sha256(
        "expected_implementation_digest",
        expected_implementation_digest,
    )
    require_sha256("expected_environment_digest", expected_environment_digest)
    root_descriptor, opened_root = _open_directory(root, require_read_only=True)
    try:
        _require_inventory(root_descriptor, FINAL_INVENTORY)
        snapshots = {
            name: _read_stable_snapshot(
                root_descriptor,
                name,
                max_bytes=_max_file_bytes(name),
                require_read_only=True,
            )
            for name in FINAL_INVENTORY
        }
        raws = {name: snapshot.raw for name, snapshot in snapshots.items()}
        final_raw = raws[FINAL_SEAL]
        final_digest = sha256(final_raw).hexdigest()
        if final_digest != expected_final_seal_sha256:
            raise TraceValidationError("C1-B0 final seal differs from external anchor")
        final_seal = parse_canonical_dataclass(
            final_raw,
            C1B0FinalSeal,
            artifact_name="C1-B0 final seal",
            max_bytes=MAX_BUNDLE_JSON_BYTES,
        )
        attempt = parse_canonical_dataclass(
            raws[ATTEMPT],
            C1B0FinalizationAttempt,
            artifact_name="C1-B0 finalization attempt",
            max_bytes=MAX_BUNDLE_JSON_BYTES,
        )
        for name, size_bytes, digest in zip(
            attempt.payload_basenames,
            attempt.payload_sizes_bytes,
            attempt.payload_sha256s,
            strict=True,
        ):
            raw = raws[name]
            if len(raw) != size_bytes or sha256(raw).hexdigest() != digest:
                raise TraceValidationError(
                    "C1-B0 payload differs from finalization attempt"
                )
        manifest_raw = raws[MANIFEST]
        manifest_digest = sha256(manifest_raw).hexdigest()
        if (
            manifest_digest != final_seal.manifest_sha256
            or len(manifest_raw) != final_seal.manifest_size_bytes
        ):
            raise TraceValidationError("C1-B0 manifest differs from final seal")
        for identity in final_seal.preseal_files:
            raw = raws[identity.basename]
            if (
                len(raw) != identity.size_bytes
                or sha256(raw).hexdigest() != identity.sha256
            ):
                raise TraceValidationError("C1-B0 preseal file identity differs")
        manifest = parse_canonical_dataclass(
            manifest_raw,
            C1B0BundleManifest,
            artifact_name="C1-B0 manifest",
            max_bytes=MAX_BUNDLE_JSON_BYTES,
        )
        if (
            manifest.bundle_id != final_seal.bundle_id
            or manifest.status != final_seal.status
            or attempt.bundle_id != manifest.bundle_id
            or attempt.protocol_digest != expected_protocol_digest
            or attempt.verifier_digest != expected_verifier_digest
            or attempt.implementation_digest != expected_implementation_digest
            or attempt.environment_digest != expected_environment_digest
            or manifest.protocol_digest != expected_protocol_digest
            or manifest.verifier_digest != expected_verifier_digest
            or manifest.implementation_digest != expected_implementation_digest
            or manifest.environment_digest != expected_environment_digest
        ):
            raise TraceValidationError("C1-B0 manifest differs from external anchors")

        previous_digest = EMPTY_SHA256
        segments: list[C1BundleSegmentCommit] = []
        for index, (layout, reference) in enumerate(
            zip(SEGMENT_LAYOUT, manifest.segments, strict=True)
        ):
            role, payload_name, commit_name = layout
            commit_raw = raws[commit_name]
            commit_digest = sha256(commit_raw).hexdigest()
            if reference.commit_sha256 != commit_digest:
                raise TraceValidationError("C1-B0 segment reference digest differs")
            segment = parse_canonical_dataclass(
                commit_raw,
                C1BundleSegmentCommit,
                artifact_name=f"C1-B0 {role.value} segment",
                max_bytes=MAX_BUNDLE_JSON_BYTES,
            )
            payload_raw = raws[payload_name]
            if (
                segment.bundle_id != manifest.bundle_id
                or segment.trace_pair_id != manifest.trace_pair_id
                or segment.segment_index != index
                or segment.previous_segment_commit_sha256 != previous_digest
                or segment.payload_size_bytes != len(payload_raw)
                or segment.payload_sha256 != sha256(payload_raw).hexdigest()
                or segment.payload_record_count
                != (
                    len(_parse_trace(payload_raw))
                    if role == C1BundleSegmentRole.TRACE
                    else 1
                )
            ):
                raise TraceValidationError("C1-B0 segment commit chain differs")
            segments.append(segment)
            previous_digest = commit_digest
        if previous_digest != final_seal.last_segment_commit_sha256:
            raise TraceValidationError("C1-B0 final segment differs from final seal")

        schedule_artifact = parse_canonical_dataclass(
            raws[SCHEDULE_PAYLOAD],
            ClosedScheduleArtifact,
            artifact_name="C1-B0 schedule payload",
            max_bytes=MAX_SIDECAR_BYTES,
        )
        lifecycle_artifact = parse_canonical_dataclass(
            raws[LIFECYCLE_PAYLOAD],
            ClosedLifecycleArtifact,
            artifact_name="C1-B0 lifecycle payload",
            max_bytes=MAX_SIDECAR_BYTES,
        )
        if type(schedule_artifact.closure) is not ReplayScheduleClosure:
            raise TraceValidationError(
                "C1-B0 bundle accepts only a controlled replay schedule"
            )
        sealed = segments[-1].sealed_trace
        if sealed is None:  # pragma: no cover - segment dataclass closes this
            raise TraceValidationError("C1-B0 trace segment lacks closure")
        records = _verify_typed_trace(raws[TRACE_PAYLOAD], sealed)
        header = records[0].payload
        if not isinstance(header, TraceHeaderPayload):
            raise TraceValidationError("C1-B0 trace lacks a header payload")
        if (
            manifest.trace_pair_id != header.trace_pair_id
            or manifest.run_id != records[0].run_id
            or manifest.schedule_id != records[0].schedule_id
            or manifest.schedule_case_id != records[0].schedule_case_id
            or schedule_artifact.trace_pair_id != manifest.trace_pair_id
            or schedule_artifact.run_id != manifest.run_id
            or schedule_artifact.schedule_id != manifest.schedule_id
            or schedule_artifact.schedule_case_id != manifest.schedule_case_id
            or lifecycle_artifact.trace_pair_id != manifest.trace_pair_id
            or lifecycle_artifact.run_id != manifest.run_id
            or header.implementation_digest != expected_implementation_digest
            or header.environment_digest != expected_environment_digest
            or lifecycle_artifact.phase != C1_B0_LIFECYCLE_PHASE
            or lifecycle_artifact.source != C1_B0_LIFECYCLE_SOURCE
            or lifecycle_artifact.implementation_digest
            != expected_implementation_digest
            or lifecycle_artifact.environment_digest != expected_environment_digest
        ):
            raise TraceValidationError("C1-B0 cross-artifact identity differs")
        lifecycle_receipt, schedule_receipt, labels = _evidence_replay(
            root,
            records,
            lifecycle_digest=sha256(raws[LIFECYCLE_PAYLOAD]).hexdigest(),
            verifier_digest=expected_verifier_digest,
        )
        if (
            lifecycle_receipt != manifest.lifecycle_validation
            or schedule_receipt != manifest.schedule_validation
            or labels != manifest.demand_labels
            or trace_stream_digest(records) != sha256(raws[TRACE_PAYLOAD]).hexdigest()
        ):
            raise TraceValidationError(
                "C1-B0 fresh evidence replay differs from manifest"
            )
        _require_inventory(root_descriptor, FINAL_INVENTORY)
        final_snapshots = {
            name: _read_stable_snapshot(
                root_descriptor,
                name,
                max_bytes=_max_file_bytes(name),
                require_read_only=True,
            )
            for name in FINAL_INVENTORY
        }
        if final_snapshots != snapshots:
            raise TraceValidationError(
                "C1-B0 bundle changed during independent verification"
            )
        _assert_directory_identity(root, root_descriptor, opened_root)
        return ValidatedC1B0Bundle(
            bundle_id=manifest.bundle_id,
            status=manifest.status,
            claim_scope=manifest.claim_scope,
            open_gates=manifest.open_gates,
            trace_pair_id=manifest.trace_pair_id,
            manifest_sha256=manifest_digest,
            final_seal_sha256=final_digest,
            schedule_sha256=sha256(raws[SCHEDULE_PAYLOAD]).hexdigest(),
            lifecycle_sha256=sha256(raws[LIFECYCLE_PAYLOAD]).hexdigest(),
            trace_sha256=sha256(raws[TRACE_PAYLOAD]).hexdigest(),
            verified_observation_ids=(lifecycle_receipt.verified_observation_ids),
            demand_labels=labels,
        )
    finally:
        os.close(root_descriptor)
