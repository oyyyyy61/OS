#!/usr/bin/env python3
"""Freeze the preregistered M2 tolerance from one closed calibration cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.aggregate_m2_calibration import _validate_run
    from tools.m2_calibration_evidence import (
        CALIBRATION_COHORT_SCHEMA,  # noqa: F401
        CALIBRATION_RUN_COUNT,  # noqa: F401
        MANIFEST_FIELDS,
        MANIFEST_RUN_FIELDS,
        MAX_FORMAL_ATOL,
        PROTOCOL_SCHEMA,  # noqa: F401
        CalibrationEvidenceError,
        validate_published_calibration_bundle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from aggregate_m2_calibration import _validate_run  # type: ignore[no-redef]
    from m2_calibration_evidence import (  # type: ignore[no-redef]
        CALIBRATION_COHORT_SCHEMA,  # noqa: F401
        CALIBRATION_RUN_COUNT,  # noqa: F401
        MANIFEST_FIELDS,
        MANIFEST_RUN_FIELDS,
        MAX_FORMAL_ATOL,
        PROTOCOL_SCHEMA,  # noqa: F401
        CalibrationEvidenceError,
        validate_published_calibration_bundle,
    )

TOLERANCE_SCHEMA = "dagkv.m2.frozen_tolerance.v2"
FORMAL_RTOL = 0.0
TOLERANCE_DERIVATION = "fixed_binary_cap_from_excluded_pilot"

COHORT_FIELDS = MANIFEST_FIELDS
COHORT_RUN_FIELDS = MANIFEST_RUN_FIELDS
TOLERANCE_FIELDS = frozenset(
    {
        "schema_version",
        "frozen",
        "frozen_at_utc",
        "atol",
        "rtol",
        "calibration_manifest_sha256",
        "reproducibility_fingerprint",
        "calibration_run_count",
        "derivation",
    }
)


class ToleranceFreezeError(RuntimeError):
    """Raised when cohort evidence cannot support an immutable tolerance."""


@dataclass(frozen=True, slots=True)
class CohortEvidence:
    manifest_sha256: str
    reproducibility_fingerprint: str
    run_count: int


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ToleranceFreezeError(message)


def validate_calibration_cohort(
    path: Path, *, expected_manifest_sha256: str | None = None
) -> CohortEvidence:
    """Revalidate and content-address one complete calibration bundle."""

    try:
        payload, manifest_sha256, evidence = validate_published_calibration_bundle(
            path,
            run_validator=_validate_run,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except CalibrationEvidenceError as exc:
        raise ToleranceFreezeError(str(exc)) from exc
    return CohortEvidence(
        manifest_sha256=manifest_sha256,
        reproducibility_fingerprint=payload["reproducibility_fingerprint"],
        run_count=len(evidence.runs),
    )


def _frozen_timestamp(value: datetime | None) -> str:
    now = datetime.now(UTC)
    timestamp = value or now
    require(
        timestamp.tzinfo is not None and timestamp.utcoffset() is not None,
        "frozen timestamp must include a timezone",
    )
    timestamp = timestamp.astimezone(UTC)
    require(timestamp <= now, "frozen timestamp cannot be in the future")
    return timestamp.isoformat()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    require(set(payload) == TOLERANCE_FIELDS, "frozen tolerance fields differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(
        not os.path.lexists(path), f"refusing to overwrite frozen tolerance: {path}"
    )
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    linked = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ToleranceFreezeError(
                f"refusing to overwrite frozen tolerance: {path}"
            ) from exc
        linked = True
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if linked:
            _fsync_parent(path)


def freeze_tolerance(
    calibration_manifest: Path,
    output_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    frozen_at: datetime | None = None,
) -> dict[str, Any]:
    """Validate cohort evidence and exclusively publish its frozen tolerance."""

    calibration_manifest = calibration_manifest.expanduser().absolute()
    raw_output = output_path.expanduser().absolute()
    require(
        not os.path.lexists(raw_output),
        f"refusing to overwrite frozen tolerance: {raw_output}",
    )
    output_path = raw_output.resolve(strict=False)
    campaign_root = calibration_manifest.parent.resolve()
    require(
        output_path != campaign_root and not output_path.is_relative_to(campaign_root),
        "frozen tolerance output must be outside the calibration campaign",
    )
    require(
        not os.path.lexists(output_path),
        f"refusing to overwrite frozen tolerance: {output_path}",
    )
    evidence = validate_calibration_cohort(
        calibration_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    payload = {
        "schema_version": TOLERANCE_SCHEMA,
        "frozen": True,
        "frozen_at_utc": _frozen_timestamp(frozen_at),
        "atol": MAX_FORMAL_ATOL,
        "rtol": FORMAL_RTOL,
        "calibration_manifest_sha256": evidence.manifest_sha256,
        "reproducibility_fingerprint": evidence.reproducibility_fingerprint,
        "calibration_run_count": evidence.run_count,
        "derivation": TOLERANCE_DERIVATION,
    }
    _write_json_exclusive(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-calibration-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output
    try:
        freeze_tolerance(
            args.calibration_manifest,
            output,
            expected_manifest_sha256=args.expected_calibration_manifest_sha256,
        )
    except (ToleranceFreezeError, OSError) as exc:
        print(f"M2 tolerance freeze failed: {exc}", file=sys.stderr)
        return 1
    print(f"M2 frozen tolerance created: {output.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
