#!/usr/bin/env python3
"""Aggregate a closed M2 calibration campaign into one frozen cohort manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_SCHEMA = "dagkv.m2.vllm_abba.v2"
CALIBRATION_COHORT_SCHEMA = "dagkv.m2.calibration_cohort.v1"
CALIBRATION_RUN_COUNT = 59
MAX_FORMAL_ATOL = 0.125
FORMAL_RTOL = 0.0

EXPECTED_MEASUREMENTS = ("A1", "G", "B1", "B2", "A2")
TOLERANT_PAIRS = (("A1", "G"), ("A1", "B1"), ("A1", "B2"))
EXACT_PAIRS = (("A1", "A2"), ("G", "B1"), ("G", "B2"), ("B1", "B2"))
EXPECTED_PAIRS = frozenset((*TOLERANT_PAIRS, *EXACT_PAIRS))
FORBIDDEN_ACCEPTANCE_FILES = frozenset(
    {
        "M2_ITEM8_FORMAL_RUN_MANIFEST.json",
        "M2_ITEM8_ACCEPTANCE_MANIFEST.json",
        "M2_ACCEPTANCE_MANIFEST.json",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "diagnostic_transfers.jsonl",
        "execution_ids.json",
        "logits_A1.npy",
        "logits_A2.npy",
        "logits_B1.npy",
        "logits_B2.npy",
        "logits_G.npy",
        "native_lifecycle.jsonl",
        "protocol.md",
        "provenance.json",
        "result.json",
    }
)


class CalibrationAggregationError(RuntimeError):
    """Raised when any campaign artifact violates the frozen cohort contract."""


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    run_id: str
    result_sha256: str
    provenance_sha256: str
    sha256sums_sha256: str
    reproducibility_fingerprint: str
    observed_max_abs_error: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationAggregationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CalibrationAggregationError(f"failed to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_digest(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationAggregationError(
            f"cannot canonicalize reproducibility components: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationAggregationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CalibrationAggregationError(f"non-finite JSON constant: {value}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except CalibrationAggregationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationAggregationError(f"invalid {label} at {path}: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _lower_sha256(value: Any, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _finite_number(value: Any, *, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def _safe_checksum_name(value: str, *, checksum_path: Path) -> str:
    require(value != "", f"empty checksum path in {checksum_path}")
    require("\\" not in value, f"non-POSIX checksum path in {checksum_path}: {value}")
    pure = PurePosixPath(value)
    require(
        not pure.is_absolute()
        and pure.as_posix() == value
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"unsafe checksum path in {checksum_path}: {value}",
    )
    require(value != "SHA256SUMS", "SHA256SUMS cannot list itself")
    return value


def _validate_sha256sums(run_dir: Path) -> tuple[dict[str, str], str]:
    checksum_path = run_dir / "SHA256SUMS"
    require(
        checksum_path.is_file() and not checksum_path.is_symlink(),
        f"missing SHA256SUMS: {run_dir}",
    )
    try:
        raw = checksum_path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise CalibrationAggregationError(
            f"invalid SHA256SUMS at {checksum_path}: {exc}"
        ) from exc
    require(raw.endswith(b"\n"), f"unterminated SHA256SUMS: {checksum_path}")
    require(b"\r" not in raw, f"SHA256SUMS must use LF newlines: {checksum_path}")
    lines = text[:-1].split("\n")
    require(lines and all(lines), f"SHA256SUMS contains a blank row: {checksum_path}")

    entries: dict[str, str] = {}
    ordered_names: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        require(
            len(line) > 66 and line[64:66] == "  ",
            f"malformed SHA256SUMS row {checksum_path}:{line_number}",
        )
        digest = _lower_sha256(
            line[:64], label=f"SHA256SUMS digest at {checksum_path}:{line_number}"
        )
        name = _safe_checksum_name(line[66:], checksum_path=checksum_path)
        require(name not in entries, f"duplicate SHA256SUMS path: {name}")
        entries[name] = digest
        ordered_names.append(name)
    require(
        ordered_names == sorted(ordered_names),
        f"SHA256SUMS paths are not sorted: {checksum_path}",
    )

    actual_names: set[str] = set()
    for path in run_dir.rglob("*"):
        require(not path.is_symlink(), f"run artifact cannot be a symlink: {path}")
        if path.is_file() and path != checksum_path:
            actual_names.add(path.relative_to(run_dir).as_posix())
    require(
        set(entries) == actual_names,
        f"SHA256SUMS coverage mismatch in {run_dir}: "
        f"missing={sorted(actual_names - set(entries))}, "
        f"extra={sorted(set(entries) - actual_names)}",
    )

    for name, expected in entries.items():
        path = run_dir / PurePosixPath(name)
        observed = sha256_file(path)
        require(
            observed == expected,
            f"checksum mismatch for {path}: expected {expected}, observed {observed}",
        )
    return entries, sha256_file(checksum_path)


def _validate_result(result: dict[str, Any], *, run_dir: Path) -> tuple[str, float]:
    require(
        result.get("schema_version") == PROTOCOL_SCHEMA,
        f"run uses a non-v2 protocol: {run_dir}",
    )
    require(result.get("mode") == "calibration", f"run is not calibration: {run_dir}")
    require(
        result.get("gate_status") == "CALIBRATED_NOT_ACCEPTED",
        f"run did not complete calibration: {run_dir}",
    )
    require(result.get("m2_accepted") is False, f"run claims M2 acceptance: {run_dir}")
    require(
        result.get("m2_item8_accepted") is False,
        f"calibration claims item 8 acceptance: {run_dir}",
    )
    require(
        result.get("formal_run_passed") is False,
        f"calibration claims a formal pass: {run_dir}",
    )
    require(
        type(result.get("within_requested_tolerance")) is bool,
        f"calibration tolerance status is invalid: {run_dir}",
    )
    run_id = result.get("run_id")
    require(isinstance(run_id, str) and run_id, f"invalid run_id: {run_dir}")

    tolerance = result.get("tolerance")
    require(isinstance(tolerance, dict), f"missing calibration tolerance: {run_dir}")
    atol = _finite_number(tolerance.get("atol"), label=f"atol in {run_dir}")
    rtol = _finite_number(tolerance.get("rtol"), label=f"rtol in {run_dir}")
    require(0.0 <= atol <= MAX_FORMAL_ATOL, f"calibration atol exceeds cap: {run_dir}")
    require(rtol == FORMAL_RTOL, f"calibration rtol must be zero: {run_dir}")

    measurements = result.get("measurements")
    require(isinstance(measurements, dict), f"measurements are missing: {run_dir}")
    require(
        set(measurements) == set(EXPECTED_MEASUREMENTS),
        f"measurement phases differ from the frozen protocol: {run_dir}",
    )
    token_ids: list[int] = []
    margins: list[float] = []
    for phase in EXPECTED_MEASUREMENTS:
        measurement = measurements[phase]
        require(
            isinstance(measurement, dict), f"invalid {phase} measurement: {run_dir}"
        )
        token_id = measurement.get("token_id")
        require(
            type(token_id) is int and token_id >= 0,
            f"invalid {phase} token_id: {run_dir}",
        )
        margin = _finite_number(
            measurement.get("top1_margin"), label=f"{phase} top1 margin in {run_dir}"
        )
        require(margin > 0.25, f"{phase} top1 margin is not above 0.25: {run_dir}")
        expected_cached_tokens = 0 if phase in {"A1", "A2"} else 16
        require(
            measurement.get("num_cached_tokens") == expected_cached_tokens,
            f"{phase} cached-token count differs from the protocol: {run_dir}",
        )
        token_ids.append(token_id)
        margins.append(margin)
    require(len(set(token_ids)) == 1, f"measurement tokens differ: {run_dir}")
    observed_minimum_margin = min(margins)
    reported_minimum_margin = _finite_number(
        result.get("minimum_top1_margin"),
        label=f"minimum_top1_margin in {run_dir}",
    )
    require(
        reported_minimum_margin == observed_minimum_margin,
        f"reported minimum top-1 margin is inconsistent: {run_dir}",
    )

    comparisons = result.get("comparisons")
    require(isinstance(comparisons, list), f"comparisons are missing: {run_dir}")
    seen: dict[tuple[str, str], float] = {}
    for comparison in comparisons:
        require(isinstance(comparison, dict), f"invalid comparison row: {run_dir}")
        pair = (comparison.get("left"), comparison.get("right"))
        require(pair in EXPECTED_PAIRS, f"unexpected comparison {pair}: {run_dir}")
        require(pair not in seen, f"duplicate comparison {pair}: {run_dir}")
        require(
            comparison.get("token_equal") is True,
            f"comparison tokens differ for {pair}: {run_dir}",
        )
        require(
            type(comparison.get("allclose")) is bool,
            f"comparison allclose is invalid for {pair}: {run_dir}",
        )
        max_abs_error = _finite_number(
            comparison.get("max_abs_error"),
            label=f"max_abs_error for {pair} in {run_dir}",
        )
        max_rel_error = _finite_number(
            comparison.get("max_rel_error"),
            label=f"max_rel_error for {pair} in {run_dir}",
        )
        require(
            max_abs_error >= 0.0 and max_rel_error >= 0.0,
            f"comparison error is negative for {pair}: {run_dir}",
        )
        if pair in EXACT_PAIRS:
            require(
                max_abs_error == 0.0,
                f"exact comparison drifted for {pair}: {run_dir}",
            )
            require(
                comparison["allclose"] is True,
                f"exact comparison is not allclose for {pair}: {run_dir}",
            )
        else:
            require(
                max_abs_error <= MAX_FORMAL_ATOL,
                f"tolerant comparison exceeds 0.125 for {pair}: {run_dir}",
            )
        seen[pair] = max_abs_error
    require(set(seen) == EXPECTED_PAIRS, f"comparison pairs are incomplete: {run_dir}")
    return run_id, max(seen.values(), default=0.0)


def _validate_provenance(
    provenance: dict[str, Any], *, run_id: str, result: dict[str, Any], run_dir: Path
) -> str:
    require(
        provenance.get("schema_version") == PROTOCOL_SCHEMA,
        f"provenance uses a non-v2 protocol: {run_dir}",
    )
    require(
        provenance.get("mode") == "calibration",
        f"provenance mode differs from calibration: {run_dir}",
    )
    require(
        provenance.get("full_provenance") is True,
        f"v2 calibration lacks full provenance: {run_dir}",
    )
    require(provenance.get("run_id") == run_id, f"provenance run_id differs: {run_dir}")
    require(
        provenance.get("prompt_token_ids") == list(range(1000, 1017))
        and provenance.get("block_size") == 16,
        f"provenance prompt profile drifted: {run_dir}",
    )
    require(
        provenance.get("tolerance") == result.get("tolerance"),
        f"result/provenance tolerance differs: {run_dir}",
    )

    components = provenance.get("reproducibility_components")
    require(
        isinstance(components, dict),
        f"reproducibility components are missing: {run_dir}",
    )
    fingerprint = _lower_sha256(
        provenance.get("reproducibility_fingerprint"),
        label=f"reproducibility_fingerprint in {run_dir}",
    )
    require(
        fingerprint == _canonical_digest(components),
        f"reproducibility fingerprint is inconsistent: {run_dir}",
    )
    require(
        result.get("reproducibility_fingerprint") == fingerprint,
        f"result/provenance reproducibility fingerprint differs: {run_dir}",
    )

    implementation = provenance.get("implementation")
    dagkv_git = provenance.get("dagkv_git")
    vllm_git = provenance.get("vllm_git")
    postflight = provenance.get("postflight")
    for label, value in (
        ("implementation", implementation),
        ("dagkv_git", dagkv_git),
        ("vllm_git", vllm_git),
        ("postflight", postflight),
    ):
        require(
            isinstance(value, dict), f"{label} is missing from provenance: {run_dir}"
        )
    assert isinstance(implementation, dict)
    assert isinstance(dagkv_git, dict)
    assert isinstance(vllm_git, dict)
    assert isinstance(postflight, dict)
    implementation_digest = _lower_sha256(
        implementation.get("manifest_sha256"),
        label=f"implementation manifest in {run_dir}",
    )
    dagkv_digest = _lower_sha256(
        dagkv_git.get("snapshot_sha256"), label=f"DAGKV snapshot in {run_dir}"
    )
    vllm_digest = _lower_sha256(
        vllm_git.get("snapshot_sha256"), label=f"vLLM snapshot in {run_dir}"
    )
    require(
        postflight.get("implementation_manifest_sha256") == implementation_digest,
        f"implementation changed during run: {run_dir}",
    )
    require(
        postflight.get("dagkv_git_snapshot_sha256") == dagkv_digest,
        f"DAGKV Git state changed during run: {run_dir}",
    )
    require(
        postflight.get("vllm_git_snapshot_sha256") == vllm_digest,
        f"vLLM Git state changed during run: {run_dir}",
    )
    require(
        postflight.get("model_file_stats_unchanged") is True,
        f"model files changed during run: {run_dir}",
    )
    require(
        postflight.get("runtime_binary_stats_unchanged") is True,
        f"runtime binaries changed during run: {run_dir}",
    )
    return fingerprint


def _validate_run(run_dir: Path) -> ValidatedRun:
    result_path = run_dir / "result.json"
    provenance_path = run_dir / "provenance.json"
    result = _read_json_object(result_path, label="result.json")
    run_id, observed_max = _validate_result(result, run_dir=run_dir)
    entries, checksum_digest = _validate_sha256sums(run_dir)
    require(
        "result.json" in entries, f"result.json is absent from SHA256SUMS: {run_dir}"
    )
    require(
        "provenance.json" in entries,
        f"provenance.json is absent from SHA256SUMS: {run_dir}",
    )
    require(
        FORBIDDEN_ACCEPTANCE_FILES.isdisjoint(entries),
        f"calibration run contains a formal acceptance artifact: {run_dir}",
    )
    require(
        REQUIRED_ARTIFACTS.issubset(entries),
        f"calibration artifact set is incomplete: {run_dir}; "
        f"missing={sorted(REQUIRED_ARTIFACTS - set(entries))}",
    )
    measurements = result["measurements"]
    for phase in EXPECTED_MEASUREMENTS:
        expected_name = f"logits_{phase}.npy"
        measurement = measurements[phase]
        require(
            measurement.get("logits_file") == expected_name,
            f"{phase} logits filename differs from the protocol: {run_dir}",
        )
        require(
            measurement.get("logits_sha256") == entries[expected_name],
            f"{phase} logits hash differs from SHA256SUMS: {run_dir}",
        )
    provenance = _read_json_object(provenance_path, label="provenance.json")
    fingerprint = _validate_provenance(
        provenance,
        run_id=run_id,
        result=result,
        run_dir=run_dir,
    )
    return ValidatedRun(
        run_id=run_id,
        result_sha256=entries["result.json"],
        provenance_sha256=entries["provenance.json"],
        sha256sums_sha256=checksum_digest,
        reproducibility_fingerprint=fingerprint,
        observed_max_abs_error=observed_max,
    )


def _discover_run_dirs(campaign_dir: Path) -> list[Path]:
    for path in campaign_dir.rglob("*"):
        require(not path.is_symlink(), f"campaign cannot contain a symlink: {path}")
    run_dirs = sorted(path for path in campaign_dir.iterdir() if path.is_dir())
    require(run_dirs, f"campaign contains no attempt directories: {campaign_dir}")
    for run_dir in run_dirs:
        require(
            (run_dir / "result.json").is_file(),
            f"campaign attempt lacks result.json: {run_dir}",
        )
        require(
            not list(run_dir.glob("*/result.json")),
            f"nested run directories are ambiguous: {run_dir}",
        )
    return run_dirs


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate_campaign(
    campaign_dir: Path, *, output_path: Path | None = None
) -> dict[str, Any]:
    """Validate every discovered run and atomically write the cohort manifest."""

    campaign_dir = campaign_dir.resolve()
    require(campaign_dir.is_dir(), f"campaign directory is missing: {campaign_dir}")
    run_dirs = _discover_run_dirs(campaign_dir)
    destination = (
        output_path.resolve()
        if output_path is not None
        else campaign_dir / "calibration_cohort.json"
    )
    require(
        destination.name not in {"result.json", "provenance.json", "SHA256SUMS"},
        f"unsafe cohort output name: {destination}",
    )
    require(
        all(not destination.is_relative_to(run_dir) for run_dir in run_dirs),
        "cohort output must be outside every run directory",
    )

    validated = [_validate_run(run_dir) for run_dir in run_dirs]
    require(
        len(validated) == CALIBRATION_RUN_COUNT,
        f"calibration cohort requires exactly {CALIBRATION_RUN_COUNT} runs; "
        f"observed {len(validated)}",
    )
    run_ids = {run.run_id for run in validated}
    result_hashes = {run.result_sha256 for run in validated}
    require(len(run_ids) == len(validated), "calibration run IDs must be unique")
    require(
        len(result_hashes) == len(validated),
        "calibration result hashes must be unique",
    )
    fingerprints = {run.reproducibility_fingerprint for run in validated}
    require(
        len(fingerprints) == 1,
        "calibration reproducibility fingerprints differ",
    )

    ordered = sorted(validated, key=lambda run: run.run_id)
    manifest = {
        "schema_version": CALIBRATION_COHORT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "pilot_excluded": True,
        "run_count": len(ordered),
        "all_passed": True,
        "failures": [],
        "observed_max_abs_error": max(run.observed_max_abs_error for run in ordered),
        "formal_atol": MAX_FORMAL_ATOL,
        "formal_rtol": FORMAL_RTOL,
        "reproducibility_fingerprint": next(iter(fingerprints)),
        "runs": [
            {
                "run_id": run.run_id,
                "result_sha256": run.result_sha256,
                "provenance_sha256": run.provenance_sha256,
                "sha256sums_sha256": run.sha256sums_sha256,
            }
            for run in ordered
        ],
    }
    _write_json_atomic(destination, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output or args.campaign_dir / "calibration_cohort.json"
    try:
        manifest = aggregate_campaign(args.campaign_dir, output_path=output)
    except (CalibrationAggregationError, OSError) as exc:
        print(f"M2 calibration aggregation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"M2 calibration cohort completed: {output.resolve()} "
        f"({manifest['run_count']} runs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
