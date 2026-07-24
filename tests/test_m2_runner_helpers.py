from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.run_m2_vllm_abba import (
    CALIBRATION_COHORT_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    MIN_CALIBRATION_RUNS,
    PROTOCOL_SCHEMA,
    TOLERANCE_DERIVATION,
    TOLERANCE_SCHEMA,
    M2ValidationError,
    _transfer_params,
    build_llm_kwargs,
    compare_logit_vectors,
    dense_logits_from_logprobs,
    load_calibration_cohort,
    load_frozen_tolerance,
    read_jsonl_strict,
    resolve_native_transfer_request_id,
    validate_abba_transfer_chain,
    validate_diagnostic_trace_closed_set,
    validate_diagnostic_transfer,
    validate_native_trace_closed_set,
    validate_native_transfer,
    validate_prefetch_result,
    validate_prompt_tokens,
)

np = pytest.importorskip("numpy")


@dataclass
class _Value:
    logprob: float


@dataclass
class _Flat:
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]

    def __len__(self) -> int:
        return len(self.start_indices)


def _endpoint(tier: str, slot: int, generation: int, digest: str) -> dict:
    return {
        "tier": tier,
        "physical_slot": slot,
        "allocation_generation": generation,
        "digest": digest,
    }


def _terminal(
    *,
    request_id: str,
    direction: str,
    source: dict,
    target: dict,
    job_id: int = 1,
) -> dict:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "terminal",
        "status": "completed",
        "run_id": "run",
        "job_id": job_id,
        "request_id": request_id,
        "direction": direction,
        "payload_bytes": 4096,
        "reported_bytes": 4096,
        "framing": "DAGKV_PAYLOAD_V1",
        "source": source,
        "target": target,
        "failure_reason": None,
    }


def test_prompt_is_exactly_one_block_plus_one() -> None:
    assert validate_prompt_tokens(range(17)) == tuple(range(17))
    with pytest.raises(M2ValidationError, match="exactly 17"):
        validate_prompt_tokens(range(16))
    with pytest.raises(M2ValidationError, match="non-negative"):
        validate_prompt_tokens([*range(16), -1])


def test_dense_logits_accepts_flat_duplicate_sampled_token() -> None:
    flat = _Flat(
        start_indices=[0],
        end_indices=[4],
        token_ids=[1, 0, 1, 2],
        logprobs=[2.0, 1.0, 2.0, -3.0],
    )
    vector = dense_logits_from_logprobs(flat, 3)
    np.testing.assert_array_equal(vector, np.array([1.0, 2.0, -3.0]))


def test_dense_logits_accepts_mapping_and_rejects_missing_vocab() -> None:
    vector = dense_logits_from_logprobs(
        [{0: _Value(1.0), 1: _Value(0.5)}],
        2,
    )
    np.testing.assert_array_equal(vector, np.array([1.0, 0.5]))
    with pytest.raises(M2ValidationError, match="cover"):
        dense_logits_from_logprobs([{0: _Value(1.0)}], 2)


def test_logit_comparison_reports_tolerance_and_token_state() -> None:
    left = np.array([1.0, 2.0])
    right = np.array([1.0, 2.0001])
    exact = compare_logit_vectors("A", 1, left, "B", 1, right, atol=0.0, rtol=0.0)
    tolerant = compare_logit_vectors("A", 1, left, "B", 2, right, atol=0.001, rtol=0.0)
    assert exact.allclose is False
    assert exact.token_equal is True
    assert tolerant.allclose is True
    assert tolerant.token_equal is False


def test_frozen_tolerance_must_predate_start(tmp_path: Path) -> None:
    fingerprint = "f" * 64
    cohort_path = tmp_path / "cohort.json"
    runs = [
        {
            "run_id": f"run-{index}",
            "result_sha256": hashlib.sha256(f"result-{index}".encode()).hexdigest(),
            "provenance_sha256": hashlib.sha256(
                f"provenance-{index}".encode()
            ).hexdigest(),
            "sha256sums_sha256": hashlib.sha256(
                f"checksums-{index}".encode()
            ).hexdigest(),
        }
        for index in range(MIN_CALIBRATION_RUNS)
    ]
    cohort_payload = {
        "schema_version": CALIBRATION_COHORT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "pilot_excluded": True,
        "run_count": MIN_CALIBRATION_RUNS,
        "all_passed": True,
        "failures": [],
        "observed_max_abs_error": 0.109375,
        "formal_atol": 0.125,
        "formal_rtol": 0.0,
        "reproducibility_fingerprint": fingerprint,
        "runs": runs,
    }
    cohort_path.write_text(json.dumps(cohort_payload), encoding="utf-8")

    tolerance_path = tmp_path / "tolerance.json"
    payload = {
        "schema_version": TOLERANCE_SCHEMA,
        "frozen": True,
        "frozen_at_utc": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        "atol": 0.125,
        "rtol": 0.0,
        "calibration_manifest_sha256": hashlib.sha256(
            cohort_path.read_bytes()
        ).hexdigest(),
        "reproducibility_fingerprint": fingerprint,
        "calibration_run_count": MIN_CALIBRATION_RUNS,
        "derivation": TOLERANCE_DERIVATION,
    }
    tolerance_path.write_text(json.dumps(payload), encoding="utf-8")
    run_started_ns = time.time_ns() + 1_000_000_000
    frozen = load_frozen_tolerance(
        tolerance_path,
        run_started_ns=run_started_ns,
    )
    assert frozen.atol == 0.125
    assert frozen.calibration_manifest_sha256 == payload["calibration_manifest_sha256"]
    cohort = load_calibration_cohort(
        cohort_path,
        frozen_tolerance=frozen,
        run_started_ns=run_started_ns,
    )
    assert cohort["run_count"] == MIN_CALIBRATION_RUNS

    with pytest.raises(M2ValidationError, match="before runner startup"):
        load_frozen_tolerance(
            tolerance_path,
            run_started_ns=tolerance_path.stat().st_mtime_ns,
        )

    cohort_payload["observed_max_abs_error"] = 0.25
    cohort_path.write_text(json.dumps(cohort_payload), encoding="utf-8")
    with pytest.raises(M2ValidationError, match="hash differs"):
        load_calibration_cohort(
            cohort_path,
            frozen_tolerance=frozen,
            run_started_ns=time.time_ns() + 1_000_000_000,
        )


def test_jsonl_reader_rejects_partial_rows(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"event":"ok"}', encoding="utf-8")
    with pytest.raises(M2ValidationError, match="unterminated"):
        read_jsonl_strict(path)


def test_prefetch_requires_one_external_block() -> None:
    validate_prefetch_result(
        {
            "started": True,
            "completed": True,
            "reason": "completed",
            "local_gpu_hit_tokens": 0,
            "external_hit_tokens": 16,
            "loaded_tokens": 16,
        }
    )
    with pytest.raises(M2ValidationError, match="exactly one external"):
        validate_prefetch_result(
            {
                "started": True,
                "completed": True,
                "reason": "completed",
                "local_gpu_hit_tokens": 0,
                "external_hit_tokens": 0,
                "loaded_tokens": 0,
            }
        )


def test_prefetch_derives_external_tokens_after_deferred_lookup() -> None:
    validate_prefetch_result(
        {
            "started": True,
            "lookup_pending": True,
            "completed": True,
            "reason": "completed",
            "local_gpu_hit_tokens": 0,
            "loaded_tokens": 16,
        }
    )

    with pytest.raises(M2ValidationError, match="exactly one external"):
        validate_prefetch_result(
            {
                "started": True,
                "completed": True,
                "reason": "completed",
                "local_gpu_hit_tokens": 0,
                "loaded_tokens": 16,
            }
        )


def test_diagnostic_transfer_requires_equal_endpoint_digests() -> None:
    digest = "a" * 64
    submitted = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "submitted",
        "run_id": "run",
        "job_id": 1,
        "request_id": "req",
        "direction": "D2H",
    }
    terminal = _terminal(
        request_id="req",
        direction="D2H",
        source=_endpoint("GPU", 3, 7, digest),
        target=_endpoint("CPU", 9, 2, digest),
    )
    assert (
        validate_diagnostic_transfer(
            [submitted, terminal],
            request_id="req",
            direction="D2H",
            run_id="run",
        )
        == terminal
    )

    terminal["target"]["digest"] = "b" * 64
    with pytest.raises(M2ValidationError, match="digest changed"):
        validate_diagnostic_transfer(
            [submitted, terminal],
            request_id="req",
            direction="D2H",
            run_id="run",
        )


def test_diagnostic_trace_rejects_extra_or_failed_dma() -> None:
    digest = "d" * 64
    submitted = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "event": "submitted",
        "status": "in_flight",
        "run_id": "run",
        "phase": "ABBA",
        "job_id": 1,
        "request_id": "req",
        "direction": "D2H",
        "framing": "DAGKV_PAYLOAD_V1",
    }
    terminal = _terminal(
        request_id="req",
        direction="D2H",
        source=_endpoint("GPU", 3, 7, digest),
        target=_endpoint("CPU", 9, 2, digest),
    )
    terminal["phase"] = "ABBA"
    validate_diagnostic_trace_closed_set(
        [submitted, terminal],
        run_id="run",
        expected_transfers=(("req", "D2H"),),
    )

    failed = dict(terminal, status="failed")
    with pytest.raises(M2ValidationError, match="did not complete"):
        validate_diagnostic_trace_closed_set(
            [submitted, failed],
            run_id="run",
            expected_transfers=(("req", "D2H"),),
        )

    extra = dict(submitted, job_id=2, request_id="extra")
    with pytest.raises(M2ValidationError, match="exactly submitted/terminal"):
        validate_diagnostic_trace_closed_set(
            [submitted, terminal, extra],
            run_id="run",
            expected_transfers=(("req", "D2H"),),
        )


def test_native_transfer_requires_scheduler_and_lifecycle_rows() -> None:
    rows = [
        {
            "event": "store_scheduled",
            "trace_id": "trace",
            "native_store_bytes": 4096,
        },
        {
            "event": "store_complete",
            "trace_id": "trace",
            "native_store_bytes": 4096,
        },
        {
            "event": "kv_lifecycle",
            "trace_id": "trace",
            "action": "save",
            "status": "scheduled",
            "event_id": "save:scheduled",
            "request_id": "engine-request",
        },
        {
            "event": "kv_lifecycle",
            "trace_id": "trace",
            "action": "save",
            "status": "completed",
            "parent_event_id": "save:scheduled",
            "observed_byte_count": 4096,
        },
    ]
    assert validate_native_transfer(rows, trace_id="trace", direction="D2H") == 4096
    assert (
        resolve_native_transfer_request_id(rows, trace_id="trace", direction="D2H")
        == "engine-request"
    )
    for row in rows:
        row["trace_id"] = "run:trace"
        if row.get("event") == "kv_lifecycle":
            row["run_id"] = "run"
    validate_native_trace_closed_set(
        rows,
        run_id="run",
        expected_transfers=(("run:trace", "D2H"),),
    )

    rows.append(dict(rows[-1], status="failed"))
    with pytest.raises(M2ValidationError, match="failed event"):
        validate_native_trace_closed_set(
            rows,
            run_id="run",
            expected_transfers=(("run:trace", "D2H"),),
        )


def test_abba_transfer_chain_requires_fresh_generations() -> None:
    digest = "c" * 64
    b1_d2h = _terminal(
        request_id="b1-producer",
        direction="D2H",
        source=_endpoint("GPU", 4, 1, digest),
        target=_endpoint("CPU", 8, 1, digest),
    )
    b1_h2d = _terminal(
        request_id="b1-prefetch",
        direction="H2D",
        source=_endpoint("CPU", 8, 1, digest),
        target=_endpoint("GPU", 4, 2, digest),
    )
    b2_d2h = _terminal(
        request_id="b2-producer",
        direction="D2H",
        source=_endpoint("GPU", 5, 1, digest),
        target=_endpoint("CPU", 8, 2, digest),
    )
    b2_h2d = _terminal(
        request_id="b2-prefetch",
        direction="H2D",
        source=_endpoint("CPU", 8, 2, digest),
        target=_endpoint("GPU", 5, 2, digest),
    )
    validate_abba_transfer_chain(b1_d2h, b1_h2d, b2_d2h, b2_h2d)

    b2_d2h["target"]["allocation_generation"] = 1
    b2_h2d["source"]["allocation_generation"] = 1
    with pytest.raises(M2ValidationError, match="reused B1 CPU"):
        validate_abba_transfer_chain(b1_d2h, b1_h2d, b2_d2h, b2_h2d)


def test_llm_kwargs_forbid_convenience_offload_size() -> None:
    sentinel = object()
    kwargs = build_llm_kwargs(Path("/model"), sentinel)
    assert kwargs["kv_transfer_config"] is sentinel
    assert kwargs["block_size"] == 16
    assert kwargs["max_logprobs"] == -1
    assert kwargs["logprobs_mode"] == "raw_logits"
    assert kwargs["attention_config"] == {
        "backend": "FLASH_ATTN",
        "flash_attn_version": 2,
    }
    assert "kv_offloading_size" not in kwargs


def test_a1_g_control_shares_lifecycle_phase_with_distinct_trace() -> None:
    params = _transfer_params(
        run_id="run",
        phase="A1_G",
        trace_id="run:G:measurement",
        native_trace=Path("/tmp/native.jsonl"),
        role="measurement",
    )

    assert params["offload_phase"] == "A1_G"
    assert params["offload_trace_id"] == "run:G:measurement"
    assert params["max_offload_tokens"] == 0
