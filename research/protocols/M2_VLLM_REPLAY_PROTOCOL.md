# M2 vLLM KV Replay Evidence Protocol v2

Status: normative pre-acceptance protocol for M2 item 8. The current runner and
artifact schemas must conform to every v2 requirement before new calibration
or formal evidence can count. A single formal execution records only one
holdout pass. Only the aggregate 20-holdout manifest can close item 8, and the
aggregate M2 gate remains **OPEN** until all nine conditions in
`research/STAGE_GATES.md` have frozen direct evidence.

## Purpose and Mainline

M2 asks one narrow systems question: does a real vLLM D2H-retain-H2D path
preserve the exact canonical KV payload and produce token-identical,
numerically bounded model output across fresh GPU allocation identities? A
no-DMA GPU prefix-hit control separates ordinary cold/prefix execution-path
numerics from DMA effects. M2 establishes lifecycle and data-plane integrity;
it makes no scheduling-policy, latency, throughput, hit-rate, or paper novelty
claim.

The policy mainline remains the claim boundary in
`research/imported/RELATED_WORK_MATRIX.md`:

- **C1:** dependence-correct probabilistic shared leases, including exclusive
  and correlated branches, first re-admission versus repeats, coalesced cost,
  and calibration or robust bounds under drift;
- **C2:** an explicitly derived capacity- and DMA-constrained joint controller
  with online dual updates, feasibility, stability, overhead, and joint
  retain/migrate/release/prefetch actions;
- **C3:** deadline-aware partial-prefix single-flight with one physical DMA,
  explicit waiters, cancellation/failure propagation, blocking control, and
  fairness.

Generic DAG-aware offload, predictive upload, TTL retention, future-access
scoring, shared-prefix aggregation, and conservative prefetch are prior work
and cannot become the primary claim through this M2 result.

## Pilot Exclusion

`/home/data/25_oyzx/dagkv_runtime/m2_vllm_abba_calibration_20260725_run03` is a
successful v1 pilot only. Its checksummed evidence recorded:

- exact D2H/H2D canonical digests and byte counts for B1 and B2;
- token 932 for A1, B1, B2, and A2;
- exact A1/A2 and B1/B2 raw-logit vectors;
- cold/replay `max_abs_error=0.109375` under BF16;
- `CALIBRATED_NOT_ACCEPTED`, `m2_item8_accepted=false`, and no acceptance
  manifest.

Run03 helped define v2 and is excluded from the 59-run calibration cohort and
the 20-run formal holdout cohort. Its B1 and B2 are state-isolated repetitions
within one engine process, not independent calibration samples. No failed or
partial v1 attempt may be promoted into either v2 cohort.

## Frozen Runtime Profile

The v2 executable must fail closed when a required capability or observation
is absent.

| Field | Frozen value |
| --- | --- |
| Model | content-addressed local Qwen3-8B checkpoint |
| Device topology | one visible GPU; TP=1, PP=1, DP=1 |
| Execution | eager |
| Prefix caching | enabled |
| vLLM KV block size | 16 tokens |
| Prompt | fixed token IDs `1000..1016` (17 tokens) |
| Decode | one greedy token, fixed seed, EOS ignored |
| Numerical observation | complete vocabulary, `raw_logits`, `max_logprobs=-1` |
| Dtype | BF16 |
| Attention | `FLASH_ATTN`, FlashAttention version 2 |
| Connector | external `DAGKVDiagnosticOffloadingConnector` |
| Connector module | `dagkv_vllm_m2.connector` |
| Spec | external `DAGKVDiagnosticCPUOffloadingSpec` |
| Spec module | `dagkv_vllm_m2.spec` |
| CPU allocation | explicit `cpu_bytes_to_use` |
| Load failure policy | fail |
| Calibration cap | `atol=0.125`, `rtol=0` |

`kv_offloading_size` is prohibited. The audited fork derives and can overwrite
connector settings from that convenience option, which would invalidate the
external diagnostic path.

## G Control and ABBA Sequence

Every process runs one no-DMA prefix control followed by the frozen ABBA
sequence. Every reset, cache observation, and closed-set trace check must
complete successfully.

1. **A1 cold:** reset GPU and connector; run the measured request with
   `max_offload_tokens=0`; require zero cached prompt tokens.
2. **G no-DMA prefix hit:** leave the GPU prefix created by A1 in place and
   immediately repeat the measured request with offload disabled; require
   exactly 16 GPU-hit tokens, zero CPU-load tokens, and no D2H/H2D event under
   the G trace ID.
3. **B1 replay:** reset GPU and connector; run a new producer with
   `max_offload_tokens=16` and `evict_after_store_complete=true`; require one
   completed D2H job; reset GPU only; call the explicit prefetch API; require a
   completed 16-token external H2D load; measure exactly 16 GPU-hit tokens.
4. **B2 replay:** repeat the complete B procedure, including connector reset
   and a new producer. B2 cannot inherit B1's CPU allocation.
5. **A2 cold:** reset GPU and connector; repeat the cold measurement and
   require zero cached prompt tokens.

Producers, explicit prefetches, and measurements have distinct trace IDs.
Measurements disable D2H, preventing result collection from creating retained
state for a later phase.

## Per-Process Correctness Contract

Each successful process must satisfy all of the following before it can appear
in a calibration or holdout manifest:

1. A1, G, B1, B2, and A2 each produce one identical greedy token and complete,
   finite raw logits for every vocabulary entry.
2. `A1 == A2` and `B1 == B2` are exact element-wise comparisons with
   `atol=0, rtol=0`. The cold/replay tolerance cannot mask within-path drift.
3. `G == B1 == B2` is exact. This isolates D2H/H2D from the ordinary GPU
   prefix-hit execution path. `A1-G`, `A1-B1`, and `A1-B2` must each satisfy
   `atol=0.125, rtol=0`.
4. The minimum top-1 logit margin across measurements is greater than
   `2 * atol`, and the decoded token remains an unconditional exact-match
   requirement.
5. G has 16 local GPU-hit tokens, zero native CPU-load tokens, and an empty DMA
   closed set. A1/A2 have zero GPU hits. B1/B2 have 16 GPU hits and perform no
   measurement-time on-demand CPU load.
6. B1 and B2 each contain exactly one submitted/terminal D2H pair and one
   submitted/terminal H2D pair. No other DMA job is permitted.
7. Every terminal reports positive bytes equal to worker-reported and native
   scheduled/completed bytes, `DAGKV_PAYLOAD_V1` framing, exact physical slots
   and allocation generations, and equal finite lowercase SHA-256 source and
   target digests.
8. Within each B trial, the D2H CPU target equals the H2D CPU source, and the
   H2D GPU target is fresh relative to the D2H GPU source. B1 and B2 produce the
   same canonical digest and use distinct CPU allocation identities.
9. Native and diagnostic transfer traces form closed event sets with unique
   IDs, valid parents, no failed terminal, and no submitted DMA left without a
   verifiable terminal after worker shutdown. Canonical runtime quiescence
   remains covered by the item 1-7 component evidence.

The maximum relative error is descriptive only. Near-zero logits make it
unstable; it cannot relax or replace the absolute cap.

## Per-Process Evidence

A successful process directory contains at least:

- `logits_A1.npy`, `logits_G.npy`, `logits_B1.npy`, `logits_B2.npy`, and
  `logits_A2.npy`;
- `result.json` with tokens, cache hits, all exact and bounded comparisons,
  prefetch results, bytes, transfer digests, top-1 margins, and gate status;
- `provenance.json` satisfying the complete provenance contract below;
- `native_lifecycle.jsonl` and `diagnostic_transfers.jsonl`;
- a copy of this protocol as `protocol.md`;
- `execution_ids.json`, freezing every producer, measurement, and transfer ID;
- `SHA256SUMS`, written only after all mode-specific checks succeed;
- in formal mode, one `M2_ITEM8_FORMAL_RUN_MANIFEST.json` that references the
  frozen calibration manifest, tolerance, provenance, checksums, and this
  process's `result.json`.

Calibration never writes a holdout or acceptance manifest. A formal process
never writes `M2_ITEM8_ACCEPTANCE_MANIFEST.json`.

## Complete Provenance Contract

Every cohort member must record and content-address:

- DAGKV commit, full worktree state, protocol, runner, connector, spec, and
  helper source files;
- vLLM base commit plus either a clean dedicated commit or a complete binary
  patch and every untracked source file; the recorded Python version and
  compiled extension hashes must resolve any version/HEAD mismatch;
- all model configuration, tokenizer, index, and weight files;
- Python, dependency lock, vLLM, PyTorch, CUDA runtime/toolkit, driver, OS,
  kernel, attention backend, GPU model, and GPU UUID;
- exact argv, relevant environment variables, seed, engine parameters,
  connector parameters, prompt, start/end timestamps, and run ID;
- every raw artifact hash and the parent calibration/tolerance hash where
  applicable.

An unrecorded dirty file, missing model shard hash, source/binary mismatch,
runtime fingerprint drift, partial trace, duplicate run ID, checksum mismatch,
or unavailable required field fails the process. Formal runs require one
identical frozen fingerprint across all 20 holdouts.

## Calibration Cohort and Frozen Tolerance

Run03 is the pilot and contributes no sample. After the v2 protocol and source
are frozen, launch exactly 59 successful calibration processes. Each process
must create a new OS process, CUDA context, vLLM engine, run ID, and output
directory. Engine reuse across samples is prohibited.

The unit of analysis is one complete process. Vocabulary entries and B1/B2 are
correlated observations and cannot inflate the sample count. With zero cap
exceedances in 59 independent processes, the one-sided exact 95% upper bound on
the future per-process exceedance probability is below 5% because
`0.95^59 < 0.05`.

The cap is preregistered at `atol=0.125, rtol=0`; it is not selected from the
largest observed calibration error. Every one of the 59 processes must pass
the complete per-process contract. The cohort aggregator writes a
content-addressed `M2_CALIBRATION_MANIFEST.json` containing the ordered closed
set of 59 result and provenance hashes, observed run-level maxima, attempt
inventory, source/runtime fingerprint, protocol hash, and selection rule.

Only then may `M2_FROZEN_TOLERANCE.json` be written. Its exact fields are:

```json
{
  "schema_version": "dagkv.m2.frozen_tolerance.v2",
  "frozen": true,
  "frozen_at_utc": "<ISO-8601 timestamp with timezone>",
  "atol": 0.125,
  "rtol": 0.0,
  "calibration_manifest_sha256": "<64 lowercase hex characters>",
  "reproducibility_fingerprint": "<64 lowercase hex characters>",
  "calibration_run_count": 59,
  "derivation": "fixed_binary_cap_from_excluded_pilot"
}
```

The tolerance file and its Git commit must predate every formal process.
Command-line overrides are prohibited.

## Formal Holdouts and Item-8 Acceptance

After the tolerance is frozen, launch exactly 20 new fresh-process formal
holdouts. They cannot overlap the pilot or calibration run IDs and must use the
same frozen source/runtime fingerprint. Each successful run writes one
`M2_ITEM8_FORMAL_RUN_MANIFEST.json`; it does not close item 8.

All 20 holdouts must pass. A separate closed-set aggregator verifies unique run
IDs, identical tolerance/protocol/runtime hashes, every per-run checksum, and
the absence of undeclared cohort members. Only that aggregator may write
`M2_ITEM8_ACCEPTANCE_MANIFEST.json`, containing the ordered 20 holdout-pass and
result hashes plus the frozen calibration/tolerance hashes. This closes item 8
alone and must record `m2_accepted=false`.

## Failure and Restart Rules

The protocol never widens `atol`, enables `rtol`, drops an outlier, replaces a
failed run silently, or promotes a partial artifact. Every launched attempt is
listed in an append-only attempt inventory outside the accepted closed set.

- A capability, correctness, trace, checksum, provenance, or cap failure
  invalidates the current candidate implementation. Diagnose and fix it.
- A material fix changes the source/runtime fingerprint and protocol evidence
  version. The 59-run calibration cohort restarts from zero.
- Any formal failure rejects the current 20-run holdout cohort. The frozen
  tolerance remains immutable; a change requires a new calibration version.
- Failed or partial process directories contain no success checksum, holdout
  pass, or acceptance manifest.

## Interpretation Limits

The diagnostic hash reads unpadded payload bytes in canonical KV order. This
readback is instrumentation cost and is excluded from performance results. The
v2 cohort estimates repeatability only for one frozen prompt, block size,
model, dtype, GPU class, software fingerprint, and protocol. Process-level
replication does not establish generalization to other prompts, context
lengths, models, dtypes, accelerators, multiple waiters, partial prefixes,
deadlines, fairness, throughput, or policy quality. Those require separate
preregistered M3 experiments and matched baselines.

M2 depends on a modified local vLLM fork. Evidence must name and fully capture
that fork and cannot be described as an upstream public vLLM capability.

## First-Order References

1. Zhuohang Bian, Feiyang Wu, Teng Ma, and Youwei Zhuo. *Tokencake: A
   KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications*.
   arXiv:2510.18586v2, 2025. DOI:
   <https://doi.org/10.48550/arXiv.2510.18586>. Tokencake already covers
   DAG-aware critical-agent reservation and predictive D2H/H2D around tool
   calls.
2. Hanchen Li, Runyuan He, Qiuyang Mang, et al. *Continuum: Efficient and
   Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live*.
   arXiv:2511.02230v6, revised 2026. DOI:
   <https://doi.org/10.48550/arXiv.2511.02230>. Continuum already covers
   history-derived TTL retention, expiry, and queue-aware cache opportunity
   cost.
3. Haoyu Zheng, Fangcheng Fu, Jia Wu, et al. *Efficient Serving for Dynamic
   Agent Workflows with Prediction-based KV-Cache Management* (PBKV).
   arXiv:2605.06472v1, 2026. DOI:
   <https://doi.org/10.48550/arXiv.2605.06472>. PBKV already covers dynamic call
   graphs, shared-node future-access aggregation, lifecycle-aware eviction, and
   conservative prefetch.

Exact local PDF hashes and the complete C1-C3 claim audit remain in
`research/REFERENCES.md` and `research/imported/RELATED_WORK_MATRIX.md`.
