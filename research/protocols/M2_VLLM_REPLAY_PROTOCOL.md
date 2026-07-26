# M2 vLLM KV Replay Evidence Protocol v3

Status: normative pre-acceptance protocol for M2 item 8. The current runner and
artifact schemas must conform to every v3 requirement before new calibration
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

`evidence/m2/PILOT_ATTEMPTS.json` is the append-only index for protocol-design
executions. It includes failed v1 ID-correlation attempts, the successful v1
run03 numerical pilot, the failed v2 A1/G phase-boundary attempt, subsequent
successful v2 validation pilots, and the failed v3 run09 loader-boundary
attempt. Every indexed attempt is excluded from the 59-run calibration cohort
and the 20-run formal holdout cohort.

The latest indexed successful v2 pilot recorded exact D2H/H2D
canonical digests and byte counts, token 932 in all five phases, exact A1/A2,
exact `G=B1=B2`, and cold/prefix `max_abs_error=0.109375` under BF16. This
localizes the observed numerical difference to the cold-prefill versus
GPU-prefix execution path for the frozen profile. It does not show general
numerical equivalence across models or prompts.

Any additional protocol-validation execution completed before the campaign
launch marker is also a pilot even if it succeeds. B1 and B2 remain
state-isolated repetitions within one engine process and cannot count as
independent calibration samples. No failed, partial, or pilot attempt may be
promoted into either cohort.

## Frozen Runtime Profile

The v3 executable must fail closed when a required capability or observation
is absent.

| Field | Frozen value |
| --- | --- |
| Model | content-addressed local Qwen3-8B checkpoint; vocabulary size 151,936 |
| Device topology | one visible GPU; TP=1, PP=1, DP=1 |
| Execution | eager |
| Prefix caching | enabled |
| vLLM KV block size | 16 tokens |
| Prompt | fixed token IDs `1000..1016` (17 tokens) |
| Decode | one greedy token, fixed seed, EOS ignored |
| Chunked prefill | enabled; the 17-token prompt fits one 64-token scheduler chunk |
| Numerical observation | exactly 151,936 logits, `raw_logits`, `max_logprobs=-1` |
| Dtype | BF16 |
| Attention | `FLASH_ATTN`, FlashAttention version 2 |
| Connector | external `DAGKVDiagnosticOffloadingConnector` |
| Connector module | `dagkv_vllm_m2.connector` |
| Spec | external `DAGKVDiagnosticCPUOffloadingSpec` |
| Spec module | `dagkv_vllm_m2.spec` |
| CPU allocation | explicit `cpu_bytes_to_use` |
| Load failure policy | fail |
| Calibration cap | `atol=0.125`, `rtol=0` |
| NVIDIA driver baseline | loaded module `580.173.02`; exact Ubuntu userspace bundle required |

`kv_offloading_size` is prohibited. The audited fork derives and can overwrite
connector settings from that convenience option, which would invalidate the
external diagnostic path.

## NVIDIA Userspace Bundle Contract

Every v3 process binds one create-only
`dagkv.nvidia_driver_userspace_bundle.v2` manifest. The current baseline is
indexed by `evidence/m2/NVIDIA_580_173_02_BASELINE.json`; campaign
preregistration freezes its absolute root, manifest SHA-256, content digest,
and expected driver version. The same four values are explicit runner argv
inputs and must agree across preregistration, argv, per-run provenance,
independent raw replay, the calibration parent, and every formal run. The
validator reconstructs both sealed Debian
packages, streams their data archives, normalizes extracted modes only by
removing write bits, and proves that their closed union equals the read-only
`rootfs` tree. It also requires the package version, loaded kernel module,
bundle SONAME targets, and `nvidia-smi` driver report to agree.

The runner requires `LD_LIBRARY_PATH` to equal the validated bundle library
directory followed by exactly `/usr/local/cuda/lib64`, invokes the bundle's
absolute `nvidia-smi`, rejects
`LD_PRELOAD` and `LD_AUDIT`, and proves from `/proc/self/maps` that its sole
mapped `libcuda` resolves to the sealed bundle inode and hash. Fresh bundle
validation occurs before CUDA initialization and after evidence construction;
the full mutation-sensitive filesystem snapshot must remain equal. A version,
digest, inode, path, mapping, or environment mismatch fails the attempt.

Runtime imports execute inside an audited spawn boundary. The runner records
the exact launch loader string, base dependency set, and initial `sys.path`.
It permits only the exact OpenCV bootstrap prefix derived from the loaded
virtual-environment module and only the exact setuptools vendor path appended
by the loaded setuptools module. The OpenCV directory must remain inside the
same Python prefix and contain no `libcuda` candidate. Before EngineCore spawn,
the runner restores the launch loader string byte-for-byte, rejects loader
injection again, revalidates the NVIDIA bundle, and binds the base, added, and
effective dependency sets in provenance. Independent raw replay reconstructs
these relationships and rejects missing, removed, reordered, or external
entries. `PYTHONPATH` preparation is idempotent so the declared integration
path appears once.

The 59-process v2 calibration under driver `580.159.03` remains immutable
historical evidence. An unattended package upgrade followed by reboot changed
the loaded driver baseline to `580.173.02`; v3 also closes a provenance gap by
binding package payloads and actual library mappings. Consequently, no v2
tolerance or cohort member is eligible as a parent or member of a v3 formal
campaign. v3 starts with an excluded pilot, then a fresh 59-process calibration
and 20-process holdout cohort.

## G Control and ABBA Sequence

Every process runs one no-DMA prefix control followed by the frozen ABBA
sequence. Every reset, cache observation, and closed-set trace check must
complete successfully.

1. **A1 cold:** reset GPU and connector; run the measured request with
   `max_offload_tokens=0`; require zero cached prompt tokens.
2. **G no-DMA prefix hit:** A1 and G use one explicit `A1_G` lifecycle phase
   because G consumes A1's live GPU allocation. Leave the GPU prefix created
   by A1 in place and immediately repeat the measured request with offload
   disabled; keep distinct A1/G trace IDs and result labels, and require exactly
   16 GPU-hit tokens, zero CPU-load tokens, and no D2H/H2D event under the G
   trace ID.
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

1. A1, G, B1, B2, and A2 each produce one identical greedy token and exactly
   151,936 finite raw logits, one for every frozen Qwen3-8B vocabulary entry.
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
   verifiable terminal after worker shutdown. Validation keys rows by transfer
   identity and event; physical JSONL row order and job-ID magnitude carry no
   cross-transfer semantics. Canonical runtime quiescence remains covered by
   the item 1-7 component evidence.

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

The process directory contains files only at its root and directly under
`source_state/`. Empty extra directories, special filesystem nodes, symlinks,
and multiply linked evidence files are invalid.

Calibration never writes a holdout or acceptance manifest. A formal process
never writes `M2_ITEM8_ACCEPTANCE_MANIFEST.json`.

## Complete Provenance Contract

Every cohort member must record and content-address:

- DAGKV HEAD, tracked diff, non-ignored untracked-file archive, protocol,
  runner, connector, spec, and helper source files;
- vLLM base commit plus either a clean dedicated commit or a complete binary
  patch and every untracked source file; the recorded Python version and
  compiled extension hashes must resolve any version/HEAD mismatch. Every
  implementation and runtime-binary entry has a lowercase SHA-256. The invoked
  executable resolves to the captured Python binary, and the runtime-binary
  root, vLLM Git root, and imported vLLM module resolve to one source tree;
- all model configuration, tokenizer, index, and weight files;
- Python, dependency lock, vLLM, PyTorch, CUDA runtime/toolkit, driver, OS,
  kernel, attention backend, GPU model, and GPU UUID; the NVIDIA package,
  manifest, normalized runtime tree, mapped `libcuda`, and absolute
  `nvidia-smi` identities are content-addressed separately;
- exact argv, relevant environment variables, seed, engine parameters,
  connector parameters, prompt, start/end timestamps, and run ID;
- every raw artifact hash and the parent calibration/tolerance hash where
  applicable.

A changed tracked file, an unrecorded non-ignored untracked file, missing model
shard hash, source/binary mismatch, runtime fingerprint drift, partial trace,
duplicate run ID, checksum mismatch, or unavailable required field fails the
process. Git-ignored entries are outside the v3 source snapshot claim. Formal
runs require one identical frozen fingerprint across all 20 holdouts.

## Calibration Cohort and Frozen Tolerance

All attempt-index entries and pre-launch validation pilots contribute no
sample. After the v3 protocol and source are frozen, launch exactly 59
successful calibration processes. Each process must create a new OS process,
CUDA context, vLLM engine, run ID, and output directory. Engine reuse across
samples is prohibited.

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

Campaign launch is a two-stage operation. Preparation creates a brand-new
campaign root containing only `CAMPAIGN_PREREGISTRATION.json`, fsyncs it, and
prints its SHA-256. Production preregistration uses schema
`dagkv.m2.calibration_campaign_preregistration.v3`, records the clean
preparation Git HEAD, and freezes the marker path. Before execution, a direct,
single-parent child commit must add only
`evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json`; the marker binds the campaign,
preregistration digest, preparation HEAD, timestamp, and calibration-only claim
scope. Execution requires the preregistration digest explicitly, obtains a
non-blocking exclusive `flock` on the campaign directory before reading it,
and holds that lock through aggregate publication and final independent replay.
It rejects every other pre-existing campaign entry and revalidates the marker
Git object, frozen protocol, runner, launcher, aggregator, shared evidence
validator, independent raw replay validator, Python entry point, model/runtime
paths, command template, environment, implementation manifest, runtime
fingerprint, selection rule, and no-retry rule before submitting `run-001`.

`ATTEMPTS.jsonl` is append-only. Its calibration prefix contains exactly 118
fsynced records: one `submitted` and one `terminal` record for each of
`run-001` through `run-059`, in sequence, with no retry or replacement. Every
submitted record uses schema `dagkv.m2.calibration_campaign_attempt.v2` and
repeats the preparation HEAD, execution HEAD, marker path, and marker SHA-256
as one `execution_binding`. Every passing terminal binds its positive process
ID, zero exit status, process
start/end timestamps, stdout/stderr size and hash, complete artifact
inventory, result/provenance/checksum hashes, implementation manifest,
reproducibility fingerprint, DAGKV Git HEAD, DAGKV snapshot SHA-256, and
observed maximum absolute error. All 59 Git HEAD values must equal the marker
execution HEAD and all 59 snapshot hashes must be equal. The first
submission must follow preregistration; every later submission must follow the
previous terminal. `run-059` terminal permanently seals the prefix by byte
length, record count, and SHA-256.

The next ledger record is the single aggregator submission. It binds the
sealed-prefix triple and the exclusive output path
`M2_CALIBRATION_MANIFEST.json`, and repeats the execution binding. After
exclusive publication, one aggregator
terminal records the manifest and aggregate-log hashes. Its `passed` status is
the terminal state of the aggregator process and its prospective candidate
validation; it is not evidence acceptance by itself. The launcher then replays
the complete 120-record bundle. A failed post-append replay makes the launcher
exit nonzero and leaves the campaign ineligible for tolerance freezing, even
though the append-only aggregate operation terminal remains `passed`. Only a
fresh successful invocation of the shared complete-bundle validator by the
tolerance freezer or a formal consumer accepts the published evidence. Any
calibration record after the sealed prefix, any additional campaign directory,
any changed log or artifact, or any missing/failed terminal invalidates the
bundle.

The v3 calibration manifest has this exact evidence shape (digest values are
abbreviated here only for readability):

```json
{
  "schema_version": "dagkv.m2.calibration_cohort.v3",
  "protocol_schema": "dagkv.m2.vllm_abba.v3",
  "campaign_id": "<preregistered campaign ID>",
  "campaign_preregistration_file": "CAMPAIGN_PREREGISTRATION.json",
  "campaign_preregistration_sha256": "<SHA-256>",
  "attempt_file": "ATTEMPTS.jsonl",
  "attempt_prefix_bytes": 123456,
  "attempt_prefix_record_count": 118,
  "attempt_prefix_sha256": "<SHA-256>",
  "protocol_sha256": "<SHA-256>",
  "implementation_manifest_sha256": "<SHA-256>",
  "selection_rule": {"ordered_run_names": ["run-001", "...", "run-059"]},
  "execution_binding": {
    "preparation_git_head": "<Git SHA-1>",
    "execution_git_head": "<Git SHA-1>",
    "launch_marker_repository_path": "evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json",
    "launch_marker_sha256": "<SHA-256>"
  },
  "dagkv_snapshot_sha256": "<SHA-256>",
  "pilot_excluded": true,
  "attempt_count": 59,
  "run_count": 59,
  "all_passed": true,
  "failures": [],
  "observed_max_abs_error": 0.0,
  "formal_atol": 0.125,
  "formal_rtol": 0.0,
  "reproducibility_fingerprint": "<SHA-256>",
  "runs": [
    {
      "sequence": 1,
      "run_name": "run-001",
      "attempt_id": "<campaign ID>:run-001",
      "run_id": "<process run ID>",
      "result_sha256": "<SHA-256>",
      "provenance_sha256": "<SHA-256>",
      "sha256sums_sha256": "<SHA-256>",
      "observed_max_abs_error": 0.0,
      "dagkv_git_head": "<execution Git SHA-1>",
      "dagkv_snapshot_sha256": "<SHA-256>"
    }
  ]
}
```

The displayed prefix byte count is illustrative. The displayed
`selection_rule` is abbreviated; the stored object also fixes
one attempt per run, calibration-only eligibility, required
`submitted`/`terminal` events, passing terminal status, zero retries, and
stop-on-first-failure. The stored run list contains all 59 full entries in
sequence.

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

The tolerance freezer requires an explicit output outside the campaign root
and every run directory, and publishes with create-only semantics. Before
freezing, and again before every formal process, the shared validator replays
the complete upstream bundle: preregistration, ledger prefix and aggregate
terminal, logs, actual run directories, checksums, manifest mapping, protocol,
implementation, and fingerprint. The runner also requires its current
implementation capture to equal the campaign implementation manifest. The
tolerance and formal consumers replay the historical execution commit directly
from the journal binding; later Git HEADs do not replace that object identity.
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
- A post-append complete-bundle replay failure invalidates the campaign and
  prohibits tolerance freezing; the aggregate process terminal records only
  the already completed publication operation.
- A material fix changes the source/runtime fingerprint and protocol evidence
  version. The 59-run calibration cohort restarts from zero.
- Any formal failure rejects the current 20-run holdout cohort. The frozen
  tolerance remains immutable; a change requires a new calibration version.
- Failed or partial process directories contain no success checksum, holdout
  pass, or acceptance manifest.

## Interpretation Limits

The diagnostic hash reads unpadded payload bytes in canonical KV order. This
readback is instrumentation cost and is excluded from performance results. The
v3 cohort estimates repeatability only for one frozen prompt, block size,
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
