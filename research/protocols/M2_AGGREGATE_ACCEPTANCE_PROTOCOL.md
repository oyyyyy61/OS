# M2 Aggregate Acceptance Protocol v1

Protocol schema: `dagkv.m2.aggregate_acceptance.v1`

Status: normative final audit protocol for aggregate M2 correctness acceptance.
It combines the independently sealed CPU component evidence for conditions 1-7
with the independently sealed v3 real-vLLM evidence for condition 8, then closes
condition 9 through a historical-source compatibility audit and a fresh external
content audit. It authorizes no performance or M3 policy claim.

## Acceptance Boundary

The only decision payload is `M2_AGGREGATE_ACCEPTANCE.json`. Its eligibility
also requires the durable sibling publication lock defined below. A successful
record has:

- `gate_status="M2_ACCEPTED_CORRECTNESS_ONLY"`;
- `verification_status="VERIFIED"`;
- `m2_item8_accepted=true` and `m2_accepted=true`;
- `performance_claims_supported=false`;
- `policy_claims_supported=false`; and
- exactly nine ordered conditions, numbered 1 through 9, each with
  `status="VERIFIED"`.

The exact claim scope is:

> M2 lifecycle and data-plane correctness for one process, one RTX 4090, and
> GPU plus primary CPU-DRAM under the frozen Qwen3-8B v3 ABBA profile. No
> latency, throughput, hit-rate, scheduling-policy, novelty, C1, C2, or C3
> claim.

The exact acceptance statement is:

> All nine M2 conditions independently replayed across the sealed component
> and formal evidence. M2 correctness is accepted within the declared scope;
> all performance and M3 policy claims remain open.

Condition 6 includes the deterministic atomic publication of every compatible
waiter mapping from one H2D operation. The acceptance adds no live
multi-request, policy-scale partial-prefix, fairness, deadline, or scheduling
benefit claim. M3-M6 remain open.

## Production Inputs

Publication requires the primary artifacts below and their operator-supplied
expected SHA-256 values. The committed component, calibration, and campaign02
success indexes restrict those arguments to the one recorded successful chain;
they cannot replace replay of either primary artifact or authorize a different,
internally consistent bundle.

| Input | Production identity |
| --- | --- |
| Component evidence root | `/home/data/25_oyzx/dagkv_runtime/m2_component_evidence_v1_20260725_run02` |
| Component manifest SHA-256 | `7502f5e09edb6f08f4ac16b216459d3db7aca5856addb18cd2c50280149dc8c1` |
| Component `SHA256SUMS` SHA-256 | `e012c739643ea1351365111c8e2c157af310f16e6c55e85640918b873e26ac63` |
| Formal campaign root | `/home/data/25_oyzx/dagkv_runtime/m2_vllm_abba_v3_formal_20260726_campaign02` |
| Formal bundle seal SHA-256 | `3f1e77164d13595b525e1962634b138b184496656489bc2b443eab9dffe1ded7` |
| Formal preregistration SHA-256 | `f49f61e2ae380f54a61510a0946a2bd1e9fe6662a455ba9320cb7e3c39405d90` |

The publisher also requires the expected clean, committed repository HEAD and
the expected SHA-256 values of `research/STAGE_GATES.md` and this protocol.
The aggregate tool, this protocol, all authority documents, and all evidence
indexes named below must already exist as blobs at that HEAD.

The validator parses those three success indexes from the recorded Git blobs
and requires exact agreement on component root/manifest/checksums, calibration
root/manifest/tolerance, formal root/campaign/seal/preregistration, DAGKV and
vLLM source identities, implementation and reproducibility fingerprints,
model and runtime-binary manifests, NVIDIA bundle/driver/libcuda identities,
and the 239-to-250 dependency boundary. Command-line expected values cannot
self-authorize an alternate evidence chain.

## Component Replay and Conditions 1-7

The aggregate validator must call the component evidence validator twice:
once with external executable and source verification enabled, and once in
raw-only mode. Both calls must return the same manifest. The sealed component
root must retain its exact read-only closed set, the three suites must report
269, 13, and 345 passing cases, and the total must remain 627 with no failure,
error, skip, duplicate testcase identity, or retry.

Suite-level success and `eligible_gate_scope=[1,...,7]` alone are insufficient.
The aggregate validator parses the three sealed JUnit XML files and requires
every identity below to exist as a passing testcase. The canonical identity
format is `suite_id:classname::name`. A testcase may support more than one
condition; every condition retains its own explicit source list in the
aggregate record.

### Condition 1: Canonical Schemas and Lifetime Identities

- `dagkv_core:tests.test_domain::test_digest_identity_is_required_and_canonical`
- `dagkv_core:tests.test_ledger::test_binding_parent_freezes_execution_identity`
- `dagkv_core:tests.test_orchestrator_failures::test_execution_reference_is_single_use_across_binding_lifetimes`
- `dagkv_vllm_adapter:integrations.vllm_m2.tests.test_contract::test_connector_captures_allocator_generation`
- `vllm_lifecycle_cpu:tests.v1.kv_offload.test_lifecycle::test_lifecycle_identity_is_explicitly_enabled`

### Condition 2: Shared-Owner Isolation and Reclaim Safety

- `dagkv_core:tests.test_orchestrator_lifecycle::test_two_owner_offload_readmission_and_reclaim`
- `dagkv_core:tests.test_orchestrator_lifecycle::test_cross_owner_release_stays_rejected_after_real_owner_release`
- `vllm_lifecycle_cpu:tests.v1.kv_offload.cpu.test_manager::test_strict_shared_load_has_independent_owner_binding`

### Condition 3: Idempotent Release and Terminal Replay

- `dagkv_core:tests.test_domain::test_transfer_validates_direction_and_exact_terminal_replay`
- `dagkv_core:tests.test_domain::test_lease_allows_only_exact_terminal_replay`
- `vllm_lifecycle_cpu:tests.v1.kv_connector.unit.offloading_connector.test_worker_metadata::test_mark_completed_identical_duplicate_is_idempotent`
- `vllm_lifecycle_cpu:tests.v1.kv_connector.unit.offloading_connector.test_scheduler::test_duplicate_and_late_rank_reports_keep_one_terminal`

### Condition 4: Stale Generation and Completion Rejection

- `dagkv_core:tests.test_engine_adapter::test_generation_mismatch_terminalizes_before_raising`
- `dagkv_core:tests.test_orchestrator_failures::test_integrity_mismatch_cleans_reservation_and_rejects_stale_completion`
- `dagkv_core:tests.test_orchestrator_failures::test_stale_drop_and_reclaim_cannot_delete_new_generation`
- `vllm_lifecycle_cpu:tests.v1.kv_offload.cpu.test_manager::test_failed_store_cleanup_emits_record_and_advances_generation`
- `vllm_lifecycle_cpu:tests.v1.kv_offload.cpu.test_manager::test_strict_capacity_reuse_closes_generation_before_next_open`

### Condition 5: Cross-Family Conservation and Independent Audit

- `dagkv_core:tests.test_ledger::test_audit_rejects_tampered_event_envelope_and_unknown_parent`
- `dagkv_core:tests.test_ledger::test_cross_family_references_block_early_free_and_failed_publish`
- `dagkv_core:tests.test_orchestrator_lifecycle::test_two_owner_offload_readmission_and_reclaim`

### Condition 6: Atomic Compatible-Waiter H2D Publication

- `dagkv_core:tests.test_orchestrator_failures::test_invalid_waiter_cannot_leave_half_scheduled_h2d`
- `dagkv_core:tests.test_orchestrator_failures::test_released_h2d_waiter_is_not_published_after_completion`
- `dagkv_core:tests.test_orchestrator_failures::test_h2d_completion_does_not_publish_a_terminal_node_waiter`
- `dagkv_core:tests.test_orchestrator_failures::test_concurrent_h2d_waiters_share_one_physical_transfer`

### Condition 7: Workflow Terminal Cleanup and DAG Running Gate

- `dagkv_core:tests.test_orchestrator_failures::test_early_workflow_failure_does_not_partially_mutate_state`
- `dagkv_core:tests.test_orchestrator_failures::test_dag_failure_cancels_parallel_node_and_skips_descendants`
- `dagkv_core:tests.test_orchestrator_failures::test_required_binding_respects_the_dag_running_gate`
- `dagkv_core:tests.test_orchestrator_failures::test_workflow_failure_closes_bindings_leases_and_execution_maps`
- `dagkv_core:tests.test_orchestrator_failures::test_workflow_failure_racing_h2d_completion_is_serializable`

These lists contain 29 condition-level references to 28 unique sealed cases;
the two-owner lifecycle case supports both conditions 2 and 5. The component
replay still validates all 627 sealed JUnit identities and semantics.

## Condition 8: Frozen GPU Cohorts

The aggregate validator must call the published formal-bundle validator with
the expected formal seal and preregistration hashes. That replay remains the
authority for the formal root closed set, 42-row journal, parent calibration,
frozen tolerance, NVIDIA bundle, all 20 formal run directories, and every raw
artifact.

After that replay, the aggregate layer independently checks the cohort boundary:

1. The parent calibration manifest contains exactly 59 ordered runs, reports
   `run_count=59` and `all_passed=true`, and replays through the shared
   calibration validator.
2. The formal seal contains exactly 20 ordered runs and retains the item-8
   acceptance mapping with `m2_item8_accepted=true`, `m2_accepted=false`, and
   `performance_claims_supported=false`.
3. Calibration and formal preregistrations both freeze
   `retry_policy="none_stop_on_first_failure"`, one attempt per run, and zero
   retries. Their sealed journals must contain no non-passing eligible terminal.
4. Every eligible run ID is a non-empty string. The 59 calibration IDs and 20
   formal IDs are internally unique and mutually disjoint; their union has
   exactly 79 members. Run names, attempt IDs, and artifact mappings must retain
   the ordered bindings replayed by their primary validators.
5. The calibration has 59 fresh processes and the formal cohort has 20 fresh
   processes. B1 and B2 are state-isolated phases inside one process and cannot
   be counted as additional runs.

The aggregate record stores the two ordered run-ID lists through a canonical
cohort-identity digest, the counts `59`, `20`, and `79`, and the zero-retry
policy. Any replacement, overlap, duplicate, partial run, later attempt, or
retry invalidates condition 8.

## Explicit Exclusion of Failed and Pilot Attempts

Only the eligible 59-run calibration and campaign02's 20 formal holdouts enter
the aggregate decision. The validator reads the following records from blobs
at the recorded aggregate repository HEAD and emits their Git blob and SHA-256
bindings under `excluded_attempts`:

- `evidence/m2/PILOT_ATTEMPTS.json`: the index contains exactly ten historical
  attempts, retains top-level `cohort_eligible=false` and
  `acceptance_claimed=false`, and repeats `cohort_eligible=false` on its run09
  and run10 entries;
- `evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json`: run09 remains
  `gate_status="FAILED"`, `cohort_eligible=false`, and
  `acceptance_claimed=false`;
- `evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json`: the successful
  diagnostic pilot remains `gate_status="CALIBRATED_NOT_ACCEPTED"`,
  `cohort_eligible=false`, and `acceptance_claimed=false`; and
- `evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json`:
  campaign01 remains `formal_cohort_eligible=false`,
  `acceptance_claimed=false`, and failed at
  `post_aggregate_candidate_replay`.

Environment smoke checks and protocol-validation executions made before their
respective launch markers are also ineligible. A passing observation in an
excluded attempt cannot be imported into an eligible cohort, used to replace a
failed run, or counted toward 59/59 or 20/20.

The validator extracts the run09 and run10 `execution.run_id` values. It also
reads campaign01's external `FORMAL_ATTEMPTS.jsonl` through the root, filename,
42-row count, and SHA-256 recorded in its committed failure index. The journal
must contain exactly 40 formal-run rows and 20 ordered passing terminal rows.
Their validation records contribute 20 more run IDs. The resulting 22 excluded
IDs must be non-empty and globally unique. Campaign01's ID must differ from
campaign02's ID, and the excluded ID set must have zero intersection with the
79 unique eligible calibration/formal IDs. The aggregate record preserves all
22 excluded IDs, the journal binding, both counts, and the zero-intersection
result.

## Condition 9: Historical Compatibility Bridge

The component evidence and formal evidence were created at different DAGKV
commits. Whole-repository snapshots can differ because evidence tooling,
protocols, and indexes evolved between those commits. The aggregate validator
therefore resolves historical objects and compares the protected runtime and
test surface directly.

The required Git relationship is:

1. the component HEAD is an ancestor of the formal preparation HEAD;
2. the formal preparation HEAD is an ancestor of the formal execution HEAD;
3. the execution HEAD has exactly one parent, the preparation HEAD; and
4. the execution commit changes only its committed launch marker, as already
   replayed by the formal validator.

The validator uses `git ls-tree` and `git cat-file` against each historical
commit. It cannot substitute files from the current checkout. The following
protected set must have the same sorted `(mode, path, Git blob)` inventory at
the component, formal preparation, and formal execution commits:

- every tracked blob under `src/dagkv/`;
- every tracked blob under `integrations/vllm_m2/dagkv_vllm_m2/`;
- `pyproject.toml` and `uv.lock`;
- `tests/test_domain.py`;
- `tests/test_engine_adapter.py`;
- `tests/test_ledger.py`;
- `tests/test_orchestrator_failures.py`;
- `tests/test_orchestrator_lifecycle.py`; and
- `integrations/vllm_m2/tests/test_contract.py`.

Missing commit, tree, or blob objects fail closed. The aggregate record stores
all three historical HEADs, the protected path policy, entry count, and one
canonical protected-tree digest.

The bridge additionally requires equal vLLM Git HEADs and reconstructable
vLLM snapshot SHA-256 values across the component and formal evidence, equal
resolved Python executable SHA-256 values, and equal normalized base dependency
sets. Dependency normalization lowercases names, replaces underscores with
hyphens, deduplicates `(name, version)` pairs, sorts them, and requires exactly
239 pairs on both sides. A changed package version, addition, removal, vLLM
patch, protected DAGKV blob, or Python binary invalidates the bridge.

## Fresh Rehash of 23 External Files

The primary formal validator proves that all 20 holdouts share the same source,
model, runtime-binary, and dependency fingerprints. The aggregate validator
then selects the first ordered formal provenance through its sealed SHA-256 and
rehashes the referenced external content at the time of aggregate publication
and at every later aggregate replay.

The mandatory set contains exactly:

- all 16 files in the closed Qwen3-8B model manifest, including five weight
  shards and eleven metadata/tokenizer files;
- all six recorded vLLM `*.so` extensions under the captured vLLM root; and
- the resolved Python executable.

Each path must be absolute at its root, use a safe relative child path where
applicable, resolve to a regular non-symlink file, and have exactly one hard
link. Its inode, size, and mtime must match provenance before hashing. Device,
inode, mode, link count, size, mtime, and ctime must remain unchanged across the
read. The freshly computed SHA-256 must equal the recorded full hash.

Before any content read, the validator snapshots the identity tuple for all 23
files. After every content hash finishes, it re-stats the complete set and
requires equality with that initial identity snapshot. It then rescans both
live closed sets. Model and `vllm/` roots cannot be symlinks, and their audited
trees cannot contain symlink or special nodes outside excluded `.git` metadata.

The live model inventory excluding `.git` must equal the recorded 16-file set,
and the live vLLM `*.so` inventory must equal the recorded six-file set. The
model and runtime-binary manifest digests are reconstructed from their ordered
entries. The aggregate record contains all 23 current path/size/SHA bindings,
the two manifest identities, a combined digest, and
`current_content_rehash_passed=true`. Metadata-only acceptance and a
`--no-external` aggregate validation mode are prohibited.

This v1 decision is a machine-local acceptance: absolute paths plus recorded
inode and mtime identities intentionally fail after relocation or restoration.
A portable paper artifact requires a later content-addressed export and a
separate reconstruction validator; portability is outside this M2 decision.

## Aggregate Repository Binding

Production publication requires a clean worktree at the exact committed HEAD
given through `--expected-repository-head`. It checks this state before the
long replay, after the input replay, and again immediately before final
publication. The validator resolves and records
the mode, Git blob ID, size, and SHA-256 of every authority below:

- `tools/m2_aggregate_acceptance.py`;
- `tools/m2_formal_evidence.py`;
- `tools/run_m2_component_evidence.py`;
- `tools/aggregate_m2_formal.py`;
- `tools/aggregate_m2_calibration.py`;
- `tools/freeze_m2_tolerance.py`;
- `tools/m2_calibration_evidence.py`;
- `tools/m2_raw_replay.py`;
- `tools/run_m2_vllm_abba.py`;
- `tools/nvidia_driver_userspace_bundle.py`;
- `research/protocols/M2_AGGREGATE_ACCEPTANCE_PROTOCOL.md`;
- `research/STAGE_GATES.md`;
- `research/M2_RUNTIME_CONTRACT.md`;
- `research/protocols/M2_COMPONENT_EVIDENCE_PROTOCOL.md`;
- `research/protocols/M2_FORMAL_CAMPAIGN_PROTOCOL.md`;
- `research/protocols/M2_VLLM_REPLAY_PROTOCOL.md`;
- `evidence/m2/M2_COMPONENT_EVIDENCE_INDEX.json`;
- `evidence/m2/PILOT_ATTEMPTS.json`;
- `evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json`;
- `evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json`;
- `evidence/m2/v3_580_173_02/M2_CALIBRATION_EVIDENCE_INDEX.json`;
- `evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json`;
- `evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN02_EVIDENCE_INDEX.json`.

Later replay may run from a descendant checkout. It still resolves every
recorded authority from the historical acceptance commit. Before using a
verdict, it requires all ten files in the executable validator closure and the
aggregate protocol in the current checkout to be regular non-symlink files with
the recorded executable mode, size, and stable SHA-256. This covers direct and
runtime imports through component replay, formal replay, calibration replay,
raw replay, NVIDIA validation, and tolerance validation. Garbage-collected,
unavailable, rewritten, or locally drifted authority makes the decision
unverifiable and causes replay to fail.

## Acceptance Schema

The JSON object has exactly these top-level fields:

- `schema_version` and `accepted_at_utc`;
- `gate_status` and `verification_status`;
- `repository`;
- `component_evidence`;
- `formal_evidence`;
- `compatibility_bridge`;
- `external_content`;
- `excluded_attempts`;
- `conditions`;
- `m2_item8_accepted` and `m2_accepted`;
- `performance_claims_supported` and `policy_claims_supported`;
- `claim_scope`; and
- `statement`.

Unknown or missing fields, duplicate JSON keys, invalid UTF-8, non-finite JSON
numbers, malformed hashes, unsafe paths, or a changed statement fail closed.
Condition sources are derived by the validator; the publisher exposes no CLI
flags for supplying verdicts or overriding cohort sizes.

The component section binds the root, manifest, `SHA256SUMS`, suite counts,
observed and required JUnit counts, and both replay modes. The formal section
binds the seal, preregistration, calibration manifest, frozen tolerance, cohort
identity, retry policy, and full replay. The compatibility, external-content,
exclusion, and repository sections carry the audits specified above.

## Create-Only Locked One-File Closed Set

The output directory must be a new absolute path under an existing non-symlink
parent. Before any long replay, the publisher exclusively creates the sibling
lock `.<OUTPUT_NAME>.m2-aggregate-publication.lock`, takes an exclusive `flock`,
writes `PREPARING\n`, fsyncs the file, sets mode `0444`, and fsyncs the parent.
The exclusive lock remains held through every replay and publication step. A
competing publisher using the same path fails at lock creation. No overwrite,
resume, repair, `--force`, or in-place update path exists.

Before creating the final output, the publisher completes all primary replays,
historical Git checks, exclusion checks, and 23-file content hashes. It creates
a randomized sibling staging directory, writes canonical strict JSON through
an exclusive temporary file, fsyncs it, seals the staged file and directory,
and performs the synchronous full SHA-pinned replay there. A failed staged
replay removes the staging directory and never creates the requested output.

After staged replay, the publisher checks the clean expected repository binding
again. It atomically creates the requested directory, hard-links the validated
file under a hidden candidate name, fsyncs it, removes the staging link so the
candidate has one hard link, and only then renames it to
`M2_AGGREGATE_ACCEPTANCE.json`. The final directory remains `0755` while the
publisher repeats the repository binding, final SHA, and closed-set checks. It
then sets the directory to `0555` and fsyncs both it and its parent. A failure
after final-directory creation quarantines any candidate or visible success name as
`INVALID_M2_AGGREGATE_ACCEPTANCE.json`; such a directory is permanently
ineligible and cannot pass the validator.

Only after all fallible final checks and fsyncs succeed does the publisher
replace the held lock contents with `PUBLISHED\n` and fsync that file. Writing
this state is the last success transition. The publisher then releases the
exclusive lock. A crash or failure that leaves `PREPARING\n` makes every
associated output ineligible, including an apparently complete success file.
If ordinary exception cleanup runs before any output becomes visible, it may
remove the unused sidecar; an abrupt crash may leave the name permanently
reserved. Either case requires a new output name for another publication.

The final directory contains exactly one regular file named
`M2_AGGREGATE_ACCEPTANCE.json`. The file has one hard link and mode `0444`; the
directory has mode `0555`. Extra files, extra directories, symlinks, special
nodes, writable modes, or a pre-existing destination invalidate the result.
The sibling lock is outside this one-file directory. It must remain a regular
non-symlink file with one hard link, mode `0444`, and exact content
`PUBLISHED\n`.
Every file chmod and hard-link-count transition is followed by an fsync of the
affected inode and directory before publication can advance.
The single file cannot carry a non-circular hash of itself, so every independent
replay requires the operator-supplied `--expected-acceptance-sha256`. A later
Git evidence index may record that SHA-256 after a separate replay succeeds.

Publication performs a synchronous full replay while the staged file is sealed.
A downstream M2 status update requires another fresh validator process against
the final path with the expected acceptance SHA-256. If failure occurs after
any requested output path becomes visible, that path remains ineligible and
cannot be repaired or reused; a material fix and a new output directory are
required. A failed publication can never combine the required success filename
with a valid `PUBLISHED` sidecar.

Every public validator derives the sibling lock path, verifies its type, link
count, mode, and stable identity, then takes a shared `flock`. It accepts only
`PUBLISHED\n` and holds the shared lock through the complete replay. There is no
public lock-bypass option. Before returning success, while still holding the
shared lock, it re-reads the locked descriptor and re-stats both the descriptor
and sidecar path. Contents and the complete device/inode/mode/link/size/mtime/
ctime identity must equal the pre-replay baseline. Unlinking or replacing the
path during replay fails even when the old locked descriptor remains valid. An
archive, backup, relocation, or restoration must preserve the decision
directory and its specifically named sibling lock as one publication unit;
this does not relax the machine-local path and inode boundary.

## Commands

Publication must run from the clean committed repository root and use a new
absolute output directory:

```bash
.venv/bin/python tools/m2_aggregate_acceptance.py publish \
  --output-dir /absolute/new/m2_aggregate_acceptance_v1 \
  --component-evidence-dir /home/data/25_oyzx/dagkv_runtime/m2_component_evidence_v1_20260725_run02 \
  --expected-component-manifest-sha256 7502f5e09edb6f08f4ac16b216459d3db7aca5856addb18cd2c50280149dc8c1 \
  --expected-component-sha256sums-sha256 e012c739643ea1351365111c8e2c157af310f16e6c55e85640918b873e26ac63 \
  --formal-seal /home/data/25_oyzx/dagkv_runtime/m2_vllm_abba_v3_formal_20260726_campaign02/M2_FORMAL_BUNDLE_SEAL.json \
  --expected-formal-seal-sha256 3f1e77164d13595b525e1962634b138b184496656489bc2b443eab9dffe1ded7 \
  --expected-formal-preregistration-sha256 f49f61e2ae380f54a61510a0946a2bd1e9fe6662a455ba9320cb7e3c39405d90 \
  --expected-repository-head COMMITTED_HEAD \
  --expected-stage-gates-sha256 STAGE_GATES_SHA256 \
  --expected-protocol-sha256 AGGREGATE_PROTOCOL_SHA256
```

Independent replay always requires the acceptance identity:

```bash
.venv/bin/python tools/m2_aggregate_acceptance.py validate \
  /absolute/new/m2_aggregate_acceptance_v1/M2_AGGREGATE_ACCEPTANCE.json \
  --expected-acceptance-sha256 ACCEPTANCE_SHA256
```

There is no GPU execution in this audit. The command replays the already sealed
GPU evidence and reads the frozen external files. Any component, formal,
historical-Git, external-content, exclusion, schema, permission, or closed-set
failure returns nonzero and leaves aggregate M2 unaccepted.

## Minimum Conformance Tests

The implementation test matrix must cover a successful publish followed by
synchronous and fresh SHA-pinned replay. It must also cover failures for:

- any missing, renamed, failed, skipped, or duplicated required JUnit identity;
- component manifest or checksum drift and disagreement between replay modes;
- calibration or formal count drift, retry, run-ID duplication or overlap, and
  a non-passing eligible terminal;
- accidental inclusion of a pilot, run09, run10, or formal campaign01, including
  an excluded/eligible run-ID intersection or campaign-ID reuse;
- missing historical Git objects, ancestry drift, marker relationship drift,
  and any protected path addition, removal, mode change, or blob change;
- vLLM snapshot, Python hash, or normalized dependency-set drift;
- model, extension, or Python content drift, metadata drift, symlink, hard link,
  concurrent mutation, or external closed-set drift;
- dirty or unexpected publication HEAD, authority blob drift, expected-hash
  drift, and claim or condition tampering; and
- pre-existing output, competing publication, extra entry, writable artifact,
  special node, wrong mode, missing/replaced/during-replay-replaced/`PREPARING`
  publication lock, wrong acceptance SHA-256, and post-publication replay
  failure.

No negative path may report aggregate success or make an output eligible for a
later status/index update.
