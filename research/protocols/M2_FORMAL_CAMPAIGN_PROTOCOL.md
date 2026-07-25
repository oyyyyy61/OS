# M2 Formal Holdout Campaign Protocol

Protocol schema: `dagkv.m2.formal_campaign.v1`

This protocol governs the orchestration evidence for the M2 item-8 formal
holdout campaign. The data-plane ABBA contract remains frozen separately in
`M2_VLLM_REPLAY_PROTOCOL.md`. This document does not change that implementation
or its calibration-derived numerical cap.

## Parent Evidence

Preparation requires a published 59-process calibration bundle and its
create-only `M2_FROZEN_TOLERANCE.json`. The shared calibration evidence
validator must replay the complete upstream campaign. The supplied files must
bind the same implementation manifest, reproducibility fingerprint,
`atol=0.125`, and `rtol=0`. Their content hashes, paths, and sizes are frozen in
the formal preregistration. The parent must use calibration preregistration v3,
the committed `CALIBRATION_V3_LAUNCH_MARKER.json` direct-child binding, one
common execution HEAD and DAGKV snapshot across all 59 runs, and the same
NVIDIA bundle root, manifest SHA-256, content digest, and driver version frozen
for formal execution.

The formal launcher also freezes the data-plane runner, formal aggregator,
independent raw replay validator, shared calibration evidence validator,
process supervisor, both protocol files, and itself. Every submission checks
these file hashes and the current data-plane implementation manifest.

## Two-Stage Launch

Preparation creates a brand-new campaign root containing only
`FORMAL_CAMPAIGN_PREREGISTRATION.json`, fsyncs the directory and file, and
returns the preregistration SHA-256. Execution requires that exact digest on
the command line. Any extra entry in the prepared root rejects execution.

Production preparation requires a clean DAGKV worktree and records its Git
HEAD before and after parent-evidence replay; both observations must agree.
The production convenience path that prepares and immediately executes is
disabled. After preparation, the preregistration digest must be written to
`evidence/m2/FORMAL_LAUNCH_MARKER.json` and committed as the only changed path
in one direct, single-parent child of the preparation HEAD. The marker has the
exact schema `dagkv.m2.formal_launch_marker.v1` and binds the campaign ID,
campaign root, preregistration digest, preparation HEAD, timestamp, and the
item-8 correctness-only claim scope.

Before the first journal write, execution requires a clean worktree at that
marker commit, verifies the committed marker object byte-for-byte, and records
the preparation HEAD, execution HEAD, marker path, and marker SHA-256 as one
execution binding. The binding is repeated in every run and aggregate
submission. It is revalidated before every submission, before aggregation,
and before seal publication. The campaign directory itself is held under one
non-blocking exclusive `flock` for the complete execution, preventing two
launchers from interleaving the append-only journal without adding an
undeclared lock file.

The preregistration fixes exactly 20 ordered names, `run-001` through
`run-020`, one attempt per name, zero retries, and stop on first failure. No
command-line run-count override exists. Test-only Python calls may inject a
smaller count and must mark the resulting preregistration as non-production.

## Fresh Holdout Processes

Each holdout uses a new output directory, OS process, CUDA context, and vLLM
engine. The command invokes the frozen runner with:

- `--mode formal`;
- the frozen `--tolerance-file` and `--calibration-manifest`;
- the frozen model, vLLM root, CPU allocation, timeout, and CUDA device;
- the NVIDIA bundle root, expected manifest SHA-256, content digest, and driver
  version; and
- `--full-provenance`.

The launcher records one fsynced `submitted` row before spawn and one fsynced
terminal row after process cleanup. It records the command, process ID,
timestamps, exit status, timeout and signal actions, stdout/stderr hashes, and
the complete output inventory. A timeout or orchestration interruption cleans
the entire process group with SIGTERM followed by SIGKILL when required. A
nonzero exit, lingering descendant, failed artifact validation, or fingerprint
drift stops the campaign permanently without replacement.

Every passing terminal binds `result.json`, `provenance.json`, `SHA256SUMS`, and
`M2_ITEM8_FORMAL_RUN_MANIFEST.json`. Validation uses both the formal
aggregator's per-run validator and the independent raw replay validator. A
single process may claim only a formal holdout pass; it cannot claim item-8 or
aggregate M2 acceptance.

Each run must record the marker execution HEAD in its DAGKV Git capture. All
20 runs must share one content-addressed DAGKV snapshot in addition to the
same execution HEAD.

## Journal and Aggregation

`FORMAL_ATTEMPTS.jsonl` is append-only. Its formal prefix contains exactly 40
records: submitted and passed terminal rows for each ordered run. The launcher
seals this prefix by byte length, record count, and SHA-256 before appending the
single aggregate submission.

The formal aggregator receives the campaign root, upstream calibration
manifest, frozen tolerance, and exclusive output path. It validates exactly 20
direct run directories and may create only
`M2_ITEM8_ACCEPTANCE_MANIFEST.json`. The launcher validates the acceptance
candidate before recording the aggregate terminal. The final journal contains
exactly 42 records: the sealed 40-row run prefix followed by one aggregate
submission and one passed aggregate terminal.

The acceptance manifest closes item 8 only. It must retain `m2_accepted=false`
and `performance_claims_supported=false`.

## Independent Bundle Seal

The formal evidence validator independently replays the root closed set,
preregistration, parent evidence, frozen files, all 42 journal rows, run and log
inventories, process identities and timestamps, per-run raw artifacts, sealed
40-row prefix, aggregate logs, and acceptance mapping. It rejects extra files,
directories, attempts, retries, symlinks, failed terminals, reordered runs, or
content drift.

It independently reconstructs the runner command, aggregate command, Python
entry point, model and vLLM paths, resource/time limits, and environment from
the frozen fields. It also resolves the preparation and marker commits from
Git objects and checks the marker-only commit relationship without trusting
the launcher. Tree inventories include inode metadata, ctime, and a SHA-256
for every file; the exact root set and critical bytes are checked again at the
end of replay.

After a successful prepublication replay, the validator publishes
`M2_FORMAL_BUNDLE_SEAL.json` with create-only semantics. The seal binds:

- the preregistration SHA-256;
- byte length, record count, and SHA-256 of the 40-row formal prefix;
- byte length, record count, and SHA-256 of the complete 42-row journal;
- the acceptance SHA-256;
- the calibration and frozen-tolerance SHA-256 values;
- the implementation manifest and reproducibility fingerprint;
- the NVIDIA bundle root, manifest SHA-256, content digest, and driver version;
- the marker execution binding and common DAGKV snapshot SHA-256;
- the ordered 20 run names, attempt IDs, run IDs, result, provenance,
  checksum, per-run formal-manifest, DAGKV HEAD, and DAGKV snapshot hashes.

The validator rescans all inputs immediately before publication. A published
seal is synchronously replayed before the publish operation can report
success. A published bundle is eligible for downstream item-8 evidence only
after a fresh validator call replays the seal and every bound input. The
acceptance file alone is not a complete campaign-orchestration record.

## Failure Rules and Claim Boundary

A failed or partial campaign remains on disk as failure evidence and cannot be
resumed. A data-plane implementation or fingerprint change requires a new
calibration version and a new formal campaign. A formal orchestration change
requires a brand-new preregistration and campaign root.

This campaign evaluates repeatability of the frozen single-prompt ABBA
correctness path. It provides no latency, throughput, cache-hit, scheduling,
fairness, deadline, partial-prefix, multi-waiter, or paper-performance claim.
