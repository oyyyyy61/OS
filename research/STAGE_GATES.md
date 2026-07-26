# DAGKV Stage Gates

Updated: 2026-07-26.

## Imported Gates

- **M0:** accepted with the limitations recorded in the imported research
  contract.
- **M1:** pass for one process, one RTX 4090, and GPU plus primary CPU-DRAM.
  The immutable authority remains the sibling M1 acceptance package identified
  by `evidence/IMPORT_MANIFEST.json`.

## M2: Canonical Runtime

Status: **ACCEPTED_CORRECTNESS_ONLY**. The create-only aggregate decision and
fresh independent replay are indexed in
`evidence/m2/v3_580_173_02/M2_AGGREGATE_ACCEPTANCE_EVIDENCE_INDEX.json`.
This status authorizes no performance, scheduling-policy, novelty, C1, C2, or
C3 claim.

M2 passes only when all conditions below have direct evidence:

1. exactly one canonical block, binding, lease, workflow, replica, transfer,
   and execution-reference schema, with historical ID uniqueness;
2. two owners can share one block without cross-owner release or early reclaim;
3. duplicate release and duplicate terminal handling are idempotent;
4. stale allocation generations and stale transfer completions fail closed;
5. refcount, binding, mapping, lease, replica, and transfer conservation pass;
6. one H2D operation can atomically publish every compatible waiter mapping;
7. workflow completion and injected failure leave no use-after-free path;
   DAG readiness also requires the owning node to be running;
8. the v3 real-vLLM replay gate passes: a no-DMA GPU prefix-hit control
   localizes cold/prefix numerics, 59 fresh-process calibrations pass the
   preregistered `atol=0.125, rtol=0` cap, and 20 new formal holdouts pass after
   the aggregate calibration manifest and tolerance are frozen; production
   calibration must also bind a clean preparation HEAD to a direct-child,
   marker-only execution commit, hold one campaign-directory `flock`, repeat
   the execution binding in every submitted row, and prove one HEAD and DAGKV
   snapshot across all 59 runs;
9. source and binary state, model files, environment, protocol, commands,
   attempt inventory, raw traces, reports, and every aggregate input are
   content-addressed and frozen; missing or inconsistent provenance fails
   closed.

Component tests can close items 1-7 only after the three-suite deterministic
contract in `research/protocols/M2_COMPONENT_EVIDENCE_PROTOCOL.md` produces a
create-only, read-only `M2_COMPONENT_EVIDENCE.json` bundle and independent
replay passes. That bundle cannot close item 8 or aggregate M2. Every attempt
indexed by `evidence/m2/PILOT_ATTEMPTS.json`, plus any protocol-validation
execution before the calibration v3 launch marker, is a pilot excluded from the
59 calibration processes. Any validation execution before the formal launch
marker is excluded from the 20 formal holdouts. Each formal run can emit only
one holdout-pass manifest. Item 8 closes only after a closed-set aggregator
verifies 20/20 new
holdouts under one frozen tolerance and writes
`M2_ITEM8_ACCEPTANCE_MANIFEST.json`. That manifest does not close aggregate M2;
all nine conditions must pass the separate aggregate protocol. That aggregate
protocol has now passed and recorded `M2_ACCEPTED_CORRECTNESS_ONLY`.

M2 establishes lifecycle and data-plane correctness only. It makes no latency,
throughput, hit-rate, scheduling-policy, or novelty claim. Tokencake already
covers DAG-aware critical-agent reservation and predictive D2H/H2D; Continuum
covers history-derived TTL and queue-aware retention; PBKV covers dynamic call
graphs, shared-node future-access aggregation, lifecycle-aware eviction, and
conservative prefetch. The exact sources and hashes remain frozen in
`research/REFERENCES.md`.

The prior 59/59 v2 calibration is retained as historical evidence under its
original driver and implementation fingerprint. The current v3 gate binds the
exact NVIDIA Debian package payloads, loaded driver `580.173.02`, and actual
mapped `libcuda`; it therefore requires a new excluded pilot, 59-process
calibration, frozen tolerance, and 20-process formal cohort. The excluded pilot
passed, the v3 calibration is 59/59, and its create-only tolerance is frozen.
The first formal campaign failed its post-aggregate replay before seal
publication and remains excluded. Campaign02 restarted from zero, passed 20/20,
published the item-8 acceptance manifest and create-only seal, and passed
independent full-bundle replay. Item 8 is closed under v3 and remains
independently scoped from the later aggregate decision. The cross-evidence
audit then replayed 627 component cases, 59 calibration runs, 20 formal
holdouts, the historical compatibility bridge, 22 excluded run IDs with zero
eligible overlap, and 23 current external files. Its create-only output passed
a fresh independent replay, closing M2 for correctness only. No v2 sample or
tolerance contributes to this item-8 or aggregate decision.

## M3: Policy Mechanisms

Status: **C1_A_IMPLEMENTED_UNSEALED**. C2 and C3 have not started.

The normative C1 boundary and later experiment gates are frozen in
`research/protocols/M3_C1_SHARED_LEASE_PROTOCOL.md`. The current component:

- binds every forecast to an immutable block snapshot and exact ledger event
  count;
- represents mutually exclusive and correlated outcomes jointly and permits
  multiplication only across declared-independent groups;
- deduplicates concurrent fanout into one physical reuse epoch;
- separates first physical re-admission probability, expected unique epochs,
  and expected repeats;
- computes exact PPM nominal values and sound total-variation drift bounds;
- rejects stale, cross-owner, ineligible-node, identity-drift, invalid-mass,
  and online-oracle inputs;
- exposes C1 nominal, C1 robust-lower, a PBKV-style additive proxy, and
  independent-union modes for matched attribution.

Sixteen focused C1 tests and eighteen evidence-validator tests pass; the full
repository regression currently collects 349 tests. This is implementation
evidence only.
C1-A remains unsealed until a clean committed source snapshot, JUnit,
checksums, aggregate M2 historical replay, and an independent component replay
are frozen. C1-B trace/calibration, C1-C paired policy replay, and C1-D
real-GPU gates remain open. No C1 performance or novelty claim is authorized.

M3 must additionally implement an explicitly derived capacity/DMA-constrained
joint controller (C2) and deadline-aware partial-prefix single-flight (C3).
Each mechanism requires an independent switch, matched first-order baselines,
and fault tests.

## M4-M6

- **M4:** trace provenance, field audit, split policy, and leakage review.
- **M5:** preregistered paired matrix with immutable schedules and raw outputs.
- **M6:** claim-to-evidence audit, paper tables, artifact reconstruction, and
  reviewer-style falsification review.
