# DAGKV Stage Gates

Updated: 2026-07-25.

## Imported Gates

- **M0:** accepted with the limitations recorded in the imported research
  contract.
- **M1:** pass for one process, one RTX 4090, and GPU plus primary CPU-DRAM.
  The immutable authority remains the sibling M1 acceptance package identified
  by `evidence/IMPORT_MANIFEST.json`.

## M2: Canonical Runtime

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
8. the v2 real-vLLM replay gate passes: a no-DMA GPU prefix-hit control
   localizes cold/prefix numerics, 59 fresh-process calibrations pass the
   preregistered `atol=0.125, rtol=0` cap, and 20 new formal holdouts pass after
   the aggregate calibration manifest and tolerance are frozen;
9. source and binary state, model files, environment, protocol, commands,
   attempt inventory, raw traces, reports, and every aggregate input are
   content-addressed and frozen; missing or inconsistent provenance fails
   closed.

Component tests can close items 1-7. Run03 is a pilot and is excluded from the
59 calibration processes and 20 formal holdouts. Each formal run can emit only
one holdout-pass manifest. Item 8 closes only after a closed-set aggregator
verifies 20/20 new holdouts under one frozen tolerance and writes
`M2_ITEM8_ACCEPTANCE_MANIFEST.json`. That manifest does not close aggregate M2;
M2 remains open until all nine conditions pass.

M2 establishes lifecycle and data-plane correctness only. It makes no latency,
throughput, hit-rate, scheduling-policy, or novelty claim. Tokencake already
covers DAG-aware critical-agent reservation and predictive D2H/H2D; Continuum
covers history-derived TTL and queue-aware retention; PBKV covers dynamic call
graphs, shared-node future-access aggregation, lifecycle-aware eviction, and
conservative prefetch. The exact sources and hashes remain frozen in
`research/REFERENCES.md`.

## M3-M6

- **M3:** dependence-correct probabilistic shared leases (C1), an explicitly
  derived capacity/DMA-constrained joint controller (C2), and deadline-aware
  partial-prefix single-flight (C3), each behind independent switches, matched
  first-order baselines, and fault tests.
- **M4:** trace provenance, field audit, split policy, and leakage review.
- **M5:** preregistered paired matrix with immutable schedules and raw outputs.
- **M6:** claim-to-evidence audit, paper tables, artifact reconstruction, and
  reviewer-style falsification review.
