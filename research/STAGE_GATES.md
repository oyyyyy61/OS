# DAGKV Stage Gates

Updated: 2026-07-24.

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
8. a real vLLM GPU replay matches the baseline tokens and logits under the
   declared tolerance;
9. the source, environment, protocol, raw traces, and reports are frozen.

Component tests can close items 1-7. Item 8 requires live GPU evidence. M2
remains open until all nine conditions pass.

## M3-M6

- **M3:** re-admission value, constrained control, and partial-prefix
  single-flight mechanisms, each behind independent switches and fault tests.
- **M4:** trace provenance, field audit, split policy, and leakage review.
- **M5:** preregistered paired matrix with immutable schedules and raw outputs.
- **M6:** claim-to-evidence audit, paper tables, artifact reconstruction, and
  reviewer-style falsification review.
