# M2 Canonical Runtime Contract

Status: accepted M2 correctness contract, 2026-07-26. The decision identity is
recorded in
`evidence/m2/v3_580_173_02/M2_AGGREGATE_ACCEPTANCE_EVIDENCE_INDEX.json`.

## Scope

The current runtime models one process, one GPU tier, and one primary CPU-DRAM
tier. It establishes lifecycle correctness before policy work. Multi-process,
multi-GPU, secondary storage, crash recovery, and performance claims remain
outside this contract.

## Authority

`LifecycleOrchestrator` is the sole writer of canonical runtime state. Engine
and workflow adapters receive immutable commands or snapshots and report
events back through owner-qualified methods. They cannot retain references to
mutable dataclasses.

The following registries form one transaction domain:

```text
workflow/node -> binding -> block -> replica/content map
                         -> lease
request binding -> execution map -> current GPU generation
                -> permanent execution-reference history
block -> in-flight transfer -> private target reservation
physical slot -> monotonically increasing generation
```

Workflow, block, and lease ID sets are derived indices. The auditor reconstructs
the opposite edge independently and rejects any mismatch.

## State Rules

1. A DAG node starts only after every predecessor is `DONE`. A request is
   executable only while its owning node is `RUNNING`; the ledger rejects both
   execution-map publication without a live node lifecycle and a node terminal
   that still owns an execution map.
2. A request binding has exactly one `ExecutionRef`; a retention binding has
   none and is the only binding kind that may own a lease. An `ExecutionRef`
   is single-use and remains reserved after its binding is released.
3. `REQUIRED` means a request has a current GPU execution mapping. `RETAINED`
   means the logical owner remains while no execution mapping is required.
   `WAITING` is entered only while attached to an H2D single-flight. Direct
   bind and state-transition calls may enter `REQUIRED` only for a running
   node. H2D waiter admission and terminal publication independently recheck
   the same gate; a node that terminates during DMA returns to `RETAINED`. A
   prefetched content mapping alone does not pass the execution-readiness gate.
   The GPU-resident fast path accepts only `RETAINED` owners without a map or an
   idempotent `REQUIRED` owner with the exact current map.
4. A content mapping describes one immutable block on one live allocation.
   Request execution mappings are separately conserved, and each activation
   uses a new per-binding generation. Payload size and digest are frozen on the
   first allocation and must match every later allocation, map, and transfer.
   A binding can open only while at least one content mapping for its block is
   live.
5. A target allocation is private to one scheduled transfer. A terminal checks
   the source generation, target generation, reservation owner, reported
   payload bytes, and reported SHA-256 digest before publication. Real DMA
   payload correctness remains part of the GPU adapter gate.
6. H2D success creates one content mapping and publishes every still-active,
   compatible waiter in the same writer critical section. Owners released or
   retained during the copy receive no mapping.
7. Failed and cancelled copies close the target allocation. The next reuse of
   that slot must increment generation, including after a failed copy. Their
   targets have no publication entitlement and cannot emit a content map.
8. Drop and reclaim commands carry the expected replica generations and GPU
   location version, so stale policy decisions cannot delete re-admitted state.
9. A block with a live CPU replica can regain GPU residency only through a
   validated `LOAD` or `PREFETCH` terminal. Producer registration cannot bypass
   that transition. The ledger assigns this entitlement to the exact target
   allocation generation and verifies it again during independent replay.
10. Final reclaim requires zero active bindings, leases, execution mappings,
   reservations, and transfers. It then closes content mappings before their
   physical allocations.

## Idempotency And Failure

Release is idempotent only after the caller supplies the exact original
`BindingHandle`. A forged workflow or request identity fails even after the
real owner released the binding.

A duplicate lease or transfer terminal is idempotent only when state,
timestamp, byte count, digest, and error payload exactly match the first
terminal. Conflicting replays fail closed and append no row. Integrity mismatch
after a transfer began records the real observation, emits failed-target
cleanup, and surfaces `TransferIntegrityError`.

Ledger IDs are lifetime identities. Allocation, content-map, binding, lease,
transfer, execution-map, and node IDs cannot reopen after a terminal event.
Every child event must preserve the parent workflow, request, node, binding
kind, execution reference, block, and physical generation fields applicable to
that lifecycle. The independent replay also validates physical-slot generation,
one allocation per block tier, one in-flight transfer per block, live
cross-family references, canonical payload metadata, and transfer-target
outcomes. Node references are keyed by workflow and node identity independently
of their operation IDs.

Every replayed row must retain the ledger schema version, run, phase, source,
canonical sequence-derived event ID, and a parent that was previously accepted.
An invalid row is excluded from replay state so it cannot legitimize a later
child.

Lease-open and terminal rows carry the same deadline. `EXPIRED` is valid only
at or after that deadline; an earlier explicit teardown uses `CANCELLED`.

## Acceptance Boundary

Component tests may close M2 conditions 1-7 in `STAGE_GATES.md`. M2 additionally
requires a forced D2H/H2D vLLM replay with exact token equality, frozen logit
tolerances, engine/version provenance, raw traces, and a read-only evidence
package. The v3 calibration/formal chain and the separate nine-condition
aggregate replay have now passed, so this contract is accepted within its
single-process, one-GPU plus primary CPU-DRAM scope. The decision makes no
latency, throughput, hit-rate, scheduling-policy, novelty, C1, C2, C3, or
paper-performance claim.
