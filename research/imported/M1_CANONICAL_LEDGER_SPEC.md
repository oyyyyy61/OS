# M1 Canonical Ledger Specification

Status: accepted M1 ledger contract. It was frozen on 2026-07-23 before the
confirmatory execution, and acceptance was recorded on 2026-07-24. This
specification supersedes the unaccepted `kv_lifecycle_event_v1` draft. No
confirmatory run used v1, so the paper-facing schema starts at
`kv_lifecycle_event_v2`.

## Scope

The ledger proves five independent conservation properties:

1. physical allocation lifetime;
2. content-to-allocation mapping lifetime;
3. logical owner binding lifetime;
4. TTL lease lifetime;
5. physical transfer terminal accounting.

One event family cannot close another. In particular, a physical eviction
cannot stand in for an owner release, and a transfer terminal cannot stand in
for allocation release.

The accepted M1 runtime scope is one process on one GPU, with GPU allocation
and the primary CPU-DRAM offload tier. Live success-path evidence is combined
with the frozen fault suite for terminal behavior. Secondary filesystem or
object tiers, process/file crash recovery, and genuine multi-rank or multi-GPU
execution are outside the M1 evidence boundary and cannot inherit its result.

## Common Identity

Every row carries the following strict fields without post-processing defaults:

```text
schema_version, event_id, parent_event_id,
run_id, phase, source, workflow_id, request_id, operation_id,
action, status, reason, timestamp_ns,
source_tier, target_tier,
blocks, block_count, byte_count, observed_byte_count
```

`run_id` identifies one immutable schedule slot. `workflow_id` identifies one
DAG instance inside that run. `request_id` identifies the runtime request that
owns or triggers the event. Event IDs are globally unique within a trace.

A physical allocation is identified by:

```text
(run_id, phase, tier, block_id, allocation_generation)
```

The physical key deliberately excludes workflow and request because one
allocation may be shared. Producer lineage remains on the event and must match
across `allocate` and `evict`. A different trigger owner is recorded only in
explicit extra metadata.

## State Machines

### Physical allocation

```text
allocate/completed -> evict/completed
```

- `allocate` contains exactly one `physical_slot` block reference.
- Its block byte count is aligned allocated capacity, not transfer payload.
- `evict` repeats the exact allocation resource tuple and byte count.
- `evict.parent_event_id` equals the corresponding allocate event ID.
- Capacity, reset, failed-store cleanup, forced release, and slot reuse are
  distinct reasons. Every reason still closes the same physical lifetime.
- A slot generation increases before every reuse and is always positive.

### Content mapping

```text
map/completed -> unmap/completed
```

- Each pair has a non-empty `mapping_id` and immutable content handle.
- Both rows reference the same live physical allocation and content identity.
- Both rows contain exactly one zero-byte `physical_slot` block reference;
  physical capacity is counted only by `allocate` and `evict`.
- `map.parent_event_id` references the allocation open event.
- `unmap.parent_event_id` references the map event.
- Hash rekey closes the old mapping and opens a new mapping on the same slot
  generation. It does not release physical bytes.
- Connector invalidation closes the affected mapping. A later physical release
  is represented independently by `evict`.

### Logical owner binding

```text
bind/completed -> release/completed
```

- Each pair has a non-empty `binding_id`.
- Each pair declares `binding_kind` as `request` or `workflow_retention`.
- Both rows contain exactly one zero-byte `physical_slot` block reference.
- `bind.parent_event_id` references the allocation open event.
- `release.parent_event_id` references the bind event.
- The binding identity is `(run_id, phase, workflow_id, request_id,
  binding_id)` and includes the physical allocation key.
- Shared allocations have one bind/release pair per logical owner.

A request binding closes when that runtime request detaches. A TTL that must
survive request completion owns a separate `workflow_retention` binding. This
prevents an open lease from referring to a closed request binding. The
retention binding closes only after all of its leases terminate.

### TTL lease

```text
lease/scheduled -> lease/{completed,failed,cancelled}
```

- Each lease has non-empty `lease_id` and `binding_id` fields. Its parent bind
  must have `binding_kind=workflow_retention`.
- The scheduled row references that bind event and carries
  `registered_timestamp_ns` and `deadline_timestamp_ns`.
- A completed terminal means the declared deadline expired.
- Refresh, workflow detach, forced eviction, invalidation, rekey, and reset
  cancel the old lease with an explicit outcome.
- Refresh then opens a new lease generation; it never overwrites history.
- The terminal references the scheduled lease event and records
  `observation_timestamp_ns`. Exactly one terminal is allowed.

### Physical transfer

```text
{save,load,prefetch}/scheduled
    -> same-action/{completed,failed,cancelled}
```

- Scheduled and terminal rows repeat the exact resource tuple.
- The terminal references the scheduled event.
- Transfer byte counts are payload DMA bytes.
- A completed terminal requires observed bytes equal declared bytes.
- Failed and cancelled terminals require exact observed bytes in the inclusive
  range `[0, declared]` after all unique worker ranks report.
- Missing worker results, conflicting rank reports, or an unknown final byte
  count fail the run; they are not repaired as zero-byte terminals.

## Required Ordering

For a physical allocation that is being removed:

1. terminate every live lease;
2. release every workflow-retention binding;
3. release every request binding;
4. close every content mapping;
5. emit the physical `evict` terminal;
6. return the slot to the free list and allow its next generation.

CPU allocation opens before the corresponding save is scheduled. A failed
store therefore closes a real allocation with `evict(reason=failed_store_cleanup)`.
GPU allocation time and reason are retained until the first strict producer
binding supplies full run/workflow identity.

Every paper-facing run ends with an explicit allocation barrier. The barrier
closes leases, bindings, mappings, and all non-null live generations whose
runtime reference count is zero, using `evict(reason=run_end)`. Any remaining
reference, binding, or unclosed lease is a hard run failure. A generation that
is closed at the barrier still increments normally when the pool assigns that
slot in a later run.

Reset records cancellation intent, requests worker flush, and continues to
aggregate rank results. A drain deadline with missing ranks is a hard runtime
failure. It cannot emit an inexact terminal and continue the measurement.

## Conservation Gates

For every accepted trace:

```text
allocates = evicts + live_allocations
maps = unmaps + live_mappings
binds = releases + live_bindings
lease_opens = lease_terminals + live_leases
transfer_scheduled = completed + failed + cancelled + inflight
```

Paper-facing completed runs require all five live terms to be zero. Byte
conservation is checked independently for aligned allocation capacity and DMA
payload. Duplicate opens, duplicate terminals, cross-workflow parent swaps,
generation aliasing, time reversal, and observed-byte overrun each invalidate
the run.

Action coverage is an aggregate M1 test-suite gate. An individual run is not
required to manufacture an expiry or failure, but every action it opens must
close, and the frozen fault suite must exercise every terminal family before
confirmatory execution.

## M1 Lifecycle and Measurement Acceptance

The accepted evidence consists of:

- `experiments/results/m1_lifecycle_v2_acceptance_20260723_215312`, which
  contains seven strict live traces, all ten lifecycle actions, zero audit
  issues, and zero live allocations, mappings, bindings, leases, or transfers;
- `experiments/results/m1_measurement_control_abba_12seed_rawindex_v2_20260723`,
  which contains the frozen 12-seed, 48-row ABBA control execution, verified
  raw-record identities, exact precomputed order, complete trace audits, and
  repeated-control relative 95% CI half-widths of 1.890365% and 1.096603%; and
- the schema, state-machine, identity, failure, cancellation, expiry, and
  accounting tests referenced by the M1 reproduction manifest.

The paired identical-label ratio is a measurement-system check and carries no
policy-benefit claim. Lifecycle conservation, sample identity, order,
independent-seed count, precision, and raw-artifact hashing pass. The final
post-M0 source archive is
`experiments/source_freezes/m1_lifecycle_v2_20260724_001249`; its top-level
`SHA256SUMS` digest is
`b2937a6c6e9164f3b82aa9078d8584ba12b5a96a81c44aca9f4738e4093f3166`.
Checksum, clean-base patch application, deterministic archive, dependency,
inventory, semantic-manifest, and read-only permission checks all pass.

M1 therefore passes for the declared single-process, single-GPU,
primary-CPU-tier scope. This result opens M2 canonical-runtime integration. It
does not extend to the explicitly excluded tiers or failure domains and does
not establish any adaptive-policy performance benefit.
