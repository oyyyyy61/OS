# DAGKV Clean-Room Architecture

Status: normative for the new implementation, 2026-07-24.

## Boundary

The sibling `offload/` directory contains useful experiments alongside several
parallel and incomplete runtime models. DAGKV does not execute or import that
code. Accepted M1 contracts and manifests are copied byte-for-byte under
`research/imported/` and `evidence/m1/`; `evidence/IMPORT_MANIFEST.json`
records their origin and digest.

## One Canonical State

`src/dagkv/domain.py` defines the only persisted runtime identities and state:

- `BlockKey`: immutable KV content identity;
- `ReplicaId`: physical tier, device, slot, and allocation generation;
- `ExecutionRef`: request-local logical block identity;
- `WorkflowSpec`: immutable DAG topology;
- `NodeRecord`: dependency-gated node execution and terminal state;
- `OwnerBinding`: one request or workflow-retention owner relationship;
- `Lease`: one explicit protection lifetime on a retention binding;
- `ContentMapping`: immutable content to one physical allocation;
- `ExecutionMapping`: request-local engine reference to one GPU generation;
- `BlockRecord`: replicas, derived owner/lease indices, transfer, and GPU
  location version;
- `Transfer`: one physical copy with exact byte and digest terminals.

External objects are translated once at adapter boundaries. They are never
stored as parallel runtime truth.

## Ownership

`LifecycleOrchestrator` is the sole writer of canonical state. A transition is
valid only when all related registries can be changed atomically. In
particular:

1. a request binding owns one execution reference while a retention binding
   owns none;
2. an execution reference is single-use for the orchestrator lifetime and
   cannot be rebound after release, including to the same content;
3. a lease names an active retention binding and the same block;
4. an execution mapping exists only while its binding is active, a valid GPU
   replica exists, and the binding's DAG node lifecycle is `RUNNING`; each map
   activation has a monotonically increasing identity; a binding cannot open
   before at least one content location is live;
5. execution readiness additionally requires the owning DAG node to be
   `RUNNING`;
6. a transfer has exactly one terminal status and validates bytes plus digest;
7. CPU-resident content can publish a GPU replica only through a successful H2D
   terminal;
8. final reclaim requires no active binding, lease, mapping, or transfer;
9. slot generations increase on reuse and stale generations cannot publish.

The append-only ledger commits compound transitions as a validated batch before
the corresponding in-memory mutation. A target slot enters a private
reservation state before DMA. A valid terminal publishes the replica and all
compatible H2D waiters while holding the writer lock. Failure or cancellation
closes the reserved allocation and consumes its generation.

The ledger permanently records every allocation, mapping, binding, lease,
transfer, and node identity after its first open. A closed identity cannot be
opened again. Parent validation freezes workflow, request, node, binding kind,
execution reference, physical resource, and transfer geometry across each
lifecycle. Cross-family replay additionally proves that content maps refer to
live allocations, execution maps refer to live GPU content, a live binding
cannot lose its final content location, and failed or cancelled transfer
targets can only be evicted. A GPU allocation opened while CPU content is live
receives no publication entitlement until its `LOAD` or `PREFETCH` completes.
Payload size and digest are canonical for a `BlockKey` across every tier and
later generation. Lease events carry their declared deadline so early expiry is
independently detectable. Replay also validates the schema, run, phase, source,
canonical event ID, sequence, and complete parent graph of every row.

Node terminal rows cannot commit while an execution mapping for that workflow
node remains live. Workflow failure batches close leases, execution mappings,
and bindings before their node terminals at the same timestamp.

## Ledger Families

`dagkv_lifecycle_event_v1` preserves M1 physical `allocate/evict`, content
`map/unmap`, logical `bind/release`, lease, and transfer families. It adds two
families needed by the canonical M2 runtime:

- `exec_map/exec_unmap` records request-local engine mappings without changing
  content-map conservation;
- `node/{scheduled,completed,failed,cancelled}` records actual DAG dependency
  execution.

Logical owner bindings in DAGKV attach to immutable content, so they can remain
valid while that content moves between tiers. This is an explicit schema delta
from the physical-allocation binding used by the frozen M1 adapter. The M2 GPU
evidence freeze must version and validate this exporter before acceptance.

## Runtime Flow

```text
DAG event
  -> binding and lease transition
  -> block policy input snapshot
  -> transfer or reclaim command
  -> transfer terminal and execution-map commit
  -> execution-map plus RUNNING-node readiness gate
  -> workflow completion or failure cleanup
  -> conservation audit
```

Policy scoring will consume immutable snapshots. It cannot mutate residency or
ownership directly. Engine adapters execute commands and report terminals back
to the orchestrator.

## Failure Model

Unknown identity, conflicting generation, conflicting terminal replay, byte
mismatch, digest mismatch, cross-owner release, and reclaim with live
references are hard errors. Exact duplicate release and terminal callbacks are
idempotent. A transfer integrity error records a failed terminal and closes its
target allocation before returning the error.
