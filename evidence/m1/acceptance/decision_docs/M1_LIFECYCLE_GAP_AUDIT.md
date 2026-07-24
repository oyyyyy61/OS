# M1 Lifecycle Gap Audit

Updated: 2026-07-24. Scope: the current local vLLM GPU block/TTL path,
CPU-DRAM offload manager, offloading connector, and DAG benchmark integration.
The paper-facing event contract is `kv_lifecycle_event_v2` in
`M1_CANONICAL_LEDGER_SPEC.md`.

## Gate Decision

The v2 implementation now closes all five lifecycle state machines in the
observed single-process, primary-tier live traces. CPU producer request bindings
now detach at request completion, in-flight reset waits for transfer terminals
before manager reset, and the live prefetch coverage phase emits real scheduled
and completed DMA events. Request-detach timing is established by unit and full
regression evidence; the named GPU artifacts establish closure and ordering for
their captured source state. The strict auditor reports zero issues and all 12
gates pass for each audited trace in its declared action scope.

M1 is **PASS for the declared scope**: one process, one GPU, and the primary
CPU-DRAM offload tier. The live runs are lifecycle evidence, and the
raw-physical-line v2 measurement-control ABBA protocol completed all 48 rows
with exact frozen order and ledger closure. Both repeated-control precision
gates pass. Its two labels intentionally execute identical control semantics,
so the observed equivalence is measurement validation only and cannot support
a policy-benefit claim. The read-only post-M0 source freeze closes the source
identity gate. Global, secondary-tier, crash-recovery, and multi-GPU boundaries
listed below remain explicitly outside the validated scope.

## Evidence Levels

| Level | What it establishes | Current evidence | Decision use |
|---|---|---|---|
| Unit | Local transition semantics, fail-stop behavior, and regression safety | Current `offload/tests`: `323 passed, 14 warnings`; control-targeted scope: `24 passed`; related accounting scope: `64 passed, 1 NVML warning` | Supports implementation claims only |
| Synthetic | Exact state-machine closure, malformed-row rejection, reset/tombstone faults, and PBKV counterexamples | v2 accounting tests and fault fixtures pass; exact PBKV oracle covers `729` states | Supports mechanism and auditor semantics only |
| Live diagnostic | Real vLLM allocation, ownership, lease, save/load/prefetch, reset, and DMA terminal events | Six reset-fixed DAG traces plus one dedicated prefetch-coverage trace pass the strict auditor | Supports observed-path lifecycle closure only |
| Confirmatory measurement control | Frozen executable order, at least five independent seeds, stable repeated control, immutable source/config, and row-by-row ledger closure | Raw-physical-line v2 protocol: `48/48` rows, `96` journal events, `0` failures, 12 clusters, all gates true, immutable accepted result root | Supports measurement validity and lifecycle closure only |

The current complete `offload/tests` run reports `323 passed, 14 warnings`.
Ruff and format checks are green for all files changed or directly relevant to
this M1 work. A separate scan of the broader historical active tree reports 443
lint diagnostics and 73 files that would be reformatted, mostly in earlier
M2/M3 prototypes. That baseline remains open and was not mechanically rewritten
during this scoped lifecycle change.

## V2 State Machines

| State machine | Canonical transition | Implemented semantics | Current validation |
|---|---|---|---|
| Physical allocation | `allocate/completed -> evict/completed` | Identity is `(run, phase, tier, slot, generation)`; aligned capacity is counted once; generation increments on reuse | Unit, synthetic, and live diagnostic |
| Content mapping | `map/completed -> unmap/completed` | Mapping IDs bind immutable content to one live generation; rekey/invalidation close mappings without releasing capacity | Unit, synthetic, and live diagnostic |
| Logical owner binding | `bind/completed -> release/completed` | `request` and `workflow_retention` are separate zero-byte references; producer request release occurs on runtime detach; removal does not double-release | Unit/synthetic detach timing; live diagnostic closure |
| TTL lease | `lease/scheduled -> lease/{completed,failed,cancelled}` | Lease IDs attach to workflow-retention bindings; refresh/reset/invalidation preserve explicit terminal history | Unit, synthetic, and live diagnostic cancellation paths |
| Physical transfer | `{save,load,prefetch}/scheduled -> same-action/{completed,failed,cancelled}` | Per-rank results drain into one tombstone-backed terminal; payload bytes and observed bytes reconcile exactly | Unit, synthetic faults, and live diagnostic success paths |

The removal order is lease terminal, workflow-retention release, request
release, unmap, evict, then slot reuse. An in-flight strict reset blocks new
admission, drains worker results, emits the transfer terminal, and only then
resets the manager. A missing or conflicting worker result poisons the run.

## Implemented Repairs

1. CPU slots retain positive generations, immutable allocation snapshots,
   payload bytes, aligned capacity, producer identity, mapping IDs, and binding
   IDs through every terminal path.
2. `CPUOffloadingManager.on_request_finished` releases each producer request
   binding exactly once while the cached allocation remains available to later
   load owners. Failed store and reset cleanup reuse the recorded release state
   and cannot double-release it.
3. GPU block allocation and TTL events carry allocation generation, owner,
   reason, and byte identity through expiry/cancellation, rekey, and eviction.
4. Connector transfer jobs preserve per-rank outcomes in durable terminal
   tombstones. Deferred reset waits for the last rank and fails closed if the
   manager reset itself fails.
5. The benchmark rejects a GPU-block override that cannot fit the largest
   request after vLLM's reserved null block, preventing the earlier 64-block
   non-progress run.
6. A dedicated `cpu_dram_prefetch_coverage` phase creates a contiguous external
   CPU prefix and real scheduler prefetches. It is explicitly tagged
   `lifecycle_action_coverage_only` and `performance_claim_eligible=false`.

## Live Diagnostic Evidence

### Reset-fixed six-phase run

Result:
`experiments/results/m1_gpu_canonical_dag_qwen3_8b_s1_b2_t256_o1_g128_resetfix_20260723_2113.json`
(SHA-256 `23213421befe7055dfa8d84b54c4e41ecfad8aa626d78603932534966007713d`).

All six phase traces pass all 12 strict gates in their observed-action scopes,
with zero issues and zero live allocation, mapping, binding, lease, or transfer
operations at audit end:

| Phase | Canonical rows | Transfer coverage | Trace SHA-256 |
|---|---:|---|---|
| `baseline_vllm` | 1,666 | none opened | `2ed3c67a499e2e9b0e789f000bb9150bdfc5e423cc4f1dc6b5d9ff8f98965b23` |
| `shared_prefix_only` | 1,466 | none opened | `cce6c5685335388fd940304b1a3d20a4ed78ca1b01463f6d31a8bd6224b6ae77` |
| `cpu_dram_offloading_baseline` | 1,750 | save `4/4`, load `3/3` | `dd016a6e3811e12ee0c2706f341f55ed9514d7c7e60b66ac8e1a4e28c5acf239` |
| `ttl_only_policy` | 1,570 | none opened | `db8886e0314147a8c9b6845cd3e842d863f180c0aa0a0d68e02cddd4adc71b68` |
| `cpu_offload_preload_policy` | 1,740 | save `1/1` | `a832e61fced9990dae4580be11dffc12115146f7c54fe11af6e2f871319ca531` |
| `cpu_offload_background_prefetch_policy` | 1,740 | save `1/1`; no prefetch hit | `cedac07add021a698582dce594aba605fc31b79524b7360f93a23e0057bf508b` |

The main policy-pressure audit remains invalid: `valid_pressure=false`,
`valid_background_prefetch=false`, and `valid_background_wall_overlap=false`.
The policy phase stores 13 keys but its three prefetch attempts report no
external KV hit. These phases provide no policy-performance evidence.

### Real prefetch action coverage

Result:
`experiments/results/m1_gpu_canonical_dag_qwen3_8b_s1_b2_t256_o1_g128_prefetchcov4_20260723_215312.json`
(SHA-256 `1816b355af23311dddb4bcd71823966c70264b6e238d08d67e07d694f46ca38f`).

Trace:
`experiments/results/native_offload_traces/cpu_dram_prefetch_coverage_1784814836502232042.jsonl`
(SHA-256 `6dd17876b1b42f98d903068ad86b802944cfcf4ca7bad85e9f58e491ea2d55b3`).

The trace has 1,898 canonical rows. It records three real prefetch jobs, all
three scheduled and completed, covering 45 blocks, 720 tokens, and 106,168,320
payload bytes in each side of the transfer equation. The same trace closes 275
allocations, 271 mappings, 376 owner bindings, 18 leases, four saves, two loads,
and the three prefetches. All 12 gates pass with zero issues. This phase exists
only to prove action coverage and is excluded from latency/speedup analysis.

## Remaining Scope Gaps

1. A process-wide run-end barrier across every CPU manager has not been shown;
   the current proof closes the managers exercised by one local engine run.
2. CPU physical-slot identity is manager-local. Concurrent managers need a
   namespaced physical key before their ledgers can be merged safely.
3. Process death or file-write failure between a runtime transition and durable
   event append has no recovery journal proof.
4. The secondary filesystem tier does not yet provide the same allocation and
   generation ledger as the primary CPU tier.
5. Live success traces do not replace the synthetic failure matrix. Rank
   failure, submit rejection, partial completion, cancellation, reset timeout,
   and reset failure remain fault-suite evidence until a frozen fault campaign
   is archived.
6. The raw-physical-line v2 measurement-control ABBA run closes order,
   precision, canonical lifecycle, and payload-conservation gates for identical
   controls. The prior v1 run stopped at `36/48` because the frozen raw JSONL
   line indices were applied as positions after eligibility filtering; its
   completed rows therefore had sample-identity drift. It is retained as the
   excluded `superseded_failed_diagnostic` record, and v2 resolves this by
   declaring raw physical-line zero-based indexing end to end.

## Frozen Measurement-Control Protocol and Execution

`experiments/results/m1_measurement_control_abba_12seed_rawindex_v2_20260723/protocol.json`
has file SHA-256
`cdbe5476061b1e8d12bb6d018f203155f36974daa86ab3feb50b1546bf8ca614`.
Its internal protocol, configuration, and schedule digests are respectively
`15fc69536f975287cb4e88c9a978a46d5f84dffe33584e9fc737e194fac4778e`,
`34d1c5124226f8c1a05be4b7266051552cc034d9c3d65e2cb07ff49dc81d6fe1`,
and `b54d10860149fdb00d46851936d4bf5cb01696d7cdd52db4071238446ee76307`.
It declares `raw_jsonl_physical_line_zero_based` sample identity.

The protocol has 48 balanced ABBA rows over 12 independent clusters.
`control_a` and `control_b` resolve to the same executable CPU-DRAM control
configuration, and `core_payloads_frozen=true`. The accepted run completed
`48/48` rows with 96 journal events and zero failures. Its report SHA-256 is
`b6d9e2033c4439724f32690cfd109d017a0e12f8a922fbf02b32a7decf74c1d3`; the
journal SHA-256 is
`730fe31210862e57afb015d763072ea4a8574389cbdc8e0d0979ded98a3da712`.

All report gates pass. The paired geometric ratio is `1.0004591871` with 95%
CI `[0.9928798409, 1.0080963917]`, entirely inside the predeclared
`[0.95, 1.05]` equivalence interval. The relative 95% CI half-width is
`1.890365%` for `control_a` and `1.096603%` for `control_b`, both below 5%.
The 48 traces contain 75,776 canonical rows, zero audit issues, zero live
objects at end, and conserved payload bytes. The accepted result root has
`SHA256SUMS` SHA-256
`feaca3dc974afbdba95c7007cbacb21501868528b614f8fa1517e7f5f64bf1d2` and
manifest SHA-256
`230abc0d4f4a5213a7f353ffe49576fabc0b54201962be1e362cf5ad218c6ce2`;
files are mode `0444` and directories mode `0555`.

This validates measurement execution, source/config identity, and lifecycle
accounting for identical controls. It supplies no adaptive-policy effect size
or policy-benefit evidence.

## M1 Closure and M2 Entry

1. The no-overwrite post-M0 source/environment freeze is
   `experiments/source_freezes/m1_lifecycle_v2_20260724_001249`. Its manifest
   digest is
   `ed9ce0383ce1ae29616ca1eca312757136d9653e93774329a9c5ef8f135c6dfb`,
   and its top-level `SHA256SUMS` digest is
   `b2937a6c6e9164f3b82aa9078d8584ba12b5a96a81c44aca9f4738e4093f3166`.
2. The freeze records the scoped lint/format gates and the full-tree historical
   baseline. It makes no whole-tree-clean claim.
3. Multi-manager execution, secondary tiers, process/file crash recovery, and
   genuine multi-GPU behavior are excluded from the M1 result. Any future
   claim over those boundaries requires its own evidence.

M2 is open for canonical schema and live-orchestrator correctness work.
Performance tuning and C1-C3 benefit claims remain closed until their ordered
correctness and mechanism gates pass.
