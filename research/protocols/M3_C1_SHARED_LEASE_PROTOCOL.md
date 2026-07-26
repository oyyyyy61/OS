# M3/C1 Dependence-Aware Shared Lease Protocol v1

## Material Passport

- Origin skill: Academic Research Suite `experiment-agent`
- Origin mode: plan
- Origin date: 2026-07-26
- Verification status: implementation protocol; performance unverified
- Version label: `m3_c1_protocol_v1`

## Status And Scope

This protocol freezes the first M3 mechanism boundary after aggregate M2
acceptance. C1 is a read-only policy calculation over an immutable projection
of canonical runtime state. It cannot mutate residency, ownership, mappings,
transfers, or leases. Any action remains an owner-qualified M2 transition.

The v1 implementation gate covers exact dependence semantics, robust
uncertainty bounds, state binding, and matched aggregation baselines. It does
not authorize latency, throughput, hit-rate, scheduling, or novelty claims.
Trace calibration belongs to M4; paired real-GPU effects belong to M5; the
claim-to-evidence decision belongs to M6.

## First-Order Boundary

The exact local sources and hashes are frozen in `research/REFERENCES.md`.

- PBKV Section 4.2.2 and Equations 1-2 sum per-workflow, per-step access
  probabilities at a shared radix node. Its score, lifecycle hierarchy, and
  conservative prefetch are first-order baselines. Generic shared-node
  probability aggregation is occupied.
- Continuum Section 4.1 and Equations 1-2 derive a per-request TTL from an
  empirical tool-duration CDF, reload and queueing benefit, and GPU occupancy
  cost. Historical TTL retention and expiry are first-order baselines.
- Tokencake Sections 4-5 use DAG state, function-call duration prediction,
  proactive D2H/H2D, and critical-agent reservation. DAG awareness and
  predictive transfer are first-order baselines.
- KVFlow's conservative next-use distance, plus LRU and an offline oracle,
  remain required controls according to
  `research/imported/RELATED_WORK_MATRIX.md`.

C1 is limited to one narrower question: does representing mutually exclusive,
correlated, concurrent, and independent future accesses as explicit joint
outcomes improve the estimate of the first physical re-admission and repeated
reuse of a shared block, especially when several logical claims coalesce into
one physical access epoch?

## Research Question And Hypotheses

**RQ-C1.** Under identical runtime state, predictor information, cache capacity,
and downstream decision rule, does dependence-correct lease aggregation reduce
first-re-admission estimation error and policy regret relative to PBKV-style
additive marginals and an independence-union approximation?

**H0-C1.** C1 has no lower paired error or regret than the strongest matched
baseline on held-out traces and adversarial dependence motifs.

**H1-C1.** C1 lowers paired first-re-admission probability error and downstream
physical-transfer regret on correlated, mutually exclusive, and coalesced
fanout cases while preserving parity on truly independent cases.

H1 is unverified until the M4 split and M5 paired schedule pass. Unit tests can
verify the mathematical implementation and failure boundary only.

## Probability Object

All probabilities use integer parts per million (`PPM=1,000,000`). Floating
point probability inputs are rejected. For block `b`, dependence group `g`
contains mutually exclusive and exhaustive outcomes `s` with mass `p[g,s]`.
Different groups are explicitly declared independent. Unknown dependence must
be represented inside one joint group.

Each outcome contains logical access claims. Claims with the same
`reuse_epoch_id` and timestamp form one physical reuse epoch, so fanout is
deduplicated before physical cost is counted. Let `N[g,s](D)` be the number of
unique epochs through deadline `D`, and let:

```text
q[g](D) = sum_s p[g,s] * 1[N[g,s](D) = 0]
```

Then the nominal statistics are:

```text
P_first(D) = 1 - product_g q[g](D)
E_epochs(D) = sum_g sum_s p[g,s] * N[g,s](D)
E_repeats(D) = E_epochs(D) - P_first(D)
```

`P_first` counts the expected first physical re-admission at most once across
all owners. `E_repeats` separates later unique reuse epochs. These quantities
are sufficient statistics for C1; C2 will attach capacity and DMA prices.

## Drift Boundary

Every dependence group may declare a total-variation radius. For a scalar
outcome statistic, C1 computes the exact lower and upper expectation obtained
by moving at most that probability mass from high to low outcomes or low to
high outcomes. Bounds combine across declared-independent groups. The repeated
reuse interval uses sound interval arithmetic and may be wider than the tight
joint extremum.

The radius is an input contract. M4 must estimate or preregister it using only
the calibration split and report held-out coverage. A radius selected after
viewing formal outcomes invalidates the robust claim.

## Runtime And Information Invariants

1. `LifecycleOrchestrator` creates the policy snapshot while holding its sole
   writer lock.
2. The snapshot contains detached active workflow-retention owners only.
   Request bindings never become forecast ownership capabilities.
3. A forecast binds the exact `BlockKey` and ledger event count. Any later
   lifecycle event makes it stale.
4. Every claim names an active retention binding, its workflow, and an eligible
   nonterminal DAG node from that snapshot.
5. A claim or physical reuse epoch cannot span groups declared independent.
6. Scenario masses are positive, outcomes are exhaustive, and each group sums
   to exactly one million PPM.
7. Repeated claim identities and epoch timestamps must remain identical across
   outcomes.
8. Forecast generation precedes every access and cannot predate its retention
   owner.
9. Oracle forecasts fail closed in online scoring. Offline evaluation must opt
   in explicitly and label the result oracle-only.
10. Aggregation produces no ledger row and mutates no canonical state.
11. A later lease action must use existing owner-qualified runtime methods;
    scoring output never grants a new ownership capability.

## Independent Switches And Baselines

The same forecast and deadline expose four independently selectable signals:

| Mode | Signal | Role |
| --- | --- | --- |
| `c1_nominal` | nominal `P_first` | C1 mechanism |
| `c1_robust_lower` | lower-bound `P_first` | C1 drift ablation |
| `pbkv_style_additive` | sum of logical claim marginals | matched additive-aggregation proxy |
| `independent_marginal_union` | union after assuming every claim independent | dependence ablation |

The proxy isolates PBKV Equation 1's additive aggregation shape while omitting
PBKV's learned predictor, survival term, confidence decay, hierarchy, and
prefetch guardrails. It cannot be labeled a PBKV reproduction. The trace and
GPU matrices must additionally include LRU, Continuum TTL, KVFlow-style
minimum next-use distance, and an offline oracle. Their predictor, capacity,
transfer implementation, request schedule, and random seed must be matched
wherever the mechanism permits. A faithful PBKV result requires the authors'
released implementation or a separately audited reproduction.

## Experiment Plan

### Variables

- Independent variables: aggregation mode, DAG dependence motif, correlation
  strength, fanout, independent workflow count, probability drift, GPU block
  capacity, PCIe load, and predictor quality.
- Primary dependent variables: first-re-admission probability error, paired
  physical H2D count and bytes, and downstream decision regret against oracle.
- Secondary dependent variables: cache hit rate, p50/p95/p99 workflow latency,
  throughput, wasted retention block-time, policy CPU time, and peak metadata.
- Controls: model, prompts, block hashes, trace split, arrival schedule, tool
  durations, GPU driver, engine revision, cache capacity, seeds, and transfer
  instrumentation.

### Workload Motifs

The synthetic correctness matrix must include exclusive OR, correlated fanout,
concurrent fanout with one coalesced epoch, correlated sequential repeats,
independent workflows, AND join, no-future-use, predictor overconfidence,
predictor underconfidence, abrupt drift, and owner cancellation during the
forecast horizon.

The trace matrix will later use frozen train/calibration/formal splits. Fields
must identify workflow instance, DAG node, binding, block, logical claim,
physical reuse epoch, prediction cutoff, outcome, and split provenance. M4
must audit that no formal future state enters the predictor or dependence
group construction.

### Statistical Analysis

- Evaluate probability forecasts with Brier score, calibration error, and
  first-re-admission MAE. Clip an additive baseline to `[0,1]` only for
  probability metrics; retain its raw value for ranking experiments.
- Use identical seeds and arrival schedules for paired policy comparisons.
- Report paired bootstrap 95% confidence intervals for mean deltas and
  workload-level distributions. Report median and tail effects separately.
- Correct families of secondary comparisons with Holm's procedure.
- Report effect sizes and confidence intervals alongside p-values.
- Analyze the complete frozen matrix; no failed case may be silently retried or
  removed. Infrastructure exclusions require a preregistered reason and an
  append-only attempt record.

No numerical success threshold is frozen in v1 because no M3 pilot measurement
has yet established scale. A pilot may size the matrix and instrumentation;
the formal minimum effect, sample count, timeout, and exclusion rules must be
committed before the formal schedule is generated.

## Stage Gates

### C1-A: Mathematical And Runtime Contract

- exhaustive small-distribution checks match exact enumeration;
- exclusive, correlated, independent, fanout, and repeated cases pass;
- TV bounds contain every enumerated distribution inside the radius;
- stale snapshot, released owner, cross-owner, invalid node, probability mass,
  identity drift, and oracle-online cases fail closed;
- aggregation leaves ledger and runtime state unchanged;
- complete M2 regression and aggregate historical replay remain valid.

Production C1-A evidence is created only from a clean committed `main` HEAD by
`tools/run_m3_c1_component_evidence.py`. The runner executes the 16-test C1
suite, the complete repository suite, Ruff lint and format checks, then
temporarily enters the M2 acceptance HEAD at the original repository root and
runs the frozen aggregate validator. It must restore the exact C1 HEAD and a
clean worktree before publication. The bundle includes exact Git blobs, a Git
archive, JUnit identities, commands, environment, raw logs, the M2 acceptance
copy, replay terminal, checksums, read-only modes, and a durable publication
sidecar. Any failed run removes its unpublished staging tree and success
filename.

The first accepted C1-A component bundle is indexed by
`evidence/m3/c1/M3_C1_COMPONENT_EVIDENCE_INDEX.json`. It binds clean source
HEAD `0467694eac84603792bc4fb5455a529297b7b2ab`, manifest schema v2, sixteen
focused C1 tests, the complete 349-test repository regression, Ruff checks,
and the historical aggregate M2 acceptance replay. A fresh process verified
the sealed checksums and durable publication sidecar. This closes C1-A only;
C1-B, C1-C, C1-D, all performance effects, and all paper claims remain open.

### C1-B: Trace And Calibration Gate

- M4 provenance and leakage audits pass;
- predictor and dependence grouping use calibration data only;
- the frozen TV radius achieves its preregistered formal coverage;
- all baselines consume the same eligible information.

### C1-C: Paired Policy Gate

- immutable CPU replay schedule and raw outputs pass reconstruction;
- the formal comparison evaluates all declared motifs and workloads;
- primary paired metrics and confidence intervals are reported regardless of
  direction;
- mechanism overhead and failure-path behavior are included.

### C1-D: Real-GPU Gate

- one frozen driver, engine, model, source, and command bundle is recorded;
- every policy runs the same paired request and tool schedule;
- raw lifecycle, prediction, decision, DMA, latency, and output-correctness
  traces are content addressed;
- independent replay reconstructs every reported aggregate.

C1 remains open until all four gates pass. C2 and C3 stay independently
disabled during the C1 attribution experiment.

## Expected Artifacts

| Artifact | Stage | Acceptance condition |
| --- | --- | --- |
| C1 component JUnit and source manifest | M3 | C1-A passes with full M2 regression |
| versioned trace schema and field audit | M4 | every required field and split hash verifies |
| calibration and formal schedules | M4/M5 | create-only, disjoint, content addressed |
| paired raw policy outputs | M5 | complete schedule, no silent retry |
| reconstructed tables and claim matrix | M6 | independent replay and reviewer falsification pass |

## Interpretation Limits

Passing C1-A establishes implementation correctness for a declared joint
distribution and uncertainty radius. It provides no evidence that a deployed
predictor can learn those distributions, that the independence declarations
hold on real workloads, or that C1 improves serving performance. Those claims
require the later gates above.
