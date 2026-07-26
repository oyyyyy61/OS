# M3/C1-B Trace And Calibration Protocol v1

## Material Passport

- Origin skill: Academic Research Suite `experiment-agent`
- Origin mode: plan
- Origin date: 2026-07-26
- Verification status: `STRUCTURAL_PROTOCOL_FROZEN`; numerical preregistration open
- Version label: `m3_c1b_trace_calibration_v1`

The Git commit that first contains this version is its structural freeze
identity. Numerical thresholds require a separate post-pilot preregistration.

## Status And Claim Boundary

This protocol freezes the C1-B data, label, split, calibration, and evidence
boundary after C1-A component acceptance. It authorizes implementation and an
excluded pilot. It does not accept C1-B, authorize a formal schedule, or
support a probability-quality, policy-benefit, GPU, latency, throughput, or
novelty claim.

C1-B must establish that a dependence-aware forecast can be constructed and
evaluated without future information, pseudoreplication, policy-mediated
labels, incomplete outcome support, or post-hoc uncertainty tuning. C1-C and
C1-D remain closed until the C1-B formal gate passes.

## First-Order Literature Boundary

The exact local PDFs and SHA-256 identities are frozen in
`research/REFERENCES.md`.

- PBKV Section 4.2.2 and Equations 1-2 define the additive shared-node
  aggregation boundary. Its offline invocation trace, global call graph,
  observed workflow prefix, multi-step distribution, survival term, horizon,
  and physical radix-node association motivate fields and a matched additive
  baseline. PBKV does not establish calibrated joint branch probabilities or
  coalesced physical reuse epochs.
- Continuum Sections 3-5 motivate timestamped history availability, sample
  count, cold-start provenance, tool-duration observations, expiry, and queue
  outcomes. Its per-tool TTL is prior art and is not a C1 reproduction.
- Tokencake Sections 4-5 motivate an event-driven timeline containing DAG
  topology, call start and finish, predicted and actual tool duration,
  transfers, bytes, and residency. DAG-aware predictive transfer remains prior
  art.

Across these three audited papers, no evaluated artifact supplies one trace
contract that jointly binds topology, canonical block identity, prediction
cutoff, joint outcome support, dependence provenance, logical claims, physical
reuse epochs, resident hits, transfers, split provenance, and realized
outcomes. C1-B may claim an auditable artifact only after its own gates pass. It
cannot attribute this contract or its calibration guarantees to those papers.

## Estimand And Observation Unit

One statistical observation is the immutable tuple:

```text
(run_id, schedule_case_id, canonical BlockKey,
 runtime_event_count, cutoff_ns, primary_horizon_duration_ns)
```

All owners, claims, dependence groups, deadlines, and realized epochs nested
under that tuple are repeated measurements or labels. They are never counted
as independent samples.

For observation `i`, let `H_i > 0` be its primary horizon duration and let
`D_i = cutoff_ns_i + H_i` be its absolute primary deadline. Let
`N_i(D_i)` be the number of distinct reference demand epochs after the cutoff
and through `D_i`. The reference epoch partition is fixed by the exogenous
schedule before a serving strategy runs. The primary policy-invariant labels
are:

```text
Y_i_first(D_i) = 1[N_i(D_i) > 0]
Y_i_epochs(D_i) = N_i(D_i)
Y_i_repeats(D_i) = max(0, N_i(D_i) - Y_i_first(D_i))
```

`Y_i_first` answers the counterfactual demand question: under the frozen
exogenous request schedule, if the canonical block were absent from GPU
immediately after the cutoff, would a first physical reuse demand occur by the
horizon? A strategy under evaluation cannot delay, suppress, add, or regroup
those scheduled demand intents. An execution that consumes a resident block is
still positive demand. Actual H2D is affected by retention, eviction, prefetch,
and single-flight decisions, so H2D count and bytes are separate C1-C/C1-D
policy outcomes and are forbidden as C1-B demand labels.

A demand intent is emitted by the workload adapter immediately before cache
serving logic. It binds a frozen schedule event, one active cutoff retention
owner, one request binding, an `ExecutionRef`, one canonical block, and one
preregistered reuse epoch. A reuse epoch contains one or more such intents for
the same block. Multiple logical consumers may share one epoch only when the
schedule manifest declared that equivalence before policy execution. Timestamp
equality, transfer identity, or JSONL adjacency alone cannot invent
coalescing. Completed `EXEC_MAP` and H2D terminals are service-outcome evidence,
not the source of the demand label. A trace that observes only cache misses,
H2D, or completed maps and omits pre-policy demand intent is `UNIDENTIFIABLE`
for C1-B.

Only one horizon-duration/deadline pair may be the formal primary endpoint.
Additional deadlines from one forecast are reported as correlated repeated
measures and cannot increase the sample count.

## Trace Record Contract

Schema `dagkv.m3.c1_trace.v1` is an append-only canonical JSONL stream. Every
row has an exact closed field set containing `schema_version`, `record_type`,
`trace_id`, `run_id`, `schedule_id`, `schedule_case_id`, `sequence`,
`record_id`, `parent_record_id`, and a type-specific payload. Observation rows
also contain one immutable `observation_id`. Duplicate keys, non-finite
numbers, blank identities, unknown fields, duplicate IDs, symlinks, invalid
UTF-8, and non-canonical ordering fail closed.

The required record types are:

| Type | Required role |
| --- | --- |
| `trace_header` | binds source, schedule, split manifest, branch grammar, feature contract, implementation, and environment digests before observation rows |
| `workflow_topology` | records immutable `WorkflowSpec`, branch-grammar digest, workflow template digest, source case digest, and connected split component before node execution |
| `cutoff` | records the complete detached `SharedLeasePolicySnapshot`, cutoff timestamp, horizon duration, derived absolute deadline, lifecycle event-count watermark, last event identity, residency, active retention owners, and eligible nodes |
| `forecast_attempt` | closed union: `PREDICTED` records the full `SharedLeaseForecast` and its digests; `ABSTAINED` records a typed reason while preserving the same common feature-view, model, artifact, and information-cutoff identities |
| `demand_intent` | records the pre-policy schedule event, canonical block, declared reuse epoch, owner-qualified request consumer, and pre-service lifecycle watermark |
| `reuse_epoch` | closes one preregistered reference demand epoch over exact demand-intent IDs and records their later service terminals plus optional actual H2D/coalescing provenance |
| `schedule_watermark` | conditionally required for `COMPLETE`; records the authoritative producer, frozen schedule digest, consumed event count, last schedule-event identity, and maximum closed absolute timestamp |
| `observation_terminal` | records the last verified completeness watermark, label-availability time, `COMPLETE`/`CENSORED`/`UNIDENTIFIABLE`/`INVALID` status, and a frozen reason |

The topology row exists because workflow registration currently creates no
ledger event. A separate sidecar with schema `dagkv.m3.lifecycle_sidecar.v1`
serializes every `LifecycleEvent` in ledger-sequence order, including nested
`WorkflowKey`, `BlockKey`, `ReplicaId`, and `ExecutionRef`, without replacing
canonical block identity with a physical slot. Both headers bind one
`trace_pair_id`; the create-only evidence manifest binds their final file
digests without rewriting either stream.

The observation state machine is:

```text
TOPOLOGY_BOUND -> CUTOFF_COMMITTED -> FORECAST_ATTEMPT_COMMITTED
               -> zero or more DEMAND_INTENT rows
               -> zero or more REUSE_EPOCH rows
               -> SCHEDULE_WATERMARK_COMMITTED -> COMPLETE_TERMINAL
               -> CENSORED | UNIDENTIFIABLE | INVALID TERMINAL
```

Only `COMPLETE_TERMINAL` requires and may consume labels. Every other terminal
records its nullable last verified watermark and remains in attempt accounting
without inventing missing provenance.

The cutoff and forecast-attempt records must be durably committed before any
demand intent referenced by their label. Each intent must itself be committed
before the serving strategy runs. An abstention creates no
`SharedLeaseForecast`, still reaches a terminal record, and remains in every
attempt-level denominator. Formal labels remain unavailable until the
predictor, outcome catalog, grouping rules, feature contract, TV radius, and
schedule are frozen and content addressed.

## Cutoff And Feature Availability

The cutoff event count is the exclusive upper bound of the allowed lifecycle
prefix. Its last event ID and timestamp must match the canonical ledger when
the prefix is nonempty. The inline snapshot is authoritative; reconstructing a
cutoff snapshot from a later workflow state is forbidden.

`SharedLeaseForecast.generated_ns` must equal `cutoff_ns`, and its absolute
`horizon_ns` must equal `primary_deadline_ns`. The snapshot, lifecycle prefix,
cutoff row, feature view, and forecast attempt are captured through one
orchestrator-held cutoff/WAL boundary. The boundary verifies the live event
count again and durably flushes all rows before releasing the sole-writer lock.
Independent snapshot, model, and file-write calls do not satisfy this contract.
A corresponding adapter gate durably flushes each demand intent before calling
`ensure_h2d` or any equivalent serving operation.

Allowed online features must be named in a create-only feature contract. Each
forecast stores one common feature-view digest. At minimum, the contract
classifies every trace field as `ONLINE_ALLOWED`, `LABEL_ONLY`,
`PROVENANCE_ONLY`, or `FORBIDDEN_PROXY`.

The following are never online features for the corresponding observation:

- post-cutoff node terminals, owner cancellation, workflow terminal, and
  actual tool duration;
- realized outcome, demand intent, reuse epoch, future `EXEC_MAP`, H2D terminal,
  hit/miss, or label-availability state;
- future ledger sequence, physical slot generation, transfer job identity,
  output-file position, or collection wall time used as a future-state proxy;
- pilot, cal-radius, or formal labels used to train the base predictor;
- formal observations used to select support, grouping, probability
  calibration, radius, threshold, sample mask, or fallback.

`ForecastSource.PREDICTED` alone does not prove this boundary. Every model and
forecast must bind the exact input observation IDs and upstream artifact
digests.

## Demand Reconstruction And Censoring

An observed logical access binds a cutoff retention owner and a request-side
`BIND` event with the same workflow, eligible node, `ExecutionRef`, and
canonical block. The demand-intent row carries exact `schedule_event_id`,
`claim_id`, `retention_binding_id`, `request_binding_id`, `ExecutionRef`, and
`reuse_epoch_id` values. It is committed with the exact pre-service event-count
watermark before `ensure_h2d` or an equivalent adapter operation runs. An epoch
contains one or more such intents. Each demand intent belongs to exactly one
epoch per observation.

Service provenance binds exactly one of resident `EXEC_MAP`, successful H2D
plus `EXEC_MAP`, failed/cancelled H2D, or request cancellation after the intent.
Optional H2D provenance records the `LOAD` or `PREFETCH` transfer, terminal
event, bytes, and actual coalesced waiter set. It describes how demand was
served and never changes `Y_i_first` or `N_i`. The declared demand-epoch
partition remains fixed even when the serving strategy coalesces a different
waiter set.

A proactive `PREFETCH` without an eligible request waiter emits no demand
intent and cannot create a positive label. A later scheduled consumer emits its
own intent even when that prefetch made the block resident.

A complete no-demand label is legal only when all of these hold:

1. a schedule-authoritative watermark exceeds the absolute primary deadline;
2. lifecycle, demand-intent, owner, reuse-epoch, and block-mapping streams cover
   the whole `(cutoff, deadline]` interval;
3. the cutoff owner and eligible-node set replays exactly;
4. no crash, truncation, dropped event, ambiguous owner mapping, unresolved
   epoch, or policy-induced schedule change prevents reconstruction.

The schedule watermark names its producer and binds the complete frozen
schedule. A replay scheduler emits it only after consuming every declared event
through its closed timestamp. A sealed natural-trace source uses its immutable
EOF, record count, and digest as the producer terminal. A lifecycle event count
or a quiet wall-clock interval cannot substitute for this watermark.

Timestamped owner cancellation and workflow termination are realized outcomes,
not censoring. Missing cancellation, trace truncation, instrumentation failure,
ambiguous epoch identity, or an unfixed schedule is censored. A demand followed
by a failed H2D remains positive demand; the transfer failure is a downstream
outcome. A demand intent emitted after the strategy starts is invalid rather
than recoverable. Censored and unidentifiable observations remain in accounting
under a preregistered, label-value-blind terminal-completeness rule. They remain
in intention-to-trace denominators, are excluded from label-required metrics,
and can never be converted to a negative label.

## Split Components And Temporal Isolation

The roles are strictly ordered:

```text
PILOT -> TRAIN -> CAL_FIT -> CAL_RADIUS -> FORMAL
```

Pilot observations are permanently ineligible for every later role. The base
predictor and nominal categorical estimator use `TRAIN`. Probability
calibration and grouping selection use `CAL_FIT`. Only `CAL_RADIUS` chooses the
TV radius and any uncertainty fallback. `FORMAL` is read once by the frozen
validator.

The split unit is a connected component. Add an undirected edge whenever two
candidate observations share any pre-policy source lineage:

- workflow instance, session, source case, or scheduled tool-execution identity;
- reference demand epoch or schedule component;
- shared random-draw lineage or shared derived example.

An entire connected component receives one role. Block-, node-, claim-, row-,
or deadline-level random splitting is forbidden. Time intervals are ordered by
role and separated by a guard gap at least
`max_primary_horizon_duration_ns + max_feature_lookback_ns`. A component
crossing a boundary invalidates the candidate campaign before labels are
revealed.

The primary temporal cohort may contain later instances of a known workflow
template. A separate template-generalization cohort assigns an entire template
to one role. Results from those cohorts are reported separately. Source-case
overlap across roles is always rejected. A widely shared system prefix or
canonical `BlockKey` is the object being studied and does not by itself create a
statistical edge. A separate content-isolated cohort assigns each declared
content-lineage family wholly to one role.

Before trace collection, the split manifest binds complete component
membership, interval bounds, guard gap, schedule hashes, source digests,
template digests, and a deterministic role-assignment algorithm. A label-blind
audit verifies the manifest before any model is fitted. Actual residence
episodes, hits, or coalesced transfers are policy outcomes and cannot change
membership. Any observed cross-role service dependency or undeclared source
edge fails the entire preregistered campaign; rows cannot be merged, moved, or
removed afterward. An immutable raw dataset file may supply disjoint cases
across roles; its artifact digest is recorded as provenance and does not by
itself create an edge.

## Branch And Outcome Grammar

`WorkflowSpec` proves precedence and acyclicity only. It contains no XOR/AND
branch semantics, path feasibility constraints, shared latent choices, or
reference-epoch rules, so DAG topology alone cannot prove complete joint
support.

Every workflow template therefore binds a create-only branch grammar artifact.
The grammar names branch variables, finite domains, XOR/AND constraints,
shared-random-source identities, feasible path predicates, terminal/no-use
outcomes, and the deterministic mapping from scheduled demand events to
reference epochs. The pilot validates the grammar by exhaustive enumeration on
synthetic motifs and source-case conservation on trace workloads. Unknown or
unbounded control flow produces typed abstention. No observed formal path may
expand or repair the grammar.

## Joint Outcome And Dependence Estimation

1. Candidate claims come only from active cutoff owners and eligible nodes.
2. Outcome grouping uses cutoff-visible DAG structure, the frozen branch
   grammar, and frozen provenance.
   Dependence is never inferred from failure to reject independence.
3. Unproven independence merges claims into one joint group. If the complete
   support then exceeds the frozen cap, the predictor abstains.
4. Every group enumerates all grammar-feasible outcomes, including
   no-use and feasible outcomes absent from training. Support is fixed before
   `CAL_RADIUS` and cannot expand after a formal label is seen.
5. Epoch partitions, time buckets, primary horizon duration/deadline, and
   outcome vocabulary are content addressed. An `OTHER` bucket is legal only
   when all members have the same first-use, unique-epoch, and repeat statistics
   at every evaluated deadline.
6. Nominal categorical masses are fitted on `TRAIN`, use a frozen positive
   pseudocount, and convert deterministically to positive PPM with
   largest-remainder rounding that sums exactly to `1,000,000`.
7. `CAL_FIT` may select a preregistered probability calibrator and grouping
   candidate. It cannot tune the TV radius.
8. Missing provenance, unsupported support, OOD context, insufficient data, or
   an invalid catalog returns a typed abstention and creates no
   `SharedLeaseForecast`.

TV mass movement cannot cover an outcome omitted from the support. A new formal
outcome outside the frozen catalog is `SUPPORT_VIOLATION`, fails C1-B, and
cannot be repaired by post-hoc support expansion or relabeling as abstention.

## TV Radius Calibration

One categorical realization cannot prove that its unknown conditional
distribution lies in a TV ball. `TV(p, delta_y) = 1 - p_y` is an outcome-level
conformity score only. C1-B therefore estimates and evaluates chronological
holdout coverage on repeated cells.

A calibration stratum is fixed by label-free fields such as outcome-catalog
digest, motif, fanout/support-size bin, predictor-output bin, and primary
horizon duration. A cell instance is `(stratum_id, nonoverlapping_time_block)`.
Every forecast group maps to exactly one cell instance. Within a cell, repeated
groups from one split component are averaged first, so each component receives
equal weight. For each `CAL_RADIUS` cell, the radius derivation records:

- the mean nominal categorical distribution;
- the empirical categorical distribution;
- their TV distance;
- a simultaneous multinomial sampling upper bound;
- within-cell predictor heterogeneity;
- PPM quantization error; and
- the final conservative cell score.

Each split component receives one calibration score equal to the maximum score
of every cell instance it contributes to. A frozen upper quantile over these
component scores selects one global radius, rounding upward to integer PPM.
The excluded pilot chooses the quantile and time-block construction; each
component is counted once in radius selection. A formal component is covered
only when every eligible cell score does not exceed that radius. Formal reports
include component coverage, cell-macro coverage, observation-micro coverage,
every mandatory motif, insufficient-evidence cells, and one-sided confidence
bounds clustered by split component.

The roles are time ordered and drift is an explicit stressor, so this protocol
makes no exchangeability-based split-conformal or distribution-free coverage
claim. Its claim is limited to preregistered chronological holdout coverage on
the sealed formal cohort. An optional exchangeable synthetic lane with known
generating probabilities is reported separately.

Coverage alone is vacuous at radius one. The formal preregistration also fixes
a maximum radius, median and p95 interval-width limits, and a minimum
non-degenerate `c1_robust_lower` fraction. Rare cells may use only a frozen,
semantically compatible pooling hierarchy. Low cell count, excessive unseen
mass, or OOD catalogs abstain and remain in coverage accounting.

Natural traces support empirical chronological-cohort coverage and
cluster-aware confidence intervals. General population, per-snapshot
conditional, or future-drift coverage requires independent replication or a
separately preregistered sequential method.

## Metrics And Information-Parity Baselines

Primary forecast metrics are first-demand Brier score, first-demand MAE,
group-level multiclass Brier score, and TV coverage plus sharpness. When one
forecast contains multiple declared-independent groups, compute the multiclass
Brier score inside each group and average groups with weight `1 / group_count`
to form one observation-level score. Cartesian-product expansion and
claim-level weighting are forbidden. ECE is secondary and uses frozen bin
edges. Unique-epoch and repeat errors are repeated-measure secondary endpoints.

The aggregation-only comparison derives these signals from the same serialized
forecast and common feature-view digest:

- `c1_nominal`;
- `c1_robust_lower`;
- `pbkv_style_additive`; and
- `independent_marginal_union`.

The additive score is clipped to `[0,1]` only for probability metrics and stays
raw for ranking. All modes share snapshot, candidate claims, joint masses,
cutoff, horizon duration, absolute deadline, sample mask, and model artifacts.
Oracle rows are a separate offline upper bound. A faithful PBKV, Continuum, or
KVFlow end-to-end baseline requires a separately audited implementation and is
secondary to the aggregation-only attribution test.

Metrics are reported on the common eligible set and with method-specific
coverage and abstention. Claim-level rows cannot inflate `n`. Cluster bootstrap
or another frozen cluster-aware interval operates on split components.
Prediction-quality metrics condition on a valid forecast and report that
denominator. Attempt-level reports additionally count every abstention,
censored terminal, and invalid attempt; selective coverage cannot hide them.

## Pilot And Formal Freeze Boundary

This structural protocol freezes split component as the inferential unit,
paired cluster-bootstrap two-sided 95% confidence intervals for mean method
deltas, a cluster-aware one-sided 95% lower confidence bound for coverage, and
Holm family-wise correction at `alpha=0.05` for secondary comparison families.
The excluded pilot cannot change those error rates or promote a secondary
endpoint.

The excluded pilot may size instrumentation and choose the following items.
Every selected value must be committed before a formal schedule is generated:

- primary horizon duration, cutoff cadence, epoch/time buckets, feature
  lookback, and guard gap;
- role interval sizes, template/content-isolated cohort inclusion, and schedule
  algorithm;
- grouping model, independence proof rule, support cap, pseudocount, and PPM
  rounding tie-break;
- probability calibrator and feature contract;
- calibration-cell definition, minimum cells/samples, pooling hierarchy,
  radius-selection quantile, and sampling-confidence method;
- maximum TV radius, sharpness limits, unseen-mass/OOD thresholds, and maximum
  abstention;
- dependent-motif minimum improvement, independent-motif equivalence margin,
  and go/no-go rule;
- formal component/cell count, timeout, censoring, exclusion, crash, and
  zero-retry rules.

Apart from the inferential error rates above, no numerical success threshold is
frozen in this v1 protocol. Selecting any listed value from `FORMAL`, reusing
pilot observations, or using the same rows for `CAL_FIT` and `CAL_RADIUS`
invalidates the C1-B formal claim.

## Expected Artifacts

| Artifact | Required evidence |
| --- | --- |
| trace schema and manifest | exact schema, implementation/source hashes, closed file inventory, canonical JSONL replay |
| lifecycle, topology, and schedule sidecars | complete event conservation, parent graph, topology digest, producer watermark, cutoff/WAL audit |
| demand reconstruction audit | pre-policy intent binding, declared reuse epochs, resident hits, H2D-separated outcomes, censor accounting |
| split manifest | connected components, role intervals, guard-gap audit, zero prohibited overlap |
| feature and leakage report | field classification, common input hashes, zero future-state access |
| branch grammar, outcome catalog, and model manifests | grammar enumeration, complete support, grouping and independence provenance, training and cal-fit IDs |
| TV calibration artifact | cal-radius cell counts, scores, bound terms, quantile index, upward PPM rounding |
| baseline parity manifest | identical eligible rows and common feature-view digest for aggregation modes |
| formal preregistration and schedule | create-only seal, frozen thresholds, ordered cases, zero retry |
| formal report | predictions, labels, support/abstention ledger, coverage, sharpness, Brier/MAE, paired intervals |
| attempt journal and evidence seal | append-only terminals, exact hashes, independent reconstruction |

## Stage Gates

### C1-B0: Schema And Reconstruction

- strict parsers and canonical serialization pass round-trip and tamper tests;
- topology, lifecycle, cutoff, forecast-attempt, demand-intent, epoch,
  schedule-watermark, and terminal rows form a closed state machine;
- cutoff and demand gates prove durable commit before serving, and abstention
  reaches terminal accounting without a `SharedLeaseForecast`;
- resident reuse demand and actual H2D are separated;
- incomplete no-use windows fail closed.

### C1-B1: Split And Leakage

- connected-component roles and temporal guard gaps replay exactly;
- source cases have zero prohibited overlap and service outcomes never alter
  split membership;
- branch grammar enumeration proves the frozen feasible support or abstains;
- feature availability has zero future-state or label leakage;
- every aggregation baseline has the same common input digest.

### C1-B2: Excluded Pilot

- pilot observations are complete and permanently excluded;
- instrumentation loss, censoring, unknown support, and abstention are reported;
- pilot-selected thresholds are frozen in a new preregistration commit.

### C1-B3: Calibration Freeze

- train, cal-fit, and cal-radius inputs are disjoint and complete;
- support, grouping, probability calibration, cell rules, and TV radius are
  create-only and independently replayed;
- formal labels remain sealed when the schedule and model artifacts freeze.

### C1-B4: Formal Coverage

- zero provenance, leakage, and formal support violations;
- frozen coverage, sharpness, OOD, and abstention gates pass;
- the complete formal matrix is reported without silent retry or removal;
- a fresh process independently reconstructs all metrics and the evidence seal.

C1-B acceptance establishes a trace and chronological-cohort calibration
contract.
It does not establish lower transfer count, lower latency, higher throughput,
or GPU benefit. A valid negative aggregation result remains publishable
evidence and stops progression to expensive C1-C unless the frozen go/no-go
rule passes.

## Initial Implementation Plan

- Language/framework: Python 3.12 standard library plus the existing test
  environment; no GPU dependency for C1-B0/B1.
- Working directory: `/home/data/25_oyzx/cagent-work/dagkv`.
- First implementation: `src/dagkv/c1_trace.py` with strict identity models,
  canonical serialization, branch grammar, ledger-bound demand reconstruction,
  schedule watermarks, and split validation; plus an orchestrator-held cutoff
  commit hook and a pre-service adapter commit gate.
- First tests: `tests/test_c1_trace.py` covering round-trip, parent/state
  closure, pre-policy intent ordering, resident hit, H2D failure separation,
  fanout coalescing, censoring, split overlap, cutoff leakage, and baseline
  parity.
- Initial success criterion: all C1-B0/B1 component tests plus the complete
  repository regression pass; no probability or performance claim.
- Pilot entry command and runtime root remain open until C1-B0/B1 evidence is
  sealed from a clean committed source snapshot.
