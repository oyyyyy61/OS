# M3/C1-B Trace And Calibration Protocol v4

## Material Passport

- Origin skill: Academic Research Suite `experiment-agent`
- Origin mode: plan
- Origin date: 2026-07-26
- Pre-data amendment date: 2026-07-27
- Verification status: `STRUCTURAL_PROTOCOL_V4_FROZEN`; numerical preregistration open
- Version label: `m3_c1b_trace_calibration_v4`
- Supersedes: v3 at Git commit
  `3b1709d93ad383c7963ed9463d24dbdc57220fd8`

The Git commit that first contains this version is its structural freeze
identity. Numerical thresholds require a separate post-pilot preregistration.
V4 was amended before excluded-pilot or formal C1-B data collection because
the post-C1-B0 implementation audit found that v3 did not close the candidate
universe, deterministic role assignment, executable branch-grammar language,
value-level feature provenance, B1/B2 ordering, or B1 evidence envelope. The
accepted C1-B0 controlled observations are component fixtures and are
permanently ineligible for `PILOT`, `TRAIN`, `CAL_FIT`, `CAL_RADIUS`, or
`FORMAL`. V3 and earlier versions remain immutable historical protocols and
cannot govern new B1 artifacts, pilot traces, or labels.

## Status And Claim Boundary

This protocol freezes the C1-B data, label, split, calibration, and evidence
boundary after C1-A component acceptance. It authorizes implementation and,
only after C1-B1 acceptance, an excluded pilot. It does not accept C1-B,
authorize a formal schedule, or support a probability-quality, policy-benefit,
GPU, latency, throughput, or novelty claim.

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

Schema `dagkv.m3.c1_trace.v3` is an append-only canonical JSONL stream. Every
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
digests without rewriting either stream. Lifecycle-event v2 assigns every row
an exact `batch_id`, zero-based index, and batch size. It records typed binding
state, waiter join/leave, block-state transitions, and complete scheduled plus
terminal transfer history. The sole writer closes the stream with one final
`STREAM_SEAL` row timestamped from its monotonic clock. No event may follow it.
The sidecar closure is derived from that row and binds the complete event count,
last event and batch, closed-through timestamp, and canonical event-stream
digest; a caller cannot supply an independent closure time. The verifier
replays only complete batches, so a prefix ending inside a batch fails closed.

`location_version` is the GPU execution-mapping ABA version. Publishing or
removing a GPU replica, or entering `FREED`, increments it exactly once per
atomic batch. D2H schedule/completion, CPU publication, H2D schedule/failure,
and waiter-only changes leave it unchanged. Residency and the complete
published-replica/inflight-transfer state are recorded separately, so this
field must not be interpreted as a general residency revision.

A separate canonical artifact with schema `dagkv.m3.schedule_sidecar.v1`
binds one trace pair, run, source artifact, schedule, and schedule case before
labels are available. It contains the exogenous demand events, the
preregistered reuse epoch partition, content-addressed closed-time checkpoints,
and a typed replay or sealed-natural-source closure. Event ordinal is
contiguous and events are ordered by `(scheduled_access_ns, event_ordinal)`.
Every event belongs to exactly one epoch; checkpoint event and epoch counts,
last event identity, and
canonical JSONL prefix digests are recomputed from the artifact. The trace
header binds the exact schedule file digest. A watermark additionally binds
one checkpoint ID, its canonical digest, both prefix digests, both counts, and
the producer identity. Copying scalar count/time fields without the referenced
checkpoint cannot establish completeness.

All schedule, cutoff, deadline, checkpoint, capture, and label-availability
timestamps use the exact clock domain `campaign_monotonic_ns`. An unknown or
different clock domain fails schema validation before any interval comparison.

For replay, the sidecar contains the finite sealed producer plan and its closure
conserves the declared event count and event-stream digest; the watermark's
`producer_artifact_digest` equals that plan-event digest. For a natural source,
normalized schedule-event count and raw source EOF record count are different
quantities and cannot be equated. The natural closure separately binds raw EOF
count/digest, capture interval, dropped-record count, and clean EOF state, and
the watermark's producer digest equals the raw EOF digest. A complete natural
observation requires clean EOF, zero dropped records, and a capture interval
covering its cutoff and referenced checkpoint. Every topology source-case
digest in the paired trace must equal the schedule sidecar source case. The
concrete natural-source gate independently reads a stable, singly linked source
artifact, verifies its complete-file EOF digest and record count, and requires
every normalized demand event's source-record digest to occur in that sealed
source. Its `source_artifact_digest` must equal the trace header source digest;
for a natural schedule it must also equal the sealed source EOF digest. A
sidecar that only self-reports these values cannot authorize labels.

The schedule-sidecar v1 concrete gate parses natural closure metadata and
audits source-file integrity, but it does not yet authorize a `COMPLETE`
natural-source label. A
source-schema-specific total normalizer must additionally prove that every
eligible raw source row maps to exactly one schedule event or a typed non-demand
row. One-way proof that each emitted event occurs in the source cannot prove
that an eligible demand was not omitted, especially for a zero-demand label.
Until that total normalizer is frozen and implemented, natural observations
fail closed; the controlled replay path is the C1-B pilot/formal lane.

Trace schema v3 supersedes v2 because every embedded lifecycle prefix now uses
lifecycle-event v2 and must end at an exact atomic batch boundary. Trace v2 and
lifecycle-event v1 files retain their historical identities and are diagnostic
only; they cannot be reinterpreted or admitted to a v3 label gate. Trace v2 had
already superseded foundation-only v1 for exact schedule checkpoint prefixes,
zero-event checkpoint semantics, and separate natural-source EOF fields.

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
the prefix is nonempty, and the final row must close its entire atomic batch.
Detached lifecycle replay must reproduce the inline block state, active
retention owners, and eligible nodes at that boundary. Reconstructing a cutoff
snapshot from a later workflow state is forbidden.

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

The B1 feature boundary uses closed create-only artifacts. A
`field_path_catalog.v1` is generated from the exact recursive trace and sidecar
schemas. A path starts with record type and payload-union variant, uses dotted
dataclass fields, represents an optional absence as typed `ABSENT`, and uses
`[*]` plus a frozen element-identity derivation rule for a sequence field. The
catalog contains schema wildcard paths and rules only; runtime element
identities occur only in `feature_view.v1`. Every schema path/union variant
occurs exactly once. A `feature_contract.v1` binds that catalog and assigns
every path exactly one of the four classifications above.

A `derivation_registry.v1` contains a finite versioned expression language,
not Python or caller-provided executable text. Each derivation binds one closed
opcode, canonical parameters, ordered typed dependency slots, and output type.
The v1 opcode set is exactly `IDENTITY`, `COUNT`, `SUM_INT`, `MIN_INT`,
`MAX_INT`, `SUB_INT`, `BOOL_ALL`, `BOOL_ANY`, `EQUAL`, `CLAMP_INT`, and
`RIGHT_CLOSED_BUCKET_INT`. Arithmetic uses exact unbounded integers;
`IDENTITY` has one dependency; `COUNT` and `SUM_INT` accept zero or more;
`MIN_INT`, `MAX_INT`, `BOOL_ALL`, and `BOOL_ANY` require one or more;
`SUB_INT` and same-typed `EQUAL` require two; `CLAMP_INT` and bucket operations
require one input plus their typed parameters.
`CLAMP_INT(x, low, high)` requires `low <= high`; a right-closed bucket returns
the first sorted edge index whose edge is at least the input, or the edge count
when no edge matches. Empty `MIN_INT`/`MAX_INT`, implicit numeric coercion,
floating-point input, and any unspecified arity fail closed.
The independent verifier evaluates the same expression from source-bound leaf
values and requires the recomputed value and digest to match. Unknown opcodes,
ambient state, file or network reads, undeclared dependencies, type coercion,
or a dependency-count mismatch fail closed.

A `feature_view.v1` records each cutoff-visible leaf by field path, typed value
and digest, source artifact, source record, stable element identity,
`committed_ns`, availability kind, optional `event_ns`, and
`lookback_start_ns`. The feature contract binds a source-schema-specific
availability-rule ID for every online path. The verifier reads the source bytes
and WAL/commit receipt, executes that rule, recomputes each value and timestamp,
and requires `lookback_start_ns = cutoff_ns - feature_lookback_ns`. Both
`IMMUTABLE_STATIC` and `WINDOWED_EVENT` values require
`committed_ns <= cutoff_ns`; only `WINDOWED_EVENT` additionally requires
`event_ns` in `[cutoff_ns - feature_lookback_ns, cutoff_ns]`.
`IMMUTABLE_STATIC` values must bind the immutable version that was active at
the cutoff and may have been committed earlier than the lookback window. The
canonical value identity is the digest of schema version, contract digest,
path, source artifact and record, element identity, typed value digest, and
availability kind.

For `feature_contract.v1`, the closed online set is limited to the nine
windowed `LifecycleEvent` leaves named by its content-addressed availability
rules. Every instance must belong to the exact lifecycle prefix and satisfy the
event-time window above. Schedule events and epochs encode future demand for
the corresponding observation and cannot be online. The cutoff trace row is
constructed only after the feature view has been computed, so it cannot serve
as that view's source artifact. Any later use of immutable cutoff-state leaves
requires a separately committed pre-attempt source artifact, a closed receipt,
and a contract version change; the final cutoff row cannot be reinterpreted as
that source.

Derived feature values name a registry entry and its ordered dependency-value
IDs. They form an acyclic graph whose leaves are source-bound
`ONLINE_ALLOWED` values. The verifier recursively reloads every leaf and
recomputes every transform and output digest. Any transitive label,
provenance-only value, forbidden proxy, missing source, post-cutoff timestamp,
or unreported dependency fails the complete view.
The derived value ID is the canonical digest of the registry digest,
derivation ID, ordered dependency-value IDs, typed output, and output digest.

A `model_input_manifest.v1` binds direct observation IDs by split role,
feature-view digests, an upstream artifact DAG, predictor, outcome catalog,
grouping rules, probability calibrator, primary horizon, and sample-mask
digest for one declared consumer purpose. The direct-row matrix is exact: base
predictor and nominal estimator consume `TRAIN`; probability calibration and
group selection consume `CAL_FIT`; TV radius and uncertainty fallback consume
`CAL_RADIUS`; the frozen validator directly consumes `FORMAL`. The upstream
DAG recursively records every producer role and input artifact: a later stage
may read frozen artifacts from earlier roles, but those artifacts cannot contain
direct rows from an unauthorized role. `PILOT` is excluded from every later
manifest and upstream closure. A role cannot be substituted merely because its
rows are earlier in time. A separate append-only access journal proves that
the formal validator reads the sealed `FORMAL` inputs exactly once; the input
manifest alone makes no read-count claim.

Every fitted or selected artifact is built in a fresh isolated process whose
read-only input mount contains exactly the manifest-closed content-addressed
artifacts, frozen source/import closure, and interpreter environment. A
create-only `artifact_build_receipt.v1` records the canonical import and
file-open closure, direct-row reads, upstream-artifact reads, command, output
digest, and terminal. Access outside that mount, an unrecorded read, a mutable
input, or a transcript mismatch invalidates the artifact. Formal execution
uses the same isolation plus a create-once execution gate and append-only access
journal.

A `baseline_parity.v1` artifact contains all four aggregation modes:
`C1_NOMINAL`, `C1_ROBUST_LOWER`, `PBKV_STYLE_ADDITIVE`, and
`INDEPENDENT_MARGINAL_UNION`. It binds the candidate-universe digest and one
common attempt ledger. Every candidate has exactly one ledger disposition.
The scoreable set is exactly the candidates with `PREDICTED` and `COMPLETE`
attempt terminals. It maps bijectively to one row in every mode; every other
candidate has one shared typed abstained, censored, invalid, or pre-observation
terminal. A structural fixture additionally requires at least one scoreable
positive-demand candidate. Thus four empty or jointly cherry-picked eligible
sets cannot pass. The exact scoreable observation IDs, feature views,
snapshot/forecast inputs, model artifacts, horizon, and sample mask must yield
one identical common-input digest. Only the aggregation-mode field may differ.
The independent leakage report recomputes field coverage, source values and
times, derivation outputs, recursive role legality, and common-input parity
from these artifacts; it never accepts digest equality as a substitute for
reading the bound inputs.

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
event, the exact `WAITER_JOIN` event for this request, bytes, and actual
coalesced waiter set. The join may follow a transfer scheduled before the
request's pre-service prefix, but it must follow the demand intent and preserve
the request binding's original workflow, node, execution, and block lineage.
It describes how demand was served and never changes `Y_i_first` or `N_i`.
The declared demand-epoch partition remains fixed even when the serving
strategy coalesces a different waiter set. V3 contains no physical-attempt
chain: the same `demand_commit_id` permits only an event-free, exact replay of
the already dispatched access batch. A later physical retry is invalid for a
`COMPLETE` observation until a separately frozen attempt-chain schema exists.

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

The independent schedule verifier selects every eligible artifact event in the
half-open/closed primary window `(cutoff_ns, deadline_ns]` and requires an exact
bijection to trace `demand_intent` rows over schedule event, access time, claim,
retention owner, request binding, workflow, node, execution, block, and reuse
epoch identities. The artifact epoch partition must then map bijectively to
trace `reuse_epoch` rows. A reference epoch that only partly intersects one
observation's eligible event set is ambiguous and fails closed. A zero-demand
label is authorized only when this independently reconstructed event and epoch
set is empty under a checkpoint strictly beyond the deadline.

Timestamped owner cancellation and workflow termination are realized outcomes,
not censoring. Missing cancellation, trace truncation, instrumentation failure,
ambiguous epoch identity, or an unfixed schedule is censored. A demand followed
by a failed H2D remains positive demand; the transfer failure is a downstream
outcome. A demand intent emitted after the strategy starts is invalid rather
than recoverable. Censored and unidentifiable observations remain in accounting
under a preregistered, label-value-blind terminal-completeness rule. They remain
in intention-to-trace denominators, are excluded from label-required metrics,
and can never be converted to a negative label.

## Candidate Universe And B1/B2 Ordering

Every B1 fixture or research campaign starts from a finite create-only
`cutoff_plan.v1` and `candidate_universe.v1`, both sealed before any trace is
collected. The cutoff plan enumerates every raw candidate slot and binds its
`candidate_slot_id`, schedule case, canonical block, pre-policy cutoff-trigger
identity, `split_time_ns`, primary horizon duration, feature lookback, and
typed lineage. `split_time_ns` is a label-blind coordinate on a sealed
`temporal_axis.v1`; it is distinct from the realized runtime `cutoff_ns` in the
statistical observation tuple. The trigger identifies the frozen schedule and
cutoff condition whose realization creates that runtime tuple.

The universe purpose is exactly one of `STRUCTURAL_FIXTURE`,
`EXCLUDED_PILOT`, or `POST_PILOT_MAIN`. It binds the complete cutoff plan,
schedule, source, workflow-template, content-lineage, normalizer, eligibility
rule, temporal axis, and method-menu digests. The normalizer maps each cutoff
plan slot exactly once to an eligible candidate or a typed, label-blind
ineligibility record. A candidate ID is the canonical digest of the schema and
purpose, cutoff-plan digest, slot ID, schedule case, block, cutoff-trigger ID,
split time, horizon duration, lookback, and complete typed lineage. It excludes
`run_id`, `runtime_event_count`, realized `cutoff_ns`, labels, and service
outcomes because those values do not exist at pre-data freeze time.

For every admitted C1-B1-or-later trace, `observation_id` equals this frozen
candidate ID. An append-only candidate-attempt journal records dispatch before
each planned trigger and one terminal for any failure before an observation
cutoff is committed. After capture closes, a create-only
`candidate_realization_audit.v1` reads that journal and the sealed traces,
reconstructs each full statistical observation tuple, and verifies that the
declared trigger, schedule case, block, horizon, and lineage match the plan.
Every eligible candidate must map to exactly one realized observation or one
typed pre-observation attempt terminal; it cannot map to both. Every realized
observation maps to exactly one eligible candidate. A missing, duplicate, or
foreign binding, trigger mismatch, or altered block/horizon invalidates the
complete campaign. Historical C1-B0 controlled fixture IDs remain governed by
their accepted v3 bundle and are never upgraded into research candidates.

Each plan slot therefore has exactly one disposition. An unmapped slot,
duplicate candidate, missing or untyped lineage field, ambiguous trigger,
post-service field, or caller-supplied runtime identity invalidates the
complete universe. Rows cannot be added, removed, or reclassified after trace
collection starts.

C1-B1 structurally validates the finite grammar and candidate universe for the
planned excluded pilot without reading labels or service outcomes. It also
validates exhaustive synthetic XOR, AND, shared-latent, terminal, and no-use
motifs. C1-B2 then checks source-case and trace conservation on the excluded
pilot and may select only entry IDs from the B1-frozen method menu plus legal
numerical parameters explicitly left open below. A
pilot mismatch fails C1-B2 and cannot repair the B1 universe or grammar.
Unknown or unbounded workload control flow remains a typed abstention. Thus no
pilot observation contributes to B1 acceptance, and B1 cannot tune a cap,
threshold, grammar rule, or eligibility decision from pilot outcomes.

The B1 evidence bundle uses `STRUCTURAL_FIXTURE`; those rows are never research
observations. The first real universe uses `EXCLUDED_PILOT`, assigns every
eligible candidate to `PILOT`, and records no later-role candidate. After B2
freezes selected menu IDs and numerical choices in a new preregistration
commit, a new
`POST_PILOT_MAIN` universe may contain only `TRAIN`, `CAL_FIT`, `CAL_RADIUS`,
and `FORMAL`. It binds the exact pilot universe and split-manifest digests plus
a separate `predecessor_exclusion_catalog.v1`. That catalog reproduces every
pilot candidate ID and its full lineage incidence from the predecessor; these
are historical exclusions, not rejected slots from the new cutoff plan. A
missing, altered, or reintroduced pilot candidate or lineage invalidates the
main universe. The role algorithm and schemas remain the B1-frozen versions,
while B2 may choose only B1-frozen menu IDs, future interval sizes, and other
explicitly open numerical parameters. No universe is amended in place.

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

Before trace collection, the split manifest binds complete candidate component
membership, interval bounds on the frozen split-time axis, guard gap, schedule
hashes, source digests, template digests, and a deterministic role-assignment
algorithm. A label-blind audit verifies the manifest before any model is
fitted. Actual residence episodes, hits, or coalesced transfers are policy
outcomes and cannot change membership. Any observed cross-role service
dependency or undeclared source edge fails the entire preregistered campaign;
rows cannot be merged, moved, or removed afterward. An immutable raw dataset
file may supply disjoint cases across roles; its artifact digest is recorded as
provenance and does not by itself create an edge.

The v1 role-assignment algorithm is closed and seedless:

1. Sort candidates by canonical candidate ID.
2. For every lineage family, require one canonical field containing an
   applicability tag and a sorted unique tuple of zero or more typed values.
   `PRESENT` requires a nonempty tuple; `ABSENT` and `NOT_APPLICABLE` require an
   empty tuple. Absence tags are audited but never create union edges. A
   candidate may name multiple reference epochs, scheduled tool executions, or
   other values in one family. Serialize every present value separately as one
   canonical incidence row `(lineage_family, lineage_value,
   sorted_member_candidate_ids)`. Union candidates that occur in the same
   incidence row. Pairwise clique/star encodings are forbidden. A raw-dataset
   digest, `BlockKey`, or widely shared system prefix is never a token in the
   primary temporal cohort.
3. Sort each component's candidate IDs and all incidence rows touching those
   candidates. Derive the component ID from the component schema version, that
   exact candidate list, and those exact canonical incidence rows. There is no
   random seed, capacity balancing, or outcome-dependent tie break.
4. Bind half-open `[start_ns, end_ns)` intervals in normative role order. A
   structural fixture binds all five roles, an excluded-pilot universe binds
   only `PILOT`, and a post-pilot main universe binds the four later roles. A
   candidate's role is determined only by its frozen `split_time_ns`, which
   must occur in exactly one interval.
5. Every member of one connected component must resolve to the same role. A
   component that spans intervals invalidates the complete candidate campaign.
6. Derive `max_primary_horizon_duration_ns` and
   `max_feature_lookback_ns` from the closed candidate universe. The gap between
   adjacent intervals must be at least their sum. Equality is valid; a one-ns
   deficit fails.

The template-generalization cohort adds workflow-template digest as a lineage
token. The content-isolated cohort instead adds declared content-lineage-family
digest. Each cohort has a separate candidate universe, split manifest, and
result; their rows cannot be pooled. Replaying a manifest recomputes every
token, union, component ID, interval assignment, maximum, and guard gap rather
than trusting serialized component or role fields.

A `POST_PILOT_MAIN` split additionally performs one predecessor-union audit.
The pilot and main universes must bind the same `temporal_axis.v1` digest. The
verifier joins their full candidate and incidence catalogs, rejects every
candidate-ID reuse and every connected component containing `PILOT` plus a
later role, and recomputes source-case overlap. It also verifies the boundary
from the pilot interval to the first `TRAIN` interval using the combined
maximum horizon duration plus combined maximum feature lookback. An
incomparable temporal axis, a one-ns gap deficit, or any shared prohibited
lineage fails the complete main campaign before labels are read. The
predecessor exclusion catalog is checked against this recomputed union and
cannot substitute for it.

## Branch And Outcome Grammar

`WorkflowSpec` proves precedence and acyclicity only. It contains no XOR/AND
branch semantics, path feasibility constraints, shared latent choices, or
reference-epoch rules, so DAG topology alone cannot prove complete joint
support.

Every workflow template therefore binds a create-only branch grammar artifact.
The grammar names branch variables, finite domains, XOR/AND constraints,
shared-random-source identities, feasible path predicates, terminal/no-use
outcomes, and deterministic mappings from template-local demand-site IDs to
template-local reference-epoch-slot IDs. A separate create-only
`grammar_instance_binding.v1` binds one schedule case to its exact assignment
ID and outcome ID. Its active-site and active-epoch-slot domains must equal the
outcome exactly. Active sites map bijectively to concrete scheduled-demand
events, and active epoch slots map bijectively to concrete reference epochs;
each active epoch has at least one mapped active site. Every other grammar site
or epoch slot occurs once in a typed inactive list and has no concrete ID. A
`NO_USE` binding therefore has empty active maps and complete inactive lists.
This keeps one template grammar reusable without placing schedule-instance
identities inside it.

B1 validates the grammar by exhaustive enumeration on synthetic motifs and
conservation of each binding's exact active domain against every pre-label
candidate schedule; B2 separately audits source-case conservation on
excluded-pilot traces. Unknown or unbounded control flow produces typed
abstention. No observed pilot or formal path may expand or repair the grammar.

Schema `branch_grammar.v1` is a finite truth-table language, not executable
source text. Variables and their nonempty finite domains are sorted and unique;
choosing one value from a variable is its XOR operation. A variable may name
one shared-random-source identity. Each table rule is a disjunction of sorted
clauses, each clause is a conjunction of typed terms, and each term restricts
one variable to a nonempty subset of its domain. A clause contains at most one
term per variable; an omitted variable is an all-domain wildcard. Term subsets
and clauses use canonical sort order. This finite DNF supplies the AND and
path-feasibility semantics without evaluating caller-provided code.

The verifier enumerates the full Cartesian product in canonical
variable/domain order. Every assignment must match exactly one
`(rule_id, clause_id)` pair; overlap between two clauses of the same rule also
fails. A rule is either `FEASIBLE`, with one frozen outcome, or `INFEASIBLE`,
with one typed reason and no outcome. Every feasible outcome records terminal
nodes, active template-local demand-site IDs, and a total many-to-one function
from each active demand site to one active predeclared epoch slot. Every active
site appears exactly once in the function domain, and every active epoch slot
has at least one site. A feasible `NO_USE` outcome with empty site and
epoch sets is explicit. Missing, overlapping, duplicate, or extra assignments;
an unmapped or multiply mapped site; an unknown or empty active epoch; and an
outcome whose terminal/site/epoch semantics change across matching assignments
all fail closed. Assignment IDs and the derived feasible-support catalog are
canonical digests, not caller labels.

The verifier has a versioned implementation safety ceiling only to bound
parser and enumeration resource use. It is not the scientific joint-support
cap, which remains open for the excluded pilot. Exceeding the safety ceiling,
unknown variables, or unbounded domains returns a typed structural abstention
and creates no support catalog. Such abstention is an accepted negative fixture
only. If the planned excluded-pilot universe has zero eligible candidates, or
if any of its grammars or instance bindings lacks a complete support catalog,
pilot readiness is `C1_B1_PLANNED_PILOT_NO_GO`, B1 stage acceptance is
forbidden, and B2 cannot start. A formal or pilot trace may match only the
already frozen catalog; an unseen outcome is `SUPPORT_VIOLATION`, never a new
rule or post-hoc abstention.

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

B1 also freezes a finite `method_menu.v1`. Each menu entry binds an
implementation or declarative specification digest, version, permitted
parameter names and domains, and compatible artifact schemas. The menu covers
every nonnumeric pilot-selectable procedure: main-schedule generators,
optional cohort definitions, epoch/time-bucket construction, grouping models,
independence-proof rules, PPM rounding, probability calibrators, predictor
feature-subset recipes over the already frozen `ONLINE_ALLOWED` paths and
derivations, calibration-cell construction, pooling, sampling-confidence
methods, OOD/abstention evaluation, go/no-go evaluation, and
censoring/exclusion/crash/zero-retry accounting. The pilot may choose an entry
and legal numeric parameters; it cannot introduce a new algorithm, field
classification, derivation, dependency, or schema. B3 reruns the complete B1
structural verifier on the selected main artifacts before any formal label is
unsealed.

Exactly one method-menu artifact and digest govern B1. The structural-fixture
lane must exercise every menu entry with at least one declared valid fixture
and its frozen invalid-domain cases. The planned-pilot universe, its split and
grammar artifacts, the B2 selection, and every post-pilot main artifact must
bind that same digest. A missing entry fixture, second menu digest, or menu
drift produces no B1 acceptance.

The excluded pilot may size instrumentation and choose the following items.
Every selected value must be committed before a formal schedule is generated:

- primary horizon duration, cutoff cadence, feature lookback, guard gap, and a
  frozen-menu epoch/time-bucket entry with legal numeric boundaries;
- role interval sizes, a frozen-menu cohort entry, and a frozen-menu schedule
  generator entry;
- frozen-menu grouping, independence-rule, and PPM-rounding entries, support
  cap, and pseudocount;
- frozen-menu probability-calibrator and predictor feature-subset entries;
- frozen-menu calibration-cell, pooling, and sampling-confidence entries,
  minimum cells/samples, and radius-selection quantile;
- frozen-menu OOD/abstention evaluator, maximum TV radius, sharpness limits,
  unseen-mass/OOD thresholds, and maximum abstention;
- frozen-menu go/no-go evaluator, dependent-motif minimum improvement, and
  independent-motif equivalence margin;
- formal component/cell count, timeout, and a frozen-menu
  censoring/exclusion/crash/zero-retry accounting entry.

Apart from the inferential error rates above, no numerical success threshold is
frozen in this v4 protocol. Selecting any listed value from `FORMAL`, reusing
pilot observations, or using the same rows for `CAL_FIT` and `CAL_RADIUS`
invalidates the C1-B formal claim.

## Expected Artifacts

| Artifact | Required evidence |
| --- | --- |
| trace schema and manifest | exact schema, implementation/source hashes, closed file inventory, canonical JSONL replay |
| lifecycle, topology, and schedule sidecars | complete event conservation, parent graph, topology digest, producer watermark, cutoff/WAL audit |
| demand reconstruction audit | pre-policy intent binding, declared reuse epochs, resident hits, H2D-separated outcomes, censor accounting |
| cutoff plan and candidate universe | complete eligible/ineligible slot normalization, candidate/observation binding, typed pre-policy lineage, closed source and schedule set |
| split manifest | canonical lineage incidence, connected components, role intervals, predecessor-union guard audit, zero prohibited overlap |
| feature and leakage report | complete path catalog, field classification, source-value replay, derivation registry recomputation, recursive role legality, common-input hashes |
| branch grammar, instance bindings, outcome catalog, and model manifests | exact clause enumeration, complete support, case conservation, grouping and independence provenance, direct and upstream role IDs |
| B1 method menu | finite implementation/specification identities and closed parameter domains for every pilot-selectable structural method |
| B1 structural bundle | fixed inventory, independent artifact replay, protocol/verifier/implementation/environment anchors, final seal and external index |
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

- the structural-fixture and planned excluded-pilot candidate universes
  conserve every eligible or typed-ineligible cutoff-plan slot without
  reading labels or service outcomes;
- connected-component roles and temporal guard gaps replay exactly;
- source cases have zero prohibited overlap and service outcomes never alter
  split membership;
- finite truth-table branch grammar enumeration and instance binding prove the
  planned pilot's frozen feasible support; typed structural abstention is legal
  only for declared negative fixtures;
- feature field coverage, value availability, derivation lineage, and role use
  have zero future-state or label leakage;
- every aggregation baseline has the same common input digest;
- a clean committed CPU-only bundle independently replays all B1 artifacts and
  publishes an external final-seal index with status `C1_B1_STAGE_ACCEPTED`.

The B1 verdict envelope has three orthogonal closed fields. `component_status`
is `C1_B1_SPLIT_GRAMMAR_LEAKAGE_COMPONENT_VERIFIED` or
`C1_B1_COMPONENT_FAILED`; `pilot_readiness` is
`C1_B1_PLANNED_PILOT_READY`, `C1_B1_PLANNED_PILOT_NO_GO`, or
`C1_B1_PLANNED_PILOT_NOT_EVALUATED`; and `stage_status` is
`C1_B1_STAGE_ACCEPTED` or `C1_B1_STAGE_NOT_ACCEPTED`. The transition table is
exact:

| Fixture component | Planned pilot | Stage | Publication |
| --- | --- | --- | --- |
| failed | not evaluated | not accepted | no index; failed attempt root is never resumed |
| verified | zero eligible candidate or typed grammar/support no-go | not accepted | sealed no-go diagnostic index after both replays |
| verified | ready with complete support and both replays passed | accepted | accepted stage index |

Every other combination fails envelope parsing. A no-go is a valid negative
readiness verdict, not a component failure or stage acceptance. Its tracked
index is `evidence/m3/c1/M3_C1_B1_NO_GO_EVIDENCE_INDEX.json`; it binds the
sealed external root and exact typed no-go reasons but cannot authorize B2. The
accepted tracked index is
`evidence/m3/c1/M3_C1_B1_STAGE_EVIDENCE_INDEX.json`.

The fixed B1 inventory contains
all bytes required for reconstruction. The structural-fixture lane includes
raw source cases, sealed schedules, cutoff plans, normalizer and eligibility
rules, temporal axis, candidate universes, split manifests, grammars and
instance bindings, derived catalogs, trace and lifecycle bytes,
candidate-realization audit, candidate attempt journal, source commit and
availability receipts, field-path catalog, feature contract, availability
rules, derivation registry, feature views, model-input and artifact-build
receipts, baseline-parity manifests, common attempt ledger, access journal, and
the shared method-menu entry fixtures. The planned-pilot lane includes
the corresponding raw source/schedule bytes, cutoff plan, normalizer and
eligibility-rule bytes, universe, split manifest, grammars and instance
bindings, field-path catalog, feature contract, availability-rule bytes,
derivation registry, and the same method-menu digest; it contains no pilot
feature view, fitted model, baseline result, label, or service outcome. The
shared method-menu bytes occur once in the sealed inventory and both lane
manifests bind them. Audit reports are derived outputs and cannot replace these
inputs.

A one-shot attempt marker binds the initial hashes; a manifest and final seal
bind the closed inventory. The production launcher must run from clean
committed `main`, freeze exact focused and full testcase identities, and record
their zero-failure/zero-skip terminals plus Ruff and compile results. It
computes Git HEAD/tree/branch/clean status and protocol, verifier,
implementation, environment, and imported-module anchors outside the inner
producer, reads the final seal independently, and runs a fresh validator
process. The outer schema freezes the canonical sorted anchor-path set, Git
blob lookup rule, transitive import closure, external-index schema, and
create-only publication order. This establishes fresh-process,
state-independent replay with one frozen verifier implementation; it makes no
claim of algorithmic independence unless a separately anchored second verifier
also reproduces the verdict.
Only after both replays succeed may a tracked external index publish
`C1_B1_STAGE_ACCEPTED` or the scoped no-go verdict under the transition table.
Every B2-B4, policy, GPU, and performance gate remains open. An indeterminate
or failed attempt publishes neither index and its root is never resumed.

### C1-B2: Excluded Pilot

- pilot observations are complete and permanently excluded;
- instrumentation loss, censoring, unknown support, and abstention are reported;
- every pilot-selected B1 menu-entry ID and legal numerical parameter is frozen
  in a new preregistration commit.

### C1-B3: Calibration Freeze

- the selected main universe, predecessor-union audit, split, grammar,
  feature subset, and method-menu identities pass the complete B1 structural
  verifier again before formal labels are available;
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
- V4 B1 implementation: `src/dagkv/c1_split.py`,
  `src/dagkv/c1_grammar.py`, `src/dagkv/c1_features.py`, and
  `src/dagkv/c1_b1.py` with closed canonical artifacts and independent replay;
  no predictor fitting or serving-policy mutation.
- B1 tests: candidate conservation, transitive components, role boundaries,
  guard gaps, service-outcome invariance, finite grammar truth tables,
  terminal/no-use and epoch conservation, field coverage, value-level cutoff
  availability, transitive derivation leakage, role isolation, baseline parity,
  create-only storage, tamper rejection, and fresh-process bundle replay.
- Initial success criterion: all C1-B0/B1 component tests plus the complete
  repository regression pass; no probability or performance claim.
- Pilot entry command and runtime root remain open until C1-B0/B1 evidence is
  sealed from a clean committed source snapshot.

### Implementation Checkpoint: 2026-07-27

The first C1-B foundation slice now implements the closed trace record model,
canonical JSONL parser, structural state machine, content-addressed lifecycle
and schedule evidence-gate interfaces, atomic cutoff view, and pre-service
demand gate. A stable `demand_commit_id` distinguishes an idempotent replay of
one scheduled access call from a new resident or H2D demand. Replay must
preserve the original timestamp, action, GPU target, and waiter identities and
cannot emit a second service event. The adapter must pass only the new logical
consumers introduced by the first invocation. Physical retry chains remain
outside v3.

This checkpoint is not C1-B0 or C1-B1 acceptance. The concrete lifecycle
sidecar verifier, schedule replay verifier, branch-grammar artifact validator,
segmented create-only commit chain and final seal, split-component builder, and
feature-leakage audit remain open gates. A typed adapter that constructs the
operation-specific canonical record batch, commits it through the durable
writer, and returns the writer-issued receipt also remains an open integration
gate; a synthetic protocol receipt cannot support a durability claim. The
current single-file writer is a component implementation: any process or I/O
failure invalidates the entire stream, which must never be resumed or used for
a formal label.

The second foundation slice implements the closed schedule sidecar, canonical
create-only writer/loader, replay and natural-source closure types, exact
checkpoint-prefix binding, and an independent demand-intent/reuse-epoch
bijection verifier. Replay labels may pass this gate. Natural closure/source
integrity is audited, while natural labels remain closed pending a total source
normalizer.

The third foundation slice upgrades the lifecycle event contract to v2 and the
containing trace contract to v3. It records exact atomic batch coordinates,
typed binding transitions, H2D waiter join/leave after-sets, transfer-terminal
waiter provenance, complete transfer history, and block boundary snapshots. A
sole-writer `STREAM_SEAL` supplies the trusted closure timestamp. The canonical
lifecycle sidecar uses create-only write/fsync/readback, stable single-link
loading, detached ledger replay, cutoff/pre-service prefix checks, and exact
resident/H2D/cancellation service reconstruction before issuing a lifecycle
receipt. Lifecycle-event v1 and trace v2 cannot issue that receipt.

This checkpoint still does not accept C1-B0. The operation-typed durable trace
committer, segmented bundle-level evidence seal, branch-grammar validator, and
split/leakage gates remain open. The bundle-level seal is distinct from the
implemented lifecycle stream seal. No pilot or formal probability result has
been collected under v3.

### Fourth Foundation Slice: Typed Commit And Component Bundle

The operation-typed canonical committer now owns record construction and the
create-only JSONL writer. It commits cutoff plus forecast attempt atomically,
derives demand intents from the frozen schedule before service, closes each
observation in one terminal batch, returns exact writer-issued receipts, and
rejects legacy trace endpoints in formal runtime mode. Requests, runtime views,
the schedule, and the trace envelope are canonical detached snapshots. Any
possibly durable response loss, post-commit application failure, receipt
mutation, lifecycle-seal ambiguity, or sealed-file drift poisons the complete
attempt.

The controlled-replay C1-B0 component bundle has a fixed schedule, lifecycle,
and trace segment chain; a durable one-shot attempt marker bound to the initial
payload hashes; typed operation boundaries; a manifest; and an externally
anchored final seal. Fresh validation rereads the exact read-only inventory,
recomputes byte and sequence boundaries with a linear hash chain, replays the
concrete lifecycle and schedule gates, and batch-reconstructs labels. Its only
success status is `C1_B0_SCHEMA_RECONSTRUCTION_COMPONENT_VERIFIED`, with C1-B1
through C1-D listed as open gates. Natural-schedule evidence is not admitted by
this component bundle.

The final-seal, protocol, verifier, implementation, and environment digests are
trust roots only when an independent launcher computes them. The external
evidence index must publish the final-seal digest only after the finalizer
returns successfully; an indeterminate finalization publishes no anchor. The
v3 demand-operation `runtime_view_digest` remains a trusted-writer assertion
because the full pre-service view is not duplicated in the trace. Demand labels
are still independently authorized from the frozen schedule, demand-intent
rows, and lifecycle prefixes and service terminals.

This slice does not yet accept the C1-B0 stage. Acceptance still requires a
fresh bundle from the clean committed implementation and an external evidence
index. Branch grammar and split/leakage remain C1-B1 gates; no excluded pilot,
calibration artifact, formal probability result, GPU result, or performance
claim has been produced.

### Fifth Pre-Data Checkpoint: B0 Acceptance And B1 Closure

After the historical fourth-slice checkpoint, the clean-source C1-B0 launcher
published a sealed CPU-only artifact from commit
`158eab9ef5b9a75e5677281fbdf22f32dd82547e`. Its exact 192-case focused set,
575-case repository regression, source and environment anchors, inner final
seal, and independent outer replay are indexed in
`evidence/m3/c1/M3_C1_B0_STAGE_EVIDENCE_INDEX.json`. This closes C1-B0 for
schema/reconstruction correctness only.

The subsequent pre-data B1 audit found the v3 structural descriptions
insufficient to reconstruct a unique candidate universe, role assignment,
branch predicate evaluator, value-level leakage proof, or evidence envelope.
V4 closes those choices before any excluded-pilot observation exists. The next
implementation slice is a controlled structural component; it cannot fit a
predictor, choose a scientific support cap, inspect pilot outcomes, or claim
calibration, policy, GPU, performance, or novelty results.

The first three V4 structural slices implement deterministic candidate/split
artifacts, exhaustive finite branch support with exact schedule-case binding,
and schema-regenerated field-path plus feature-contract artifacts. The feature
catalog expands only legal trace payload and sidecar union variants, binds
sequence-element identity rules, and is regenerated during validation so a
self-consistent omission fails. The contract classifies every path exactly
once, freezes the complete four-class profile by digest, and conserves the
online allowlist, content-addressed availability rules, and path assignments
exactly. The conservative v1 boundary exposes nine windowed lifecycle-prefix
leaves and excludes future schedule data plus the post-attempt cutoff row.
Source-byte value/time replay, a pre-attempt source for any later static cutoff
state, finite derivations, recursive role legality, baseline parity, B1 verdict
and envelope construction, and clean-source evidence remain open.
