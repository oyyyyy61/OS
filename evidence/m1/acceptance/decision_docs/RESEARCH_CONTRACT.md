# Research Contract: Stateful PBKV for Dynamic Agent DAGs

Status: M0 frozen and M1 passed for its declared scope, 2026-07-24. The
intended scope is a focused systems paper suitable for a B-tier venue. Venue
selection remains open until the mechanism and hardware gates are measured.

## Decision

The original C1 and C2 claims are superseded. PBKV already predicts multi-step
agent invocations, aggregates reuse probability across active workflows at a
shared radix node, retires cache at workflow completion, ranks active cache for
eviction, and performs capacity/bandwidth-gated proactive prefetch. KVFlow and
TokenCake independently occupy much of the deterministic DAG and predictive
prefetch space.

The revised paper studies a narrower systems question:

> How should a two-tier KV manager value and schedule a shared block when one
> first reuse triggers re-admission, later reuses see a different cache state,
> multiple DAG branches may execute concurrently, and GPU capacity, DMA
> bandwidth, and workflow deadlines are all binding?

The primary baseline is the algorithm in PBKV equations (1) and (2), followed
by its retired-first eviction and conservative prefetch budget. Until an
official implementation is available and behaviorally aligned, the local
baseline is named **PBKV-formula**.

Primary PBKV source: `../../2605.06472v1.pdf`, SHA-256
`eb2cc2a7750f972a228519376ad27214cc4d674686c2e7209207c8d47b0da301`.
Relevant boundaries are Section 4.2 (score), Section 4.3 (prefetch), Section 5
(limitations), and Appendix G (eviction-only analysis).

## Core Observation

PBKV defines a node score proportional to a discounted expected count of future
accesses. Appendix G proves this is an expected miss count under an
**eviction-only** abstraction in which an evicted node is not re-admitted during
the K-step horizon. Its deployed hierarchy has a different transition: a host
hit reloads the node into GPU memory. One H2D operation can therefore serve
later accesses until another eviction. Under that state machine, the first
reuse probability, repeated reuse value, and subsequent eviction probability
have different costs.

PBKV also models each active workflow as invoking exactly one agent at a step.
This excludes parallel fanout in which several successors can request
overlapping blocks and share one in-flight transfer. Its evaluation reports
top-1 prediction accuracy, while calibration, deadline misses, duplicate DMA,
and transfer cancellation/failure are outside the reported contract.

These observations are hypotheses about an opportunity. They do not establish
that a new controller improves end-to-end performance.

## Formal Model

At decision epoch `t`, each canonical block `b` has:

- immutable content identity and allocation generation;
- size `s_b`, GPU/CPU/absent residency, and any in-flight transfer;
- a set of workflow owners and lease terminal conditions;
- future demand events `E_b`, each with a block set, time distribution,
  deadline, dependency relation, and joint-probability group;
- profiled H2D, D2H, and recompute costs, including queue and exposed latency.

Let `N_b(H)` be the number of demands for `b` within horizon `H`. Define

```text
p_first(b, H) = Pr[N_b(H) >= 1]
r_repeat(b, H) = E[max(N_b(H) - 1, 0)]
```

`p_first` prices the first re-admission or recomputation. `r_repeat` is valued
only through the post-admission state and the probability of another eviction;
it cannot be charged another full load unconditionally. For mutually exclusive
branches, the union uses a sum over disjoint events. For concurrent or
correlated branches, the controller uses a declared joint model or conservative
Frechet bounds. Independent workflows may use a product complement. Events
sharing one physical transfer are coalesced before transfer value is counted.

The online decision selects `retain`, `save`, `load`, `release`, or `prefetch`
for eligible blocks. The declared optimization target is expected exposed
critical-path latency plus a deadline-miss penalty, subject to hard GPU and DMA
budgets:

```text
min_x E[critical_path_delay(x)] + kappa * E[deadline_misses(x)]
s.t.  sum_b s_b * gpu_resident_b(t) <= G(t)
      sum_{jobs in window t} bytes(job) <= D(t)
```

Any quantity called a shadow price must arise from the Lagrangian. The initial
controller will use projected dual updates:

```text
lambda_G(t+1) = [lambda_G(t) + eta_G * (gpu_used(t) - G(t))]_+
lambda_D(t+1) = [lambda_D(t) + eta_D * (dma_used(t) - D(t))]_+
```

Step sizes, update windows, projections, admission reserve, and maximum budget
violation must be frozen before confirmatory runs. A tuned weighted sum without
this derivation is named a heuristic.

## Candidate Contributions

### C1: Re-admission-aware shared-block value

Separate first reuse from repeated reuse, represent parallel/correlated DAG
demand, and deduct coalesced transfer cost once. Establish a counterexample in
which PBKV-formula ranks two blocks incorrectly under the measured two-tier
state machine, then measure ranking regret against a trace oracle.

### C2: GPU/DMA-priced online lifecycle control

Use the stateful value inside a constrained online controller. Report budget
violation, convergence/stability, decision time, metadata footprint, and
sensitivity to update period in addition to cache hit rate and latency.

### C3: Deadline-aware partial-prefix single-flight

Transfer the maximal compatible block prefix that can finish before a consumer
deadline, attach all compatible waiters to one physical job, and define partial
completion, timeout, cancellation, primary failure, retry, and starvation
behavior. Existing exact-signature local vLLM coalescing is the implementation
baseline.

### C4: Auditable artifact

Provide a canonical lifecycle ledger, reproducible run-order manifests, a
PBKV-formula baseline, adversarial correlation/re-admission fixtures, and real
workflow replay. This remains an artifact contribution unless a field-level
public-data audit justifies a dataset claim.

## Falsifiable Hypotheses

| ID | Hypothesis | Primary evidence | Failure interpretation |
|---|---|---|---|
| H1 | PBKV-formula has nonzero ranking regret when repeated demand shares one re-admission or one coalesced transfer. | Exhaustive small-state oracle plus controlled fanout/retry traces. | If no realistic rank inversion appears, C1 is demoted to an analysis note. |
| H2 | Stateful value reduces oracle decision regret and duplicate DMA bytes relative to PBKV-formula under fanout/retry pressure. | Paired randomized replay; regret and physical bytes from the event ledger. | Lower regret without end-to-end benefit indicates a weak or hidden mechanism. |
| H3 | Dual control respects declared GPU/DMA limits and improves primary JCT without increasing deadline misses. | Paired live runs, price trajectory, budget violation, JCT and miss rate. | Persistent violation invalidates the shadow-price claim; no latency effect limits the work to policy analysis. |
| H4 | Partial-prefix single-flight lowers exposed load time and duplicate bytes when consumers overlap. | Fault-injected component test and live concurrent replay. | A hit-rate-only change cannot support C3. |

## Baselines

All paper-facing comparisons use one vLLM commit, model/tokenizer, trace,
connector, block size, dtype, and hardware constraint.

1. GPU-only/no offload.
2. LRU and LFU.
3. Belady trace oracle, labeled as offline.
4. Current vLLM on-demand exact-prefix loading.
5. Local exact-signature prefetch/single-flight.
6. Continuum-style cost TTL.
7. TokenCake temporal policy and the frozen negative spatial variant.
8. KVFlow-style deterministic steps-to-execution.
9. PBKV-formula, with a line-by-line formula and state-machine conformance test.
10. C1, C2, C3, pairwise combinations, and the full system.

InferCept naming is reserved for a faithful implementation. Otherwise the
baseline is called paused-request swap.

## Frozen Outcomes and Statistics

Primary end-to-end outcome: paired workflow JCT log ratio in a predeclared
high-pressure dynamic fanout regime. Confirmatory success requires a 95% CI
strictly above 1.0 and at least 5% geometric-mean improvement over
PBKV-formula. A low-pressure guardrail forbids more than 2% JCT regression.

Secondary outcomes: TTFT, TPOT, deadline miss, throughput, peak GPU KV, exposed
load time, duplicate physical DMA bytes, prefetch timeliness, coalescing ratio,
cache hit rate, and decision regret. Mechanism claims require directionally
consistent end-to-end and mechanism outcomes.

- Design: paired randomized for multi-policy matrices; ABBA or BAAB for a
  two-policy confirmatory comparison.
- Unit of inference: independent workload seed/trace cluster. Reusing a seed
  across regimes does not increase the independent sample count.
- Effect: lower-is-better geometric ratio `control/candidate`, computed from
  paired log effects. Absolute paired deltas are also reported.
- CI: two-sided 95% Student-t interval over seed-cluster means. At least five
  paired seed clusters are required; more are added until the precision gate or
  declared resource limit is reached.
- Outliers: no deletion. Crashes, OOM, timeout, failed guardrails, and ledger
  violations remain in the run table with a reason; valid reruns receive new
  run IDs.
- Control gate: repeated-control relative 95% CI half-width must be at most 5%
  in every primary regime.
- Multiplicity: H1-H4 each has its own mechanism endpoint; only H3/JCT is the
  primary performance claim. Other comparisons are exploratory unless the
  contract is amended before their data are collected.

## B-Tier Scope

The minimum defensible package is one production serving substrate, one 8B
model on RTX 4090 for mechanism development, two real workflow families, one
controlled adversarial suite, and complete paired evidence. Claims are limited
to single-GPU behavior if no data-center GPU is obtained. Cross-hardware and
multi-GPU generality require at least one data-center GPU class and a genuine
multi-GPU experiment.

The predictor may initially use calibrated trace probabilities or an oracle
input so that lifecycle-control value is isolated. Training a new neural
predictor becomes paper scope only after the controller passes H1/H2. This keeps
the optimization paper centered on the systems gap left by PBKV.

## Stage Gates

- **M0:** provenance, related-work matrix, this contract, and claim ledger are
  frozen; PBKV and adjacent semantic-lifecycle work are first-order boundaries.
- **M1:** canonical event conservation passes; run order is precomputed; at
  least five independent paired seeds are available; control precision is at
  most 5%; all raw files and source state are hashed.
- **M2:** one canonical block/lease/workflow schema and live orchestrator pass
  ownership, release, use-after-free, isolation, and output checks.
- **M3:** H1 precedes C1 implementation; C1 precedes C2; C3 has explicit fault
  semantics; each component has independent switches and mechanism coverage.
- **M4-M6:** real trace audit, preregistered matrix, immutable aggregation, and
  claim-to-evidence review complete.

Current gate: M0 has a recoverable source/dependency freeze and a revised claim
boundary. M1 lifecycle conservation, precomputed order, 12 independent paired
seed clusters, repeated-control precision, raw-data identity, artifact hashing,
and post-M0 source reconstruction pass for the declared single-process,
single-GPU, primary-CPU-tier scope. The read-only source freeze is
`experiments/source_freezes/m1_lifecycle_v2_20260724_001249`, with top-level
`SHA256SUMS` digest
`b2937a6c6e9164f3b82aa9078d8584ba12b5a96a81c44aca9f4738e4093f3166`.
M2 canonical-runtime implementation is open. Performance tuning and proposed
policy claims remain gated by M2 correctness and the ordered M3 mechanism
tests.
