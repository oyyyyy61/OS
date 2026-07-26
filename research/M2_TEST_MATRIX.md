# M2 Component Test Matrix

Status: living pre-acceptance index for the v3 evidence protocol, 2026-07-26.

The three CPU suites passed as one create-only, read-only bundle bound to clean
DAGKV commit `67e69ffb3159c66805018fe78b13a7b29f011adf`. External and raw-only
independent replay both passed. The manifest and checksum identities are indexed
in `evidence/m2/M2_COMPONENT_EVIDENCE_INDEX.json`; this directly supports M2
items 1-7 and leaves item 8 and aggregate M2 open.

The exact excluded v3 `run09` pilot command, environment, input hashes, and
no-retry policy remain frozen in
`evidence/m2/M2_V3_PILOT_PREREGISTRATION.json`. The attempt completed all ABBA
computations and then failed the exact loader-environment postflight after an
OpenCV import mutation. Its immutable failure hashes and diagnostic-only
observations are indexed in
`evidence/m2/M2_V3_RUN09_FAILURE_EVIDENCE_INDEX.json`. The corrected, separately
preregistered v3 `run10` completed and passed independent raw replay under the
580.173.02 bundle. Its immutable evidence is indexed in
`evidence/m2/M2_V3_RUN10_PILOT_EVIDENCE_INDEX.json`; it remains excluded from
both cohorts.

The v3 calibration campaign then completed 59/59 fresh processes with zero
retries under one execution binding, implementation manifest, runtime
fingerprint, DAGKV/vLLM snapshot pair, and NVIDIA bundle. The 120-record bundle
passed independent replay and produced a create-only v3 tolerance. The exact
hashes are indexed in
`evidence/m2/v3_580_173_02/M2_CALIBRATION_EVIDENCE_INDEX.json`.

Formal campaign01 completed 20/20 run computations and its aggregator exited
zero, then the launcher rejected candidate replay because its prefix helper
mistakenly required the 41-record candidate journal to contain only the 40-run
prefix. The append-only terminal is `validation_failed`, no bundle seal exists,
and the entire campaign remains ineligible. Its hashes and the exact restart
boundary are indexed in
`evidence/m2/v3_580_173_02/M2_FORMAL_CAMPAIGN01_FAILURE_EVIDENCE_INDEX.json`.

| Gate | Direct component or pilot evidence | Remaining gate-closing evidence |
| --- | --- | --- |
| Canonical identities | `test_domain.py`, repository import-boundary AST check, the diagnostic contract, v3 run10, and 59 fresh calibration processes bound live GPU allocation generations and exact CPU allocation records | repeat across the 20 holdout processes |
| DAG dependency gate | diamond fanout failure, pending-child bind rejection, orphan-waiter fast-path rejection, ledger node-lifecycle enforcement, mapped-node terminal rejection, and terminal-during-DMA waiter filtering | live workflow adapter |
| Shared-owner isolation | two-workflow D2H/H2D lifecycle and cross-owner release; the current vLLM CPU manager test proves that shared loads receive independent owner bindings | live engine requests |
| Idempotency | duplicate release, lease terminal, transfer terminal, and early-expiry rejection; the diagnostic adapter contract proves shutdown drain and repeated-shutdown idempotency; current vLLM scheduler/worker tests cover identical duplicate rank reports, late rank reports, repeated reset, and one terminal | live callback fault injection |
| Historical identity | released `ExecutionRef` ABA rejection, unique mapping activations, ledger ID replay, and current vLLM strict terminal tombstones retained across run IDs | live delayed-callback injection |
| Generation safety | failed-target generation consumption and slot checks; the diagnostic adapter contract rejects unallocated GPU generations and missing CPU allocation records; current vLLM manager tests prove failed-store generation advance and capacity-reuse close-before-open | live allocator ABA injection |
| Transfer metadata integrity | component tests cover exact terminal bytes/digests, failed-target no-publish, cleanup, payload framing, padding exclusion, and adapter-side digest/byte mismatch rejection; current vLLM scheduler/worker tests reject missing bytes, cross-rank byte disagreement, malformed ranks, conflicting duplicates, and unverifiable reset drains; v3 run10 and all 59 calibration processes independently replayed four exact `2,359,296`-byte DMA terminal pairs | all 20 v3 formal traces under the frozen 580.173.02 fingerprint |
| GPU re-admission path | runtime and independent-ledger rejection of direct CPU-only publication, plus successful H2D retry; current vLLM scheduler tests defer async preempt/re-admit until transfer output; historical v2 run08 independently replayed fresh GPU target identities and CPU generations 1 then 2 | live adapter API isolation and fresh v3 cohort evidence |
| H2D coalescing | concurrent eight-waiter single-flight component test | live multi-request batch; policy-scale partial-prefix coalescing remains M3/C3 |
| Conservation | runtime/ledger reconciliation, cross-family reference replay, event-envelope/parent tamper rejection, dangling-binding rejection, quiescent audits, and complete replay of all 59 v3 calibration traces | quiescent replay of every formal holdout trace before acceptance |
| No-DMA prefix control | v3 run10 and all 59 calibration processes observed 16 local GPU-hit tokens and zero DMA in G, while B1/B2 each used 16 external H2D-loaded tokens; every process produced exact `G=B1=B2` raw logits and measured `A1-G max_abs=0.109375` | repeat in 20 frozen v3 holdouts |
| Output correctness | all 59 v3 calibration processes decoded token 932 in every phase with exact within-path repeats and `max_abs=0.109375`; every raw NPY vector passed bundle replay | re-establish the complete contract in 20 v3 holdouts under the 580.173.02 bundle |
| Provenance | the 59-run v3 bundle binds one execution HEAD, DAGKV/vLLM snapshot pair, implementation manifest, runtime fingerprint, mapped `libcuda`, audited 239-to-250 dependency boundary, and NVIDIA 580.173.02 bundle; complete replay passed and the v3 tolerance is frozen | preregister the formal campaign, commit only its launch marker as a direct child, then repeat the frozen identity across all 20 formal holdouts |
| Item-8 aggregate acceptance | open; the v3 calibration is 59/59 with a frozen tolerance and v3 formal holdouts remain 0/20; the old v2 tolerance is ineligible as a v3 parent | 20/20 holdout-pass manifests, closed-set audit, and one aggregate `M2_ITEM8_ACCEPTANCE_MANIFEST.json` |

All indexed pilot attempts and environment smoke checks are excluded from both
v3 cohorts. B1 and B2 are
state-isolated repeats inside one engine process; they are not independent
calibration runs. A single formal execution can produce only a holdout-pass
record and cannot close item 8. Passing this matrix establishes component
behavior only; `STAGE_GATES.md` remains the acceptance authority.

The root development suite and the diagnostic adapter contract are separate
because the root environment deliberately has no PyTorch or vLLM dependency.
The adapter and frozen-fork CPU commands are recorded in `README.md`. The
current snapshot passed 288 root tests, 13 diagnostic-adapter tests, and 345
current-vLLM CPU tests. These CPU results did not replace the successful
real-engine v3 pilot and cannot replace the required 20-run CUDA formal cohort.
