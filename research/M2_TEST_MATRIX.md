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

| Gate | Direct component or pilot evidence | Remaining gate-closing evidence |
| --- | --- | --- |
| Canonical identities | `test_domain.py`, repository import-boundary AST check, the diagnostic contract, and v3 run10 live engine field binding of GPU allocation generations and exact CPU allocation records | repeat across the 59 calibration and 20 holdout processes |
| DAG dependency gate | diamond fanout failure, pending-child bind rejection, orphan-waiter fast-path rejection, ledger node-lifecycle enforcement, mapped-node terminal rejection, and terminal-during-DMA waiter filtering | live workflow adapter |
| Shared-owner isolation | two-workflow D2H/H2D lifecycle and cross-owner release; the current vLLM CPU manager test proves that shared loads receive independent owner bindings | live engine requests |
| Idempotency | duplicate release, lease terminal, transfer terminal, and early-expiry rejection; the diagnostic adapter contract proves shutdown drain and repeated-shutdown idempotency; current vLLM scheduler/worker tests cover identical duplicate rank reports, late rank reports, repeated reset, and one terminal | live callback fault injection |
| Historical identity | released `ExecutionRef` ABA rejection, unique mapping activations, ledger ID replay, and current vLLM strict terminal tombstones retained across run IDs | live delayed-callback injection |
| Generation safety | failed-target generation consumption and slot checks; the diagnostic adapter contract rejects unallocated GPU generations and missing CPU allocation records; current vLLM manager tests prove failed-store generation advance and capacity-reuse close-before-open | live allocator ABA injection |
| Transfer metadata integrity | component tests cover exact terminal bytes/digests, failed-target no-publish, cleanup, payload framing, padding exclusion, and adapter-side digest/byte mismatch rejection; current vLLM scheduler/worker tests reject missing bytes, cross-rank byte disagreement, malformed ranks, conflicting duplicates, and unverifiable reset drains; v3 run10 independently replayed four exact `2,359,296`-byte DMA terminal pairs | 59 new v3 calibration traces and all 20 v3 formal traces under the frozen 580.173.02 fingerprint |
| GPU re-admission path | runtime and independent-ledger rejection of direct CPU-only publication, plus successful H2D retry; current vLLM scheduler tests defer async preempt/re-admit until transfer output; historical v2 run08 independently replayed fresh GPU target identities and CPU generations 1 then 2 | live adapter API isolation and fresh v3 cohort evidence |
| H2D coalescing | concurrent eight-waiter single-flight component test | live multi-request batch; policy-scale partial-prefix coalescing remains M3/C3 |
| Conservation | runtime/ledger reconciliation, cross-family reference replay, event-envelope/parent tamper rejection, dangling-binding rejection, and quiescent audits | quiescent replay of every candidate v3 calibration and holdout trace before acceptance |
| No-DMA prefix control | v3 run10 observed 16 local GPU-hit tokens and zero DMA in G, while B1/B2 each used 16 external H2D-loaded tokens; the phases produced exact `G=B1=B2` raw logits and measured `A1-G max_abs=0.109375` | repeat in 59 new v3 calibrations and 20 frozen v3 holdouts |
| Output correctness | v3 run10 decoded token 932 in all phases with exact within-path repeats and `max_abs=0.109375`; every 151,936-entry NPY vector passed independent replay | re-establish the complete contract in 59 new v3 calibrations and 20 v3 holdouts under the 580.173.02 bundle |
| Provenance | v3 run10 binds the create-only 580.173.02 bundle, mapped `libcuda`, model/runtime hashes, vLLM snapshot, and audited 239-to-250 runtime-import dependency boundary; SHA256SUMS and independent replay passed | prepare the campaign, commit only `CALIBRATION_V3_LAUNCH_MARKER.json` as its direct child, hold the campaign lock, bind one HEAD/snapshot and runtime identity across 59 runs, freeze a new tolerance, then repeat across all 20 formal holdouts |
| Item-8 aggregate acceptance | open; the v3 calibration remains 0/59 and v3 formal holdouts remain 0/20; the old v2 tolerance is ineligible as a v3 parent | 59/59 calibration, new create-only tolerance, 20/20 holdout-pass manifests, closed-set audit, and one aggregate `M2_ITEM8_ACCEPTANCE_MANIFEST.json` |

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
real-engine v3 pilot and cannot replace the required 59-run/20-run CUDA cohorts.
