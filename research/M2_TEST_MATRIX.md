# M2 Component Test Matrix

Status: living pre-acceptance index for the v2 evidence protocol, 2026-07-25.

| Gate | Direct component or pilot evidence | Remaining gate-closing evidence |
| --- | --- | --- |
| Canonical identities | `test_domain.py`, repository import-boundary AST check | adapter field audit |
| DAG dependency gate | diamond fanout failure, pending-child bind rejection, orphan-waiter fast-path rejection, ledger node-lifecycle enforcement, mapped-node terminal rejection, and terminal-during-DMA waiter filtering | live workflow adapter |
| Shared-owner isolation | two-workflow D2H/H2D lifecycle and cross-owner release | live engine requests |
| Idempotency | duplicate release, lease terminal, transfer terminal, and early-expiry rejection | callback fault injection |
| Historical identity | released `ExecutionRef` ABA rejection, unique mapping activations, ledger ID replay | live delayed-callback injection |
| Generation safety | failed-target generation consumption and slot checks | live allocator ABA injection |
| Transfer metadata integrity | component tests cover exact terminal bytes/digests, failed-target no-publish, and cleanup; run08 and all 59 calibration processes independently replayed four exact `2,359,296`-byte DMA terminal pairs; the validator reconciles diagnostic and native ABBA identities, generations, bytes, and digests | all 20 formal holdout traces under the frozen calibration fingerprint |
| GPU re-admission path | runtime and independent-ledger rejection of direct CPU-only publication, plus successful H2D retry; run08 independently replayed fresh GPU target identities and CPU generations 1 then 2 | live adapter API isolation and v2 cohort evidence |
| H2D coalescing | concurrent eight-waiter single-flight component test | live multi-request batch; policy-scale partial-prefix coalescing remains M3/C3 |
| Conservation | runtime/ledger reconciliation, cross-family reference replay, event-envelope/parent tamper rejection, dangling-binding rejection, and quiescent audits | quiescent replay of every accepted v2 calibration and holdout trace |
| No-DMA prefix control | run08 and 59/59 calibration processes observed 16 GPU-hit tokens, zero extra DMA, exact `G=B1=B2` raw logits, and `A1-G max_abs=0.109375` | repeat across 20 frozen formal holdouts |
| Output correctness | 59/59 independent fresh processes decoded token 932 in all five phases, preserved exact A1/A2 and G/B1/B2 vectors, and passed `atol=0.125, rtol=0`; every 151,936-entry NPY set was independently replayed | 20 new frozen formal holdouts with exact within-path repeats and exact token equality |
| Provenance | the 59-run cohort has one implementation/runtime fingerprint, 59 unique run IDs and result hashes, a sealed 118-record prefix, a 120-record complete journal, and an independently replayed cohort manifest; the create-only tolerance binds its manifest SHA | repeat the frozen fingerprint across all 20 formal holdouts and seal their aggregate inputs |
| Item-8 aggregate acceptance | open; 59/59 calibration passed and `M2_FROZEN_TOLERANCE.json` is frozen; formal holdouts remain 0/20 | 20/20 holdout-pass manifests, closed-set audit, and one aggregate `M2_ITEM8_ACCEPTANCE_MANIFEST.json` |

All indexed pilot attempts are excluded from both v2 cohorts. B1 and B2 are
state-isolated repeats inside one engine process; they are not independent
calibration runs. A single formal execution can produce only a holdout-pass
record and cannot close item 8. Passing this matrix establishes component
behavior only; `STAGE_GATES.md` remains the acceptance authority.
