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
| Transfer metadata integrity | component tests cover exact terminal bytes/digests, failed-target no-publish, and cleanup; run08 recorded and independently replayed four exact `2,359,296`-byte DMA terminal pairs with one canonical digest; the validator reconciles diagnostic and native ABBA identities, generations, bytes, and digests | all 59 calibration and 20 formal holdout traces under one frozen source/runtime fingerprint |
| GPU re-admission path | runtime and independent-ledger rejection of direct CPU-only publication, plus successful H2D retry; run08 independently replayed fresh GPU target identities and CPU generations 1 then 2 | live adapter API isolation and v2 cohort evidence |
| H2D coalescing | concurrent eight-waiter single-flight component test | live multi-request batch; policy-scale partial-prefix coalescing remains M3/C3 |
| Conservation | runtime/ledger reconciliation, cross-family reference replay, event-envelope/parent tamper rejection, dangling-binding rejection, and quiescent audits | quiescent replay of every accepted v2 calibration and holdout trace |
| No-DMA prefix control | run08 observed 16 GPU-hit tokens, zero extra DMA, exact `G=B1=B2` raw logits, and `A1-G max_abs=0.109375` | repeat across the frozen 59+20 cohorts |
| Output correctness | run08 decoded token 932 in all five phases and recorded exact A1/A2 and G/B1/B2 vectors; the independent validator reloads all five 151,936-entry NPY files with pickle disabled and recomputes tokens, margins, and seven comparisons | 59 independent fresh-process calibrations at `atol=0.125, rtol=0`; then 20 new frozen formal holdouts with exact within-path repeats and exact token equality |
| Provenance | run08 hashes the clean DAGKV tree, complete dirty-vLLM patch/untracked set, 16 model files, five weight shards, six native extensions, dependencies, environment, and every raw artifact; cohort v2 binds preregistration, the 118-record calibration prefix, process logs, source-state archives, frozen validators, and exclusive publication | repeat one fingerprint across the complete cohorts and freeze aggregate inputs |
| Item-8 aggregate acceptance | open; all entries in `evidence/m2/PILOT_ATTEMPTS.json` are pilot-only | frozen 59-run calibration manifest and tolerance, 20/20 holdout-pass manifests, closed-set audit, and one aggregate `M2_ITEM8_ACCEPTANCE_MANIFEST.json` |

All indexed pilot attempts are excluded from both v2 cohorts. B1 and B2 are
state-isolated repeats inside one engine process; they are not independent
calibration runs. A single formal execution can produce only a holdout-pass
record and cannot close item 8. Passing this matrix establishes component
behavior only; `STAGE_GATES.md` remains the acceptance authority.
