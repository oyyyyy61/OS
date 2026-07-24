# M2 Component Test Matrix

Status: living pre-acceptance index, 2026-07-24.

| Gate | Direct component evidence | Remaining evidence |
| --- | --- | --- |
| Canonical identities | `test_domain.py`, repository import-boundary AST check | adapter field audit |
| DAG dependency gate | diamond fanout failure, pending-child bind rejection, orphan-waiter fast-path rejection, ledger node-lifecycle enforcement, mapped-node terminal rejection, and terminal-during-DMA waiter filtering | live workflow adapter |
| Shared-owner isolation | two-workflow D2H/H2D lifecycle and cross-owner release | live engine requests |
| Idempotency | duplicate release, lease terminal, transfer terminal, and early-expiry rejection | callback fault injection |
| Historical identity | released `ExecutionRef` ABA rejection, unique mapping activations, ledger ID replay | live delayed-callback injection |
| Generation safety | failed-target generation consumption and slot checks | live allocator ABA injection |
| Transfer metadata integrity | canonical payload size/digest, exact terminal reports, failed-target no-publish, and cleanup | real DMA payload trace |
| GPU re-admission path | runtime and independent-ledger rejection of direct CPU-only publication, plus successful H2D retry | live adapter API isolation |
| H2D coalescing | concurrent eight-waiter single-flight | live multi-request batch |
| Conservation | runtime/ledger reconciliation, cross-family reference replay, event-envelope/parent tamper rejection, dangling-binding rejection, and quiescent audits | frozen M2 replay report |
| Output correctness | open | forced vLLM token/logit replay |

Passing this matrix establishes component behavior only. `STAGE_GATES.md`
remains the acceptance authority.
