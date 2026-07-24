# DAGKV

DAGKV is the clean implementation of a lifecycle-safe KV-cache runtime for
multi-agent DAG workflows. It starts after the accepted M1 measurement and
ledger gate preserved in the sibling historical `offload/` tree.

The project is intentionally independent:

- `src/dagkv/domain.py` is the only canonical runtime schema.
- `src/dagkv/orchestrator.py` owns workflow, binding, lease, residency,
  execution-mapping, transfer, and reclaim state.
- external serving engines will enter through narrow adapter protocols;
- old prototype code is excluded from runtime imports and represented only by
  hashed evidence metadata.

## Current gate

M1 is imported as scoped evidence. M2 is active under the v2 evidence protocol.
M2 requires a single-process GPU plus CPU-DRAM lifecycle loop with shared-owner
isolation, idempotent release, generation safety, transfer integrity, no
use-after-free, and output correctness. Performance mechanisms remain out of
scope until M2 passes.

The clean runtime now contains the canonical schema, transactional lifecycle
ledger, DAG node gate, generation-safe transfer reservations, shared-owner
orchestrator, and conservation auditor. These are component results for M2
items 1-7. `evidence/m2/PILOT_ATTEMPTS.json` records every current real-GPU
attempt and its resolution. The latest successful v2 pilot (run06) established
exact canonical DMA digests, equal greedy tokens, exact A1/A2 repeats, exact
`G=B1=B2` logits, and a BF16 cold/prefix maximum absolute logit difference of
`0.109375`. This localizes that numerical difference to the cold-prefill versus
GPU-prefix execution path; the observed DMA path added no logit difference.
Every indexed run remains excluded from calibration and formal cohorts and
closes no M2 gate.

The v2 item-8 gate adds a no-DMA GPU prefix-hit control (`G`) to separate
prefix-path numerics from transfer effects. It then requires 59 fresh-process
calibration runs under the preregistered cap `atol=0.125, rtol=0`, a frozen
aggregate calibration manifest and tolerance file, and 20 new fresh-process
formal holdouts. One formal run can record only one holdout pass. Item 8 closes
only after the complete 20-run holdout set passes a closed-set audit and is
referenced by an aggregate acceptance manifest. Exact source, binary, model,
environment, command, and raw-evidence provenance is fail-closed throughout.
M2 therefore remains open.

This correctness work remains substrate for the narrowed research mainline:
dependence-aware shared leases (C1), an explicitly constrained joint
controller (C2), and deadline-aware partial-prefix single-flight (C3).
Tokencake, Continuum, and PBKV already occupy generic DAG-aware offload,
predictive upload, TTL retention, shared-node future-access scoring, and
conservative prefetch.

## Development

```bash
uv sync --frozen
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

See `research/ARCHITECTURE.md`, `research/M2_RUNTIME_CONTRACT.md`,
`research/STAGE_GATES.md`, `research/protocols/M2_VLLM_REPLAY_PROTOCOL.md`,
`research/REFERENCES.md`, and
`evidence/IMPORT_MANIFEST.json` for the normative boundaries and first-order
prior-work constraints.

## License

No open-source license has been selected yet. The repository contents remain
under their default copyright terms until the project owner records a license.
