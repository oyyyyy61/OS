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

M1 is imported as scoped evidence. M2 is active. M2 requires a single-process,
GPU plus CPU-DRAM lifecycle loop with shared-owner isolation, idempotent
release, generation safety, transfer integrity, no use-after-free, and output
correctness. Performance mechanisms remain out of scope until M2 passes.

The clean runtime now contains the canonical schema, transactional lifecycle
ledger, DAG node gate, generation-safe transfer reservations, shared-owner
orchestrator, and conservation auditor. These are component results for M2
items 1-7. The real vLLM GPU token/logit replay and immutable M2 evidence pack
remain open, so M2 has not passed.

## Development

```bash
uv sync --frozen
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

See `research/ARCHITECTURE.md`, `research/M2_RUNTIME_CONTRACT.md`,
`research/STAGE_GATES.md`, `research/REFERENCES.md`, and
`evidence/IMPORT_MANIFEST.json` for the normative boundaries and first-order
prior-work constraints.

## License

No open-source license has been selected yet. The repository contents remain
under their default copyright terms until the project owner records a license.
