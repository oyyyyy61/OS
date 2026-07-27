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

M1 is imported as scoped evidence. M2 is accepted for correctness only under
the v3 evidence protocol. The accepted scope is one process, one RTX 4090, and
GPU plus primary CPU-DRAM with shared-owner isolation, idempotent release,
generation safety, transfer integrity, no use-after-free, and output
correctness. Latency, throughput, hit-rate, scheduling-policy, novelty, C1, C2,
C3, and paper-performance claims remain open for M3-M6.

The clean runtime now contains the canonical schema, transactional lifecycle
ledger, DAG node gate, generation-safe transfer reservations, shared-owner
orchestrator, and conservation auditor. These are component results for M2
items 1-7. `evidence/m2/PILOT_ATTEMPTS.json` records every current real-GPU
attempt and its resolution. The post-evidence-chain protocol-freeze v2 pilot
(run08) independently re-established
exact canonical DMA digests, equal greedy tokens, exact A1/A2 repeats, exact
`G=B1=B2` logits, and a BF16 cold/prefix maximum absolute logit difference of
`0.109375`. Independent raw replay reproduced every token, margin, comparison,
DMA identity, byte count, digest, and source/runtime binding. This localizes
that numerical difference to the cold-prefill versus GPU-prefix execution
path; the observed DMA path added no logit difference.
Every indexed run remains excluded from calibration and formal cohorts and
closes no M2 gate.

The v3 item-8 gate adds a no-DMA GPU prefix-hit control (`G`) to separate
prefix-path numerics from transfer effects. It then requires 59 fresh-process
calibration runs under the preregistered cap `atol=0.125, rtol=0`, a frozen
aggregate calibration manifest and tolerance file, and 20 new fresh-process
formal holdouts. One formal run can record only one holdout pass. Item 8 closes
only after the complete 20-run holdout set passes a closed-set audit and is
referenced by an aggregate acceptance manifest. Exact source, binary, model,
environment, command, and raw-evidence provenance is fail-closed throughout.
Campaign02 satisfies this item-8 boundary. The subsequent cross-evidence audit
verified all nine M2 conditions and published the create-only aggregate
decision indexed in
`evidence/m2/v3_580_173_02/M2_AGGREGATE_ACCEPTANCE_EVIDENCE_INDEX.json`.

The calibration path now uses a two-stage, no-retry campaign launcher and a
closed cohort validator. `tools/m2_raw_replay.py` independently reloads the
five NumPy logit vectors with pickle disabled, recomputes tokens, margins, and
all seven comparisons, and reconciles the four diagnostic DMA pairs with the
native lifecycle trace and source-state provenance. Production preparation
records a clean Git HEAD. Execution requires a direct single-parent commit
whose only changed path is
`evidence/m2/CALIBRATION_V3_LAUNCH_MARKER.json`, holds a non-blocking exclusive
lock on the campaign directory, and repeats one execution binding in every run
and aggregate submission. All 59 runs must share that Git HEAD and one DAGKV
snapshot, and the calibration and formal paths must bind the same NVIDIA
bundle root, manifest, content digest, and driver version.

The earlier v2 cohort passed 59/59 under its historical driver and froze a v2
tolerance with `atol=0.125, rtol=0`. It cannot supply a v3 parent. The fresh v3
calibration passed 59/59 under driver 580.173.02 and froze a new v3 tolerance
with the same fixed cap. Formal campaign01 completed all 20 run computations
but failed the fail-closed post-aggregate replay before seal publication and
remains excluded. Campaign02 restarted from zero, passed 20/20 holdouts, closed
the exact 42-record journal, published the item-8 acceptance manifest and
create-only bundle seal, and passed independent full-bundle replay. Item 8 is
closed. The separately published aggregate decision passed a fresh full replay,
so M2 correctness is accepted; every performance and M3 policy claim remains
open.

C1-B trace and calibration work is governed by
`research/protocols/M3_C1_TRACE_CALIBRATION_PROTOCOL.md`. It records pre-policy
demand intent separately from resident/H2D service outcomes, freezes
connected-component and temporal split rules, and reserves all numerical
calibration thresholds for an excluded pilot before formal preregistration.
The current pre-data v3 trace contract includes lifecycle-event v2, exact
atomic batch coordinates, binding and H2D waiter transitions, block boundary
snapshots, full transfer-history reconciliation, a sole-writer lifecycle stream
seal, and a create-only lifecycle sidecar with independent replay. The
operation-typed durable committer, formal runtime endpoints, and segmented
create-only C1-B0 bundle are now implemented with focused fault, tamper,
concurrency, and fresh-process replay tests. Their serialized status is limited
to `C1_B0_SCHEMA_RECONSTRUCTION_COMPONENT_VERIFIED`. C1-B0 stage acceptance
still requires a bundle produced from a clean committed source snapshot and an
independently published external anchor; C1-B1 through C1-B4 remain open.
`tools/run_m3_c1_b0_evidence.py` is the CPU-only clean-source launcher for that
gate. It freezes the exact 192-case focused set and 575-case repository set,
binds Git blobs, the protocol, the virtual environment, installed
distributions, and imported module origins, and generates plus replays the
inner bundle in separate processes. The launcher publishes only outside the
repository and does not authorize B1 calibration or any GPU/performance claim.

This correctness work remains substrate for the narrowed research mainline:
dependence-aware shared leases (C1), an explicitly constrained joint
controller (C2), and deadline-aware partial-prefix single-flight (C3).
Tokencake, Continuum, and PBKV already occupy generic DAG-aware offload,
predictive upload, TTL retention, shared-node future-access scoring, and
conservative prefetch.

M3/C1 component work is specified under
`research/protocols/M3_C1_SHARED_LEASE_PROTOCOL.md`. The first component
implements exact PPM joint-outcome aggregation, explicit independence groups,
fanout coalescing, first-versus-repeated reuse statistics, total-variation
drift bounds, an oracle information barrier, and snapshot-staleness rejection.
It exposes PBKV-style additive and independent-marginal baselines behind
separate modes. The sealed C1-A bundle passed sixteen focused mechanism tests,
eighteen evidence-validator tests, the complete 349-test repository regression,
Ruff checks, and the frozen aggregate M2 replay. Its exact external root and
hashes are recorded in
`evidence/m3/c1/M3_C1_COMPONENT_EVIDENCE_INDEX.json`. C1-B trace calibration,
C1-C paired policy effects, C1-D GPU performance, and every paper claim remain
open.

## Development

```bash
uv sync --frozen
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

After committing the launcher source, create the C1-B0 stage artifact from a
clean `main` checkout:

```bash
.venv/bin/python -m tools.run_m3_c1_b0_evidence run \
  --output-dir /home/data/25_oyzx/dagkv_runtime/m3_c1_b0_stage_evidence_v1_20260727_run01 \
  --python .venv/bin/python
```

Record the published manifest, checksum, final-seal, source-commit, and
publication-lock identities in a tracked evidence index only after the fresh
outer replay passes.

After committing a clean C1-A source snapshot, create its CPU-only evidence
bundle with the frozen M2 acceptance identity:

```bash
.venv/bin/python tools/run_m3_c1_component_evidence.py run \
  --output-dir /absolute/new/m3_c1_component_evidence \
  --python "$PWD/.venv/bin/python" \
  --m2-acceptance /absolute/M2_AGGREGATE_ACCEPTANCE.json \
  --expected-m2-sha256 5a226b083ac34a2691017b2d1745c55e8c2968b51651fe8221f6355e40d8aee0 \
  --accepted-m2-head 8331c4fccbcac95890becf63211123c5c9ebccc8
```

The runner records focused and full JUnit, source Git blobs and archive,
commands, environment, lint and format checks, and an exact historical M2
replay. It publishes with create-only rename and a durable `PUBLISHED`
sidecar, then validates the read-only bundle again.

The dependency-free project environment does not install PyTorch or vLLM, so
the diagnostic adapter contract has a separate CPU-only check under the frozen
vLLM environment:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH="$PWD/integrations/vllm_m2" \
/path/to/vllm/.venv/bin/python -m pytest \
  integrations/vllm_m2/tests/test_contract.py
```

Both suites are required before a v3 pilot or campaign launch.

The current M2 vLLM fork also carries the CPU-side lifecycle implementation used by
the adapter. Run its complete M2-relevant CPU set separately; the CUDA worker
tests remain part of the real-GPU gate:

```bash
cd /path/to/vllm
PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES= \
.venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/v1/kv_connector/unit/offloading_connector/test_events.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py \
  tests/v1/kv_offload/cpu/test_manager.py \
  tests/v1/kv_offload/cpu/test_shared_offload_region.py \
  tests/v1/kv_offload/test_factory.py \
  tests/v1/kv_offload/test_fanout_planner.py \
  tests/v1/kv_offload/test_lifecycle.py
```

After committing all component-test changes, seal the three suites into one
create-only evidence bundle:

```bash
.venv/bin/python tools/run_m2_component_evidence.py run \
  --output-dir /absolute/new/m2_component_evidence \
  --dagkv-python "$PWD/.venv/bin/python" \
  --vllm-python /path/to/vllm/.venv/bin/python \
  --vllm-root /path/to/vllm
```

The bundle independently replays Git snapshots, commands, environments,
JUnit, and checksums before and after read-only sealing. It supports M2 items
1-7 component evidence only; the v3 GPU and item-8 gates remain separate.

See `research/ARCHITECTURE.md`, `research/M2_RUNTIME_CONTRACT.md`,
`research/STAGE_GATES.md`, `research/protocols/M2_VLLM_REPLAY_PROTOCOL.md`,
`research/protocols/M2_COMPONENT_EVIDENCE_PROTOCOL.md`,
`research/protocols/M2_AGGREGATE_ACCEPTANCE_PROTOCOL.md`,
`research/REFERENCES.md`, and
`evidence/IMPORT_MANIFEST.json` for the normative boundaries and first-order
prior-work constraints.

## License

No open-source license has been selected yet. The repository contents remain
under their default copyright terms until the project owner records a license.
