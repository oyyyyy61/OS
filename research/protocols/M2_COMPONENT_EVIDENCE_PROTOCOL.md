# M2 CPU Component Evidence Protocol v1

Status: normative pre-acceptance protocol for deterministic CPU evidence that
may support M2 items 1-7. It cannot close item 8, aggregate M2, a CUDA worker
path, a performance gate, or any C1-C3 paper claim.

## Scope

The component contract executes three closed test sets:

1. the dependency-free DAGKV runtime tests under `tests/`;
2. the diagnostic adapter contract under
   `integrations/vllm_m2/tests/test_contract.py`, using the recorded vLLM
   Python environment with no visible GPU;
3. the current M2-relevant vLLM scheduler, worker-metadata, CPU-manager,
   shared-region, factory, fanout, and lifecycle tests listed explicitly in
   `tools/run_m2_component_evidence.py`.

The minimum test counts are 269, 13, and 345 respectively. These floors prevent
silent coverage erosion. The manifest records the full observed counts. Every
test must pass; skips, errors, duplicate JUnit identities, and retries are
prohibited.

## Execution Contract

Production execution requires a clean DAGKV worktree and explicit absolute
paths for the DAGKV Python executable, vLLM Python executable, vLLM Git root,
and a new output directory. The vLLM worktree may be dirty because the current
integration depends on a captured local patch, but its complete tracked diff
and every non-ignored untracked file are archived before execution and must be
unchanged after all three suites.

Each suite runs in a new process with:

- `CUDA_VISIBLE_DEVICES` empty;
- `PYTHONDONTWRITEBYTECODE=1` and pytest's cache provider disabled;
- `PYTHONNOUSERSITE=1`;
- Hugging Face and Transformers offline modes enabled;
- a fixed `/usr/bin:/bin` `PATH`;
- no inherited `LD_PRELOAD`, `LD_AUDIT`, or other ambient variables;
- one hard timeout and no retry.

The adapter suite alone receives the recorded DAGKV integration directory on
`PYTHONPATH`. The protocol captures both Python launchers and resolved binaries,
sorted `importlib.metadata` distribution inventories with editable-source
metadata, the imported vLLM module path and hash, platform identity, exact
argv, exact child environments, UTC timestamps, stdout, stderr, and JUnit XML.
Distribution source URLs are hashed inside the child process; evidence stores
only the canonical SHA-256, editable flag, and URL scheme, so local paths and
embedded credentials never enter the bundle.

## Evidence Closed Set

A successful create-only output contains exactly:

- `M2_COMPONENT_EVIDENCE.json` with schema
  `dagkv.m2.component_evidence.v1`;
- `SHA256SUMS`, covering every other file in sorted order;
- `source_state/` with reconstructable DAGKV and vLLM tracked patches and
  non-ignored untracked-file archives;
- `environment/` with both dependency snapshots and the normalized vLLM
  runtime probe;
- `logs/` with stdout, stderr, and JUnit XML for each suite.

Before publication, the runner independently replays the complete tree,
checksums, Git archives, JUnit semantics, commands, environments, runtime
binding, and postflight snapshots. It then sets files to mode `0444` and
directories to `0555` and replays the sealed directory again. Symlink roots,
symlink members, hard-linked files, special nodes, extra paths, writable modes,
oversized logs, duplicate JSON keys, non-finite JSON, and checksum or semantic
drift fail closed.

The manifest records the original absolute evidence root because it appears in
the frozen JUnit argv. File replay uses the directory supplied to the validator,
so a byte-for-byte relocated bundle remains valid in `--no-external` mode.

A failed run produces no successful manifest or checksum seal. If validation
fails after candidate files exist, they are renamed with an `INVALID_` prefix
and `FAILURE.json` records the error. The directory is ineligible and cannot be
repaired in place; diagnosis and a material fix must precede any new
create-only attempt.

## Claim Boundary

The exact manifest claim is:

> M2 items 1-7 deterministic CPU component contract only; no item-8,
> performance, scheduling-policy, C1, C2, C3, or aggregate-M2 claim

Even a complete passing bundle remains component evidence. M2 item 8 still
requires the excluded v3 GPU pilot, 59 fresh-process calibration runs, a new
frozen tolerance, and 20 formal holdouts under the NVIDIA `580.173.02` bundle.
Live multi-request, allocator callback, and CUDA behavior remain limited to the
direct evidence named in `research/M2_TEST_MATRIX.md`.

## Commands

Run from a clean DAGKV root with a new absolute output path:

```bash
.venv/bin/python tools/run_m2_component_evidence.py run \
  --output-dir /absolute/new/m2_component_evidence \
  --dagkv-python "$PWD/.venv/bin/python" \
  --vllm-python /absolute/vllm/.venv/bin/python \
  --vllm-root /absolute/vllm
```

Replay with current external binaries:

```bash
.venv/bin/python tools/run_m2_component_evidence.py validate \
  /absolute/new/m2_component_evidence
```

Raw-only replay may omit external executable re-hashing while retaining the
recorded hashes and every sealed-file check:

```bash
.venv/bin/python tools/run_m2_component_evidence.py validate \
  --no-external /absolute/new/m2_component_evidence
```
