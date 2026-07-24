# DAGKV Project Instructions

This directory is a clean research implementation. The sibling `offload/`
tree is a historical prototype and evidence source.

- Never import Python modules from `offload`, Agentrix, or vLLM into
  `src/dagkv`. Integration must use explicit adapter protocols implemented in
  this project.
- Keep one canonical runtime schema in `src/dagkv/domain.py`. Adapters project
  external data into that schema and never persist a second runtime state.
- Fail closed on ambiguous identity, ownership, generation, transfer, or
  terminal state. Do not infer missing identities or byte counts.
- Every state transition requires focused tests. Shared ownership, duplicate
  release, stale generation, failed transfer, and workflow cleanup are
  mandatory regression cases.
- Historical M0/M1 artifacts may be copied only under `evidence/` or
  `research/imported/`, with source hashes recorded in
  `evidence/IMPORT_MANIFEST.json`.
- No policy-performance claim may be added until its declared stage gate has
  immutable paired evidence.
