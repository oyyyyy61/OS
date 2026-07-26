# Imported Evidence

This directory contains metadata and document snapshots only. It contains no
runtime implementation from the historical `offload/` prototype.

`IMPORT_MANIFEST.json` is the authority for source paths, source hashes, copied
hashes, scope, and the explicit no-code-import boundary. Full raw M1 artifacts
remain in their immutable sibling evidence roots.

`m1/acceptance/` is a complete, read-only copy of the small M1 decision
package. It includes its original `SHA256SUMS`, manifest checksum, validation
report, and claim/reproduction documents. From that directory, run:

```bash
sha256sum -c SHA256SUMS
sha256sum -c MANIFEST.sha256
```

The larger lifecycle, ABBA, and source-freeze payloads are identified by their
root hashes in the copied manifest. Their original `SHA256SUMS` and
`MANIFEST.sha256` inventories are copied under `m1/external_roots/`; referenced
payload files remain external. A stable public artifact URI remains open before
a paper artifact release.

`m3/c1/M3_C1_COMPONENT_EVIDENCE_INDEX.json` records the first accepted C1-A
component bundle, its exact external root and hashes, fresh-process replay,
frozen M2 historical regression, and the two prepublication failures that left
no public artifact. The acceptance is limited to mathematical and runtime
component correctness; later trace, policy, GPU, and paper gates remain open.

Frozen documents retain their original relative paths, some of which point to
the historical directory layout. Use `research/REFERENCES.md` and the
`relocation` section of `IMPORT_MANIFEST.json` as resolvers. The frozen files
remain unchanged so their recorded hashes stay authoritative.
