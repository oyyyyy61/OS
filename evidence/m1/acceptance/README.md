# M1 Scoped Acceptance Record

Created: `2026-07-24T18:10:58+08:00`

Decision: **M1 PASS for the declared scope**. The accepted scope is one
process, one RTX 4090, GPU allocation, and the primary CPU-DRAM offload tier.
This record opens M2 canonical-runtime correctness work.

## Acceptance authority

M1 is supported jointly by three immutable evidence roots:

1. `../m1_lifecycle_v2_acceptance_20260723_215312` proves strict canonical
   lifecycle closure for seven live traces, aggregate coverage of all ten
   actions, zero audit issues, and zero final live objects. Its `SHA256SUMS`
   digest is
   `bc48cf42daa9c55ebfa76ec55bb81a5e72e6f82f91aee1500125e6743897453d`.
2. `../m1_measurement_control_abba_12seed_rawindex_v2_20260723` proves frozen
   raw-record identity, exact ABBA order, 48/48 completion over 12 independent
   clusters, complete trace auditing, and repeated-control precision below 5%.
   Its `SHA256SUMS` digest is
   `feaca3dc974afbdba95c7007cbacb21501868528b614f8fa1517e7f5f64bf1d2`.
3. `../../source_freezes/m1_lifecycle_v2_20260724_001249` proves scoped source
   and environment reconstruction. Its `SHA256SUMS` digest is
   `b2937a6c6e9164f3b82aa9078d8584ba12b5a96a81c44aca9f4738e4093f3166`.

The final decision documents are copied under `decision_docs/`; their hashes
are recorded in `MANIFEST.json` and covered by this directory's checksums.

## Gate result

- Canonical event conservation: pass.
- Precomputed, frozen run order: pass.
- Independent paired clusters: 12, exceeding the minimum of five.
- Repeated-control relative 95% CI half-width: 1.890365% and 1.096603%, both
  below 5%.
- Raw files, record identity, configuration, results, and source state hashed:
  pass.
- Post-M0 source reconstruction and read-only sealing: pass.

The two ABBA labels execute identical controls. Their geometric ratio of
`1.0004591871` with 95% CI `[0.9928798409, 1.0080963917]` validates the
measurement system and supplies no adaptive-policy benefit claim.

## Excluded boundaries

Multi-manager execution, secondary filesystem/object tiers, process or file
crash recovery, and genuine multi-rank or multi-GPU execution are outside this
M1 result. Output/logit equivalence through the new canonical orchestrator is a
required M2 gate. Proposed C1-C3 performance claims remain closed.

## Verification

Run `sha256sum -c SHA256SUMS` in this directory, then verify
`MANIFEST.json` with the predicate in `VALIDATION.md`. The three referenced
evidence roots retain their own checksum files and must also validate. Files in
this package are mode `0444`; directories are mode `0555`.
