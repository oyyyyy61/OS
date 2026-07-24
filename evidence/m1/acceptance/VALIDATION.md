# M1 Acceptance Validation

The package was accepted only after the following checks returned success:

- `sha256sum -c SHA256SUMS` in this directory;
- `sha256sum -c SHA256SUMS` in the lifecycle, ABBA-v2, and source-freeze
  evidence roots;
- exact hashes for all five decision-document snapshots;
- the manifest predicate below;
- `0444` for every package file, `0555` for every package directory, and no
  group/other writable path.

```bash
jq -e '
  .schema_version == "m1_scoped_acceptance_manifest_v1" and
  .status == "accepted_scoped" and
  .stage_decision.m1 == "pass_for_declared_scope" and
  .stage_decision.m2 == "open_for_canonical_runtime_correctness" and
  .stage_gates.all_m1_gates_passed == true and
  .stage_gates.observed_independent_clusters >=
    .stage_gates.minimum_independent_clusters and
  .stage_gates.control_a_relative_95ci_half_width <=
    .stage_gates.control_precision_limit_relative_95ci_half_width and
  .stage_gates.control_b_relative_95ci_half_width <=
    .stage_gates.control_precision_limit_relative_95ci_half_width and
  .evidence.lifecycle_acceptance.issues == 0 and
  .evidence.measurement_control_abba.completed_rows == 48 and
  .evidence.measurement_control_abba.failed_rows == 0 and
  .evidence.measurement_control_abba.accounting_issues == 0 and
  .evidence.measurement_control_abba.policy_benefit_claim_eligible == false and
  .evidence.source_environment_freeze.read_only_permissions_passed == true and
  (.decision_documents | length) == 5
' MANIFEST.json
```

This validation closes M1 only within `declared_scope`. Excluded claims in the
manifest remain open or blocked at later stage gates.
