# G3.1 Protocol Decision Register — audit-only

Status: G3.1 INCONCLUSIVE | threshold TO_LOCK | Confirm NOT ACCESSED | no protocol lock or confirmatory claim.
G3.2 is recorded as CLOSED in the existing plan; this register does not change that state.

This register records evidence and unresolved owner decisions. It is not an approval, authorization, training instruction, or permission to access Confirm/TEST.

## Evidence and integrity facts

- `artifacts/g31/pilot.jsonl` contains 100 transitions across five episodes (seeds 2001–2005), Pendulum-v1 / Gymnasium 1.4.0, observation shape `[3]`, action shape `[1]`, action bounds `[-2, 2]`, and `bounded_uniform` collection. Its raw-byte SHA-256 was computed in this audit and matches `pilot_manifest.json`: `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e`.
- `pilot_manifest.json` says `dataset_status = NEW DATASET — NOT ORIGINAL`, `confirm_accessed = false`, and `scored = false`. The provenance wording still needs owner clarification; no interpretation is assumed.
- Candidate A is recorded as `g31_verified_train_checkpoint.pt` (3,386,259 bytes; report SHA-256 `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147`). B1 is recorded as `g31_b1_checkpoint.pt` (3,383,579 bytes; report SHA-256 `136017b650fc910043c559822d47fea900391e2823324413327de815a4d8c018`). These are report-declared digests; a hash-only CI test now checks raw bytes against the reports. No checkpoint was deserialized in this audit.
- The previous 44-character B1 digest was malformed and did not match the report's 64-character value. Do not reuse it.
- `pilot_b1_predictions.jsonl`, `pilot_b1_metrics.json`, and `pilot_b1_report.json` have the same Git blob identities as the baseline A prediction, metrics, and report files. They are copies, not independent B1 evidence.
- `staging_b1_predictions.jsonl` contains 100 rows with the same `(episode_id, timestep)` keys, applied actions, and physical targets as the baseline predictions; `predicted_physical` differs in all 100 rows. Its per-row checkpoint digest matches the staging report's B1 digest.
- Recorded descriptive mean theta RMSE is 1.3707174687 for A and 1.4035570433 for staging B1 over five pilot episodes. This is not a significance test, acceptance decision, or confirmatory result; the gate remains INCONCLUSIVE and the threshold remains TO_LOCK.
- No Confirm/TEST run, training run, benchmark, or model-weight deserialization was performed in this session.

## Owner decisions still unresolved

1. Primary metric: approve theta RMSE, velocity RMSE, or a defined combination and rationale.
2. Aggregation: approve equal-episode mean, median, pooled metric, and required dispersion reporting.
3. Comparison direction and pairing: approve B1-versus-A direction and the `(episode_id, timestep)` pairing contract.
4. Dataset provenance: clarify whether `NEW DATASET — NOT ORIGINAL` is expected and whether it is eligible for confirmatory use.
5. Sample size: justify five episodes / 100 transitions or specify a larger confirmatory sample.
6. Acceptance criterion: define a numeric threshold and independent scientific justification before examining confirmatory results.
7. Statistical analysis: explicitly permit or prohibit tests; if permitted, define method, alpha, and multiple-comparison handling.
8. Missing/non-finite/exclusion policy: define whether invalid predictions abort or are excluded and how exclusions are reported.
9. Stop/abort/rollback: define failure thresholds and recovery rules.
10. Evaluator versioning: decide whether the evaluator source hash/version must be frozen for confirmation.
11. Checkpoint approval: approve exact A/B1 checkpoint identities for confirmatory use.
12. Confirm authorization: explicit owner authorization is required before any Confirm access.

## Guardrails

- Keep pilot and confirmatory data separate.
- Do not use the byte-identical `pilot_b1_*` artifacts as B1 evidence.
- Do not infer improvement or statistical significance from the five-episode staging report.
- Do not lock a threshold based on observed pilot values.
- Do not access Confirm or TEST, retrain, replace checkpoints, or alter datasets without the required protocol and authorization.

Decision: NOT READY TO LOCK. No owner choices have been made on their behalf.
