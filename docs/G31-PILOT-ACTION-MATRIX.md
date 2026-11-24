# G3.1 PILOT action-channel matrix

This is an exploratory evaluation harness for the frozen G3.1 checkpoint. It extends the checkpoint-replay diagnostic to the pinned external PILOT split without changing training or protocol decisions.

## Run

From the repository root:

\`\`\`bash
python -m soul.g31.pilot_action_matrix --seed 0
\`\`\`

The default report is \`artifacts/g31/pilot_action_matrix_report.json\`. Existing reports are never overwritten; choose a new \`--report\` path for another run. \`--checkpoint\` accepts only the A/B1 checkpoint hashes already allowlisted by the scorer.

## Evidence and safety boundaries

- Reads the pinned \`artifacts/g31/train.jsonl\` and \`artifacts/g31/pilot.jsonl\` snapshots only.
- Verifies the TRAIN digest against the frozen checkpoint provenance and verifies the exact PILOT digest before evaluation.
- Loads model weights only after the checkpoint has passed the bounded regular-file and allowlisted-SHA checks. This is an actual evaluation, not package inspection.
- Does not open CONFIRM or TEST, does not train, performs zero optimizer updates, and verifies that the model state is unchanged.
- Keeps \`DRAFT_NOT_LOCKED\`, \`TO_LOCK\`, and \`NO_DECISION\`. This run cannot lock a threshold or promote a checkpoint.

## Matrix and pairing

The six interventions are \`real\`, \`zero\`, \`shuffle\`, \`lag\`, \`lead\`, and \`negate\`. Shuffle is per episode and preserves the action multiset. Lag and lead use explicit zero boundaries. Sign reversal is valid for the symmetric Pendulum action domain.

The KL/value-channel pass reuses the diagnostic's paired CPU/CUDA latent RNG protocol. Each intervention starts from the same latent-noise stream; action permutations use dedicated generators seeded by \`seed + 1000 + intervention_index\`. The physical prediction pass recreates those same per-intervention generators and consumes them in the same episode order, so its shuffle is paired with the KL pass.

The first stored row is the training-aligned warm-up and is not scored. Every later observed state is a valid target, including a terminal observation. The replay diagnostic consumes \`Episode.continues\` as a sequence-validity mask; because each PILOT episode is already separated and contains no padding, the harness sets that mask to all ones instead of accidentally dropping a terminal target.

## Report interpretation

The machine-readable report retains per-episode KL, prior-decoder value-channel MSE, physical theta/velocity metrics, paired effects versus the real-action baseline, dispersion, episode order, and valid-transition count. The physical pass is teacher-forced: every intervention receives the same observed next state, while its recurrent context evolves under that intervention's action. It is not a free-running rollout.

The current five-episode PILOT sample is exploratory only. A near-zero action effect is a negative or inconclusive result, not evidence that the action channel is correctly conditioned and not a generalization claim.
