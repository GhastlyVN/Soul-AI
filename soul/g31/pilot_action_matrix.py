"""Exploratory G3.1 action matrix on the pinned external PILOT split.

This module evaluates an allowlisted frozen checkpoint. It never trains, updates
optimizer state, or opens CONFIRM/TEST. The protocol remains DRAFT_NOT_LOCKED.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from soul.config import Config
from soul.diag_actioncausal import (
    INTERVENTIONS,
    _write_json_report_atomic,
    intervene_actions,
    measure_ablation,
    summarize_ablation,
)
from soul.g31.evaluator import (
    evaluate_one_step,
    physical_metrics,
    prime_episode_state,
    raw_observation_to_physical,
)
from soul.g31.score_pilot import (
    DEFAULT_CHECKPOINT_PATH,
    PILOT_HASH,
    PILOT_PATH,
    TRAIN_PATH,
    aggregate_episode_metrics,
    load_frozen_checkpoint,
    load_rows_snapshot,
    paired_differences,
    validate_checkpoint_training_provenance,
    validate_output_paths,
    verify_model_unchanged,
)
from soul.g31.train_data import (
    PENDULUM_ACTION_DIM,
    PENDULUM_ENV_ID,
    PENDULUM_OBS_SHAPE,
    load_train_jsonl_snapshot,
    pendulum_model_config,
    rows_to_episodes,
    validate_pendulum_action,
)
from soul.model import SoulModel
from soul.obs import ObsEncoder

PILOT_SEEDS = tuple(range(2001, 2006))
DEFAULT_REPORT_PATH = Path("artifacts/g31/pilot_action_matrix_report.json")


def _finite_observation(value, *, field: str, row_number: int) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != PENDULUM_OBS_SHAPE[0]:
        raise ValueError(f"{field} must have shape {PENDULUM_OBS_SHAPE} at PILOT row {row_number}")
    if any(
        isinstance(component, (bool, np.bool_))
        or not isinstance(component, (int, float, np.integer, np.floating))
        for component in value
    ):
        raise ValueError(f"{field} must contain numeric values at PILOT row {row_number}")
    try:
        observation = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field} is invalid at PILOT row {row_number}") from error
    if not np.isfinite(observation).all():
        raise ValueError(f"{field} contains non-finite values at PILOT row {row_number}")
    return observation


def validate_pilot_rows(rows, expected_seeds=PILOT_SEEDS):
    """Validate and group every PILOT row, including the unscored warm-up action."""
    expected = tuple(expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_seeds must be a non-empty set of unique seeds")
    if not rows:
        raise ValueError("PILOT dataset is empty")

    grouped = {}
    episode_for_seed = {}
    seed_for_episode = {}
    seen_seeds = set()

    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"PILOT row {row_number} must be an object")
        if row.get("split") != "pilot":
            raise ValueError(f"non-PILOT row at line {row_number}")

        seed = row.get("episode_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in expected:
            raise ValueError(f"unexpected PILOT episode seed at line {row_number}")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError(f"invalid PILOT episode_id at line {row_number}")
        previous_episode = episode_for_seed.setdefault(seed, episode_id)
        if previous_episode != episode_id:
            raise ValueError(f"PILOT seed {seed} is assigned to multiple episodes")
        previous_seed = seed_for_episode.setdefault(episode_id, seed)
        if previous_seed != seed:
            raise ValueError(f"PILOT episode {episode_id} changes seed")
        seen_seeds.add(seed)

        timestep = row.get("timestep")
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError(f"invalid PILOT timestep at line {row_number}")
        if not isinstance(row.get("terminal"), bool):
            raise ValueError(f"PILOT terminal must be boolean at line {row_number}")

        _finite_observation(
            row.get("raw_observation_before"),
            field="raw_observation_before",
            row_number=row_number,
        )
        _finite_observation(
            row.get("raw_observation_after"),
            field="raw_observation_after",
            row_number=row_number,
        )
        validate_pendulum_action(
            row.get("applied_action"),
            field=f"PILOT applied action at line {row_number}",
        )

        reward = row.get("reward")
        if (
            isinstance(reward, (bool, np.bool_))
            or not isinstance(reward, (int, float, np.integer, np.floating))
            or not np.isfinite(float(reward))
        ):
            raise ValueError(f"invalid PILOT reward at line {row_number}")
        environment = row.get("environment_config")
        if not isinstance(environment, dict) or environment.get("env_id") != PENDULUM_ENV_ID:
            raise ValueError(f"wrong PILOT environment at line {row_number}")

        grouped.setdefault(episode_id, []).append(row)

    if seen_seeds != set(expected):
        raise ValueError("PILOT dataset does not contain exactly the expected episode seeds")

    result = []
    for episode_id, episode_rows in grouped.items():
        ordered = sorted(episode_rows, key=lambda item: item["timestep"])
        timesteps = [item["timestep"] for item in ordered]
        if len(ordered) < 2:
            raise ValueError(f"{episode_id}: at least two rows are required")
        if timesteps != list(range(len(ordered))):
            raise ValueError(f"{episode_id}: timesteps must be contiguous from zero")
        if any(item["terminal"] for item in ordered[:-1]):
            raise ValueError(f"{episode_id}: terminal transition must be the final row")

        for previous, current in zip(ordered, ordered[1:]):
            previous_after = _finite_observation(
                previous["raw_observation_after"],
                field="raw_observation_after",
                row_number=previous["timestep"] + 1,
            )
            current_before = _finite_observation(
                current["raw_observation_before"],
                field="raw_observation_before",
                row_number=current["timestep"] + 1,
            )
            if not np.allclose(previous_after, current_before, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"{episode_id}: observation discontinuity before timestep {current['timestep']}"
                )
        result.append((episode_id, ordered))
    return result


def prepare_pilot_episodes(rows, encoder, goal_dim, expected_seeds=PILOT_SEEDS):
    """Convert PILOT rows and mark every stored observation as sequence-valid.

    Episode.continues stores the terminal target, but measure_episode_kl_series
    consumes that field as a padding/validity mask. Episodes are already split,
    so a terminal observation is still a valid target and must not be masked out.
    """
    groups = validate_pilot_rows(rows, expected_seeds=expected_seeds)
    ordered_rows = [row for _, episode_rows in groups for row in episode_rows]
    episodes = rows_to_episodes(ordered_rows, encoder, goal_dim)
    if len(episodes) != len(groups):
        raise ValueError("PILOT episode conversion changed the episode count")
    for episode in episodes:
        episode.continues = [1.0] * episode.length
    return groups, episodes


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def paired_action_generators(seed):
    """Create the same per-intervention shuffle streams used by measure_ablation."""
    _validate_seed(seed)
    return {
        kind: torch.Generator(device="cpu").manual_seed(seed + 1000 + offset)
        for offset, kind in enumerate(INTERVENTIONS)
    }


def build_intervened_actions(actions, generators):
    """Apply the shared six-way intervention policy to one complete episode."""
    matrix = np.asarray(actions, dtype=np.float32)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != PENDULUM_ACTION_DIM
        or matrix.shape[0] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("actions must be a finite [time, action_dim] matrix")
    for action in matrix:
        validate_pendulum_action(action)

    tensor = torch.as_tensor(matrix, dtype=torch.float32, device="cpu").unsqueeze(0)
    valid = torch.ones((1, matrix.shape[0]), dtype=torch.float32, device="cpu")
    return {
        kind: intervene_actions(
            tensor,
            valid,
            kind,
            generator=generators[kind],
        )[0].cpu().numpy().copy()
        for kind in INTERVENTIONS
    }


def measure_physical_ablation(model, groups, encoder, seed):
    """Run paired teacher-forced physical predictions for every intervention."""
    generators = paired_action_generators(seed)
    per_episode = {kind: {} for kind in INTERVENTIONS}

    for episode_id, rows in groups:
        actions = np.stack([
            validate_pendulum_action(row["applied_action"])
            for row in rows
        ])
        action_sequences = build_intervened_actions(actions, generators)
        states = {
            kind: prime_episode_state(
                model,
                rows[0]["raw_observation_after"],
                encoder,
                device="cpu",
            )
            for kind in INTERVENTIONS
        }
        predictions = {kind: [] for kind in INTERVENTIONS}
        targets = []

        # Row zero is the training-aligned warm-up. Every later row is a
        # scored transition, including a terminal row's valid next observation.
        for row_index in range(1, len(rows)):
            row = rows[row_index]
            current = _finite_observation(
                row["raw_observation_before"],
                field="raw_observation_before",
                row_number=row_index + 1,
            )
            next_observation = _finite_observation(
                row["raw_observation_after"],
                field="raw_observation_after",
                row_number=row_index + 1,
            )
            targets.append(raw_observation_to_physical(next_observation))

            for kind in INTERVENTIONS:
                predicted, next_state = evaluate_one_step(
                    model,
                    states[kind],
                    current,
                    action_sequences[kind][row_index],
                    obs_next=next_observation,
                    device="cpu",
                    feature_encoder=encoder,
                )
                predictions[kind].append(predicted)
                states[kind] = next_state

        target_matrix = np.stack(targets)
        for kind in INTERVENTIONS:
            per_episode[kind][episode_id] = physical_metrics(
                np.stack(predictions[kind]),
                target_matrix,
            )

    baseline = per_episode["real"]
    return {
        "per_episode": per_episode,
        "aggregate_equal_episode": {
            kind: aggregate_episode_metrics(values)
            for kind, values in per_episode.items()
        },
        "paired_effects_vs_real": {
            kind: paired_differences(values, baseline)
            for kind, values in per_episode.items()
            if kind != "real"
        },
    }



def resolve_code_commit(value=None):
    """Resolve a full Git commit SHA or fail before producing a non-reproducible report."""
    candidate = value or os.environ.get("GITHUB_SHA")
    if candidate is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "could not resolve the code commit; pass --code-commit or run inside a Git checkout"
            ) from error
        candidate = result.stdout.strip()
    if not isinstance(candidate, str) or re.fullmatch(r"[0-9a-fA-F]{40}", candidate) is None:
        raise ValueError("code commit must be a full 40-character hexadecimal Git SHA")
    return candidate.lower()


def run(checkpoint_path=None, report_path=None, seed=0, code_commit=None):
    """Score the pinned PILOT split without training or touching CONFIRM/TEST."""
    _validate_seed(seed)
    code_commit = resolve_code_commit(code_commit)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    report_path = Path(report_path) if report_path else DEFAULT_REPORT_PATH

    validate_output_paths(
        (report_path,),
        (checkpoint_path, PILOT_PATH, TRAIN_PATH),
    )
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(
            f"report already exists; choose a new path instead of overwriting: {report_path}"
        )

    pilot_rows, pilot_sha256 = load_rows_snapshot(
        PILOT_PATH,
        "pilot",
        PILOT_SEEDS,
        expected_sha256=PILOT_HASH,
    )
    train_rows, train_sha256 = load_train_jsonl_snapshot(TRAIN_PATH)

    # The loader verifies a bounded regular-file snapshot and an allowlisted
    # SHA-256 before torch.load is permitted.
    checkpoint, checkpoint_sha256 = load_frozen_checkpoint(checkpoint_path)
    validate_checkpoint_training_provenance(checkpoint, train_sha256, train_rows)

    cfg = pendulum_model_config(Config.from_dict(checkpoint["config_dict"]))
    encoder = ObsEncoder(cfg.obs)
    encoder.load_schema(checkpoint["enc_schema"])
    model = SoulModel(cfg, encoder.input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    groups, episodes = prepare_pilot_episodes(
        pilot_rows,
        encoder,
        cfg.model.goal_dim,
    )
    episode_ids = [episode_id for episode_id, _ in groups]
    scorer = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        replay=SimpleNamespace(episodes=episodes),
    )
    kl_result = measure_ablation(
        scorer,
        seed,
        interventions=INTERVENTIONS,
    )

    expected_indices = list(range(len(episode_ids)))
    expected_transitions = sum(len(rows) - 1 for _, rows in groups)
    if kl_result.get("episode_indices") != expected_indices:
        raise ValueError("KL matrix did not preserve the complete PILOT episode order")
    if kl_result.get("valid_transitions") != expected_transitions:
        raise ValueError("KL matrix valid-transition count does not match PILOT rows")

    physical_result = measure_physical_ablation(model, groups, encoder, seed)
    verify_model_unchanged(before, model)
    kl_summary = summarize_ablation(kl_result)

    payload = {
        "schema_version": 1,
        "protocol": "DRAFT_NOT_LOCKED",
        "threshold": "TO_LOCK",
        "decision": "NO_DECISION",
        "access": {
            "train_read": True,
            "pilot_read": True,
            "confirm_read": False,
            "test_read": False,
        },
        "training": {
            "performed": False,
            "optimizer_updates": 0,
            "checkpoint_mutated": False,
        },
        "provenance": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "train_path": str(TRAIN_PATH),
            "train_sha256": train_sha256,
            "train_rows": len(train_rows),
            "pilot_path": str(PILOT_PATH),
            "pilot_sha256": pilot_sha256,
            "pilot_rows": len(pilot_rows),
            "code_commit": code_commit,
        },
        "reproducibility": {
            "seed": seed,
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "latent_rng": (
                "measure_ablation forks CPU/CUDA RNG and restores the same initial "
                "latent-noise stream before each intervention"
            ),
            "shuffle_rng": (
                "dedicated per-intervention generators seeded with seed+1000+offset; "
                "the physical pass reuses the same seeds and episode order"
            ),
        },
        "episode_order": episode_ids,
        "episode_count": len(episode_ids),
        "valid_transitions": expected_transitions,
        "interventions": list(INTERVENTIONS),
        "kl": {
            "summary": kl_summary["interventions"],
            "paired_effects_vs_real": kl_summary["paired_effects_vs_real"],
            "per_episode": kl_result["series"],
        },
        "prior_decoder_value_mse": {
            "summary": kl_summary["physical_interventions"],
            "paired_effects_vs_real": kl_summary["paired_physical_effects_vs_real"],
            "per_episode": kl_result["physical_series"],
        },
        "physical_predictions": physical_result,
        "rollout_semantics": (
            "teacher-forced per-intervention recurrent evaluation: every condition "
            "receives the same observed next state, while its recurrent state evolves "
            "under that condition's intervened action; this is not a free-running rollout"
        ),
        "limitations": [
            "The current five-episode PILOT sample is exploratory and is not evidence of generalization.",
            "The protocol remains DRAFT_NOT_LOCKED and the threshold remains TO_LOCK.",
            "A near-zero action effect is a negative or inconclusive result, not proof of correct conditioning.",
            "No CONFIRM or TEST data is read; no training or optimizer update is performed.",
        ],
    }
    _write_json_report_atomic(
        report_path,
        payload,
        (checkpoint_path, PILOT_PATH, TRAIN_PATH),
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-commit", help="full Git commit SHA for reproducible local runs")
    args = parser.parse_args(argv)
    payload = run(args.checkpoint, args.report, args.seed, args.code_commit)
    print(json.dumps({
        "report": str(args.report),
        "checkpoint_sha256": payload["provenance"]["checkpoint_sha256"],
        "train_sha256": payload["provenance"]["train_sha256"],
        "pilot_sha256": payload["provenance"]["pilot_sha256"],
        "episodes": payload["episode_count"],
        "valid_transitions": payload["valid_transitions"],
        "protocol": payload["protocol"],
        "threshold": payload["threshold"],
        "decision": payload["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
