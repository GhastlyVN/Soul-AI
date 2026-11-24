import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import soul.g31.pilot_action_matrix as matrix
from soul.diag_actioncausal import INTERVENTIONS
from soul.g31.train_data import PENDULUM_ENV_ID


def make_episode(seed=2001, actions=(0.1, 0.2), terminal_last=True):
    observations = [
        [1.0 - 0.01 * index, 0.1 * index, 0.2 * index]
        for index in range(len(actions) + 1)
    ]
    rows = []
    for timestep, action in enumerate(actions):
        rows.append({
            "split": "pilot",
            "episode_id": f"pilot-{seed}",
            "episode_seed": seed,
            "timestep": timestep,
            "raw_observation_before": observations[timestep],
            "raw_observation_after": observations[timestep + 1],
            "applied_action": [action],
            "reward": 1.0,
            "terminal": bool(terminal_last and timestep == len(actions) - 1),
            "environment_config": {"env_id": PENDULUM_ENV_ID},
        })
    return rows


class FakeFeatureEncoder:
    def encode(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        return {
            "values": values,
            "mask": np.ones_like(values),
            "text": np.zeros(0, dtype=np.float32),
        }


def test_pilot_validation_checks_the_unscored_warmup_action():
    rows = make_episode()
    rows[0]["applied_action"] = [2.01]

    with pytest.raises(ValueError, match="outside"):
        matrix.validate_pilot_rows(rows, expected_seeds=(2001,))


def test_pilot_validation_rejects_nonfinite_observations_and_bad_terminal_flags():
    rows = make_episode()
    rows[0]["raw_observation_after"] = [1.0, float("nan"), 0.0]
    with pytest.raises(ValueError, match="non-finite"):
        matrix.validate_pilot_rows(rows, expected_seeds=(2001,))

    rows = make_episode()
    rows[0]["terminal"] = "false"
    with pytest.raises(ValueError, match="terminal must be boolean"):
        matrix.validate_pilot_rows(rows, expected_seeds=(2001,))


def test_pilot_validation_rejects_seed_episode_aliases_and_early_terminal_rows():
    rows = make_episode()
    rows[1]["episode_id"] = "different-episode"
    with pytest.raises(ValueError, match="assigned to multiple episodes"):
        matrix.validate_pilot_rows(rows, expected_seeds=(2001,))

    rows = make_episode(actions=(0.1, 0.2, 0.3))
    rows[0]["terminal"] = True
    with pytest.raises(ValueError, match="terminal transition must be the final row"):
        matrix.validate_pilot_rows(rows, expected_seeds=(2001,))


def test_terminal_observation_remains_valid_for_the_kl_sequence_mask():
    rows = make_episode()
    groups, episodes = matrix.prepare_pilot_episodes(
        rows,
        FakeFeatureEncoder(),
        goal_dim=0,
        expected_seeds=(2001,),
    )

    assert [episode_id for episode_id, _ in groups] == ["pilot-2001"]
    assert episodes[0].length == 2
    # The terminal flag describes continuation, not whether the observed
    # terminal state is a valid prediction target in this unpadded episode.
    assert episodes[0].continues == [1.0, 1.0]


def test_action_sequences_keep_explicit_boundaries_and_paired_shuffle_streams():
    actions = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32)
    first_generators = matrix.paired_action_generators(17)
    second_generators = matrix.paired_action_generators(17)

    first = matrix.build_intervened_actions(actions, first_generators)
    repeated = matrix.build_intervened_actions(actions, second_generators)

    assert np.array_equal(first["real"], actions)
    assert np.array_equal(first["zero"], np.zeros_like(actions))
    assert np.allclose(first["lag"].reshape(-1), [0.0, 0.1, 0.2])
    assert np.allclose(first["lead"].reshape(-1), [0.2, 0.3, 0.0])
    assert np.allclose(first["negate"].reshape(-1), [-0.1, -0.2, -0.3])
    assert np.allclose(
        np.sort(first["shuffle"].reshape(-1)),
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )
    assert np.array_equal(first["shuffle"], repeated["shuffle"])

    # Reusing the same generator objects advances one deterministic stream
    # across episodes, matching measure_ablation's episode iteration order.
    next_first = matrix.build_intervened_actions(actions, first_generators)
    next_repeated = matrix.build_intervened_actions(actions, second_generators)
    assert np.array_equal(next_first["shuffle"], next_repeated["shuffle"])


def test_physical_matrix_scores_terminal_target_for_all_six_interventions(monkeypatch):
    rows = make_episode()
    captured = []

    monkeypatch.setattr(
        matrix,
        "prime_episode_state",
        lambda *_args, **_kwargs: SimpleNamespace(step=0),
    )

    def fake_evaluate(_model, state, _current, action, **_kwargs):
        captured.append(float(action[0]))
        return np.asarray([0.0, 0.0], dtype=np.float32), SimpleNamespace(step=state.step + 1)

    monkeypatch.setattr(matrix, "evaluate_one_step", fake_evaluate)
    result = matrix.measure_physical_ablation(
        model=object(),
        groups=[("pilot-2001", rows)],
        encoder=object(),
        seed=5,
    )

    assert len(captured) == len(INTERVENTIONS)
    assert captured[0] == pytest.approx(0.2)  # real
    assert captured[1] == pytest.approx(0.0)  # zero
    assert captured[3] == pytest.approx(0.1)  # one-step lag
    assert captured[4] == pytest.approx(0.0)  # one-step lead boundary
    assert captured[5] == pytest.approx(-0.2)  # sign-reversed
    assert set(result["per_episode"]) == set(INTERVENTIONS)
    assert all("pilot-2001" in result["per_episode"][kind] for kind in INTERVENTIONS)



def test_code_commit_resolution_uses_explicit_environment_or_git_head(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=("a" * 40) + "\\n"),
    )
    assert matrix.resolve_code_commit() == "a" * 40
    assert matrix.resolve_code_commit("B" * 40) == "b" * 40

    monkeypatch.setenv("GITHUB_SHA", "C" * 40)
    assert matrix.resolve_code_commit() == "c" * 40


def test_code_commit_resolution_fails_closed_without_a_full_revision(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="short"),
    )
    with pytest.raises(ValueError, match="full 40-character"):
        matrix.resolve_code_commit()

    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    with pytest.raises(RuntimeError, match="--code-commit"):
        matrix.resolve_code_commit()


def test_run_writes_paired_provenance_without_reading_confirm_or_test(tmp_path, monkeypatch):
    pilot_rows = make_episode()
    train_rows = [{"episode_seed": 1001}]
    groups = [("pilot-2001", pilot_rows)]
    episodes = [SimpleNamespace(length=2)]
    checkpoint = {
        "config_dict": {},
        "enc_schema": {},
        "model_state_dict": {},
    }
    calls = []

    class FakeConfig:
        def __init__(self):
            self.obs = object()
            self.model = SimpleNamespace(action_dim=1, discrete_dim=0, goal_dim=0)

    class FakeEncoder:
        input_dim = 3

        def load_schema(self, _schema):
            return None

    class FakeModel:
        def __init__(self, *_args):
            self.weight = torch.tensor([1.0])

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

        def state_dict(self):
            return {"weight": self.weight}

    def load_pilot(path, split, seeds, expected_sha256=None):
        calls.append(("pilot", path, split))
        assert path == matrix.PILOT_PATH
        assert split == "pilot"
        assert tuple(seeds) == matrix.PILOT_SEEDS
        assert expected_sha256 == matrix.PILOT_HASH
        return pilot_rows, "a" * 64

    def load_train(path):
        calls.append(("train", path))
        assert path == matrix.TRAIN_PATH
        return train_rows, "b" * 64

    monkeypatch.setattr(matrix, "load_rows_snapshot", load_pilot)
    monkeypatch.setattr(matrix, "load_train_jsonl_snapshot", load_train)
    monkeypatch.setattr(
        matrix,
        "load_frozen_checkpoint",
        lambda _path: (checkpoint, "c" * 64),
    )
    monkeypatch.setattr(
        matrix,
        "validate_checkpoint_training_provenance",
        lambda _checkpoint, digest, _rows: calls.append(("provenance", digest)),
    )
    monkeypatch.setattr(matrix.Config, "from_dict", lambda _payload: FakeConfig())
    monkeypatch.setattr(matrix, "ObsEncoder", lambda _config: FakeEncoder())
    monkeypatch.setattr(matrix, "SoulModel", lambda *_args: FakeModel())
    monkeypatch.setattr(
        matrix,
        "prepare_pilot_episodes",
        lambda *_args, **_kwargs: (groups, episodes),
    )

    fake_result = {
        "series": {kind: [float(index + 1)] for index, kind in enumerate(INTERVENTIONS)},
        "physical_series": {kind: [float(index + 2)] for index, kind in enumerate(INTERVENTIONS)},
        "episode_indices": [0],
        "valid_transitions": 1,
    }
    monkeypatch.setattr(matrix, "measure_ablation", lambda *_args, **_kwargs: fake_result)
    monkeypatch.setattr(
        matrix,
        "measure_physical_ablation",
        lambda *_args, **_kwargs: {
            "per_episode": {},
            "aggregate_equal_episode": {},
            "paired_effects_vs_real": {},
        },
    )

    report_path = tmp_path / "pilot-action-matrix.json"
    payload = matrix.run(
        checkpoint_path=tmp_path / "frozen-checkpoint.pt",
        report_path=report_path,
        seed=23,
    )
    written = json.loads(report_path.read_text(encoding="utf-8"))

    assert calls == [
        ("pilot", matrix.PILOT_PATH, "pilot"),
        ("train", matrix.TRAIN_PATH),
        ("provenance", "b" * 64),
    ]
    assert payload == written
    assert written["protocol"] == "DRAFT_NOT_LOCKED"
    assert written["threshold"] == "TO_LOCK"
    assert written["decision"] == "NO_DECISION"
    assert written["access"]["confirm_read"] is False
    assert written["access"]["test_read"] is False
    assert written["training"]["performed"] is False
    assert written["training"]["optimizer_updates"] == 0
    assert written["provenance"]["checkpoint_sha256"] == "c" * 64
    assert written["provenance"]["train_sha256"] == "b" * 64
    assert written["provenance"]["code_commit"] == "d" * 40
    assert written["provenance"]["pilot_sha256"] == "a" * 64
    assert written["episode_order"] == ["pilot-2001"]
    assert written["valid_transitions"] == 1
