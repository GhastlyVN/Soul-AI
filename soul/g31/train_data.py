"""G3.1 Pendulum contract and TRAIN-only JSONL loader."""
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Dict, List

import numpy as np
from soul.memory import Episode
from soul.obs import ObsEncoder

PENDULUM_ENV_ID = 'Pendulum-v1'
PENDULUM_ACTION_DIM = 1
PENDULUM_OBS_SHAPE = (3,)
TRAIN_SEEDS = frozenset(range(1001, 1006))


def pendulum_model_config(cfg):
    """Return a copy-like config configured for the scalar Pendulum action."""
    cfg.model.action_dim = PENDULUM_ACTION_DIM
    cfg.model.discrete_dim = 0
    return cfg


MAX_TRAIN_JSONL_BYTES = 4 * 1024 * 1024
MAX_TRAIN_JSONL_ROWS = 10_000


def _reject_duplicate_json_keys(pairs):
    """Reject ambiguous JSON objects instead of silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result


def parse_jsonl_object(line: str, source: Path, line_no: int) -> Dict:
    """Parse one JSONL object and reject duplicate keys at every nesting level."""
    row = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(row, dict):
        raise ValueError(f'row must be a JSON object at {source}:{line_no}')
    return row



def _file_identity(metadata):
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


def read_file_snapshot(path, max_bytes):
    """Read one bounded regular-file snapshot and hash the exact bytes returned."""
    source = Path(path)
    path_before = source.lstat()
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f'dataset source must be a regular non-symlink file: {source}')
    with source.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(path_before):
            raise RuntimeError(f'dataset changed before it could be read: {source}')
        if before.st_size > max_bytes:
            raise ValueError(f'dataset exceeds the maximum snapshot size of {max_bytes} bytes: {source}')
        raw = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > max_bytes:
        raise ValueError(f'dataset exceeds the maximum snapshot size of {max_bytes} bytes: {source}')
    try:
        path_after = source.lstat()
    except OSError as error:
        raise RuntimeError(f'dataset path changed while it was being read: {source}') from error
    if (
        len(raw) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise RuntimeError(f'dataset changed while it was being read: {source}')
    return raw, hashlib.sha256(raw).hexdigest()


def load_train_jsonl_snapshot(path: str):
    """Parse and hash one bounded TRAIN snapshot; the digest describes these exact rows."""
    source = Path(path)
    raw, digest = read_file_snapshot(source, MAX_TRAIN_JSONL_BYTES)
    rows = []
    for line_no, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        row = parse_jsonl_object(line, source, line_no)
        if row.get('split') != 'train':
            raise ValueError(f'non-TRAIN row at {source}:{line_no}')
        episode_seed = row.get('episode_seed')
        if (
            isinstance(episode_seed, bool)
            or not isinstance(episode_seed, int)
            or episode_seed not in TRAIN_SEEDS
        ):
            raise ValueError(f'non-TRAIN seed at {source}:{line_no}')
        episode_id = row.get('episode_id')
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError(f'invalid episode_id at {source}:{line_no}')
        timestep = row.get('timestep')
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError(f'invalid timestep at {source}:{line_no}')
        if not isinstance(row.get('terminal'), bool):
            raise ValueError(f'terminal must be a boolean at {source}:{line_no}')
        before = np.asarray(row['raw_observation_before'], dtype=np.float32)
        after = np.asarray(row['raw_observation_after'], dtype=np.float32)
        action = np.asarray(row['applied_action'], dtype=np.float32)
        if before.shape != PENDULUM_OBS_SHAPE or after.shape != PENDULUM_OBS_SHAPE:
            raise ValueError(f'bad observation shape at {source}:{line_no}')
        if action.shape != (PENDULUM_ACTION_DIM,):
            raise ValueError(f'bad applied action shape at {source}:{line_no}')
        if not np.isfinite(before).all() or not np.isfinite(after).all() or not np.isfinite(action).all():
            raise ValueError(f'nonfinite observation or action at {source}:{line_no}')
        if 'reward' not in row:
            raise ValueError(f'missing reward at {source}:{line_no}')
        reward_value = row['reward']
        if isinstance(reward_value, bool) or not isinstance(reward_value, (int, float)):
            raise ValueError(f'invalid reward at {source}:{line_no}')
        reward = float(reward_value)
        if not np.isfinite(reward):
            raise ValueError(f'nonfinite reward at {source}:{line_no}')
        environment = row.get('environment_config')
        if not isinstance(environment, dict) or environment.get('env_id') != PENDULUM_ENV_ID:
            raise ValueError(f'wrong environment at {source}:{line_no}')
        rows.append(row)
        if len(rows) > MAX_TRAIN_JSONL_ROWS:
            raise ValueError(f'TRAIN JSONL exceeds the row limit of {MAX_TRAIN_JSONL_ROWS}: {source}')
    if not rows:
        raise ValueError(f'empty TRAIN JSONL: {source}')
    return rows, digest


def load_train_jsonl(path: str) -> List[Dict]:
    """Compatibility wrapper returning validated TRAIN rows from one snapshot."""
    rows, _ = load_train_jsonl_snapshot(path)
    return rows


def rows_to_episodes(rows: List[Dict], encoder: ObsEncoder,
                     goal_dim: int) -> List[Episode]:
    """Convert validated TRAIN rows without joining episode boundaries."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['episode_id'], []).append(row)
    episodes = []
    for episode_rows in grouped.values():
        ordered = sorted(episode_rows, key=lambda row: row['timestep'])
        if [row['timestep'] for row in ordered] != list(range(len(ordered))):
            raise ValueError('episode timesteps must be contiguous from zero')
        for previous, current in zip(ordered, ordered[1:]):
            previous_after = np.asarray(previous['raw_observation_after'], dtype=np.float32)
            current_before = np.asarray(current['raw_observation_before'], dtype=np.float32)
            if (
                previous_after.shape != PENDULUM_OBS_SHAPE
                or current_before.shape != PENDULUM_OBS_SHAPE
                or not np.allclose(previous_after, current_before, rtol=0.0, atol=1e-6)
            ):
                raise ValueError(
                    f'episode {current["episode_id"]}: observation discontinuity '
                    f'before timestep {current["timestep"]}'
                )
        episode = Episode()
        for row in ordered:
            encoded = encoder.encode(row['raw_observation_after'])
            episode.append(
                encoded,
                np.asarray(row['applied_action'], dtype=np.float32),
                float(row['reward']),
                0.0 if row['terminal'] else 1.0,
                np.zeros(goal_dim, dtype=np.float32),
            )
        episodes.append(episode)
    return episodes
