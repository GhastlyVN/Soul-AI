"""G3.1 Pendulum contract and TRAIN-only JSONL loader."""
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Dict, List

import numpy as np
from soul.memory import Episode
from soul.obs import ObsEncoder

PENDULUM_ENV_ID = 'Pendulum-v1'
PENDULUM_ACTION_DIM = 1
PENDULUM_ACTION_BOUNDS = (-2.0, 2.0)
PENDULUM_OBS_SHAPE = (3,)
TRAIN_SEEDS = frozenset(range(1001, 1006))


def validate_pendulum_action(value, *, field='Pendulum applied action'):
    """Validate the scalar action before float32 conversion can round it into bounds.

    JSON booleans and numeric strings are not action numbers, even though NumPy can coerce them.
    """
    raw = np.asarray(value, dtype=object)
    if raw.shape != (PENDULUM_ACTION_DIM,):
        raise ValueError(f'{field} must have shape (1,)')
    scalar = raw[0]
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f'{field} must contain a numeric value')
    try:
        numeric = float(scalar)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f'{field} must contain a finite numeric value') from error
    if not math.isfinite(numeric):
        raise ValueError(f'{field} must be finite')
    if numeric < PENDULUM_ACTION_BOUNDS[0] or numeric > PENDULUM_ACTION_BOUNDS[1]:
        raise ValueError(f'{field} outside {PENDULUM_ACTION_BOUNDS}')
    action = np.asarray([numeric], dtype=np.float32)
    if not np.isfinite(action).all():
        raise ValueError(f'{field} cannot be represented as finite float32')
    return action


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
    identity = (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns,
    )
    # Windows st_ctime is creation time, not a reliable mutation timestamp.
    if os.name != 'nt':
        return (*identity, metadata.st_ctime_ns)
    return identity


def _is_link_or_reparse_point(metadata):
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, 'st_file_attributes', 0) & 0x400
    )


def read_file_snapshot(path, max_bytes):
    """Read one bounded regular-file snapshot without following links or blocking on FIFOs."""
    source = Path(path)
    try:
        path_before = source.lstat()
    except OSError as error:
        raise ValueError(f'dataset source is unavailable: {source}') from error
    if _is_link_or_reparse_point(path_before) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f'dataset source must be a regular non-symlink file: {source}')

    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f'dataset source could not be opened safely: {source}: {error}') from error

    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if (
            _is_link_or_reparse_point(before)
            or not stat.S_ISREG(before.st_mode)
            or _file_identity(before) != _file_identity(path_before)
        ):
            raise RuntimeError(f'dataset changed before it could be read: {source}')
        try:
            path_after_open = source.lstat()
        except OSError as error:
            raise RuntimeError(f'dataset path changed while it was being opened: {source}') from error
        if (
            _is_link_or_reparse_point(path_after_open)
            or not stat.S_ISREG(path_after_open.st_mode)
            or _file_identity(path_after_open) != _file_identity(before)
        ):
            raise RuntimeError(f'dataset path changed while it was being opened: {source}')
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
            or _is_link_or_reparse_point(path_after)
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise RuntimeError(f'dataset changed while it was being read: {source}')
    return raw, hashlib.sha256(raw).hexdigest()


def _observation_vector(value, source: Path, line_no: int):
    """Validate raw JSON numbers before NumPy can coerce booleans or numeric strings."""
    if not isinstance(value, list) or len(value) != PENDULUM_OBS_SHAPE[0]:
        raise ValueError(f'bad observation shape at {source}:{line_no}')
    if any(
        isinstance(component, bool) or not isinstance(component, (int, float))
        for component in value
    ):
        raise ValueError(f'invalid observation value at {source}:{line_no}')
    try:
        observation = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f'invalid observation value at {source}:{line_no}') from error
    if not np.isfinite(observation).all():
        raise ValueError(f'nonfinite observation at {source}:{line_no}')
    return observation


def load_train_jsonl_snapshot(path: str):
    """Parse and hash one bounded TRAIN snapshot; the digest describes these exact rows."""
    source = Path(path)
    raw, digest = read_file_snapshot(source, MAX_TRAIN_JSONL_BYTES)
    rows = []
    episode_for_seed = {}
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
        previous_episode = episode_for_seed.setdefault(episode_seed, episode_id)
        if previous_episode != episode_id:
            raise ValueError(
                f'TRAIN seed {episode_seed} is assigned to multiple episodes: '
                f'{previous_episode!r} and {episode_id!r}'
            )
        timestep = row.get('timestep')
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError(f'invalid timestep at {source}:{line_no}')
        if not isinstance(row.get('terminal'), bool):
            raise ValueError(f'terminal must be a boolean at {source}:{line_no}')
        before = _observation_vector(row['raw_observation_before'], source, line_no)
        after = _observation_vector(row['raw_observation_after'], source, line_no)
        try:
            action = validate_pendulum_action(row['applied_action'], field='applied action')
        except ValueError as error:
            if 'shape' in str(error):
                raise ValueError(f'bad applied action shape at {source}:{line_no}') from error
            raise ValueError(f'{error} at {source}:{line_no}') from error
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError(f'nonfinite observation at {source}:{line_no}')
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
    grouped: dict[str, list[Dict]] = {}
    episode_seeds: dict[str, int] = {}
    seed_episodes: dict[int, str] = {}
    for row in rows:
        episode_id = row['episode_id']
        seed = row['episode_seed']
        previous_episode = seed_episodes.setdefault(seed, episode_id)
        if previous_episode != episode_id:
            raise ValueError(
                f'TRAIN seed {seed} is assigned to multiple episodes: '
                f'{previous_episode!r} and {episode_id!r}'
            )
        previous_seed = episode_seeds.setdefault(episode_id, seed)
        if previous_seed != seed:
            raise ValueError(
                f'episode {episode_id}: episode_seed changes within episode'
            )
        grouped.setdefault(episode_id, []).append(row)

    episodes = []
    for episode_id, episode_rows in grouped.items():
        ordered = sorted(episode_rows, key=lambda row: row['timestep'])
        timesteps = [row['timestep'] for row in ordered]
        if timesteps != list(range(len(ordered))):
            raise ValueError('episode timesteps must be contiguous from zero')
        if any(row['terminal'] for row in ordered[:-1]):
            raise ValueError(
                f'episode {episode_id}: terminal transition must be the final timestep'
            )
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
