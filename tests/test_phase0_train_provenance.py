"""Phase 0 training must persist provenance without reloading its checkpoint."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from scripts import phase0_train


class _Encoder:
    schema = {'kind': 0}
    overflow = 0

    def encode(self, observation):
        return np.zeros(2, dtype=np.float32)


class _Replay:
    def __init__(self):
        self.episodes = []

    def add_episode(self, episode):
        self.episodes.append(episode)


class _Soul:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enc = _Encoder()
        self.replay = _Replay()
        self.updates = 0

    def learn(self, force=False):
        self.updates += 1
        return {'total': 0.0}

    def save(self, path, extra=None):
        assert extra is not None, 'provenance must be passed to save()'
        Path(path).write_text(
            json.dumps(extra, ensure_ascii=False, sort_keys=True) + '\n',
            encoding='utf-8')


def test_phase0_train_attaches_provenance_at_save_time(tmp_path, monkeypatch):
    rows = []
    for tick in (1, 2):
        rows.append({
            'schema_version': 2,
            'split': 'train',
            'seed': 0,
            'episode': 0,
            'tick': tick,
            'observation': {'x': tick - 1},
            'next_observation': {'x': tick},
            'candidate_action': 1,
            'action': 1,
            'reward': 0.0,
            'done': tick == 2,
            'safety': {'blocked': False},
        })
    data = tmp_path / 'train.jsonl'
    data.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    monkeypatch.setattr(phase0_train, 'Soul', _Soul)
    monkeypatch.setattr(
        phase0_train, '_validate_transition_rows',
        lambda rows: OrderedDict({
            (0, 0): [(index, row) for index, row in enumerate(rows)]}))

    def _no_reload(*args, **kwargs):
        raise AssertionError('checkpoint must not be reloaded after save')

    monkeypatch.setattr(phase0_train.torch, 'load', _no_reload)

    checkpoint = tmp_path / 'soul.pt'
    result = phase0_train.train(
        data, checkpoint, updates=1, seq_len=1, batch_size=1)

    saved = json.loads(checkpoint.read_text(encoding='utf-8'))
    assert saved['phase0_provenance']['data_sha256'] == result['data_sha256']
    assert saved['phase0_provenance']['status'] == 'TRAIN_ONLY'
    assert result['checkpoint_sha256'] == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()