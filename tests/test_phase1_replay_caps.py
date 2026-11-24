"""Phase 1 replay traces are bounded before they are materialized."""

import json

import pytest

from scripts import phase1_replay_compare


def test_load_replay_rejects_oversized_trace_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(phase1_replay_compare, 'MAX_TRACE_BYTES', 32)
    path = tmp_path / 'trace.jsonl'
    path.write_text(json.dumps({'a': 'x' * 64}) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='too large'):
        phase1_replay_compare._load_replay(path, 'defend')


def test_load_replay_rejects_too_many_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(phase1_replay_compare, 'MAX_TRACE_ROWS', 2)
    path = tmp_path / 'trace.jsonl'
    path.write_text('{}\n{}\n{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='too many rows'):
        phase1_replay_compare._load_replay(path, 'defend')