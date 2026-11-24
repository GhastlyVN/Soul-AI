"""Phase 0 training JSONL is bounded and hashed as it is read."""

import hashlib

import pytest

from scripts import phase0_train


def test_load_rows_hashes_the_exact_bytes(tmp_path):
    path = tmp_path / 'train.jsonl'
    path.write_bytes(b'{"a": 1}\n\n')
    rows, digest = phase0_train._load_rows(path)
    assert rows == [{'a': 1}]
    assert digest == hashlib.sha256(b'{"a": 1}\n\n').hexdigest()


def test_load_rows_rejects_oversized_input(tmp_path, monkeypatch):
    monkeypatch.setattr(phase0_train, 'MAX_DATA_BYTES', 8)
    path = tmp_path / 'train.jsonl'
    path.write_bytes(b'{"a": 1}\n')
    with pytest.raises(ValueError, match='too large'):
        phase0_train._load_rows(path)


def test_load_rows_rejects_too_many_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(phase0_train, 'MAX_DATA_ROWS', 2)
    path = tmp_path / 'train.jsonl'
    path.write_bytes(b'{}\n{}\n{}\n')
    with pytest.raises(ValueError, match='too many rows'):
        phase0_train._load_rows(path)