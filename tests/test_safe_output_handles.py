"""Output helpers refuse links and path swaps."""

import os
from pathlib import Path

import pytest

from soul import safe_io
from soul.safe_io import open_output


def test_write_output_creates_a_regular_file(tmp_path):
    target = tmp_path / 'report.json'
    with open_output(target, append=False) as stream:
        stream.write('first\n')
    with open_output(target, append=False) as stream:
        stream.write('second\n')
    assert target.read_text(encoding='utf-8') == 'second\n'
    assert not target.is_symlink()


def test_write_output_refuses_a_static_symlink(tmp_path):
    victim = tmp_path / 'victim.json'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'report.json'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        pytest.skip(f'symlink creation is unavailable: {error}')
    with pytest.raises(ValueError, match='regular non-link'):
        with open_output(alias, append=False):
            pass
    assert victim.read_text(encoding='utf-8') == 'keep\n'


def test_write_output_rejects_symlink_swap_between_check_and_open(tmp_path, monkeypatch):
    target = tmp_path / 'report.json'
    moved = tmp_path / 'report-original.json'
    victim = tmp_path / 'victim.json'
    target.write_text('existing\n', encoding='utf-8')
    victim.write_text('keep\n', encoding='utf-8')
    real_open = os.open

    def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
        os.rename(target, moved)
        try:
            target.symlink_to(victim)
        except OSError as error:
            pytest.skip(f'symlink creation is unavailable: {error}')
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_io.os, 'open', replace_before_open)
    with pytest.raises(ValueError, match='could not be opened safely|not a regular file|changed while'):
        with open_output(target, append=False):
            pass
    assert moved.read_text(encoding='utf-8') == 'existing\n'
    assert victim.read_text(encoding='utf-8') == 'keep\n'


def test_binary_output_writes_bytes(tmp_path):
    target = tmp_path / 'checkpoint.pt'
    with open_output(target, append=False, binary=True) as stream:
        stream.write(b'bytes')
    assert target.read_bytes() == b'bytes'