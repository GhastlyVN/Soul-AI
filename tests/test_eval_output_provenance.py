"""Evaluation output paths must not follow links or race with path replacement."""
import argparse
import os
from pathlib import Path

import pytest

from soul.eval import _open_append_output, _output_path


def test_folder_output_is_refused(tmp_path):
    parser = argparse.ArgumentParser()
    target = tmp_path / 'eval.jsonl'
    target.mkdir()

    with pytest.raises(SystemExit):
        _output_path(parser, str(target))


def test_regular_output_is_accepted(tmp_path):
    parser = argparse.ArgumentParser()
    target = tmp_path / 'eval.jsonl'

    assert _output_path(parser, str(target)) == target


def test_append_output_creates_a_new_regular_file(tmp_path):
    target = tmp_path / 'eval.jsonl'

    with _open_append_output(target) as stream:
        stream.write('record\n')

    assert target.read_text(encoding='utf-8') == 'record\n'
    assert target.is_file()
    assert not target.is_symlink()


def test_append_output_refuses_a_static_symlink(tmp_path):
    victim = tmp_path / 'victim.jsonl'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'eval.jsonl'
    try:
        alias.symlink_to(victim)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f'symlink creation is unavailable: {error}')

    with pytest.raises(ValueError, match='regular non-link file'):
        with _open_append_output(alias):
            pass

    assert victim.read_text(encoding='utf-8') == 'keep\n'


def test_append_output_rejects_symlink_swap_between_check_and_open(tmp_path, monkeypatch):
    target = tmp_path / 'eval.jsonl'
    moved = tmp_path / 'eval-original.jsonl'
    victim = tmp_path / 'victim.jsonl'
    target.write_text('existing\n', encoding='utf-8')
    victim.write_text('keep\n', encoding='utf-8')
    real_open = os.open
    swapped = False

    def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if Path(path) == target and not swapped:
            target.replace(moved)
            try:
                target.symlink_to(victim)
            except (OSError, NotImplementedError) as error:
                pytest.skip(f'symlink creation is unavailable: {error}')
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, 'open', replace_before_open)
    with pytest.raises(ValueError, match='safely|changed while it was being opened'):
        with _open_append_output(target) as stream:
            stream.write('must not reach victim\n')

    assert swapped
    assert moved.read_text(encoding='utf-8') == 'existing\n'
    assert victim.read_text(encoding='utf-8') == 'keep\n'
