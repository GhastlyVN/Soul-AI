"""Core config and memory saves refuse linked or swapped output paths."""

import pytest

from soul.config import Config
from soul.memory import WorkingMemory


def test_config_save_refuses_symlink(tmp_path):
    victim = tmp_path / 'victim.json'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'config.json'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        pytest.skip(f'symlink creation is unavailable: {error}')
    with pytest.raises(ValueError):
        Config().save(str(alias))
    assert victim.read_text(encoding='utf-8') == 'keep\n'


def test_memory_save_refuses_symlink(tmp_path):
    victim = tmp_path / 'victim.json'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'memory.json'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        pytest.skip(f'symlink creation is unavailable: {error}')
    with pytest.raises(ValueError):
        WorkingMemory().save(str(alias))
    assert victim.read_text(encoding='utf-8') == 'keep\n'