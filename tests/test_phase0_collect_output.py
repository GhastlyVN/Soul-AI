"""The Phase 0 collector writes only to a verified output handle."""

import pytest

from scripts import phase0_collect


def test_collect_refuses_symlinked_output(tmp_path):
    victim = tmp_path / 'victim.jsonl'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'train.jsonl'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        pytest.skip(f'symlink creation is unavailable: {error}')
    with pytest.raises(ValueError, match='regular non-link'):
        phase0_collect.collect(alias, [0], 1)
    assert victim.read_text(encoding='utf-8') == 'keep\n'