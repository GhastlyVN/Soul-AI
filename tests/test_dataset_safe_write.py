"""Sample CSV generation writes through a verified handle."""

import pytest

from soul.dataset import make_sample_csv


def test_make_sample_csv_refuses_symlink(tmp_path):
    victim = tmp_path / 'victim.csv'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'sample.csv'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        pytest.skip(f'symlink creation is unavailable: {error}')
    with pytest.raises(ValueError):
        make_sample_csv(str(alias), n_ep=1, length=2)
    assert victim.read_text(encoding='utf-8') == 'keep\n'


def test_make_sample_csv_still_writes_usable_csv(tmp_path):
    target = tmp_path / 'sample.csv'
    path = make_sample_csv(str(target), n_ep=1, length=2)
    assert path == str(target)
    assert target.read_text(encoding='utf-8').startswith('ep,done,reward')