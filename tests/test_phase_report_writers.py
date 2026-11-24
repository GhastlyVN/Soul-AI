"""Phase 0/1 report writers must not use path-based writes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WRITERS = (
    'scripts/phase0_alpha_run.py',
    'scripts/phase1_compressed_skill.py',
    'scripts/phase1_distill_skill.py',
    'scripts/phase1_mapped_skill.py',
    'scripts/phase1_skill_smoke.py',
    'scripts/phase1_skill_sweep.py',
)


def test_report_writers_use_verified_handles():
    for name in WRITERS:
        text = (ROOT / name).read_text(encoding='utf-8')
        assert '.write_text(' not in text, name
        assert ".open('w'" not in text, name


def test_write_text_safe_refuses_a_symlink(tmp_path):
    from soul.safe_io import write_text_safe

    victim = tmp_path / 'victim.json'
    victim.write_text('keep\n', encoding='utf-8')
    alias = tmp_path / 'report.json'
    try:
        alias.symlink_to(victim)
    except OSError as error:
        import pytest
        pytest.skip(f'symlink creation is unavailable: {error}')
    try:
        write_text_safe(alias, 'overwritten\n')
    except ValueError:
        pass
    else:
        raise AssertionError('write_text_safe followed a symlink')
    assert victim.read_text(encoding='utf-8') == 'keep\n'