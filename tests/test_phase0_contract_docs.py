"""The Phase 0 contract must show a runnable scorer command."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase0_contract_scorer_example_supplies_checkpoint_digest():
    text = (ROOT / 'docs' / 'PHASE0_CONTRACT.md').read_text(encoding='utf-8')
    commands = [line.strip() for line in text.splitlines()
                if 'scripts.phase0_score' in line]
    assert commands
    for command in commands:
        assert '--checkpoint-sha256' in command
        assert 'results/phase0_train_report.json' in text