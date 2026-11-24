"""CI actions must be pinned to immutable commits."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workflow_actions_are_pinned_to_commit_shas():
    text = (ROOT / '.github/workflows' / 'ci.yml').read_text(encoding='utf-8')
    uses = [line.strip() for line in text.splitlines()
            if line.strip().startswith('- uses:')]
    assert uses
    for line in uses:
        reference = line.split('- uses:', 1)[1].strip().split()[0]
        owner, name, revision = reference.split('/')
        assert owner and name
        assert re.fullmatch(r'[0-9a-f]{40}', revision), line