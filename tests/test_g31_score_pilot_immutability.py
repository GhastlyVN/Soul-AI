"""Behavioural guard for frozen-checkpoint evaluator immutability."""
from types import SimpleNamespace

import pytest
import torch

from soul.g31.score_pilot import verify_model_unchanged


def test_model_immutability_guard_fails_closed_only_on_change():
    before = {'weight': torch.tensor([1.0])}
    unchanged = SimpleNamespace(state_dict=lambda: {'weight': torch.tensor([1.0])})
    changed = SimpleNamespace(state_dict=lambda: {'weight': torch.tensor([2.0])})
    verify_model_unchanged(before, unchanged)
    with pytest.raises(RuntimeError, match='frozen checkpoint model changed') as info:
        verify_model_unchanged(before, changed)
    assert 'weight' in str(info.value)
