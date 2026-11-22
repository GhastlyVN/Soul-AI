"""Evaluation output paths must not point at a folder or a link."""
import argparse

import pytest

from soul.eval import _output_path


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
