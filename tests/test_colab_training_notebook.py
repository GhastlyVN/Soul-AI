from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "SOUL_General_Training_Lab_90m.ipynb"
)


def _notebook_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    return notebook["cells"]


def test_general_training_notebook_code_cells_parse_as_python():
    cells = _notebook_cells()
    code_cells = [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
    ]
    assert code_cells
    for index, source in code_cells:
        if source.strip():
            ast.parse(source, filename=f"{NOTEBOOK_PATH}#cell-{index}")


def test_general_training_notebook_fails_closed_at_checkout_and_checkpoint_boundaries():
    cells = _notebook_cells()
    sources = {
        index: "".join(cell.get("source", []))
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
    }

    checkout = sources[2]
    assert "Refusing to replace an existing checkout" in checkout
    assert "shutil.rmtree(REPO_DIR)" not in checkout

    training = sources[7]
    assert "timeout=int(train_budget)" in training
    assert "refusing to force a 500-update minimum" in training

    checkpoint = sources[9]
    assert checkpoint.index("checkpoint_snapshot(") < checkpoint.index("torch.load(")
    assert "checkpoint_snapshot_path.unlink(missing_ok=True)" in checkpoint

    memory = sources[9]
    assert "serialized_sha256_before" in memory
    assert "serialized_sha256_after" in memory
