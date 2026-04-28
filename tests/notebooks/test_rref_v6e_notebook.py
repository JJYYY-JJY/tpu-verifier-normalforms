import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = Path("notebooks/rref_v6e_smoke_training.ipynb")


def _load_notebook() -> dict[str, Any]:
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _source_text(notebook: dict[str, Any]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    chunks: list[str] = []
    for cell in cells:
        assert isinstance(cell, dict)
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.extend(str(part) for part in source)
        else:
            chunks.append(str(source))
    return "".join(chunks)


def test_rref_v6e_notebook_has_jupyter_schema() -> None:
    notebook = _load_notebook()

    assert {"cells", "metadata", "nbformat", "nbformat_minor"} <= notebook.keys()
    assert isinstance(notebook["cells"], list)
    assert isinstance(notebook["metadata"], dict)
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert notebook["cells"]


def test_rref_v6e_notebook_mentions_current_cli_contract() -> None:
    source = _source_text(_load_notebook())

    required_snippets = (
        "nf-agent --help",
        "data make-rref-shard",
        "train rref-pivot",
        "rollout rref-neural",
        "benchmark rref",
        "--hidden-size",
        "configs/rref_8x8_mod101.yaml",
        "/tmp/rref_8x8_train_smoke.npz",
        "/tmp/rref_pivot_ckpt",
    )
    for snippet in required_snippets:
        assert snippet in source


def test_rref_v6e_notebook_does_not_encode_fallback_workarounds() -> None:
    source = _source_text(_load_notebook()).lower()

    prohibited_snippets = (
        "except exception",
        "bare except",
        "silent fallback",
        "hidden fallback",
        "deterministic fallback",
        "teacher fallback",
    )
    for snippet in prohibited_snippets:
        assert snippet not in source
