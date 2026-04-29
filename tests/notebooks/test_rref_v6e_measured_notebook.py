import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = Path("notebooks/rref_v6e_measured_run.ipynb")


def _load_notebook() -> dict[str, Any]:
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _source_text(notebook: dict[str, Any]) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        assert isinstance(cell, dict)
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.extend(str(part) for part in source)
        else:
            chunks.append(str(source))
    return "".join(chunks)


def test_rref_v6e_measured_notebook_has_clear_jupyter_schema() -> None:
    notebook = _load_notebook()

    assert {"cells", "metadata", "nbformat", "nbformat_minor"} <= notebook.keys()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_rref_v6e_measured_notebook_runs_large_colab_profile() -> None:
    source = _source_text(_load_notebook())

    required_snippets = (
        "colab-v6e1-large",
        "scripts/rref_measured_run.py",
        "JAX_PLATFORMS",
        "tpu,cpu",
        "/tmp/nf-rref-colab-v6e1-large",
        "/tmp/rref_8x8_mod101_colab_v6e1_large.json",
        "/tmp/rref_8x8_mod101_colab_v6e1_large.md",
        "results/measured/rref_8x8_mod101_colab_v6e1_large.json",
        "results/measured/rref_8x8_mod101_colab_v6e1_large.md",
        "files.download",
        "getpass",
        "remote\", \"set-url\", \"origin\"",
    )
    for snippet in required_snippets:
        assert snippet in source


def test_rref_v6e_measured_notebook_requires_python_312_and_tpu() -> None:
    notebook = _load_notebook()
    source = _source_text(notebook)
    metadata = notebook["metadata"]
    assert isinstance(metadata, dict)
    language_info = metadata["language_info"]
    assert isinstance(language_info, dict)

    assert "sys.version_info[:2] != (3, 12)" in source
    assert "expected TPU backend" in source
    assert language_info["version"] == "3.12"
