import importlib.util
import json
from pathlib import Path
import tarfile


def test_export_is_source_only_and_sanitizes_notebooks(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/export_code.py"
    spec = importlib.util.spec_from_file_location("export_code", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project"
    (root / "crest_pipeline").mkdir(parents=True)
    (root / "crest_pipeline/main.py").write_text("print('source')\n")
    (root / "crest_pipeline/data.npy").write_bytes(b"data")
    (root / "data").mkdir()
    (root / "data/hidden.py").write_text("should not be exported")
    (root / "weights").mkdir()
    (root / "weights/model.pth").write_bytes(b"weights")
    (root / "legacy").mkdir()
    (root / "legacy/example.ipynb").write_text(json.dumps({
        "cells": [{"cell_type": "code", "source": ["print(1)"], "metadata": {},
                   "outputs": [{"data": {"image/png": "IMAGE DATA"}}], "execution_count": 1}],
        "metadata": {"widgets": "DATA"}, "nbformat": 4, "nbformat_minor": 0}))
    archive = tmp_path / "code.tar.gz"
    assert module.export(root, archive) == 2
    with tarfile.open(archive) as handle:
        names = handle.getnames()
        assert set(names) == {"crest-reconstruction/crest_pipeline/main.py", "crest-reconstruction/legacy/example.ipynb"}
        notebook = json.load(handle.extractfile(names[1]))
        assert notebook["metadata"] == {}
        assert notebook["cells"][0]["outputs"] == []
        assert notebook["cells"][0]["source"] == ["print(1)"]
