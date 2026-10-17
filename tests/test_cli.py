import subprocess
import sys

import pytest
import torch

from geo_ood.cli import dataset_loading


def test_help_is_lightweight():
    p = subprocess.run(
        [sys.executable, "-m", "geo_ood", "train", "--help"], capture_output=True, text=True
    )
    assert p.returncode == 0
    assert "--use_occ" in p.stdout and "--output" in p.stdout


def test_import_does_not_train_or_parse_arguments():
    p = subprocess.run(
        [sys.executable, "-c", "import geo_ood.main, geo_ood.discuss"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    assert "Epoch:" not in p.stdout


def test_torch_loader_restored_after_failure():
    original = torch.load
    with pytest.raises(RuntimeError), dataset_loading():
        assert torch.load is not original
        raise RuntimeError("test")
    assert torch.load is original


@pytest.mark.parametrize("mode", ["detect", "classify"])
def test_cli_writes_artifacts_and_restores_process_state(tmp_path, monkeypatch, mode):
    import json
    from pathlib import Path
    import geo_ood.main as training
    from geo_ood.cli import run
    from test_model import graph

    monkeypatch.setattr(training, "load_dataset", lambda args: (graph(), graph(), graph()))
    output = tmp_path / mode
    cwd = Path.cwd()
    loader = torch.load
    run(
        "train",
        [
            "--cpu",
            "--epochs",
            "2",
            "--runs",
            "2",
            "--method",
            "gnnsafe",
            "--use_occ",
            "--mode",
            mode,
            "--output",
            str(output),
        ],
    )
    assert Path.cwd() == cwd and torch.load is loader
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(metrics["values"]) == 2
    assert (output / "run.log").is_file() and (output / "config.json").is_file()
    if mode == "detect":
        assert len(metrics["values"][0]) == 4
    else:
        assert isinstance(metrics["values"][0], float)
