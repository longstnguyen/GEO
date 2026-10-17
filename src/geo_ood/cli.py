"""Command dispatch and run artifacts around the GEO experiment pipelines."""

import argparse
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def dataset_loading():
    """Preserve the source loader's PyTorch compatibility, scoped to this run."""
    import torch

    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def run(command, argv):
    from .parse import parser_add_main_args

    parser = argparse.ArgumentParser(prog=f"geo {command}")
    parser_add_main_args(parser)
    if command == "discuss":
        parser.add_argument(
            "--dis_type",
            default="margin",
            choices=["margin", "lamda", "prop", "backbone", "time", "vis_energy"],
        )
    settings = parser.parse_args(argv)
    if min(settings.epochs, settings.runs, settings.display_step) < 1:
        parser.error("epochs, runs and display_step must be positive")
    if settings.use_occ and not 0 < settings.nu <= 1:
        parser.error("nu must lie in (0, 1]")
    if command == "discuss" and settings.mode != "detect":
        parser.error("diagnostic studies require --mode detect")
    if settings.T <= 0:
        parser.error("T must be positive")
    output = Path(
        settings.output or ("outputs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    ).resolve()
    data = str(Path(settings.data_dir).resolve()) + os.sep
    output.mkdir(parents=True, exist_ok=False)
    (output / "results").mkdir()
    (output / "config.json").write_text(json.dumps(vars(settings), indent=2) + "\n")
    with dataset_loading():
        from . import discuss, main as training
        import torch
        import torch_geometric

        environment = {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "torch_geometric": torch_geometric.__version__,
            "cuda": torch.version.cuda,
            "data_dir": data,
            "command": command,
        }
        (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        previous = Path.cwd()
        try:
            os.chdir(output)
            with (output / "run.log").open("w") as log, redirect_stdout(Tee(sys.stdout, log)):
                execute = training.execute if command == "train" else discuss.execute
                args, results = execute([*argv, "--data_dir", data])
                values = results.detach().cpu().tolist()
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "mode": args.mode,
                            "values": values,
                            "layout": (
                                "triples of AUROC, AUPR, FPR95 for each OOD cohort, "
                                "then ID test score; percentages"
                            )
                            if args.mode == "detect"
                            else "validation-selected test classification score per run; percentages",
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
        finally:
            os.chdir(previous)
    print(f"Artifacts: {output}")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"train", "discuss"}:
        return run(arguments[0], arguments[1:])
    parser = argparse.ArgumentParser(prog="geo", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=["train", "discuss"])
    parser.parse_args(arguments)
