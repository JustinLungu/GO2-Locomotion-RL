# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from dataclasses import dataclass

from IPython.display import Markdown, display

LOG_ROOT = Path("../logs")


@dataclass
class Run:
    path: Path
    experiment: str
    variant: str
    created: str

    @property
    def checkpoint(self) -> Path:
        checkpoints = list(self.path.glob("model_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {self.path}")
        return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))

    def eval_files(self) -> list[Path]:
        return sorted((self.path / "eval").glob("*.npz"))

    def eval_file(self, name: str) -> Path:
        path = self.path / "eval" / f"{name}_{self.checkpoint.stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No evaluation {name!r} for {self.checkpoint.name} in {self.path}"
            )
        return path


def create_run(run_dir: str | Path) -> Run:
    return Run(
        path=run_dir.resolve(),
        experiment=run_dir.parent.name,
        variant="_".join(run_dir.name.split("_")[2:]),
        created="_".join(run_dir.name.split("_")[:2]),
    )


def runs(
    experiment: str | None = None,
    variant: str | None = None,
) -> list[Run]:
    found = []

    for exp_dir in sorted(Path(LOG_ROOT).iterdir()):
        for var_dir in sorted(exp_dir.iterdir()):
            if var_dir.is_dir() and (var_dir / "params").exists():
                run = create_run(var_dir)
                found.append(run)

    # Filter experiments
    if experiment is not None:
        found = [run for run in found if run.experiment == experiment]

    # Filter variants
    if variant is not None:
        found = [run for run in found if run.variant == variant]

    return found


def show(runs: list[Run]) -> None:
    lines = ["| experiment | variant | created |", "| --- | --- | --- |"]
    for run in runs:
        lines.append(f"| {run.experiment} | {run.variant} | {run.created} |")
    display(Markdown("\n".join(lines)))
