# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import re
import numpy as np
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_training_logs(
    run_dir: str | Path, patterns: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    logs = {}
    for tag in accumulator.Tags()["scalars"]:
        if any(re.fullmatch(p, tag) for p in patterns):
            events = sorted(accumulator.Scalars(tag), key=lambda e: e.step)
            logs[tag] = (
                np.array([e.step for e in events]),
                np.array([e.value for e in events]),
            )
    return logs
