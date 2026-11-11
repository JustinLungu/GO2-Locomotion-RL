# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, fields, field


@dataclass(kw_only=True)
class Rollout:
    """Everything logged during a rollout, as numpy arrays.
    N robots, T logged steps (after warm-up).
    """

    dt: float  # policy timestep (s)
    cmd: np.ndarray  # (N, 3) command of each robot (vx, vy, wz)
    alive: np.ndarray  # (N,) True if robot never fell
    mass: np.ndarray  # (N,) robot mass including randomizations (kg)
    base_lin_vel: np.ndarray  # (N, T, 3) base linear velocity, body frame (m/s)
    base_ang_vel: np.ndarray  # (N, T, 3) base angular velocity, body frame (rad/s)
    base_pos: np.ndarray  # (N, T, 3) base position, world frame (m)
    base_quat: np.ndarray  # (N, T, 4) base orientation, world frame (quaternion)
    joint_pos: np.ndarray  # (N, T, 12) joint positions (rad)
    joint_vel: np.ndarray  # (N, T, 12) joint velocities (rad/s)
    joint_torque: np.ndarray  # (N, T, 12) joint torques (N m)
    action: np.ndarray  # (N, T, 12) policy actions
    foot_contact: (
        np.ndarray
    )  # (N, T, 4) True if foot touches the ground (FL, FR, RL, RR)
    foot_height: np.ndarray  # (N, T, 4) foot centre height, world frame (m)
    foot_grf: (
        np.ndarray
    )  # (N, T, 4, 3), ground reaction forces on each foot, world frame (N)
    meta: dict = field(
        default_factory=dict
    )  # evaluation settings (checkpoint, commands, ...)

    @property
    def num_robots(self) -> int:
        return self.cmd.shape[0]

    def select(self, mask: np.ndarray) -> Rollout:
        """Keep a subset of robots, e.g., rollout.select(rollout.alive)."""
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            per_robot = isinstance(v, np.ndarray) and v.shape[0] == self.num_robots
            values[f.name] = v[mask] if per_robot else v
        return Rollout(**values)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            f.name: np.asarray(getattr(self, f.name))
            for f in fields(self)
            if f.name != "meta"
        }
        np.savez_compressed(path, **arrays, meta=np.array(json.dumps(self.meta)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Rollout:
        with np.load(path, allow_pickle=False) as data:
            values = {k: data[k] for k in data.files if k != "meta"}
            values["dt"] = float(values["dt"])
            values["meta"] = json.loads(str(data["meta"]))
        return cls(**values)
