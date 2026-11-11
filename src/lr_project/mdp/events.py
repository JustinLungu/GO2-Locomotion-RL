# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def resolve_env_ids(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    return env_ids


def _sample_se3_range(
    range_dict: dict[str, tuple[float, float]] | None,
    shape: tuple[int, ...],
    device: str,
) -> torch.Tensor:
    range_dict = range_dict or {}
    range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
    ranges = torch.tensor(range_list, device=device)
    return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def reset_root_pose_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    env_ids = resolve_env_ids(env, env_ids)

    asset: Entity = env.scene[asset_cfg.name]

    # Pose sampling
    pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    positions = (
        root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )

    velocities = root_states[:, 7:13]
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)

    # Call forward so the data is up to date
    env.sim.forward()


def reset_joints_by_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    env_ids = resolve_env_ids(env, env_ids)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
