# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    return torch.any(data.found, dim=-1)


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return (
        torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > limit_angle
    )


def root_height_below_minimum(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < minimum_height


def terrain_edge_reached(
    env: ManagerBasedRlEnv,
    threshold_fraction: float = 0.95,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    asset: Entity = env.scene[asset_cfg.name]
    displacement = (
        asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ).abs()

    half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
    half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

    at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

    # Don't fire on the first 2 steps after reset (position may be stale).
    at_edge &= env.episode_length_buf > 2

    return at_edge
