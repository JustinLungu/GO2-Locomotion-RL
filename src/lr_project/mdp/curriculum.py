# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    edge_term_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None
    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # Promote: walked far enough, or got stopped by the terrain edge
    # (which is success: it only fires once the robot has crossed the tile).
    move_up = distance > terrain_generator.size[0] / 2
    if edge_term_name is not None:
        move_up |= env.termination_manager.get_term(edge_term_name)[env_ids]

    # Demote: covered less than half the distance the command asked for
    # *in the time the robot actually had* (episodes can end early).
    elapsed_s = env.episode_length_buf[env_ids].float() * env.step_dt
    required = torch.norm(command[env_ids, :2], dim=1) * elapsed_s * 0.5
    move_down = (distance < required) & ~move_up

    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)

    # Compute per-terrain-type mean levels.
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # In curriculum mode num_cols == num_terrains (one column per type),
    # so the column index directly maps to the sub-terrain name.
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result
