# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor, ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#####################################
# JOINT STATES
#####################################


def joint_positions(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    jnt_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos
    joint_pos_rel = joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]
    return joint_pos_rel


def joint_velocities(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    joint_vel = asset.data.joint_vel[:, jnt_ids]
    return joint_vel


def joint_accelerations(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    joint_acc = asset.data.joint_acc[:, jnt_ids]
    return joint_acc


def joint_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    joint_tau = asset.data.qfrc_actuator[:, jnt_ids]
    return joint_tau


#####################################
# BASE STATES (w = world frame, b = base frame)
#####################################


def base_pos_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w


def base_quat_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_quat_w


def base_lin_vel_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


def base_ang_vel_w(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


def base_lin_vel_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def base_ang_vel_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def projected_gravity_b(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


#####################################
# POLICY STATES
#####################################


def last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.action


def last_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.prev_action


def last_last_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.prev_prev_action


#####################################
# COMMANDS
#####################################


def generated_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    assert type(command) is not None
    return command


#####################################
# EXTEROCEPTION
#####################################


def height_scan(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    sensor: RayCastSensor = env.scene[sensor_name]
    miss_value = sensor.cfg.max_distance

    data = sensor.data
    F, N = sensor.num_frames, sensor.num_rays_per_frame
    B = data.distances.shape[0]

    frame_z = data.frame_pos_w[:, :, 2:3]  # [B, F, 1]
    hit_z = data.hit_pos_w[..., 2].view(B, F, N)  # [B, F, N]
    heights = (frame_z - hit_z).view(B, F * N)

    miss_mask = data.distances < 0
    return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)


#####################################
# FEET
#####################################


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


#####################################
# BUILTIN SENSORS
#####################################


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    return sensor.data


def builtin_sensors(
    env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    return torch.cat([env.scene[name].data for name in sensor_names], dim=-1)
