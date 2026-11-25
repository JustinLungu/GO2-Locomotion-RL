# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#############################
# TASK
#############################


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity without penalizing z-velocity"""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    return torch.exp(-xy_error / std**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others without penalizing pitch/roll."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-z_error / std**2)


#############################
# PENALTIES
#############################


def body_linear_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_b
    z_error = lin_vel[:, 2]
    return torch.square(z_error)


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_link_ang_vel_b
    ang_vel_xy = ang_vel[:, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    return angmom_magnitude_sq


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1
    )


def joint_power(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    mech = tau * qd
    mech_abs = torch.abs(mech)
    return torch.sum(mech_abs, dim=1)


def sensor_acc_l2(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: BuiltinSensor = env.scene[sensor_name]
    acc = sensor.data.clone()
    acc[:, 2] -= 9.81  # remove gravity
    return torch.sum(torch.square(acc), dim=1)


def sensor_acc_norm(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: BuiltinSensor = env.scene[sensor_name]
    acc = sensor.data.clone()
    acc[:, 2] -= 9.81  # remove gravity
    return torch.norm(acc, dim=1)


def sensors_acc_l2(
    env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    acc = torch.stack([env.scene[name].data for name in sensor_names], dim=1)
    acc = acc.clone()
    acc[..., 2] -= 9.81  # remove gravity
    return torch.sum(torch.square(acc), dim=(1, 2))


def sensors_acc_norm(
    env: ManagerBasedRlEnv, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    acc = torch.stack([env.scene[name].data for name in sensor_names], dim=1)
    acc = acc.clone()
    acc[:, 2] -= 9.81  # remove gravity
    return torch.norm(acc, dim=1)


def contacts_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    return torch.sum(sensor.data.found, dim=1)


def any_contacts_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    return torch.any(sensor.data.found, dim=1).float()


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale

    return reward
