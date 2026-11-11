# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from lr_project.rollout import Rollout

import torch
import copy
import itertools
from tqdm import trange
from pathlib import Path
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor

from lr_project import utils, policies, scene

FEET = ("FL", "FR", "RL", "RR")  # Order of the feet in go2.xml


def command_grid(vx=(0.0,), vy=(0.0,), wz=(0.0,)) -> list[tuple[float, float, float]]:
    """All combinations, e.g. command_grid(vx=[0.5, 1.0], wz=[0.0, 1.0]) -> 4 commands."""
    return [tuple(map(float, c)) for c in itertools.product(vx, vy, wz)]


def collect_rollouts(
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RslRlOnPolicyRunnerCfg,
    checkpoint: str | Path,
    commands: list[tuple[float, float, float]],
    *,
    name: str,
    robots_per_command: int = 10,
    duration_s: float = 10.0,
    command_name: str = "joystick",
    seed: int = 42,
    overwrite: bool = False,
) -> Rollout:
    checkpoint = Path(checkpoint).resolve(True)
    meta = {
        "checkpoint": str(checkpoint),
        "commands": [list(map(float, c)) for c in commands],
        "robots_per_command": robots_per_command,
        "duration_s": duration_s,
        "command_name": command_name,
        "seed": seed,
    }

    path = checkpoint.parent / "eval" / f"{name}_{checkpoint.stem}.npz"
    if path.exists() and not overwrite:
        try:
            cached = Rollout.load(path)
        except TypeError:
            cached = None
        if cached is not None and cached.meta == meta:
            print(f"[evaluate] loaded {path} (already evaluated)")
            return cached
        print(
            f"[evaluate] {path.name} is outdated or has different settings, re-running"
        )

    num_envs = len(commands) * robots_per_command
    print(
        f"[evaluate] creating {len(commands)} commands x {robots_per_command} robots = {num_envs} environments"
    )
    cfg = _make_eval_env_cfg(env_cfg, num_envs, command_name, seed)
    device = utils.device()
    env = ManagerBasedRlEnv(cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    policy = policies.make_inference_policy(env, rl_cfg, checkpoint)

    robot: Entity = env.scene["robot"]
    contact: ContactSensor = env.scene[scene.FEET_CONTACT_SENSOR]
    site_ids, _ = robot.find_sites(FEET, preserve_order=True)

    # Generate commands
    command_term = env.command_manager.get_term(command_name)
    command_index = torch.arange(len(commands), device=device).repeat_interleave(
        robots_per_command
    )
    cmd = torch.tensor(commands, dtype=torch.float32, device=device)[command_index]

    # Initialize environments
    vec_env.reset()
    command_term.vel_command_b[:] = cmd
    obs = vec_env.get_observations()

    # Setup buffers
    num_steps = int(round(duration_s / env.step_dt))
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    base_lin_vel = torch.zeros(num_envs, num_steps, 3, dtype=torch.float, device=device)
    base_ang_vel = torch.zeros(num_envs, num_steps, 3, dtype=torch.float, device=device)
    base_pos = torch.zeros(num_envs, num_steps, 3, dtype=torch.float, device=device)
    base_quat = torch.zeros(num_envs, num_steps, 4, dtype=torch.float, device=device)
    joint_pos = torch.zeros(num_envs, num_steps, 12, dtype=torch.float, device=device)
    joint_vel = torch.zeros(num_envs, num_steps, 12, dtype=torch.float, device=device)
    joint_torque = torch.zeros(
        num_envs, num_steps, 12, dtype=torch.float, device=device
    )
    action = torch.zeros(num_envs, num_steps, 12, dtype=torch.float, device=device)
    foot_contact = torch.zeros(num_envs, num_steps, 4, dtype=torch.bool, device=device)
    foot_grf = torch.zeros(num_envs, num_steps, 4, 3, dtype=torch.float, device=device)
    foot_height = torch.zeros(num_envs, num_steps, 4, dtype=torch.float, device=device)

    try:
        with torch.inference_mode():
            for step in trange(num_steps):
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                alive &= ~dones.bool()
                command_term.vel_command_b[:] = cmd

                # Collect buffer data
                base_lin_vel[alive, step, :] = robot.data.root_link_lin_vel_b[alive]
                base_ang_vel[alive, step, :] = robot.data.root_link_ang_vel_b[alive]
                base_pos[alive, step, :] = robot.data.root_link_pos_w[alive]
                base_quat[alive, step, :] = robot.data.root_link_quat_w[alive]
                joint_pos[alive, step, :] = robot.data.joint_pos[alive]
                joint_vel[alive, step, :] = robot.data.joint_vel[alive]
                joint_torque[alive, step, :] = robot.data.qfrc_actuator[alive]
                action[alive, step, :] = actions[alive]
                foot_contact[alive, step, :] = contact.data.found[alive] > 0
                foot_grf[alive, step, :] = contact.data.force[alive]
                foot_height[alive, step, :] = robot.data.site_pos_w[alive][
                    :, site_ids, 2
                ]

            mass = _robot_mass(env, robot)

    finally:
        vec_env.close()

    rollout = Rollout(
        dt=env.step_dt,
        cmd=cmd.cpu().numpy(),
        alive=alive.cpu().numpy(),
        mass=mass.cpu().numpy(),
        base_lin_vel=base_lin_vel.cpu().numpy(),
        base_ang_vel=base_ang_vel.cpu().numpy(),
        base_pos=base_pos.cpu().numpy(),
        base_quat=base_quat.cpu().numpy(),
        joint_pos=joint_pos.cpu().numpy(),
        joint_vel=joint_vel.cpu().numpy(),
        joint_torque=joint_torque.cpu().numpy(),
        action=action.cpu().numpy(),
        foot_contact=foot_contact.cpu().numpy(),
        foot_grf=foot_grf.cpu().numpy(),
        foot_height=foot_height.cpu().numpy(),
        meta=meta,
    )

    rollout.save(path)

    return rollout


def _make_eval_env_cfg(
    env_cfg: ManagerBasedRlEnvCfg, num_envs: int, command_name: str, seed: int
) -> ManagerBasedRlEnvCfg:
    cfg = copy.deepcopy(env_cfg)
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 1e6  # no time-outs: every termination is a fall

    # Remove curriculum
    cfg.curriculum = {}

    # Disable sampling commands
    command = cfg.commands[command_name]
    command.resampling_time_range = (1e9, 1e9)
    for attr in ("rel_standing_envs", "rel_heading_envs", "rel_forward_envs"):
        if hasattr(command, attr):
            setattr(command, attr, 0.0)

    return cfg


def _robot_mass(env: ManagerBasedRlEnv, robot: Entity) -> torch.Tensor:
    body_mass = env.sim.model.body_mass
    if body_mass.ndim == 1:
        body_mass = body_mass.expand(env.scene.num_envs, -1)
    return body_mass[:, robot.indexing.body_ids].sum(dim=-1)
