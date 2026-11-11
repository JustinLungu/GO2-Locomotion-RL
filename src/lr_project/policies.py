# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch
from pathlib import Path
from dataclasses import asdict
from typing import Callable, Literal

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, MjlabOnPolicyRunner

Policy = Callable[[torch.Tensor], torch.Tensor] | Literal["zero", "random"]


def make_policy(
    env: ManagerBasedRlEnv,
    policy: Policy,
    *,
    rl_cfg: RslRlOnPolicyRunnerCfg | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if not isinstance(policy, (str, os.PathLike)):
        return policy
    if policy == "zero":
        return make_zero_policy(env)
    elif policy == "random":
        return make_random_policy(env)
    elif rl_cfg is not None:
        return make_inference_policy(env, rl_cfg, policy)
    else:
        raise ValueError(
            f"Unknown policy type: {policy!r} (if checkpoint path please specify rl_cfg)"
        )


def make_zero_policy(env: ManagerBasedRlEnv):
    return lambda obs: torch.zeros(env.action_space.shape, device=env.unwrapped.device)


def make_random_policy(env: ManagerBasedRlEnv):
    return (
        lambda obs: 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device)
        - 1
    )


def make_inference_policy(
    env: ManagerBasedRlEnv, rl_cfg: RslRlOnPolicyRunnerCfg, checkpoint: str | Path
):
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(
        vec_env, asdict(rl_cfg), log_dir=None, device=env.device
    )
    runner.load(str(checkpoint), map_location=env.device)
    return runner.get_inference_policy(device=env.device)
