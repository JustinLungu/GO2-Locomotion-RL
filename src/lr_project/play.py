# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import copy
from IPython.display import IFrame, display

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.viewer.viser import ViserPlayViewer

from lr_project import utils, policies


def interactive(
    env_cfg: ManagerBasedRlEnvCfg,
    policy: policies.Policy,
    *,
    num_envs: int = 8,
    rl_cfg: RslRlOnPolicyRunnerCfg | None = None,
    frame_rate: int = 30,
    height_px: int = 600,
    local_display: bool = False,
):
    # Update number of environments
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.scene.num_envs = num_envs

    # Build the environment
    env = ManagerBasedRlEnv(env_cfg, device=utils.device())
    env.reset()

    # Make the policy
    policy = policies.make_policy(env, policy, rl_cfg=rl_cfg)

    # Instantiate viewer with internal web server
    viewer = ViserPlayViewer(env=env, policy=policy, frame_rate=frame_rate)

    # URL depending on whether we are on gnoto
    port = viewer._server.get_port()
    if utils.is_gnoto():
        src = f"https://gnoto.epfl.ch/user-redirect/proxy/{port}"
    else:
        src = f"http://localhost:{port}"

    # Display IFrame BEFORE starting the blocking loop
    if local_display:
        display(IFrame(src=src, width="100%", height=f"{height_px}px"))
    print(f"URL to interactive simulation: {src}")

    # Blocks main thread, interrupts are caught by viewer.run
    viewer.run(catch_sigint=True)

    viewer.close()

    print("Viewer closed!")

    # Clean up
    del env
    del policy
    del viewer
