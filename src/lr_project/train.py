# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import copy
import re
import socket
import subprocess
import sys
import time
import matplotlib.pyplot as plt

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from IPython.display import IFrame, display

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from lr_project import utils

LOG_ROOT = Path("../logs")

# Handle of the TensorBoard process started by `tensorboard()`, reused on re-runs.
_tensorboard: tuple[subprocess.Popen, int] | None = None


def train(
    env_cfg: ManagerBasedRlEnvCfg,
    rl_cfg: RslRlOnPolicyRunnerCfg,
    *,
    experiment: str,
    variant: str,
    max_minutes: float | None = None,
    resume_from: str | Path | None = None,
    log_root: str | Path = LOG_ROOT,
) -> Path:
    # Copy so the caller's configs are never modified.
    env_cfg = copy.deepcopy(env_cfg)
    rl_cfg = copy.deepcopy(rl_cfg)
    rl_cfg.logger = "tensorboard"
    rl_cfg.upload_model = False
    rl_cfg.experiment_name = experiment
    rl_cfg.run_name = variant

    # Run directory.
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{stamp}_{variant}"
    run_dir = (Path(log_root) / rl_cfg.experiment_name / dir_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(run_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(run_dir / "params" / "agent.yaml", asdict(rl_cfg))

    device = utils.device()
    print(f"[train] run dir: {run_dir}")
    print(
        f"[train] device: {device}, envs: {env_cfg.scene.num_envs}, seed: {rl_cfg.seed}"
    )

    # Environment and runner.
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(vec_env, asdict(rl_cfg), str(run_dir), device)

    if resume_from is not None:
        print(f"[train] initializing weights from {resume_from}")
        runner.load(
            str(resume_from),
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
            },
            map_location=device,
        )

    # Train. Without a time budget this is a single runner.learn() call. With a
    # budget, train in chunks and check the clock between chunks (each chunk
    # adds a TensorBoard event file to the same run dir; TensorBoard merges them).
    start = time.time()
    try:
        if max_minutes is None:
            runner.learn(rl_cfg.max_iterations, init_at_random_ep_len=True)
        else:
            chunk = max(1, rl_cfg.save_interval // 5)
            done = 0
            while done < rl_cfg.max_iterations:
                n = min(chunk, rl_cfg.max_iterations - done)
                runner.learn(n, init_at_random_ep_len=(done == 0))
                # learn() leaves the counter on the last iteration it ran.
                runner.current_learning_iteration += 1
                done += n
                if (time.time() - start) / 60 > max_minutes:
                    # learn() already saved a checkpoint at the end of the chunk.
                    print(
                        f"[train] time budget of {max_minutes} min reached after {done} iterations"
                    )
                    break
    finally:
        vec_env.close()

    print(f"[train] finished in {(time.time() - start) / 60:.1f} min")
    print(f"[train] latest checkpoint: {latest_checkpoint(run_dir)}")
    return run_dir


def latest_checkpoint(run_dir: str | Path) -> Path:
    checkpoints = list(Path(run_dir).glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))


def tensorboard(
    log_root: str | Path = LOG_ROOT,
    *,
    height_px: int = 800,
    reload_interval_s: int = 10,
) -> None:
    """Start TensorBoard in the background and show it inside the notebook.

    Run this cell *before* the training cell: the curves update live while
    training runs. Re-running the cell reuses the same TensorBoard server.
    """
    global _tensorboard
    log_root = Path(log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    if _tensorboard is None or _tensorboard[0].poll() is not None:
        port = _free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(log_root),
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
                "--reload_interval",
                str(reload_interval_s),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_port(port, timeout_s=30)
        _tensorboard = (process, port)

    port = _tensorboard[1]
    if utils.is_gnoto():
        src = f"https://gnoto.epfl.ch/user-redirect/proxy/{port}/"
    else:
        src = f"http://localhost:{port}/"
    print(f"Tensorboard server running! Monitor your training here: {src}")


def stop_tensorboard() -> None:
    global _tensorboard
    if _tensorboard is not None:
        _tensorboard[0].terminate()
        _tensorboard = None


def read_curves(
    run_dir: str | Path, patterns: list[str]
) -> dict[str, tuple[list[int], list[float]]]:
    """Read logged scalars whose tag matches any regex in `patterns`.

    Returns {tag: (iterations, values)}. Useful tags:
      "Train/mean_reward", "Train/mean_episode_length",
      "Episode_Reward/.*" (one curve per reward term),
      "Episode_Termination/.*", "Loss/.*", "Policy/mean_std".
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves = {}
    for tag in accumulator.Tags()["scalars"]:
        if any(re.fullmatch(p, tag) for p in patterns):
            events = sorted(accumulator.Scalars(tag), key=lambda e: e.step)
            curves[tag] = ([e.step for e in events], [e.value for e in events])
    return curves


def plot_curves(
    run_dirs: str | Path | dict[str, str | Path],
    patterns: list[str] = ["Train/mean_reward", "Train/mean_episode_length"],
    *,
    ncols: int = 3,
):
    """Static plots of training curves, saved in the notebook output.

    run_dirs: one run dir, or {label: run_dir} to overlay several runs.
    """
    if not isinstance(run_dirs, dict):
        run_dirs = {Path(run_dirs).name: run_dirs}
    all_curves = {label: read_curves(d, patterns) for label, d in run_dirs.items()}
    tags = sorted({tag for curves in all_curves.values() for tag in curves})
    if not tags:
        print(f"No scalars matching {patterns}")
        return

    nrows = (len(tags) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False
    )
    for ax, tag in zip(axes.flat, tags):
        for label, curves in all_curves.items():
            if tag in curves:
                ax.plot(*curves[tag], label=label)
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("iteration")
    for ax in list(axes.flat)[len(tags) :]:
        ax.axis("off")
    if len(run_dirs) > 1:
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"TensorBoard did not start on port {port} within {timeout_s} s")
