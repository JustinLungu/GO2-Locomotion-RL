# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch


def device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def is_gnoto() -> bool:
    # Check for common JupyterHub or Jupyter environment flags
    if "JUPYTERHUB_USER" in os.environ or "JPY_PARENT_PID" in os.environ:
        return True

    home_dir = os.path.expanduser("~")
    if "gnoto" in home_dir or "/home/jovyan" in home_dir:
        return True

    return False
