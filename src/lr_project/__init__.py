# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import os
from lr_project.utils import is_gnoto

if is_gnoto():
    # This has to happen BEFORE mjlab imports!
    os.environ["MUJOCO_GL"] = "disable"
