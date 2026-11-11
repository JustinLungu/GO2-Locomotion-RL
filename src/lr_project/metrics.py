# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from lr_project.rollout import Rollout

GRAVITY = 9.81  # m/s^2
FOOT_RADIUS = 0.022  # m


#############################
# HELPERS (provided)
#############################


def touchdowns(contact: np.ndarray) -> np.ndarray:
    """Step indices where a 1-D contact signal switches from False to True."""
    return np.flatnonzero(~contact[:-1] & contact[1:]) + 1


def lift_offs(contact: np.ndarray) -> np.ndarray:
    """Step indices where a 1-D contact signal switches from True to False."""
    return np.flatnonzero(contact[:-1] & ~contact[1:]) + 1


def phases(contact: np.ndarray, stance: bool) -> list[tuple[int, int]]:
    """Complete stance (or swing) phases of a 1-D contact signal, as
    (start, end) step indices: phase = contact[start:end]. Phases cut off by
    the start or end of the rollout are ignored."""
    starts = touchdowns(contact) if stance else lift_offs(contact)
    ends = lift_offs(contact) if stance else touchdowns(contact)
    out = []
    for start in starts:
        later = ends[ends > start]
        if len(later):
            out.append((int(start), int(later[0])))
    return out


def body_weight(r: Rollout) -> np.ndarray:
    """Weight of each robot, m g, in N. Shape (N,)."""
    return r.mass * GRAVITY


#############################
# EXAMPLE (provided): signal + metric pair
#############################


def lin_vel_error_per_step(r: Rollout) -> np.ndarray:
    """Norm of the (vx, vy) tracking error at each step, in m/s. Shape (N, T)."""
    actual = r.base_lin_vel[..., :2]  # (N, T, 2)
    command = r.cmd[:, None, :2]  # (N, 1, 2), broadcast over time
    return np.linalg.norm(actual - command, axis=-1)


def lin_vel_error(r: Rollout) -> np.ndarray:
    """RMS (vx, vy) tracking error of each robot over its rollout, in m/s. Shape (N,)."""
    error = lin_vel_error_per_step(r)
    return np.sqrt(np.mean(error**2, axis=1))


#############################
# PER-STEP SIGNALS, shape (N, T)
#############################


def yaw_rate_error_per_step(r: Rollout) -> np.ndarray:
    """Absolute yaw rate tracking error at each step, in rad/s. Shape (N, T)."""
    # OUR: Broadcast each robot's commanded yaw rate over time and take the absolute error.
    return np.abs(r.base_ang_vel[..., 2] - r.cmd[:, None, 2])


def mechanical_power(r: Rollout) -> np.ndarray:
    """Sum over joints of |torque * joint velocity| at each step, in W. Shape (N, T)."""
    # OUR: Sum the absolute torque times velocity across joints at each step.
    return np.sum(np.abs(r.joint_torque * r.joint_vel), axis=-1)


def total_vertical_grf(r: Rollout) -> np.ndarray:
    """Sum of the vertical GRF of the four feet at each step, in body weights. Shape (N, T)."""
    # OUR: Sum vertical foot forces and normalize by each robot's body weight.
    return np.sum(r.foot_grf[..., 2], axis=-1) / body_weight(r)[:, None]


#############################
# PER-ROBOT METRICS, shape (N,)
#############################


def yaw_rate_error(r: Rollout) -> np.ndarray:
    """RMS yaw rate tracking error of each robot, in rad/s. Shape (N,)."""
    # TODO: use yaw_rate_error_per_step
    raise NotImplementedError


def cost_of_transport(r: Rollout) -> np.ndarray:
    """Mechanical cost of transport of each robot:
    mean power / (m g mean horizontal speed). Dimensionless. Shape (N,)."""
    # TODO: use mechanical_power and body_weight
    raise NotImplementedError


def duty_factor(r: Rollout) -> np.ndarray:
    """Fraction of time the feet are on the ground, averaged over the four
    feet (> 0.5 walk, < 0.5 trot/run). Shape (N,)."""
    # TODO
    raise NotImplementedError


def stride_frequency(r: Rollout) -> np.ndarray:
    """Strides per second of each robot: 1 / mean time between two
    touchdowns of the same foot, averaged over the feet. Shape (N,)."""
    # TODO: use touchdowns
    raise NotImplementedError


def stride_length(r: Rollout) -> np.ndarray:
    """Distance travelled per stride: mean horizontal speed / stride frequency, in m. Shape (N,)."""
    # TODO: use stride_frequency
    raise NotImplementedError


def foot_clearance(r: Rollout) -> np.ndarray:
    """Mean over swing phases of the maximum foot height above the ground,
    in m (flat ground only). Shape (N,)."""
    # TODO: use phases(..., stance=False) and FOOT_RADIUS
    raise NotImplementedError


def peak_vertical_grf(r: Rollout) -> np.ndarray:
    """Mean over stance phases of the peak vertical GRF, in body weights. Shape (N,)."""
    # TODO: use phases(..., stance=True) and body_weight
    raise NotImplementedError


def weight_support(r: Rollout) -> np.ndarray:
    """Mean total vertical GRF in body weights. About 1 for steady walking
    (sanity check of the force measurement). Shape (N,)."""
    # TODO: use total_vertical_grf
    raise NotImplementedError


def action_rate(r: Rollout) -> np.ndarray:
    """Mean norm of the change in action between two control steps. Shape (N,)."""
    # TODO
    raise NotImplementedError


# TODO: add your own metrics, following the same convention (per step: (N, T),
# per robot: (N,)). For example, how would you quantify base stability?
