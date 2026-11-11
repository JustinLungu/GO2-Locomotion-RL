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
    # OUR: Take the root mean square of each robot's yaw rate error over time.
    error = yaw_rate_error_per_step(r)
    return np.sqrt(np.mean(error**2, axis=1))


def cost_of_transport(r: Rollout) -> np.ndarray:
    """Mechanical cost of transport of each robot:
    mean power / (m g mean horizontal speed). Dimensionless. Shape (N,)."""
    # TODO: use mechanical_power and body_weight
    # OUR: Normalize mean power by weight times mean speed; zero speed is undefined.
    mean_power = np.mean(mechanical_power(r), axis=1)
    mean_speed = np.mean(np.linalg.norm(r.base_lin_vel[..., :2], axis=-1), axis=1)
    denominator = body_weight(r) * mean_speed
    return np.divide(
        mean_power,
        denominator,
        out=np.full_like(mean_power, np.nan),
        where=denominator > 0,
    )


def duty_factor(r: Rollout) -> np.ndarray:
    """Fraction of time the feet are on the ground, averaged over the four
    feet (> 0.5 walk, < 0.5 trot/run). Shape (N,)."""
    # TODO
    # OUR: Average the contact indicators over time and all four feet.
    return np.mean(r.foot_contact, axis=(1, 2))


def stride_frequency(r: Rollout) -> np.ndarray:
    """Strides per second of each robot: 1 / mean time between two
    touchdowns of the same foot, averaged over the feet. Shape (N,)."""
    # TODO: use touchdowns
    # OUR: Average inverse mean touchdown intervals over feet with at least two touchdowns.
    frequency = np.full(r.num_robots, np.nan)
    for robot in range(r.num_robots):
        foot_frequencies = []
        for foot in range(r.foot_contact.shape[2]):
            steps = touchdowns(r.foot_contact[robot, :, foot])
            if steps.size >= 2:
                foot_frequencies.append(1.0 / (np.mean(np.diff(steps)) * r.dt))
        if foot_frequencies:
            frequency[robot] = np.mean(foot_frequencies)
    return frequency


def stride_length(r: Rollout) -> np.ndarray:
    """Distance travelled per stride: mean horizontal speed / stride frequency, in m. Shape (N,)."""
    # TODO: use stride_frequency
    # OUR: Divide mean horizontal speed by stride frequency when strides are available.
    mean_speed = np.mean(np.linalg.norm(r.base_lin_vel[..., :2], axis=-1), axis=1)
    frequency = stride_frequency(r)
    return np.divide(
        mean_speed,
        frequency,
        out=np.full_like(mean_speed, np.nan),
        where=frequency > 0,
    )


def foot_clearance(r: Rollout) -> np.ndarray:
    """Mean over swing phases of the maximum foot height above the ground,
    in m (flat ground only). Shape (N,)."""
    # TODO: use phases(..., stance=False) and FOOT_RADIUS
    # OUR: Average complete swing peaks above the foot radius; missing swings stay NaN.
    clearance = np.full(r.num_robots, np.nan)
    for robot in range(r.num_robots):
        peaks = []
        for foot in range(r.foot_contact.shape[2]):
            for start, end in phases(r.foot_contact[robot, :, foot], stance=False):
                peaks.append(np.max(r.foot_height[robot, start:end, foot]) - FOOT_RADIUS)
        if peaks:
            clearance[robot] = np.mean(peaks)
    return clearance


def peak_vertical_grf(r: Rollout) -> np.ndarray:
    """Mean over stance phases of the peak vertical GRF, in body weights. Shape (N,)."""
    # TODO: use phases(..., stance=True) and body_weight
    # OUR: Average complete stance force peaks and normalize by each robot's body weight.
    peak_force = np.full(r.num_robots, np.nan)
    weight = body_weight(r)
    for robot in range(r.num_robots):
        peaks = []
        for foot in range(r.foot_contact.shape[2]):
            for start, end in phases(r.foot_contact[robot, :, foot], stance=True):
                peaks.append(np.max(r.foot_grf[robot, start:end, foot, 2]))
        if peaks:
            peak_force[robot] = np.mean(peaks) / weight[robot]
    return peak_force


def weight_support(r: Rollout) -> np.ndarray:
    """Mean total vertical GRF in body weights. About 1 for steady walking
    (sanity check of the force measurement). Shape (N,)."""
    # TODO: use total_vertical_grf
    # OUR: Average the total force in body weights over each robot's rollout.
    return np.mean(total_vertical_grf(r), axis=1)


def action_rate(r: Rollout) -> np.ndarray:
    """Mean norm of the change in action between two control steps. Shape (N,)."""
    # TODO
    # OUR: Average consecutive action change norms; fewer than two steps is undefined.
    if r.action.shape[1] < 2:
        return np.full(r.num_robots, np.nan)
    return np.mean(np.linalg.norm(np.diff(r.action, axis=1), axis=-1), axis=1)


# TODO: add your own metrics, following the same convention (per step: (N, T),
# per robot: (N,)). For example, how would you quantify base stability?
