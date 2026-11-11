# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal

from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv

from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
from mjlab.sensor import RayCastSensorCfg, ObjRef, GridPatternCfg, BuiltinSensorCfg

from lr_project.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg

# Contact sensors
FEET_CONTACT_SENSOR = "feet_contact"
BASE_CONTACT_SENSOR = "base_contact"
LEG_CONTACT_SENSOR = "leg_contact"

# Exteroception
TERRAIN_SCAN_SENSOR = "terrain_scan"

# Feet sensors
FOOT_SITES = ("FL", "FR", "RL", "RR")
FOOT_ACC_SENSORS = tuple(f"{foot}_foot_acc" for foot in FOOT_SITES)
FOOT_VEL_SENSORS = tuple(f"{foot}_foot_vel" for foot in FOOT_SITES)

# mujoco builtin sensors
BASE_IMU_ANG_VEL = "robot/imu_ang_vel"
BASE_IMU_LIN_VEL = "robot/imu_lin_vel"
BASE_IMU_LIN_ACC = "robot/imu_lin_acc"
ROBOT_ANGULAR_MOMENTUM = "robot/root_angmom"


def make_scene_cfg(
    num_envs: int,
    terrain: Literal["flat", "stairs"],
    *,
    robot_entity_name: str = "robot",
    control_mode: Literal["pd", "torque"] = "pd",
) -> SceneCfg:
    # Foot contact sensors
    feet_contact = ContactSensorCfg(
        name=FEET_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="geom", pattern=(".*_foot_collision",), entity=robot_entity_name
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        num_slots=1,
    )

    # Base contact sensor
    base_contact = ContactSensorCfg(
        name=BASE_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=("base.*_collision",),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
    )

    # Leg contact sensor
    leg_contact = ContactSensorCfg(
        name=LEG_CONTACT_SENSOR,
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=(".*_thigh_collision", ".*_hip_collision", ".*_calf1_collision"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
    )

    sensors = [feet_contact, base_contact, leg_contact]

    for foot, acc_name, vel_name in zip(FOOT_SITES, FOOT_ACC_SENSORS, FOOT_VEL_SENSORS):
        site = ObjRef(type="site", name=f"{robot_entity_name}/{foot}")
        sensors += [
            BuiltinSensorCfg(name=acc_name, sensor_type="framelinacc", obj=site),
            BuiltinSensorCfg(name=vel_name, sensor_type="framelinvel", obj=site),
        ]

    if terrain == "stairs":
        # Add terrain scan if doing stairs
        terrain_scan = RayCastSensorCfg(
            name=TERRAIN_SCAN_SENSOR,
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            ray_alignment="yaw",
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.2),
            max_distance=5.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),  # Terrain only.
            debug_vis=True,
        )
        sensors.append(terrain_scan)

    return SceneCfg(
        num_envs=num_envs,
        terrain=_make_terrain_cfg(terrain),
        entities={robot_entity_name: get_go2_robot_cfg(control_mode)},
        sensors=sensors,
    )


def _make_terrain_cfg(terrain: Literal["flat", "stairs"]) -> TerrainEntityCfg:
    if terrain == "flat":
        return TerrainEntityCfg(terrain_type="plane")
    elif terrain == "stairs":
        return TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(12.0, 12.0),
                difficulty_range=(0.0, 1.0),
                border_width=10.0,
                curriculum=True,
                num_rows=10,
                num_cols=2,
                sub_terrains={
                    "stairs_down": pyramid_stairs(
                        proportion=0.5,
                        step_height_range=(0.0, 0.2),
                        border_width=3.0,
                        platform_width=1.5,
                    ),
                    "stairs_up": pyramid_stairs_inv(
                        proportion=0.5,
                        step_height_range=(0.0, 0.2),
                        border_width=3.0,
                        platform_width=1.5,
                    ),
                },
            ),
        )
    else:
        raise ValueError(f"Unknown terrain type: {terrain!r}")


def make_sim_cfg(timestep_s: float) -> SimulationCfg:
    return SimulationCfg(
        njmax=300,
        contact_sensor_maxmatch=64,
        mujoco=MujocoCfg(
            timestep=timestep_s,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=20,
        ),
    )
