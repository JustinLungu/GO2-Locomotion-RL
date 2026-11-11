# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal
from importlib.resources import files

import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_XML = str(files("lr_project").joinpath("assets/robots/unitree_go2/xml/go2.xml"))


def get_spec():
    return mujoco.MjSpec.from_file(GO2_XML)  # type: ignore


def get_go2_initial_state() -> EntityCfg.InitialStateCfg:
    return EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.36),
        joint_pos={
            ".*_thigh_joint": 0.67,
            ".*_calf_joint": -1.3,
            ".*_hip_joint": 0.0,
        },
    )


def get_go2_articulation(
    control_mode: Literal["torque", "pd"],
) -> EntityArticulationInfoCfg:
    if control_mode == "torque":
        hip_actuator_cfg = BuiltinMotorActuatorCfg(
            target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
            effort_limit=23.5,
            armature=0.01,
        )
        thigh_actuator_cfg = BuiltinMotorActuatorCfg(
            target_names_expr=(".*_calf_joint",),
            effort_limit=45,
            armature=0.02,
        )
        return EntityArticulationInfoCfg(
            actuators=(hip_actuator_cfg, thigh_actuator_cfg),
            soft_joint_pos_limit_factor=0.9,
        )
    elif control_mode == "pd":
        hip_actuator_cfg = BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
            stiffness=20.0,
            damping=1.0,
            effort_limit=23.5,
            armature=0.01,
        )
        thigh_actuator_cfg = BuiltinPositionActuatorCfg(
            target_names_expr=(".*_calf_joint",),
            stiffness=40.0,
            damping=2.0,
            effort_limit=45,
            armature=0.02,
        )
        return EntityArticulationInfoCfg(
            actuators=(hip_actuator_cfg, thigh_actuator_cfg),
            soft_joint_pos_limit_factor=0.9,
        )
    else:
        raise ValueError(f"Unknown control mode: {control_mode!r}")


def get_go2_full_collisions():
    # This enables all collisions, excluding self collisions.
    # Foot collisions are given custom condim, friction and solimp.
    _foot_regex = "^[FR][LR]_foot_collision$"
    _collision_regex = r".*_collision"
    return CollisionCfg(
        geom_names_expr=(_collision_regex,),
        condim={_foot_regex: 3, _collision_regex: 1},
        priority={_foot_regex: 1, _collision_regex: 0},
        friction={_foot_regex: (0.6,)},
        solimp={_foot_regex: (0.9, 0.95, 0.023)},
        contype=1,
        conaffinity=0,
    )


def get_go2_robot_cfg(control_mode: Literal["torque", "pd"]) -> EntityCfg:
    return EntityCfg(
        init_state=get_go2_initial_state(),
        collisions=(get_go2_full_collisions(),),
        spec_fn=get_spec,
        articulation=get_go2_articulation(control_mode),
    )


# For visual inspection of the model
if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity import Entity

    robot = Entity(get_go2_robot_cfg(control_mode="pd"))
    viewer.launch(robot.spec.compile())
