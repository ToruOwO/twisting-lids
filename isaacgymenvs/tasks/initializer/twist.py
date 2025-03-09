"""Updated Allegro joint 14 init qpos"""

from isaacgym import gymutil, gymtorch, gymapi
from scipy.spatial.transform import Rotation as R
from isaacgymenvs.tasks.initializer.base import EnvInitializer, Pose


class Jan8EnvInitializer(EnvInitializer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hand QPos
        self.cfg = kwargs.get("cfg")
        left_hand_init_qpos = {
            "joint_0.0": -0.008,
            "joint_1.0": 0.9478,
            "joint_2.0": 0.6420,
            "joint_3.0": -0.0330,
            "joint_12.0": 0.667,  # 0.600,
            "joint_13.0": 1.167,  # 1.1630,
            "joint_14.0": 0.75,  # 1.000,
            "joint_15.0": 0.45,  # 0.480,
            "joint_4.0": 0.0530,
            "joint_5.0": 0.7163,
            "joint_6.0": 0.9606,
            "joint_7.0": 0.0000,
            "joint_8.0": 0.0000,
            "joint_9.0": 0.7811,
            "joint_10.0": 0.7868,
            "joint_11.0": 0.3454,
        }
        right_hand_init_qpos = {}
        for k, v in left_hand_init_qpos.items():
            right_hand_init_qpos[k + "_r"] = v
        self.hand_init_qpos = {**left_hand_init_qpos, **right_hand_init_qpos}

        self.dof_velocity = 1.0

        # Pose [px, py, pz, qx, qy, qz, qw]
        self.hand_base_init_pose = Pose([0.0, 0.0, 1.0], [0, 0, -0.7071068, 0.7071068])
        self.cube_base_init_pose = Pose(
            [0.70, -0.03, 1.24], [0, -0.7071068, 0, 0.7071068]
        ).post_multiply_euler("z", [-90])
        return

    def get_ur_base_init_pos(self):
        # TODO: get base init pos by solving IK from wrist init pose
        return [
            # 830
            -1.567662541066305,
            -2.4176141224303187,
            -1.470444917678833,
            -0.8341446679881592,
            0.894737720489502,
            0.08133087307214737,
            # 828
            -4.674656931553976,
            -0.6805991691401978,
            1.5093582312213343,
            -2.377801080743307,
            -0.8824575583087366,
            -0.06327754655946904,
        ]

    def initialize_object_dof(self, cube_dof_props):
        # print(cube_dof_props)
        if len(cube_dof_props["driveMode"]) == 2:
            # print("ENTER!")
            cube_dof_props["driveMode"][0] = gymapi.DOF_MODE_EFFORT
            cube_dof_props["velocity"][0] = 3.0
            cube_dof_props["effort"][0] = 1.0
            cube_dof_props["stiffness"][0] = 0.0
            cube_dof_props["damping"][0] = 0.0
            cube_dof_props["friction"][0] = 0.0
            cube_dof_props["armature"][0] = 0.0001

            cube_dof_props["driveMode"][1] = gymapi.DOF_MODE_NONE
            cube_dof_props["velocity"][1] = 3.0
            cube_dof_props["effort"][1] = 1.0
            cube_dof_props["stiffness"][1] = 0.00
            cube_dof_props["damping"][1] = 0.1
            cube_dof_props["friction"][1] = self.cfg["obj_dof_friction"]
            cube_dof_props["armature"][1] = 0.0001
        else:
            for i in range(1):
                cube_dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
                cube_dof_props["velocity"][i] = 3.0
                cube_dof_props["effort"][i] = 10.0
                cube_dof_props["stiffness"][i] = 0.01
                cube_dof_props["damping"][i] = 20.0
                cube_dof_props["friction"][i] = self.cfg["obj_dof_friction"]
                cube_dof_props["armature"][i] = 0.0001


def build(**kwargs):
    return Jan8EnvInitializer(**kwargs)
