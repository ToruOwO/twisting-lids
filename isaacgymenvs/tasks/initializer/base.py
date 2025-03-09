from isaacgym import gymutil, gymtorch, gymapi
from scipy.spatial.transform import Rotation as R


class Pose:
    def __init__(self, pos, quat):
        # pos:  LIST [FLOAT * 3] xyz  format
        # quat: LIST [FLOAT * 4] xyzw format
        self.pos = pos
        self.quat = quat

    def to_isaacgym_pose(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.pos)
        pose.r = gymapi.Quat(*self.quat)
        return pose

    def to_list(self):
        return self.pos + self.quat

    def post_multiply_quat(self, quat):
        """
        self.quat = quat * self.quat
        Fortunately, scipy also uses xyzw format.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
        """
        r = R.from_quat(self.quat)
        q = R.from_quat(quat)
        r = q * r
        self.quat = list(r.as_quat().reshape(-1))
        return self

    def post_multiply_euler(self, mode="zyx", angle=[0, 0, 0]):
        """
        Transform the pose with an Eulerian
        """
        r = R.from_quat(self.quat)
        q = R.from_euler(mode, angle, degrees=True)
        r = q * r
        self.quat = list(r.as_quat().reshape(-1))
        return self


class EnvInitializer:
    def __init__(self, **kwargs):
        # Hand QPos

        self.hand_init_qpos = {
            "joint_0.0": -0.008,
            "joint_1.0": 0.9478,
            "joint_2.0": 0.6420,
            "joint_3.0": -0.0330,
            "joint_12.0": 1.067,  # 0.600,
            "joint_13.0": 1.167,  # 1.1630,
            "joint_14.0": 0.02,  # 1.000,
            "joint_15.0": 1.52,  # 0.480,
            "joint_4.0": 0.0530,
            "joint_5.0": 0.7163,
            "joint_6.0": 0.9606,
            "joint_7.0": 0.0000,
            "joint_8.0": 0.0000,
            "joint_9.0": 0.7811,
            "joint_10.0": 0.7868,
            "joint_11.0": 0.3454,
        }

        self.dof_velocity = 3.14
        # Pose [px, py, pz, qx, qy, qz, qw]
        self.right_hand_base_init_pose = Pose(
            [0.0, 0.12, 0.5], [0.7071068, -0.7071068, 0, 0]
        )
        self.left_hand_base_init_pose = Pose(
            [0.0, -0.12, 0.5], [0.7071068, 0.7071068, 0, 0]
        )
        self.cube_base_init_pose = Pose(
            [0.01, 0.0, 0.51], [0, -0.7071068, 0, 0.7071068]
        )
        return

    def initialize_object_dof(self, cube_dof_props):
        for i in range(1):
            cube_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            cube_dof_props["velocity"][i] = 3.0
            cube_dof_props["effort"][i] = 0.0
            cube_dof_props["stiffness"][i] = 0.01
            cube_dof_props["damping"][i] = 0.0
            cube_dof_props["friction"][i] = 2000.0
            cube_dof_props["armature"][i] = 0.0

    def get_dof_velocity(self):
        return self.dof_velocity

    def get_hand_base_init_pose(self):
        try:
            return self.hand_base_init_pose
        except:
            return Pose([0.0, 0.0, 0.0], [0.7071068, -0.7071068, 0, 0])

    def get_hand_init_qpos(self):
        return self.hand_init_qpos

    def get_left_hand_base_init_pose(self):
        return self.left_hand_base_init_pose

    def get_right_hand_base_init_pose(self):
        return self.right_hand_base_init_pose

    def get_left_hand_init_qpos(self):
        return self.hand_init_qpos

    def get_right_hand_init_qpos(self):
        return self.hand_init_qpos

    def get_cube_init_pose(self):
        return self.cube_base_init_pose

    def get_hand_info(self):
        info = {"stiffness": 0.0, "damping": 0.0, "armature": 0.001}
        return info

    def get_ur_base_init_pos(self):
        return [0] * 12


def build(**kwargs):
    return EnvInitializer(**kwargs)
