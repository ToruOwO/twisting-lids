import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask
import torch
import torch.nn.functional as F
import importlib


class AssetManager:
    def __init__(self, asset_root, model_root_path) -> None:
        self.asset_root = asset_root
        self.model_root_path = model_root_path  # STR or LIST
        self.asset_files = self._scan()
        self.assets = []
        self.assets_dof_props = []
        self.rigid_body_counts = 0
        self.rigid_shape_counts = 0

        pass

    def __len__(self):
        # How many object instances do we have.
        return len(self.asset_files)

    def _scan(self):
        from pathlib import Path

        if isinstance(self.model_root_path, str):
            self.model_root_path = [self.model_root_path]

        for root in self.model_root_path:
            root_folder = Path(self.asset_root) / root  # self.model_root_path
            asset_files = []
            for item in root_folder.iterdir():
                print("Iterating", item)
                if os.path.isdir(item) and os.path.exists(item / "model.urdf"):
                    urdf_path = item / "model.urdf"
                    urdf_relative_path = os.path.relpath(
                        urdf_path, Path(self.asset_root)
                    )

                    asset_files.append(urdf_relative_path)
                    print(urdf_relative_path)
        return asset_files

    def load(self, env, asset_option, initializer):
        for f in self.asset_files:
            # print(f.as_posix())
            asset = env.gym.load_asset(env.sim, self.asset_root, f, asset_option)
            cube_dof_props = env.gym.get_asset_dof_properties(asset)
            initializer.initialize_object_dof(cube_dof_props)
            # print("CUBE_LR", cube_dof_props['lower'][0], cube_dof_props['upper'][0])
            self.assets.append(asset)
            self.assets_dof_props.append(cube_dof_props)

            self.rigid_body_counts += env.gym.get_asset_rigid_body_count(asset)
            self.rigid_shape_counts += env.gym.get_asset_rigid_shape_count(asset)

    def get_asset_rigid_body_count(self):
        return self.rigid_body_counts

    def get_asset_rigid_shape_count(self):
        return self.rigid_shape_counts

    def get_random_asset(self):
        i = np.random.randint(0, len(self.assets), 1)[0]
        return i, self.assets[i], self.assets_dof_props[i]

    def get_markers(self):
        # Marker handles
        cap_marker_handle_names = [
            "l10",
            "l11",
            "l12",
            "l13",
            "l10b",
            "l11b",
            "l12b",
            "l13b",
        ]
        base_marker_handle_names = [
            "l20",
            "l21",
            "l22",
            "l23",
            "l24",
            "l25",
            "l26",
            "l27",
        ]
        base_marker_handle_names += [
            "l30",
            "l31",
            "l32",
            "l33",
            "l34",
            "l35",
            "l36",
            "l37",
        ]

        return cap_marker_handle_names, base_marker_handle_names


class DualURBottle(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        self.save_init_grasp = self.cfg["env"].get("save_init_grasp", False)

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        # Hand spec.
        self.action_moving_average = self.cfg["env"]["controller"]["actionEMA"]
        self.controller_action_scale = self.cfg["env"]["controller"][
            "controllerActionScale"
        ]
        self.p_gain_val = self.cfg["env"]["controller"]["kp"]
        self.d_gain_val = self.cfg["env"]["controller"]["kd"]

        # Create dicts to pass to reward function
        self.reward_settings = self.cfg["env"]["reward_setup"]
        # Count the number of assets
        self.num_objects = len(
            AssetManager(
                asset_root=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "../../assets"
                ),
                model_root_path=cfg["env"]["object_asset_root_folder"],
            )
        )
        print("Total Training Objects", self.num_objects)

        # TODO(): Make this more elegant
        self.cam_w = 128
        self.cam_h = 128
        self.im_size = 128

        # Import the reward function
        self.rewarder_name = self.cfg["env"]["rewarder"]
        reward_module_name = f"isaacgymenvs.tasks.rewarder.{self.rewarder_name}"
        reward_module = importlib.import_module(reward_module_name)
        reward_function_builder = getattr(reward_module, "build")
        # build a pseudo reward function
        reward_function = reward_function_builder(
            device="cpu", num_envs=4, reward_settings=self.reward_settings
        )

        self.num_wrist_joints = 0
        self.use_real_allegro_limit = self.cfg["env"]["use_real_allegro_limit"]

        self.n_stack_frame = self.cfg["env"]["n_stack_frame"]
        self.single_frame_obs_dim = (
            106
            + reward_function.obs_dim()
            + self.num_objects
            + 6 * self.num_wrist_joints
        )
        self.cfg["env"]["numObservations"] = int(
            self.single_frame_obs_dim * self.n_stack_frame
        )  # Just set a bit larger

        self.full_state = self.cfg["env"]["fullState"]
        if self.cfg["env"]["computeState"]:
            if self.cfg["env"]["fullState"]:
                self.cfg["env"]["numStates"] = 500
            else:
                if self.num_objects > 20:
                    self.cfg["env"]["numStates"] = 280  # TODO(): Compute this.
                else:
                    self.cfg["env"]["numStates"] = 200
        else:
            self.cfg["env"]["numStates"] = 0

        if reward_function.obs_dim() > 0:
            self.use_reward_obs = True
        else:
            self.use_reward_obs = False

        # this task is designed for controlling both arm and hand
        # 2 x 16 hand and 2 x 6 arm
        self.whole_ur_hand_control = False
        self.cfg["env"]["numActions"] = 44 if self.whole_ur_hand_control else 32
        self.total_hand_dof = 32
        self.total_arm_dof = 12

        # Values to be filled in at runtime
        self.states = {}  # will be dict filled with relevant states to use for reward calculation
        self.handles = {}  # will be dict mapping names to relevant sim handles
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._init_cube_state = None  # Initial state of cube for the current env
        self.cube_id = None  # Actor ID corresponding to cube for a given env

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, n_actors, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof, 2)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._global_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = 1 / 60.0
        self.torque_control = True

        self.use_mirrored_urdf = self.cfg["env"]["use_mirrored_urdf"]
        self.use_updated_urdf = self.cfg["env"]["use_updated_urdf"]
        self.initializer_name = self.cfg["env"]["initializer"]
        if self.use_mirrored_urdf:
            print("Using Mirrored URDF.")
            print(f"Initalizer: {self.initializer_name}")

        # Import the initializer
        initializer_name = f"isaacgymenvs.tasks.initializer.{self.initializer_name}"
        initializer_module = importlib.import_module(initializer_name)
        initializer_function_builder = getattr(initializer_module, "build")
        self.initializer = initializer_function_builder(
            cfg=self.cfg["env"]["init_setup"]
        )

        # Import the randomizer
        self.randomizer_name = self.cfg["env"]["randomizer"]
        randomizer_name = f"isaacgymenvs.tasks.randomizer.{self.randomizer_name}"
        randomizer_module = importlib.import_module(randomizer_name)
        randomizer_function_builder = getattr(randomizer_module, "build")
        self.randomizer = randomizer_function_builder(
            cfg=self.cfg["env"]["randomization_setup"]
        )

        # For Sim2Real Testing
        self.disable_gravity = self.cfg["env"].get("disable_gravity", True)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Setup reward function
        self.reward_function = reward_function_builder(
            device=self.device,
            num_envs=self.num_envs,
            reward_settings=self.reward_settings,
        )

        # set up default hand initialization.
        self.hand_default_dof_pos = torch.zeros(self.total_hand_dof, device=self.device)
        self.ur_default_dof_pos = to_torch(
            self.initializer.get_ur_base_init_pos(),
            device=self.device,
        )
        self._post_init()

        # set tensors and buffers.
        self.last_actions = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )

        self.p_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.p_gain_val
        )
        self.d_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.d_gain_val
        )
        self.last_cube_dof_pos = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

        # setup random forces.
        self.num_bodies = self._rigid_body_state.shape[1]
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.left_control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.right_control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )
        self.control_work = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.float
        )

        # object brake joint torque
        self.brake_torque = self.cfg["env"]["brake_torque"]
        self.object_brake_torque = torch.full(
            (self.num_envs,), self.brake_torque, dtype=torch.float, device=self.device
        )

        self.force_prob = self.cfg["env"]["randomization"]["force_prob"]
        self.force_decay = self.cfg["env"]["randomization"]["force_decay"]
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )

        self.force_decay_interval = self.cfg["env"]["randomization"][
            "force_decay_interval"
        ]

        self.force_scale = self.cfg["env"]["randomization"]["force_scale"]
        self.force_scale_x = self.cfg["env"]["randomization"]["force_scale_x"]
        self.force_scale_y = self.cfg["env"]["randomization"]["force_scale_y"]
        self.force_scale_z = self.cfg["env"]["randomization"]["force_scale_z"]
        self.force_horizon_decay = self.cfg["env"]["randomization"][
            "force_horizon_decay"
        ]
        self.force_progress_buf = torch.zeros_like(self.progress_buf)

        # refresh
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.obs_setting = self.cfg["env"]["obs_setting"]
        # refresh tensors
        self._refresh()

    def _post_init(self):
        for i, (idx, qpos) in enumerate(self.hand_default_qpos_info[:32]):
            print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
            self.hand_default_dof_pos[i] = qpos
        for i, (idx, qpos) in enumerate(self.hand_default_qpos_info[32:]):
            print("UR QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
            self.ur_default_dof_pos[i] = qpos

        self.cube_init_pose = to_torch(
            self.initializer.get_cube_init_pose().to_list(),
            dtype=torch.float,
            device=self.device,
        )

    def _update_states(self):
        self.cube_state = self._env_root_state[:, self.cube_handle, :]

        self.states.update(
            {
                # robot
                "dof_pos": self.hand_dof_pos[:, :],
                "dof_vel": self.hand_dof_vel[:, :],
                "cube_pos": self.cube_state[:, :3],
                "cube_quat": self.cube_state[:, 3:7],
                "cube_vel": self.cube_state[:, 7:10],
                "prev_cube_quat": self.prev_cube_state[:, 3:7],
                "cube_angvel": self.cube_state[:, 10:13],
                "left_tips_pos": self._env_rigid_body_state[
                    :, self.left_tip_handles, :3
                ],
                "right_tips_pos": self._env_rigid_body_state[
                    :, self.right_tip_handles, :3
                ],
                # we need two separate groups.
                "left_thumb_tips_pos": self._env_rigid_body_state[
                    :, self.left_thumb_tip_handles, :3
                ],
                "right_thumb_tips_pos": self._env_rigid_body_state[
                    :, self.right_thumb_tip_handles, :3
                ],
                "left_nonthumb_tips_pos": self._env_rigid_body_state[
                    :, self.left_nonthumb_tip_handles, :3
                ],
                "right_nonthumb_tips_pos": self._env_rigid_body_state[
                    :, self.right_nonthumb_tip_handles, :3
                ],
                "cube_dof_pos": self.all_dof_pos[:, self.cube_joint_id],
                "cube_dof_vel": self.all_dof_vel[:, self.cube_joint_id],
                "last_cube_dof_pos": self.last_cube_dof_pos.clone(),
                "cube_base_pos": self._env_rigid_body_state[
                    :, self.bottle_base_handle, :3
                ],
                "cube_cap_pos": self._env_rigid_body_state[
                    :, self.bottle_cap_handle, :3
                ],
                "cube_base_marker_pos": self._env_rigid_body_state[
                    :, self.bottle_base_marker_handles, :3
                ],
                "cube_cap_marker_pos": self._env_rigid_body_state[
                    :, self.bottle_cap_marker_handles, :3
                ],
                "left_work": self.left_control_work,
                "right_work": self.right_control_work,
                "work": self.control_work,
            }
        )

    def get_state(self):
        # For asymmetric training.
        cursor = 0
        self.states_buf = torch.zeros_like(self.states_buf)  # Clear this first
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["dof_pos"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["dof_vel"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["cube_pos"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf, self.states["cube_quat"], cursor
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["left_tips_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["right_tips_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["cube_base_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["cube_cap_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["cube_base_marker_pos"].reshape(self.num_envs, -1),
            cursor,
        )
        self.states_buf, cursor = self._fill(
            self.states_buf,
            self.states["cube_cap_marker_pos"].reshape(self.num_envs, -1),
            cursor,
        )  # W = 147

        # New states (Jan21)
        if self.full_state:
            self.states_buf, cursor = self._fill(
                self.states_buf, self.states["cube_vel"], cursor
            )

            self.states_buf, cursor = self._fill(
                self.states_buf, self.states["cube_angvel"], cursor
            )

            self.states_buf, cursor = self._fill(
                self.states_buf,
                self.rb_forces[:, self.bottle_base_handle, :].reshape(
                    self.num_envs, -1
                ),
                cursor,
            )

            self.states_buf, cursor = self._fill(
                self.states_buf,
                self.env_physics_setup.reshape(self.num_envs, -1),
                cursor,
            )

            self.states_buf, cursor = self._fill(
                self.states_buf,
                self.object_brake_torque.reshape(self.num_envs, -1),
                cursor,
            )

            randomizer_info = self.randomizer.get_randomize_state()

            if randomizer_info is not None:
                self.states_buf, cursor = self._fill(
                    self.states_buf, randomizer_info.reshape(self.num_envs, -1), cursor
                )
            else:
                print("Randomizer is not initialized.")

        self.states_buf, cursor = self._fill(
            self.states_buf, self.cube_shape_id.reshape(self.num_envs, -1), cursor
        )
        self.states_buf = torch.nan_to_num(
            self.states_buf, nan=0.0, posinf=1.0, neginf=-1.0
        )

        return self.states_buf

    @staticmethod
    def _fill(buf, x, start_pos):
        width = x.size(1)
        buf[:, start_pos : start_pos + width] = x
        return buf, start_pos + width

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.hand_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :,
            self.allegro_dof_handles,
        ]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.ur_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :, self.ur_dof_handles
        ]
        self.ur_dof_pos = self.ur_dof_state[..., 0]
        self.ur_dof_vel = self.ur_dof_state[..., 1]

        # refresh states
        self._update_states()


    def _create_camera(self, env_ptr):
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.cam_w
        cam_props.height = self.cam_h

        cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
        self.env_camera_handles.append(cam_handle)
        cam_pos = gymapi.Vec3(1.0, 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0.4)
        result = self.gym.set_camera_location(cam_handle, env_ptr, cam_pos, cam_target)


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs))
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # --------------------------------------------------------------------------------------
        #                                   Load Assets
        # --------------------------------------------------------------------------------------
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../assets"
        )

        if self.use_mirrored_urdf:
            asset_file = "urdf/ur5e_allegro/robots/dual_ur5e_allegro.urdf"
        elif self.use_updated_urdf:
            asset_file = "urdf/ur5e_allegro/robots/dual_ur5e_allegro_real_v2.urdf"
        else:
            asset_file = "urdf/ur5e_allegro/robots/dual_ur5e_allegro_real.urdf"

        object_asset_manager = AssetManager(
            asset_root=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../../assets"
            ),
            model_root_path=self.cfg["env"]["object_asset_root_folder"],
        )
        self.object_asset_manager = object_asset_manager

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = self.disable_gravity
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        dual_ur_allegro_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        allegro_dof_names = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
        ]
        allegro_dof_names = allegro_dof_names + [
            dof_name + "_r" for dof_name in allegro_dof_names
        ]
        ur_dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        ur_dof_names = ur_dof_names + [dof_name + "_r" for dof_name in ur_dof_names]
        self.allegro_dof_handles = to_torch(
            [
                self.gym.find_asset_dof_index(dual_ur_allegro_asset, allegro_dof_name)
                for allegro_dof_name in allegro_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.ur_dof_handles = to_torch(
            [
                self.gym.find_asset_dof_index(dual_ur_allegro_asset, ur_dof_name)
                for ur_dof_name in ur_dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        # Load Cube Assets
        bottle_asset_options = gymapi.AssetOptions()
        bottle_asset_options.flip_visual_attachments = False
        bottle_asset_options.collapse_fixed_joints = False
        bottle_asset_options.disable_gravity = False
        bottle_asset_options.thickness = 0.001
        bottle_asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            bottle_asset_options.use_physx_armature = True
        bottle_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # --------------------------------------------------------------------------------------
        #                                   Setup Both Hands
        # --------------------------------------------------------------------------------------

        self.num_allegro_bodies = self.gym.get_asset_rigid_body_count(
            dual_ur_allegro_asset
        )
        self.num_allegro_dofs = self.gym.get_asset_dof_count(dual_ur_allegro_asset)

        print("num Allegro Bodies: ", self.num_allegro_bodies)
        print("num Allegro Dofs: ", self.num_allegro_dofs)

        # set franka dof properties
        allegro_dof_props = self.gym.get_asset_dof_properties(dual_ur_allegro_asset)

        if self.use_real_allegro_limit:
            print("Change lower limit of joint14 at index 12,34 according to init")
            allegro_dof_props["lower"][12] = self.initializer.hand_init_qpos[
                "joint_14.0"
            ]
            allegro_dof_props["lower"][34] = self.initializer.hand_init_qpos[
                "joint_14.0_r"
            ]

        self.hand_dof_upper_limits = []
        self.hand_dof_lower_limits = []

        # we only set hand dof properties, arm property is inheritied from URDF
        for i in self.ur_dof_handles:
            allegro_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS

        for i in self.allegro_dof_handles:
            allegro_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            allegro_dof_props["velocity"][i] = self.initializer.get_dof_velocity()
            allegro_dof_props["effort"][i] = 0.5
            allegro_dof_props["stiffness"][i] = self.initializer.get_hand_info()[
                "stiffness"
            ]  # 0.0# 2
            allegro_dof_props["damping"][i] = self.initializer.get_hand_info()[
                "damping"
            ]  # 0.0 #0.1
            allegro_dof_props["friction"][i] = 0.01
            allegro_dof_props["armature"][i] = self.initializer.get_hand_info()[
                "armature"
            ]  # 0.001 #0.002

            self.hand_dof_lower_limits.append(allegro_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(allegro_dof_props["upper"][i])

        self.hand_dof_upper_limits = to_torch(
            self.hand_dof_upper_limits, device=self.device
        )
        self.hand_dof_lower_limits = to_torch(
            self.hand_dof_lower_limits, device=self.device
        )
        self.allegro_dof_speed_scales = torch.ones_like(self.hand_dof_lower_limits)

        allegro_start_pose = (
            self.initializer.get_hand_base_init_pose().to_isaacgym_pose()
        )

        # --------------------------------------------------------------------------------------
        #                                   Setup Bottle.
        # --------------------------------------------------------------------------------------

        object_asset_manager.load(self, bottle_asset_options, self.initializer)
        cube_start_pose = self.initializer.get_cube_init_pose().to_isaacgym_pose()

        # --------------------------------------------------------------------------------------
        #                              Initialize Vec Environment
        # --------------------------------------------------------------------------------------

        # compute aggregate size
        num_hand_bodies = (
            self.gym.get_asset_rigid_body_count(dual_ur_allegro_asset)
            + object_asset_manager.get_asset_rigid_body_count()
        )
        num_hand_shapes = (
            self.gym.get_asset_rigid_shape_count(dual_ur_allegro_asset)
            + object_asset_manager.get_asset_rigid_shape_count()
        )
        max_agg_bodies = num_hand_bodies + 3  # 1 for table, table stand, cube
        max_agg_shapes = num_hand_shapes + 3  # 1 for table, table stand, cube

        self.allegro = []
        self.cube = []
        self.envs = []

        self.hand_indices = []
        self.all_cube_indices = []

        self.cube_shape_id = []

        if self.enable_camera_sensors:
            # self.cams = []
            self.cam_tensors = []
            self.cam_tensors_wrist = []

        self.env_camera_handles = []
        self.env_physics_setup = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Begin Routine
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            env_hand_actor_indices = []

            # Create hand actor
            allegro_actor = self.gym.create_actor(
                env_ptr,
                dual_ur_allegro_asset,
                allegro_start_pose,
                "allegro",
                i,
                -1,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_dof_props)

            prop = self.gym.get_actor_rigid_shape_properties(env_ptr, allegro_actor)
            # prop[0].restitution = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, allegro_actor, prop)

            hand_idx = self.gym.get_actor_index(
                env_ptr, allegro_actor, gymapi.DOMAIN_SIM
            )
            env_hand_actor_indices.append(hand_idx)
            self.hand_indices.append(env_hand_actor_indices)

            # Create cube actor
            (
                cube_shape_idx,
                cube_asset,
                cube_dof_props,
            ) = object_asset_manager.get_random_asset()
            self.cube_shape_id.append(cube_shape_idx)

            cube_actor = self.gym.create_actor(
                env_ptr, cube_asset, cube_start_pose, "cube", i, 1, 0
            )
            cube_body_prop = self.gym.get_actor_rigid_body_properties(
                env_ptr, cube_actor
            )

            mass_dict = self.randomizer.get_random_bottle_mass(
                cube_body_prop[0].mass, cube_body_prop[1].mass
            )
            cube_body_prop[0].mass = mass_dict["body_mass"]
            cube_body_prop[1].mass = mass_dict["cap_mass"]
            mass_scaling = mass_dict["mass_scaling"]
            # print("MASS", cube_body_prop[0].mass,  cube_body_prop[1].mass)

            # Random Object Physics
            friction_rescaling = self.randomizer.get_random_object_scaling("friction")

            cube_scaling = self.randomizer.get_random_object_scaling("scale")

            cube_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_actor)
            cube_prop[0].restitution = 0.0
            cube_prop[0].rolling_friction = 1.0
            cube_prop[0].friction = 1.5 * friction_rescaling

            cube_prop[1].restitution = 0.0
            cube_prop[1].rolling_friction = 1.0
            cube_prop[1].friction = 1.5 * friction_rescaling

            # Random Object Scale
            object_scale = self.cfg["env"]["cube_scale"] * cube_scaling

            self.gym.set_actor_rigid_body_properties(
                env_ptr, cube_actor, cube_body_prop
            )
            self.gym.set_actor_rigid_shape_properties(env_ptr, cube_actor, cube_prop)
            self.gym.set_actor_scale(env_ptr, cube_actor, object_scale)
            self.gym.set_actor_dof_properties(env_ptr, cube_actor, cube_dof_props)

            self.cube.append(cube_actor)
            cube_idx = self.gym.get_actor_index(env_ptr, cube_actor, gymapi.DOMAIN_SIM)
            self.all_cube_indices.append(cube_idx)

            self.env_physics_setup.append(
                [mass_scaling, friction_rescaling, cube_scaling]
            )
            # End Routine
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro.append(allegro_actor)

            if self.enable_camera_sensors:
                self._create_camera(env_ptr)

        self.brake_joint_id = self.gym.find_actor_dof_handle(
            env_ptr, cube_actor, "brake_joint"
        )
        self.cube_joint_id = self.gym.find_actor_dof_handle(
            env_ptr, cube_actor, "b_joint"
        )

        tip_names = ["link_7.0_tip", "link_15.0_tip", "link_3.0_tip", "link_11.0_tip"]
        nonthumb_tip_names = ["link_7.0_tip", "link_3.0_tip", "link_11.0_tip"]
        thumb_tip_names = ["link_15.0_tip"]

        self.left_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, tip_name)
            for tip_name in tip_names
        ]
        self.left_tip_handles = to_torch(
            self.left_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_tip_handles = [
            self.gym.find_actor_rigid_body_handle(
                env_ptr, allegro_actor, tip_name + "_r"
            )
            for tip_name in tip_names
        ]
        self.right_tip_handles = to_torch(
            self.right_tip_handles, dtype=torch.long, device=self.device
        )

        self.left_thumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, tip_name)
            for tip_name in thumb_tip_names
        ]
        self.left_thumb_tip_handles = to_torch(
            self.left_thumb_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_thumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(
                env_ptr, allegro_actor, tip_name + "_r"
            )
            for tip_name in thumb_tip_names
        ]
        self.right_thumb_tip_handles = to_torch(
            self.right_thumb_tip_handles, dtype=torch.long, device=self.device
        )

        self.left_nonthumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, allegro_actor, tip_name)
            for tip_name in nonthumb_tip_names
        ]
        self.left_nonthumb_tip_handles = to_torch(
            self.left_nonthumb_tip_handles, dtype=torch.long, device=self.device
        )
        self.right_nonthumb_tip_handles = [
            self.gym.find_actor_rigid_body_handle(
                env_ptr, allegro_actor, tip_name + "_r"
            )
            for tip_name in nonthumb_tip_names
        ]
        self.right_nonthumb_tip_handles = to_torch(
            self.right_nonthumb_tip_handles, dtype=torch.long, device=self.device
        )

        self.env_physics_setup = to_torch(
            self.env_physics_setup, dtype=torch.float, device=self.device
        )

        # Set handles
        self.bottle_cap_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, cube_actor, "link1"
        )
        # self.bottle_cap_handles = to_torch(self.bottle_cap_handles, dtype=torch.long, device=self.device)
        self.bottle_base_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, cube_actor, "link2"
        )
        print(
            self.bottle_cap_handle,
            self.bottle_base_handle,
            self.gym.find_actor_rigid_body_handle(env_ptr, cube_actor, "l10"),
        )

        (
            cap_marker_handle_names,
            base_marker_handle_names,
        ) = self.object_asset_manager.get_markers()
        self.bottle_cap_marker_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, cube_actor, name)
            for name in cap_marker_handle_names
        ]
        self.bottle_cap_marker_handles = to_torch(
            self.bottle_cap_marker_handles, dtype=torch.long, device=self.device
        )
        self.bottle_base_marker_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, cube_actor, name)
            for name in base_marker_handle_names
        ]
        self.bottle_base_marker_handles = to_torch(
            self.bottle_base_marker_handles, dtype=torch.long, device=self.device
        )

        print("CAP_MARKER_HANDLES", self.bottle_cap_marker_handles)
        print("BASE_MARKER_HANDLES", self.bottle_base_marker_handles)

        # Set shape_idx observation
        self.cube_shape_id = to_torch(
            self.cube_shape_id, dtype=torch.long, device=self.device
        )
        self.cube_shape_id = F.one_hot(self.cube_shape_id, num_classes=self.num_objects)

        # Set handles
        self.cube_handle = cube_actor  # this is the local handle idx in one env. not the global one. Global one should be determined by get_actor_index()

        # Set the default qpos
        hand_qpos_default_dict = self.initializer.get_hand_init_qpos()
        self.hand_default_qpos_info = [
            (
                self.gym.find_actor_dof_handle(env_ptr, allegro_actor, finger_name),
                hand_qpos_default_dict[finger_name],
            )
            for finger_name in hand_qpos_default_dict
        ]
        # [print(finger_name, self.gym.find_actor_dof_handle(env_ptr, allegro_right_actor, finger_name)) for finger_name in right_hand_qpos_default_dict]

        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )  # shape = [num_envs, 2]. It stores the global index of left_hand & right_hand.
        self.all_cube_indices = to_torch(
            self.all_cube_indices, dtype=torch.long, device=self.device
        )  # shape = [num_envs]. It stores the global index of the cube.

        # Setup data
        self.init_data()

    def init_data(self):
        self.handles = {}

        # get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.total_hand_dof), dtype=torch.float, device=self.device
        )

        # get gym GPU state tensors
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # setup tensor buffers
        self._root_state = gymtorch.wrap_tensor(
            _actor_root_state_tensor
        ).view(
            -1, 13
        )  # TODO()??? gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._contact_forces = gymtorch.wrap_tensor(_contact_forces).view(
            self.num_envs, -1, 3
        )

        self._env_root_state = self._root_state.view(self.num_envs, -1, 13)
        self._env_rigid_body_state = self._rigid_body_state.view(self.num_envs, -1, 13)

        self.all_dof_pos = self._dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.all_dof_vel = self._dof_state.view(self.num_envs, -1, 2)[..., 1]

        # 46 dof, 2x16 hand + 2x6 arm
        self.hand_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :,
            self.allegro_dof_handles,
        ]
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]

        self.ur_dof_state = self._dof_state.view(self.num_envs, -1, 2)[
            :, self.ur_dof_handles
        ]
        self.ur_dof_pos = self.ur_dof_state[..., 0]
        self.ur_dof_vel = self.ur_dof_state[..., 1]

        # initialize prev cube states.
        self.prev_cube_state = self._env_root_state[:, self.cube_handle, :].clone()

        # initialize actions
        self.hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        # initialize indices
        self._global_indices = torch.arange(
            self.num_envs * 4, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

    def compute_reward(self, actions):
        # print(self.states["cube_pos"].shape)
        self.rew_buf[:], self.reset_buf[:], info = self.reward_function.forward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.reward_settings,
            self.max_episode_length,
        )
        return info

    def compute_observations(self):
        self._refresh()

        # Normalized hand dof position. [0:32]
        # dof_pos_scaled = self.states["dof_pos"]
        self.extras["dof_pos"] = self.states["dof_pos"].clone()

        dof_pos_scaled = (
            2.0
            * (
                self.randomizer.randomize_dofpos(self.states["dof_pos"])
                - self.hand_dof_lower_limits
            )
            / (self.hand_dof_upper_limits - self.hand_dof_lower_limits)
            - 1.0
        )

        # Hand dof velocity. [32:64]
        dof_vel_scaled = self.states["dof_vel"] * self.dof_vel_scale

        if self.obs_setting["no_dof_vel"]:
            dof_vel_scaled = torch.zeros_like(dof_vel_scaled)

        # Bottle Body Pos. [64:67]
        # Bottle Cap Pos. [71:74]
        self.cap_base_pos = self.states["cube_cap_pos"].reshape(self.num_envs, -1)

        cube_pos, self.cap_base_pos = self.randomizer.randomize_bottle_observation(
            self.states["cube_base_pos"], self.cap_base_pos
        )


        if self.obs_setting["no_obj_pos"]:
            cube_pos = torch.zeros_like(cube_pos)

        if self.obs_setting["no_cap_base"]:
            self.cap_base_pos = torch.zeros_like(self.cap_base_pos)

        # Cube Quat. [67:71]
        cube_quat = self.states["cube_quat"]

        if self.obs_setting["no_obj_quat"]:
            cube_quat = torch.zeros_like(cube_quat)

        obs_prev_target = self.prev_targets.clone()
        randomized_prev_target = self.randomizer.randomize_prev_target(obs_prev_target)

        # Prev_target. [74:106]
        frame_obs_buf = torch.cat(
            (
                dof_pos_scaled,
                dof_vel_scaled,
                cube_pos,
                cube_quat,
                self.cap_base_pos,
                randomized_prev_target,
            ),
            dim=-1,
        )

        if self.use_reward_obs:
            frame_obs_buf = torch.cat(
                (frame_obs_buf, self.reward_function.get_observation()), dim=-1
            )

        # Concatenate object id.
        if self.obs_setting["no_obj_id"]:
            cube_id_obs = torch.zeros_like(self.cube_shape_id)
        else:
            cube_id_obs = self.cube_shape_id

        frame_obs_buf = torch.cat((frame_obs_buf, cube_id_obs), dim=-1)

        if torch.isnan(frame_obs_buf).int().sum() > 0:
            print("Nan Detected in IsaacGym simulation.")

        frame_obs_buf = torch.nan_to_num(
            frame_obs_buf, nan=0.0, posinf=1.0, neginf=-1.0
        )
        frame_obs_buf = torch.clamp(frame_obs_buf, -100.0, 100.0)

        frame_obs_buf = self.randomizer.randomize_frame_obs_buffer(frame_obs_buf)
        if self.n_stack_frame == 1:
            self.obs_buf = frame_obs_buf.clone()
        else:
            self.obs_buf = torch.cat(
                (
                    frame_obs_buf[:, : self.single_frame_obs_dim],
                    self.obs_buf[:, : -self.single_frame_obs_dim],
                ),
                dim=-1,
            )
        if self.enable_camera_sensors:
            self.compute_pixel_obs()
        return self.obs_buf

    def compute_pixel_obs(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            image = self.gym.get_camera_image(
                self.sim, self.envs[i], self.env_camera_handles[i], gymapi.IMAGE_COLOR
            )

            image = image.reshape(self.cam_h, self.cam_w, -1)[:, crop_l:crop_r, :3]

            tensor_image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

            self.pix_obs_buf[i] = tensor_image
            if i == 0 and self.viewer:
                import cv2

                cv2.imshow("image", image[:, :, ::-1])
                cv2.waitKey(10)

        self.gym.end_access_image_tensors(self.sim)
        # print(self.pix_obs_buf[0], torch.max(self.pix_obs_buf[0]))
        return {"third-person": self.pix_obs_buf}

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # reset controller setup
        p_lower, p_upper, d_lower, d_upper = self.randomizer.get_pd_gain_scaling_setup()
        self.randomize_p_gain_lower = self.p_gain_val * p_lower  # 0.30
        self.randomize_p_gain_upper = self.p_gain_val * p_upper  # 0.60
        self.randomize_d_gain_lower = self.d_gain_val * d_lower  # 0.75
        self.randomize_d_gain_upper = self.d_gain_val * d_upper  # 1.05

        self.p_gain[env_ids] = torch_rand_float(
            self.randomize_p_gain_lower,
            self.randomize_p_gain_upper,
            (len(env_ids), self.num_actions),
            device=self.device,
        ).squeeze(1)
        self.d_gain[env_ids] = torch_rand_float(
            self.randomize_d_gain_lower,
            self.randomize_d_gain_upper,
            (len(env_ids), self.num_actions),
            device=self.device,
        ).squeeze(1)

        # reset hand dof.
        pos = self.hand_default_dof_pos.unsqueeze(0)
        self.hand_dof_pos[env_ids, :] = pos.clone()
        self.ur_dof_pos[env_ids] = self.ur_default_dof_pos[None]

        # only randomize hand, fix arm
        self.hand_dof_pos[env_ids] = self.randomizer.randomize_hand_init_qpos(
            self.hand_dof_pos[env_ids]
        )
        self.hand_dof_vel[env_ids, :] = torch.zeros_like(self.hand_dof_vel[env_ids])

        # reset cube dof
        self.all_dof_pos[env_ids, self.cube_joint_id] = 0.0
        self.all_dof_vel[env_ids, self.cube_joint_id] = 0.0
        self.all_dof_pos[env_ids, self.brake_joint_id] = 0.01
        self.all_dof_vel[env_ids, self.brake_joint_id] = 0.0

        # Cube Cartesian Position
        self._root_state[
            self.all_cube_indices[env_ids], 0:3
        ] = self.cube_init_pose.unsqueeze(0)[:, 0:3]
        self._root_state[
            self.all_cube_indices[env_ids], 0:3
        ] = self.randomizer.randomize_object_init_pos(
            self._root_state[self.all_cube_indices[env_ids], 0:3]
        )

        # Cube Orientation
        self._root_state[
            self.all_cube_indices[env_ids], 3:7
        ] = self.cube_init_pose.unsqueeze(0)[:, 3:7]

        # print("SHAPE", self.cube_init_pose.unsqueeze(0).shape, self._root_state.shape)
        rotation = self.randomizer.randomize_object_init_quat(
            self.cube_init_pose.repeat(len(env_ids), 1)[:, 3:7]
        )

        self._root_state[self.all_cube_indices[env_ids], 3:7] = rotation

        # Cube 6DOF Velocity
        self._root_state[self.all_cube_indices[env_ids], 7:13] = torch.zeros_like(
            self._root_state[self.all_cube_indices[env_ids], 7:13]
        )

        reset_object_indices = (self.all_cube_indices[env_ids]).int()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(reset_object_indices),
            len(reset_object_indices),
        )

        reset_hand_indices = self.hand_indices[env_ids].to(torch.int32).reshape(-1)
        reset_actor_indices = torch.cat((reset_hand_indices, reset_object_indices))
        self._dof_state.view(self.num_envs, -1, 2)[
            :, self.ur_dof_handles, 0
        ] = self.ur_default_dof_pos[None]
        self._dof_state.view(self.num_envs, -1, 2)[:, self.ur_dof_handles, 1] = 0
        self._dof_state.view(self.num_envs, -1, 2)[
            :, self.allegro_dof_handles, 0
        ] = self.hand_dof_pos.clone()
        self._dof_state.view(self.num_envs, -1, 2)[
            :, self.allegro_dof_handles, 1
        ] = self.hand_dof_vel.clone()
        # Reset the hands' dof and cube's dof jointly.
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(reset_actor_indices),
            len(reset_actor_indices),
        )

        # Reset observation
        self.obs_buf[env_ids, :] = 0.0
        self.states_buf[env_ids, :] = 0.0

        # Reset buffer.
        self.last_actions[env_ids, :] = 0.0
        self.last_cube_dof_pos[env_ids] = 0.0

        # Reset controller target.
        self.prev_targets[env_ids, :] = pos
        self.cur_targets[env_ids, :] = pos

        # Rest cube state recorder
        self.prev_cube_state[env_ids, ...] = self._env_root_state[
            env_ids, self.cube_handle, :
        ].clone()

        self.progress_buf[env_ids] = 0
        self.force_progress_buf[env_ids] = -1000

        # Reset torque setup
        (
            torque_lower,
            torque_upper,
        ) = self.randomizer.get_object_dof_friction_scaling_setup()
        torque_lower, torque_upper = (
            self.brake_torque * torque_lower,
            self.brake_torque * torque_upper,
        )
        self.object_brake_torque[env_ids] = (
            torch.rand(len(env_ids)).to(self.device) * (torque_upper - torque_lower)
            + torque_lower
        )

        # Reset reward function (goal resets)
        self.reward_function.reset(env_ids)
        self.randomizer.reset(env_ids)

        self.left_control_work = torch.zeros_like(self.left_control_work)
        self.right_control_work = torch.zeros_like(self.right_control_work)
        self.control_work = torch.zeros_like(self.control_work)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.actions = self.randomizer.randomize_action(self.actions)

        self.actions = torch.clamp(self.actions, -1, 1)
        assert torch.isnan(self.actions).int().sum() == 0, "nan detected"

        # smooth our action.
        self.actions = self.actions * self.action_moving_average + self.last_actions * (
            1.0 - self.action_moving_average
        )

        self.cur_targets[:] = (
            self.cur_targets + self.controller_action_scale * self.actions
        )
        self.cur_targets[:] = tensor_clamp(
            self.cur_targets,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

        self.prev_targets = self.cur_targets.clone()
        self.last_actions = self.actions.clone().to(self.device)
        self.last_cube_dof_pos = self.all_dof_pos[:, -1].clone()

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(
                self.force_decay, self.dt / self.force_decay_interval
            )

            obj_mass = to_torch(
                [
                    self.gym.get_actor_rigid_body_properties(
                        env, self.gym.find_actor_handle(env, "cube")
                    )[0].mass
                    for env in self.envs
                ],
                device=self.device,
            )

            prob = self.force_prob
            force_indices_candidate = (
                torch.less(torch.rand(self.num_envs, device=self.device), prob)
            ).nonzero()
            last_force_progress = self.force_progress_buf[force_indices_candidate]
            current_progress = self.progress_buf[force_indices_candidate]

            valid_indices = torch.where(
                current_progress
                > last_force_progress
                + torch.randint(
                    20, 50, (len(force_indices_candidate),), device=self.device
                ).unsqueeze(-1)
            )[0]


            force_indices = force_indices_candidate[valid_indices]

            self.force_progress_buf[force_indices] = self.progress_buf[force_indices]

            step = self.progress_buf[force_indices]  # [N, 1]
            horizon_decay = torch.pow(self.force_horizon_decay, step)

            for i, axis_scale in enumerate(
                [self.force_scale_x, self.force_scale_y, self.force_scale_z]
            ):
                self.rb_forces[force_indices, self.bottle_base_handle, i] = (
                    torch.randn(
                        self.rb_forces[force_indices, self.bottle_base_handle, i].shape,
                        device=self.device,
                    )
                    * horizon_decay
                    * obj_mass[force_indices]
                    * axis_scale
                    * self.force_scale
                )
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.rb_forces),
                None,
                gymapi.ENV_SPACE,
            )

    def update_controller(self):
        previous_dof_pos = self.hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._refresh()

        dof_pos = self.hand_dof_pos
        dof_vel = (dof_pos - previous_dof_pos) / self.dt

        self.dof_vel_finite_diff = dof_vel.clone()
        hand_torques = (
            self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
        )

        torques = torch.zeros(self.num_envs, self.num_dofs).to(self.device)
        torques[:, self.allegro_dof_handles] = hand_torques
        torques = torch.clip(torques, -1.0, 1.0)

        # Brake applies the force
        torques[
            :, self.brake_joint_id
        ] = self.object_brake_torque  # -10.0 # adjust this to apply different friction.

        all_work = torques[:, self.allegro_dof_handles] * self.dof_vel_finite_diff
        self.left_control_work += all_work[:, :16].abs().sum(-1) * self.dt
        self.right_control_work += all_work[:, 16:].abs().sum(-1) * self.dt
        self.control_work +=  (self.left_control_work + self.right_control_work)
      
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(torques)
        )
        ur_target = torch.zeros(
            self.num_envs, self.num_dofs, device=self.cur_targets.device
        )
        ur_target[:, self.ur_dof_handles] = self.ur_default_dof_pos[None]
        ur_target[:, self.allegro_dof_handles] = 0
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(ur_target)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh()

        reward_info = self.compute_reward(self.actions)
        self.extras.update(reward_info)

        self.left_control_work = torch.zeros_like(self.left_control_work)
        self.right_control_work = torch.zeros_like(self.right_control_work)
        self.control_work = torch.zeros_like(self.control_work)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

        self.prev_cube_state = self._env_root_state[:, self.cube_handle, :].clone()

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.reward_function.render(self, self.envs[i], i)
                CAP_Y = 0.00
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [-1, CAP_Y, 1.24, 1, CAP_Y, 1.24],
                    [1, 0, 0],
                )

                CAP_X = 0.70
                self.gym.add_lines(
                    self.viewer,
                    self.envs[i],
                    1,
                    [CAP_X, -1, 1.24, CAP_X, 1, 1.24],
                    [1, 0, 0],
                )



    def update_curriculum(self, steps):
        # print("update reward curriculum")
        if not self.reward_settings["use_curriculum"]:
            return

        if "screw_curriculum" in self.reward_settings:
            low, high = self.reward_settings["screw_curriculum"]
            r_low, r_high = self.reward_settings["screw_curriculum_reward_scale"]
            if steps < low:
                scale = r_low

            elif steps > high:
                scale = r_high

            else:
                scale = r_low + (r_high - r_low) * (steps - low) / (high - low)
            self.reward_settings["rotation_reward_scale"] = scale
        # print(self.reward_settings)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat(
        [
            vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
            torch.cos(angle[idx, :] / 2.0),
        ],
        dim=-1,
    )

    # Reshape and return output
    quat = quat.reshape(
        list(input_shape)
        + [
            4,
        ]
    )
    return quat
