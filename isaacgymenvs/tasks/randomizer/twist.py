import torch
import numpy as np
from isaacgymenvs.tasks.randomizer.base import BaseRandomizer


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def wxyz_xyzw_qmul(q, r):
    """
    Multiply quaternion(s) q (wxyz convention) with quaternion(s) r (xyzw convention).
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    # xyzw to wxyz
    r = r[..., [3, 0, 1, 2]]

    qr = quaternion_multiply(q, r)

    # wxyz to xyzw
    qr = qr[..., [1, 2, 3, 0]]
    return qr


class Randomizer(BaseRandomizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg = kwargs.get("cfg")
        print("Jan19 Randomizer")
        self.last_action = None
        self.last_bottle_obs_shift = None
        self.last_frame_obs = None

        self.noise_setup = cfg.get("noise", {})

        self.frame_latency_prob = cfg.get("frame_latency_prob", 0.1)
        self.action_latency_prob = cfg.get("action_latency_prob", 0.1)
        self.action_drop_prob = cfg.get("action_drop_prob", 0.1)  # TODO()

        self.observation_noise_scale = cfg.get("observation_noise_scale", 0.0005)

        self.hand_init_qpos_noise_scale = cfg.get("hand_init_qpos_noise_scale", 0.005)

        self.object_init_pos_noise_x_scale = cfg.get(
            "object_init_pos_noise_x_scale", 0.0
        )
        self.object_init_pos_noise_y_scale = cfg.get(
            "object_init_pos_noise_y_scale", 0.0
        )
        self.object_init_pos_noise_z_scale = cfg.get(
            "object_init_pos_noise_z_scale", 0.02
        )
        self.object_init_quat_noise_x_scale = cfg.get(
            "object_init_quat_noise_x_scale", 0.001
        )
        self.object_init_quat_noise_y_scale = cfg.get(
            "object_init_quat_noise_y_scale", 0.0
        )
        self.object_init_quat_noise_z_scale = cfg.get(
            "object_init_quat_noise_z_scale", 0.001
        )

        self.bottle_obs_shift_scale = cfg.get("bottle_obs_shift_scale", 0.01)
        self.bottle_obs_shift_reset_prob = cfg.get("bottle_obs_shift_reset_prob", 0.05)
        # self.bottle_obs_noise_scale = cfg.get("bottle_obs_noise_scale", 0.0005)

        self.object_param_random_scales = {}

        self.scale_randomization_lower = cfg.get("scale_randomization_lower", 0.9)
        self.scale_randomization_upper = cfg.get("scale_randomization_upper", 1.5)
        self.object_param_random_scales["scale"] = (
            self.scale_randomization_lower,
            self.scale_randomization_upper,
        )

        self.mass_randomization_lower = cfg.get("mass_randomization_lower", 0.9)
        self.mass_randomization_upper = cfg.get("mass_randomization_upper", 1.1)
        self.object_param_random_scales["mass"] = (
            self.mass_randomization_lower,
            self.mass_randomization_upper,
        )

        self.randomize_mass_by_value = cfg.get("randomize_mass_by_value", False)
        self.mass_value_lower = cfg.get("mass_value_lower", 0.030)  # 30g
        self.mass_value_upper = cfg.get("mass_value_upper", 0.150)  # 150g

        self.object_param_random_scales["mass_value"] = (
            self.mass_value_lower,
            self.mass_value_upper,
        )

        self.friction_randomization_lower = cfg.get("friction_randomization_lower", 0.9)
        self.friction_randomization_upper = cfg.get("friction_randomization_upper", 1.1)
        self.object_param_random_scales["friction"] = (
            self.friction_randomization_lower,
            self.friction_randomization_upper,
        )

        # CONTROLLER
        self.p_gain_lower = cfg.get("p_gain_lower")
        self.p_gain_upper = cfg.get("p_gain_upper")
        self.d_gain_lower = cfg.get("d_gain_lower")
        self.d_gain_upper = cfg.get("d_gain_upper")

        # DOF
        self.object_dof_friction_lower = cfg.get("object_dof_friction_lower")
        self.object_dof_friction_upper = cfg.get("object_dof_friction_upper")
        self._display_attributes()

        # Per-step noises
        self.dofpos_noise_scale = cfg.get("dofpos_noise_scale", 0.1)
        self.prev_target_noise_scale = cfg.get("prev_target_noise_scale", 0.1)
        self.bottle_obs_noise_scale = cfg.get("bottle_obs_noise_scale", 0.0005)
        self.action_noise_scale = cfg.get("action_noise_scale", 0.05)
        
        self.separate_bottle_perstep_noise = cfg.get("separate_bottle_perstep_noise", True)

        # Episode noises
        self.set_episode_additive_noise_scale(
            cfg.get("action_episode_additive_noise_scale", 0.0), "action"
        )
        self.set_episode_affine_noise_scale(
            cfg.get("action_episode_affine_noise_scale", 0.0), "action"
        )

        self.set_episode_additive_noise_scale(
            cfg.get("dofpos_episode_additive_noise_scale", 0.0), "dofpos"
        )
        self.set_episode_affine_noise_scale(
            cfg.get("dofpos_episode_affine_noise_scale", 0.0), "dofpos"
        )

        self.set_episode_additive_noise_scale(
            cfg.get("prev_target_episode_additive_noise_scale", 0.0), "prev_target"
        )
        self.set_episode_affine_noise_scale(
            cfg.get("prev_target_episode_affine_noise_scale", 0.0), "prev_target"
        )

        self.set_episode_additive_noise_scale(
            cfg.get("bottle_episode_additive_noise_scale", 0.0), "bottle"
        )
        self.set_episode_affine_noise_scale(
            cfg.get("bottle_episode_affine_noise_scale", 0.0), "bottle"
        )

        self.episode_additive_noise_names = []
        self.episode_affine_noise_names = []

        # Measure noise deltas.
        self.last_action_noise_info = None
        self.last_dofpos_noise_info = None
        self.last_object_noise_info = None
        self.last_prevtarget_noise_info = None

        # TODO: randomize gravity?
        return

    def _display_attributes(self):
        for key, value in self.__dict__.items():
            if key == "cfg":
                continue

            print(f"[Randomization setup] {key}: {value}")

    def get_random_bottle_mass(self, body_mass, cap_mass):
        if self.randomize_mass_by_value:
            mass_scaling = self.get_random_object_scaling("mass_value")
            ratio = np.random.uniform(0.8, 0.95)
            body_mass = mass_scaling * ratio
            cap_mass = mass_scaling * (1 - ratio)
        else:
            mass_scaling = self.get_random_object_scaling("mass")
            body_mass = body_mass * mass_scaling
            cap_mass = cap_mass * mass_scaling
        return {
            "mass_scaling": mass_scaling,
            "body_mass": body_mass,
            "cap_mass": cap_mass,
        }

    def get_random_object_scaling(self, key):
        if key not in self.object_param_random_scales:
            print(f"WARNING: Randomization param {key} not found.")
            return 1.0

        lower, upper = self.object_param_random_scales[key]
        return np.random.uniform(lower, upper, 1)[0]

    def get_object_dof_friction_scaling_setup(self):
        return 0.9999, 1.0000

    def _add_white_noise(self, x, scale):
        return x + torch.randn_like(x) * scale

    def _get_white_noise(self, x, scale):
        return torch.randn_like(x) * scale

    def randomize_dofpos(self, dofpos):
        noise_info, randomized_dofpos = self.randomize_with_noise(
            dofpos,
            perstep_noise_scale=self.dofpos_noise_scale,
            apply_episode_additive_noise=True,
            apply_episode_affine_noise=True,
            name="dofpos",
            return_raw=True,
        )
        self.last_dofpos_noise_info = noise_info[
            "additive"
        ]  # randomized_dofpos - dofpos
        return randomized_dofpos

    def randomize_observation(self, observation):
        # TODO: model observation delay and latency, add per-episode correlated noise
        assert False, "Legacy code."
        return self._add_white_noise(observation, self.observation_noise_scale)

    def randomize_action(self, action):
        if self.last_action is None:
            self.last_action = action

        latency_mask = (
            (torch.rand_like(action) < self.action_latency_prob)[:, 0:1]
        ).float()
        action = self.last_action * latency_mask + action * (1 - latency_mask)

        # mask = torch.rand_like(action) < self.action_drop_prob

        # action[mask] = 0  # action[mask] * 0.1
        # action[~mask] = self._add_white_noise(action[~mask], self.action_noise_scale)
        noise_info, randomized_action = self.randomize_with_noise(
            action,
            perstep_noise_scale=self.action_noise_scale,
            apply_episode_additive_noise=True,
            apply_episode_affine_noise=True,
            name="action",
            return_raw=True,
        )

        # action = self._add_white_noise(action, self.action_noise_scale)
        self.last_action = randomized_action
        self.last_action_noise_info = noise_info["additive"]

        return randomized_action

    def randomize_hand_init_qpos(self, qpos):
        return self._add_white_noise(qpos, self.hand_init_qpos_noise_scale)

    def randomize_object_init_pos(self, pos):
        # print("POS BEFORE", pos)
        for i, axis_noise_scale in enumerate(
            [
                self.object_init_pos_noise_x_scale,
                self.object_init_pos_noise_y_scale,
                self.object_init_pos_noise_z_scale,
            ]
        ):
            pos[:, i] += torch.randn_like(pos[:, i]) * axis_noise_scale
        # print("POS AFTER", pos)
        return pos

    def randomize_object_init_quat(self, quat):
        B = quat.size(0)
        # print("ANGLE BEFORE", torch.rad2deg(quaternion_to_axis_angle(quat[:, [3, 0, 1, 2]])))
        for i, axis_noise_scale in enumerate(
            [
                self.object_init_quat_noise_x_scale,
                self.object_init_quat_noise_y_scale,
                self.object_init_quat_noise_z_scale,
            ]
        ):
            axis = torch.tensor([[0, 0, 0]])
            axis[0, i] = 1
            angle = (torch.rand(B, 1) - 0.5) * 2 * axis_noise_scale
            axis_angle = (axis * angle).to(quat.device)
            rotation_quat = axis_angle_to_quaternion(axis_angle)
            quat = wxyz_xyzw_qmul(rotation_quat, quat)
        # print("ANGLE AFTER", torch.rad2deg(quaternion_to_axis_angle(quat[:, [3, 0, 1, 2]])))
        return quat

    def get_pd_gain_scaling_setup(self):
        return (
            self.p_gain_lower,
            self.p_gain_upper,
            self.d_gain_lower,
            self.d_gain_upper,
        )

    def randomize_bottle_observation(self, base_obs, cap_obs):
        raw_randomization, _ = self.randomize_with_noise(
            cap_obs,
            perstep_noise_scale=self.bottle_obs_noise_scale,
            apply_episode_additive_noise=True,
            apply_episode_affine_noise=True,
            name="bottle",
            return_raw=True,
        )
        affine = raw_randomization["affine"]
        perstep = raw_randomization["perstep"]
        additive = raw_randomization["additive"]

        if self.separate_bottle_perstep_noise:
            base_perstep = self._get_white_noise(base_obs, self.bottle_obs_noise_scale)
            cap_perstep = self._get_white_noise(cap_obs, self.bottle_obs_noise_scale)
        else:
            base_perstep = perstep
            cap_perstep = perstep

        base_obs_randomized = base_obs * affine + base_perstep + additive
        cap_obs_randomized = cap_obs * affine + cap_perstep + additive
        self.last_object_noise_info = additive 
        return base_obs_randomized, cap_obs_randomized

    # Additive Noise
    def set_episode_additive_noise_scale(self, scale, name):
        setattr(self, f"{name}_episode_additive_noise_scale", scale)

    def get_episode_additive_noise_scale(self, name):
        return getattr(self, f"{name}_episode_additive_noise_scale")

    def set_episode_additive_noise(self, x, name):
        setattr(self, f"episode_{name}_additive_noise", x)
        if name not in self.episode_additive_noise_names:
            self.episode_additive_noise_names.append(name)

    def get_episode_additive_noise(self, name):
        try:
            return getattr(self, f"episode_{name}_additive_noise")
        except Exception:
            return None

    # Affine Noise
    def set_episode_affine_noise_scale(self, scale, name):
        setattr(self, f"{name}_episode_affine_noise_scale", scale)

    def get_episode_affine_noise_scale(self, name):
        return getattr(self, f"{name}_episode_affine_noise_scale")

    def set_episode_affine_noise(self, x, name):
        setattr(self, f"episode_{name}_affine_noise", x)
        if name not in self.episode_affine_noise_names:
            self.episode_affine_noise_names.append(name)

    def get_episode_affine_noise(self, name):
        try:
            return getattr(self, f"episode_{name}_affine_noise")
        except Exception:
            return None

    def reset(self, env_ids):
        for episode_noise_name in self.episode_additive_noise_names:
            episode_noise = self.get_episode_additive_noise(episode_noise_name)
            episode_noise_scale = self.get_episode_additive_noise_scale(
                episode_noise_name
            )
            episode_noise[env_ids] = self._get_white_noise(
                episode_noise[env_ids], episode_noise_scale
            )

        for episode_noise_name in self.episode_affine_noise_names:
            episode_noise = self.get_episode_affine_noise(episode_noise_name)
            episode_noise_scale = self.get_episode_affine_noise_scale(
                episode_noise_name
            )
            episode_noise[env_ids] = self._get_white_noise(
                episode_noise[env_ids], episode_noise_scale
            )

    def randomize_with_noise(
        self,
        x,
        perstep_noise_scale,
        episode_additive_noise_scale=None,
        episode_affine_noise_scale=None,
        apply_episode_additive_noise=False,
        apply_episode_affine_noise=False,
        apply_latency=False,
        name="",
        return_raw=False,
    ):
        # Initialize affine noise and additive noise, if necessary.
        if apply_episode_additive_noise and name != "":
            if self.get_episode_additive_noise(name) is None:
                if episode_additive_noise_scale is None:
                    episode_additive_noise_scale = (
                        self.get_episode_additive_noise_scale(name)
                    )
                episode_additive_noise = self._get_white_noise(
                    x, episode_additive_noise_scale
                )
                self.set_episode_additive_noise(episode_additive_noise, name)

        if apply_episode_affine_noise and name != "":
            if self.get_episode_affine_noise(name) is None:
                if episode_affine_noise_scale is None:
                    episode_affine_noise_scale = self.get_episode_affine_noise_scale(
                        name
                    )
                episode_affine_noise = self._get_white_noise(
                    x, episode_affine_noise_scale
                )
                self.set_episode_affine_noise(episode_affine_noise, name)

        if apply_episode_affine_noise:
            affine_coeff = 1.0 + self.get_episode_affine_noise(name)
        else:
            affine_coeff = 1.0

        # Get episode noise
        if apply_episode_additive_noise:
            episode_additive_noise = self.get_episode_additive_noise(name)
        else:
            episode_additive_noise = 0.0

        # print("Affine", affine_coeff)
        # print("Additive", episode_additive_noise)

        # Get perstep noise
        perstep_noise = self._get_white_noise(x, perstep_noise_scale)

        # Create randomization
        randomized_x = affine_coeff * x + episode_additive_noise + perstep_noise

        if return_raw:
            return {
                "affine": affine_coeff,
                "additive": episode_additive_noise,
                "perstep": perstep_noise,
            }, randomized_x

        else:
            return randomized_x


    def randomize_prev_target(self, x):
        noise_info, randomized_prev_target = self.randomize_with_noise(
            x,
            perstep_noise_scale=self.prev_target_noise_scale,
            apply_episode_additive_noise=True,
            apply_episode_affine_noise=True,
            name="prev_target",
            return_raw=True,
        )
        self.last_prevtarget_noise_info = noise_info["additive"]
        return randomized_prev_target

    def randomize_frame_obs_buffer(self, frame_obs_buffer):
        if self.last_frame_obs is None:
            self.last_frame_obs = frame_obs_buffer.clone()
        latency_mask = (
            (torch.rand_like(frame_obs_buffer) < self.frame_latency_prob)[:, 0:1]
        ).float()

        randomized_frame_obs_buffer = (
            self.last_frame_obs * latency_mask + frame_obs_buffer * (1 - latency_mask)
        )

        self.last_frame_obs = frame_obs_buffer.clone()
        return randomized_frame_obs_buffer

    def get_randomize_state(self):
        tensors = []
        if self.last_action_noise_info is not None:
            tensors.append(self.last_action_noise_info)

        if self.last_dofpos_noise_info is not None:
            tensors.append(self.last_dofpos_noise_info)

        if self.last_object_noise_info is not None:
            tensors.append(self.last_object_noise_info)

        if self.last_prevtarget_noise_info is not None:
            tensors.append(self.last_prevtarget_noise_info)

        if len(tensors) == 0:
            return None

        return torch.cat(tensors, dim=-1)


def build(**kwargs):
    return Randomizer(**kwargs)
