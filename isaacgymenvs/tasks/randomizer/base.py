import torch
import numpy as np


class BaseRandomizer:
    def __init__(self, **kwargs):
        return

    def reset(self, env_ids):
        return

    def get_random_object_scaling(self, key):
        return 1.0

    def get_object_dof_friction_scaling_setup(self):
        return 0.9999, 1.0000

    def _add_white_noise(self, x, scale):
        return x

    def randomize_observation(self, observation):
        return observation

    def randomize_cube_observation(self, observation):
        return observation

    def randomize_cap_observation(self, observation):
        return observation

    def randomize_action(self, action):
        return action

    def randomize_hand_init_qpos(self, qpos):
        return qpos

    def randomize_object_init_pos(self, pos):
        return pos

    def randomize_object_init_quat(self, quat):
        return quat

    def get_pd_gain_scaling_setup(self):
        return 0.999, 1.0, 0.999, 1.0

    def randomize_bottle_observation(self, base_obs, cap_obs):
        return base_obs, cap_obs

    def randomize_frame_obs_buffer(self, frame_obs_buffer):
        return frame_obs_buffer

    def randomize_dofpos(self, dofpos):
        return dofpos

    def randomize_prev_target(self, x):
        return x

    def get_randomize_state(self):
        return None

    def get_random_bottle_mass(self, body_mass, cap_mass):
        print("NO MASS RANDOMIZATION")
        return {
            "mass_scaling": 1.0,
            "body_mass": body_mass,
            "cap_mass": cap_mass,
        }


def build(**kwargs):
    return BaseRandomizer(**kwargs)
