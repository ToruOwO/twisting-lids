import torch
import torch.nn
import torch.nn.functional as F


class BaseRewardFunction:
    def __init__(self, **kwargs) -> None:
        self.device = kwargs['device']
        self.num_envs = kwargs['num_envs']
        self.reward_settings = kwargs['reward_settings']
        return

    def obs_dim(self):
        return 0
    
    def get_observation(self):
        '''
            get goal vector.
        '''
        return None
        
    def reset(self, env_ids):
        return

    def forward(self, reset_buf, progress_buf, actions, states, reward_settings, max_episode_length):
        reset_buf = torch.where(states["cube_pos"][:, 2] < 0.3, torch.ones_like(reset_buf), reset_buf)
        reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
        
        # distance from left hand to the cube
        reward = torch.zeros_like(reset_buf).reshape(-1)

        # Failing Penalty
        reward = torch.where(states["cube_pos"][:, 2] < 0.3, reward + reward_settings['failure_penalty'], reward)

        reward = reward + states['cube_dof_vel'] * reward_settings['rotation_reward_scale']

        # Action penalty.
        action_penalty = torch.sum(actions ** 2, dim=-1)
        reward = reward + action_penalty * reward_settings['action_penalty_scale']
        return reward, reset_buf, None

    def render(self, env, env_ptr, env_id:int):
        return
    
def build(**kwargs):
    return BaseRewardFunction(**kwargs)
