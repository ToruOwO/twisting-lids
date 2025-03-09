import torch
import torch.nn
import torch.nn.functional as F
from isaacgymenvs.tasks.rewarder.base import BaseRewardFunction


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def xyzw_to_wxyz(quat):
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


class RewardFunction(BaseRewardFunction):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rot_score = torch.zeros((self.num_envs,)).to(self.device)
        self.z_score = torch.zeros((self.num_envs,)).to(self.device)

        return

    def forward(
        self,
        reset_buf,
        progress_buf,
        actions,
        states,
        reward_settings,
        max_episode_length,
    ):
        info_dict = {}

        # Early termination.
        reset_buf = torch.where(
            states["cube_pos"][:, 2] < reward_settings["drop_threshold"],
            torch.ones_like(reset_buf),
            reset_buf,
        )
        reset_buf = torch.where(
            progress_buf >= max_episode_length - 1,
            torch.ones_like(reset_buf),
            reset_buf,
        )

        reward = torch.zeros_like(reset_buf).reshape(-1)
        reward = reward.float()

        # Distance Penalty
        # distance_to_center = torch.norm(states["cube_pos"] - torch.FloatTensor([0.0, 0.0, 0.5]).to(self.device), dim=-1, p=2)
        # penalty = - distance_to_center
        # penalty = torch.clamp(penalty, -0.05, 0)
        # reward = reward + penalty * reward_settings['distance_penalty_scale']

        # Failing Penalty
        assert reward_settings["failure_penalty"] <= 0, "penalty must <= 0"
        reward = torch.where(
            states["cube_pos"][:, 2] < reward_settings["drop_threshold"],
            reward + reward_settings["failure_penalty"],
            reward,
        )

        # Encourage rotating the face.
        # I checked the rollout of random policy. The dof_vel scale is usually in 1e-7 (not rotating) - 1e-1 (rotating).
        rotation_reward = (
            states["cube_dof_pos"] - states["last_cube_dof_pos"]
        )  # diff = 0.01 is good enough.
        # print(states['cube_dof_pos'])
        # print(states['cube_dof_pos'])
        # print(rotation_reward)
        # print(rotation_reward)

        rotation_reward = torch.nan_to_num(
            rotation_reward, nan=0.0, posinf=1.0, neginf=-1.0
        )
        rotation_reward = torch.clamp(rotation_reward, -0.02, 0.02)

        info_dict["cube_dof_pos_delta"] = rotation_reward
        # rot_score_update_idx = torch.where(progress_buf < 200)
        self.rot_score += rotation_reward

        # rotation_reward = torch.clamp(rotation_reward, -15, 15)
        reward = reward + rotation_reward * reward_settings["rotation_reward_scale"]
        # print(rotation_reward)

        # Encourage dof force
        # dof_force = states["dof_force"]
        # reward = reward + torch.clamp(dof_force, -0.5, 0.5) * reward_settings["force_reward_scale"]
        # print(dof_force)

        # Encourage finger reaching
        left_tips_pos = states["left_tips_pos"]
        right_nonthumb_tips_pos = states["right_nonthumb_tips_pos"]
        right_thumb_tips_pos = states["right_thumb_tips_pos"]

        base_marker_pos = states["cube_base_marker_pos"]  # [N, K1, 3]
        cap_marker_pos = states["cube_cap_marker_pos"]  # [N, K2, 3]

        # print(base_pos.shape, cap_pos.shape)
        # cube_pos = states["cube_pos"]
        # base_pos_expanded = base_pos.reshape(left_tips_pos.size(0), 1, 3).repeat(1, left_tips_pos.size(1), 1)
        # cap_pos_expanded = cap_pos.reshape(left_tips_pos.size(0), 1, 3).repeat(1, left_tips_pos.size(1), 1)

        base_marker_pos = base_marker_pos.unsqueeze(1)  # [N, 1, K1, 3]
        cap_marker_pos = cap_marker_pos.unsqueeze(1)  # [N, 1, K2, 3]

        # print("BAS", base_marker_pos[0])

        # print("CAP", cap_marker_pos[0])
        # print("LEF", left_tips_pos[0])
        left_tips_pos = left_tips_pos.unsqueeze(2)  # [N, F1, 1, 3]
        right_thumb_tips_pos = right_thumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]
        right_nonthumb_tips_pos = right_nonthumb_tips_pos.unsqueeze(2)  # [N, F2, 1, 3]

        dist_base_to_left = torch.norm(
            left_tips_pos - base_marker_pos, dim=-1, p=2
        )  # (N, F1, K1)
        dist_cap_to_right_nonthumb = torch.norm(
            right_nonthumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)
        dist_cap_to_right_thumb = torch.norm(
            right_thumb_tips_pos - cap_marker_pos, dim=-1, p=2
        )  # (N, F2, K2)

        dist_base_to_left = torch.min(dist_base_to_left, -1)[0]
        dist_cap_to_right_nonthumb = torch.min(dist_cap_to_right_nonthumb, -1)[0]
        dist_cap_to_right_thumb = torch.min(dist_cap_to_right_thumb, -1)[0]
        # print(dist_base_to_left.shape, dist_cap_to_right_nonthumb.shape, reward_cap_dist_nonthumb.shape)
        # print(dist_base_to_left)
        reward_base_dist = 0.1 / (dist_base_to_left * 2 + 0.03)
        reward_base_dist = torch.clamp(
            reward_base_dist, 0, 0.1 / (0.01 * 2 + 0.03)
        ).sum(dim=-1)
        # print("DIST", dist_cap_to_right_thumb)
        reward_cap_dist_nonthumb = 0.1 / (dist_cap_to_right_nonthumb * 2 + 0.03)
        reward_cap_dist_nonthumb = torch.clamp(
            reward_cap_dist_nonthumb, 0, 0.1 / (0.01 * 2 + 0.03)
        ).sum(dim=-1)
        # print(reward_cap_dist_nonthumb)

        reward_cap_dist_thumb = 0.1 / (dist_cap_to_right_thumb * 2 + 0.03)
        reward_cap_dist_thumb = torch.clamp(
            reward_cap_dist_thumb, 0, 0.1 / (0.01 * 2 + 0.03)
        ).mean(dim=-1)

        # print(reward_cap_dist_nonthumb, reward_cap_dist_thumb)
        # print(reward_base_dist.shape, reward_cap_dist_thumb.shape, reward_cap_dist_nonthumb.shape)
        # print(dist_cap_to_right)

        reward_dist = (
            reward_base_dist
            + reward_cap_dist_nonthumb * reward_settings["cap_mult"]
            + reward_cap_dist_thumb
            * reward_settings["cap_mult"]
            * reward_settings["thumb_mult"]
        )

        # Sparsify.
        # reward_mask = torch.where(
        #     progress_buf % max([1, int(reward_settings["grasp_reward_freq"])]) != 0
        # )
        # reward_dist[reward_mask] = 0.0
        # print(reward_dist)

        reward = reward + reward_dist * reward_settings["finger_distance_reward_scale"]

        # Encourage pose matching.
        cube_quat = states["cube_quat"]
        cube_rotation_matrix = quaternion_to_matrix(
            xyzw_to_wxyz(cube_quat)
        )  # [B, 3, 3]
        z_axis = cube_rotation_matrix[:, :, 2]  # [B, 3]

        # Point to the right.
        target_vector = (
            torch.FloatTensor([0.0, 1.0, 0.0])
            .to(self.device)
            .reshape(-1, 3)
            .repeat(z_axis.size(0), 1)
        )
        angle_difference = torch.arccos(torch.sum(z_axis * target_vector, dim=-1))
        angle_penalty = -torch.clamp(angle_difference, 0.0, 1.0)
        # print(angle_difference)
        info_dict["cube_z_angle_difference"] = angle_difference

        z_score_update_idx = torch.where(progress_buf > 100)
        self.z_score[z_score_update_idx] += angle_difference[z_score_update_idx]

        # if episode length is over 100, reset if the angle is too large
        if reward_settings["reset_by_z_angle"]:
            reset_buf = torch.where(
                torch.logical_and(progress_buf >= 100, angle_difference > 0.2),
                torch.ones_like(reset_buf),
                reset_buf,
            )

        reward = reward + angle_penalty * reward_settings["angle_penalty_scale"]

        # Action penalty.
        left_action_penalty = -torch.sum(actions[:, :16] ** 2, dim=-1)
        right_action_penalty = -torch.sum(actions[:, 16:] ** 2, dim=-1)
        action_penalty = (
            left_action_penalty * reward_settings["left_action_penalty_scale"]
            + right_action_penalty * reward_settings["right_action_penalty_scale"]
        )
        reward = reward + action_penalty * reward_settings["action_penalty_scale"]

        # penalize work
        # print("WORK", states["work"]) # work is usually around 0.01-0.1

        work_penalty = (
            -states["left_work"] * reward_settings["left_work_penalty_scale"]
            - states["right_work"] * reward_settings["right_work_penalty_scale"]
        )

        reward = reward + work_penalty * reward_settings["work_penalty_scale"]
        # print(states['work'])

        # Add a distance reward to encourage getting closer to the center.
        goal_pos = torch.tensor(reward_settings["cap_center_point"]).to(
            states["cube_cap_pos"].device
        )
        distance_to_center = torch.norm(
            states["cube_cap_pos"][:, :2] - goal_pos, dim=-1, p=2
        )
        distance_reward = (
            -distance_to_center * 100
        )  # 0.1 / (0.03 + distance_to_center * 20.0)
        reward = reward + distance_reward * reward_settings["distance_reward_scale"]

        # print(reward)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)
        # print("ROT_S", self.rot_score)
        # print("Z___S", self.z_score)

        info_dict["rot_score"] = (
            self.rot_score / progress_buf.float()
        )  # torch.clip(self.progress_buf.float(), min=1, max=200) # + 1e-4)
        info_dict["z_score"] = self.z_score / torch.clip(
            progress_buf.float() - 100, min=1
        )  # + 1e-4)

        return reward, reset_buf, info_dict

    def reset(self, env_ids):
        self.z_score[env_ids] = 0
        self.rot_score[env_ids] = 0
        return


def build(**kwargs):
    return RewardFunction(**kwargs)
