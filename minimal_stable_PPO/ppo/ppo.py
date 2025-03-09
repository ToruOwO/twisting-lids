import os
import math
import time
import torch
import torch.distributed as dist

import wandb
import imageio
import numpy as np
from collections import defaultdict

from .experience import ExperienceBuffer
from .models import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd

from tensorboardX import SummaryWriter


class PPO(object):
    def __init__(self, env, output_dir, full_config):
        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.state_dim = self.env.num_states  # state dimension

        # ---- Model ----
        try:
            self.asymmetric = self.ppo_config["asymmetric"]
        except Exception:
            self.asymmetric = False

        if self.asymmetric:
            assert (
                self.state_dim > 0
            ), "Enviroment state dimension must > 0 in asymmetric training"

        if hasattr(self.network_config, "value_mlp"):
            value_mlp_units = self.network_config.value_mlp.units
        else:
            value_mlp_units = self.network_config.mlp.units

        net_config = {
            "asymmetric": self.asymmetric,  # For asymmetric training.
            "state_input_shape": self.env.num_states,  # For asymmetric training.
            "actor_units": self.network_config.mlp.units,
            "value_units": value_mlp_units,  # For asymmetric training.
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "separate_value_mlp": self.network_config.get("separate_value_mlp", True),
        }

        self.model = ActorCritic(net_config)

        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)

        if self.asymmetric:
            self.state_running_mean_std = RunningMeanStd(self.state_dim).to(self.device)

        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "nn")
        self.tb_dif = os.path.join(self.output_dir, "tb")
        self.video_save_dir = os.path.join(self.output_dir, "video")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        os.makedirs(self.video_save_dir, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config["learning_rate"])
        self.last_lr = float(self.ppo_config["learning_rate"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-8, weight_decay=self.ppo_config["weight_decay"]
        )
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config["e_clip"]
        self.clip_value = self.ppo_config["clip_value"]
        self.entropy_coef = self.ppo_config["entropy_coef"]
        self.critic_coef = self.ppo_config["critic_coef"]
        self.bounds_loss_coef = self.ppo_config["bounds_loss_coef"]
        self.gamma = self.ppo_config["gamma"]
        self.tau = self.ppo_config["tau"]
        self.truncate_grads = self.ppo_config["truncate_grads"]
        self.grad_norm = self.ppo_config["grad_norm"]
        self.value_bootstrap = self.ppo_config["value_bootstrap"]
        self.normalize_advantage = self.ppo_config["normalize_advantage"]
        self.normalize_input = self.ppo_config["normalize_input"]
        self.normalize_value = self.ppo_config["normalize_value"]
        self.reward_scale_value = self.ppo_config["reward_scale_value"]
        self.clip_value_loss = self.ppo_config["clip_value_loss"]
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config["minibatch_size"]
        self.mini_epochs_num = self.ppo_config["mini_epochs"]
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config["lr_schedule"]
        if self.lr_schedule == "kl":
            self.kl_threshold = self.ppo_config["kl_threshold"]
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == "linear":
            self.scheduler = LinearScheduler(
                self.init_lr, self.ppo_config["max_agent_steps"]
            )
        # ---- Snapshot
        self.save_freq = self.ppo_config["save_frequency"]
        self.save_best_after = self.ppo_config["save_best_after"]
        self.num_video_envs = self.ppo_config["num_video_envs"]
        self.save_video_freq = self.ppo_config["save_video_freq"]
        self.eval_freq = self.ppo_config["eval_freq"]
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)

        self.reward_info_keys = ["z_score", "rot_score"]

        self.reward_info_meters = {
            k: AverageScalarMeter(200) for k in self.reward_info_keys
        }

        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.device,
            self.state_dim,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config["max_agent_steps"]
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            "performance/RLTrainFPS": self.agent_steps / self.rl_train_time,
            "performance/EnvStepFPS": self.agent_steps / self.data_collect_time,
            "losses/actor_loss": torch.mean(torch.stack(a_losses)).item(),
            "losses/bounds_loss": torch.mean(torch.stack(b_losses)).item(),
            "losses/critic_loss": torch.mean(torch.stack(c_losses)).item(),
            "losses/entropy": torch.mean(torch.stack(entropies)).item(),
            "info/last_lr": self.last_lr,
            "info/e_clip": self.e_clip,
            "info/kl": torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f"{k}"] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

            if self.asymmetric:
                self.state_running_mean_std.eval()

        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()

            if self.asymmetric:
                self.state_running_mean_std.train()

        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict["obs"])
        input_dict = {
            "obs": processed_obs,
        }

        if self.asymmetric:
            processed_state = self.state_running_mean_std(obs_dict["states"])
            input_dict["states"] = processed_state

        res_dict = self.model.act(input_dict)
        res_dict["values"] = self.value_mean_std(res_dict["values"], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()

        self.agent_steps = (
            self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size
        )

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1

            if self.epoch_num % self.save_video_freq == 0:
                num_video_envs = self.num_video_envs
            else:
                num_video_envs = 0

            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch(
                num_video_envs=num_video_envs
            )
            self.storage.data_dict = None

            if self.lr_schedule == "linear":
                self.last_lr = self.scheduler.update(self.agent_steps)

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = (
                    self.batch_size
                    if not self.multi_gpu
                    else self.batch_size * self.rank_size
                ) / (time.time() - _last_t)
                _last_t = time.time()
                info_string = (
                    f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                    f"Last FPS: {last_fps:.1f} | "
                    f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                    f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                print(info_string)

                self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

                mean_rewards = self.episode_rewards.get_mean()
                mean_lengths = self.episode_lengths.get_mean()
                self.writer.add_scalar(
                    "metrics/episode_rewards_per_step", mean_rewards, self.agent_steps
                )
                self.writer.add_scalar(
                    "metrics/episode_lengths_per_step", mean_lengths, self.agent_steps
                )
                wandb_dict = {
                    "metrics/episode_rewards_per_step": mean_rewards,
                    "metrics/episode_lengths_per_step": mean_lengths,
                }
                for k, meters in self.reward_info_meters.items():
                    wandb_dict[f"reward_metrics/{k}"] = meters.get_mean()

                wandb.log(
                    wandb_dict,
                    step=self.agent_steps,
                )
                checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"

                if self.save_freq > 0:
                    if (self.epoch_num % self.save_freq == 0) and (
                        mean_rewards <= self.best_rewards
                    ):
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, f"last"))

                if mean_rewards > self.best_rewards:
                    print(f"save current best reward: {mean_rewards:.2f}")
                    # remove previous best file
                    prev_best_ckpt = os.path.join(
                        self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth"
                    )
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(
                        os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}")
                    )

        print("max steps achieved")

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
            if self.asymmetric:
                weights[
                    "state_running_mean_std"
                ] = self.state_running_mean_std.state_dict()
        if self.value_mean_std:
            weights["value_mean_std"] = self.value_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.value_mean_std.load_state_dict(checkpoint["value_mean_std"])
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        if self.asymmetric:
            self.state_running_mean_std.load_state_dict(
                checkpoint["state_running_mean_std"]
            )

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.value_mean_std.load_state_dict(checkpoint["value_mean_std"])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.asymmetric:
            self.state_running_mean_std.load_state_dict(
                checkpoint["state_running_mean_std"]
            )

    def test(self, video_fn="eval", video_length=200, save_trajs=False):
        self.set_eval()
        obs_dict = self.env.reset()
        num_video_envs = self.num_video_envs
        # print("num_video_envs", num_video_envs)
        # print("video save dir", self.video_save_dir)

        self.video_buff = defaultdict(list)

        finished = torch.zeros(self.num_actors, dtype=bool).to(device=self.device)
        rewards = torch.zeros(self.num_actors, dtype=float).to(device=self.device)
        lengths = torch.zeros(self.num_actors, dtype=float).to(device=self.device)
        num_steps = 0

        info_keys = {
            "dof_pos",
            "dones",
            "mu",
            "obs",
            "obs_rms",
        }
        extras = {k: [] for k in info_keys}

        while not finished.all() or num_steps < video_length:
            extras["obs"].append(obs_dict["obs"])
            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]),
            }
            extras["obs_rms"].append(input_dict["obs"])
            if self.asymmetric:
                # During model action inference, we actually do not need state.
                # print(obs_dict['states'].device)
                input_dict["states"] = self.state_running_mean_std(
                    obs_dict["states"].to(obs_dict["obs"].device)
                )

            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, dones, info = self.env.step(mu)

            for k, v in info.items():
                if k in reward_info_keys:
                    reward_extras[k].append(v)
            extras["dof_pos"].append(info["dof_pos"])
            extras["dones"].append(dones.unsqueeze(-1))
            extras["mu"].append(mu)

            finished = torch.logical_or(finished, dones)
            not_dones = 1.0 - finished.float()

            rewards += r * not_dones
            lengths += not_dones
            num_steps += 1

            if num_video_envs > 0:
                pixel_obs_dict = self.env.compute_pixel_obs()
                for key, obs in pixel_obs_dict.items():
                    obs = obs[:num_video_envs].permute(0, 2, 3, 1).cpu().numpy()
                    self.video_buff[key].append(obs)

        for k, v in reward_extras.items():
            reward_extras[k] = torch.stack(v)
        for k, v in extras.items():
            extras[k] = torch.stack(v)

        extras["reward"] = torch.mean(rewards)
        extras["length"] = torch.mean(lengths.float())

        return extras

    def train_epoch(self, num_video_envs=0):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps(num_video_envs=num_video_envs)
        self.data_collect_time += time.time() - _t
        if num_video_envs > 0:
            videos = []
            for key, video in self.video_buff.items():
                video = np.stack(video, axis=1)
                videos.append(video)
            videos = np.concatenate(videos, axis=-3)
            for i in range(num_video_envs):
                path = os.path.join(self.video_save_dir, f"train-latest-{i}.mp4")
                imageio.mimsave(path, videos[i], fps=5)
            wandb.log(
                {
                    f"video/train": wandb.Video(
                        (np.clip(videos, 0, 1).transpose((0, 1, 4, 2, 3)) * 255).astype(
                            np.int
                        ),
                        fps=5,
                        format="mp4",
                    )
                },
                step=self.agent_steps,
            )

        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    state,
                ) = self.storage[i]

                # some check for nans
                assert (
                    torch.isnan(advantage).int().sum() == 0
                ), "advantage contains Nan!"
                assert torch.isnan(returns).int().sum() == 0, "return contains Nan"
                assert (
                    torch.isnan(old_action_log_probs).int().sum() == 0
                ), "action_log_prob contains Nan"
                assert torch.isnan(obs).int().sum() == 0, "obs contains Nan"

                obs = self.running_mean_std(obs)

                if self.asymmetric:
                    state = self.state_running_mean_std(state)

                batch_dict = {
                    "prev_actions": actions,
                    "obs": obs,
                    "states": state,  # Note: state is None when we use normal training mode (value net use obs).
                }

                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds + (values - value_preds).clamp(
                        -self.e_clip, self.e_clip
                    )
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(
                                    param.grad.data
                                )
                                / self.rank_size
                            )
                            offset += param.numel()

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            kls.append(av_kls)

            if self.lr_schedule == "kl":
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == "cos":
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.last_lr

        update_curriculum = getattr(self.env, "update_curriculum", None)
        if callable(update_curriculum):
            self.env.update_curriculum(self.agent_steps)

        self.rl_train_time += time.time() - _t
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self, num_video_envs=0):
        self.video_buff = defaultdict(list)
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data("obses", n, self.obs["obs"])
            if self.asymmetric:
                self.storage.update_data("states", n, self.obs["states"])

            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)
            self.obs, r, self.dones, infos = self.env.step(actions)

            if num_video_envs > 0:
                pixel_obs_dict = self.env.compute_pixel_obs()
                for key, obs in pixel_obs_dict.items():
                    obs = obs[:num_video_envs].permute(0, 2, 3, 1).cpu().numpy()
                    self.video_buff[key].append(obs)

            rewards = r.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data("dones", n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * infos["time_outs"].unsqueeze(1).float()
                )
            self.storage.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            # print("CURRENT REWARD", self.current_rewards)
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            assert isinstance(infos, dict), "Info Should be a Dict"
            for k, v in infos.items():
                # only log scalars
                if (
                    isinstance(v, float)
                    or isinstance(v, int)
                    or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                ):
                    self.extra_info[k] = v

            if len(done_indices) > 0:
                for k in self.reward_info_keys:
                    if k in infos:
                        self.reward_info_meters[k].update(infos[k][done_indices])

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict["values"]

        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict["returns"]
        values = self.storage.data_dict["values"]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns

    def adjust_learning_rate_cos(self, epoch):
        lr = (
            self.init_lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (self.agent_steps + epoch / self.mini_epochs_num)
                    / self.max_agent_steps
                )
            )
        )
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
