import isaacgym

import datetime
import hydra
import gym
import os
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from ppo.ppo import PPO
import random
import string


def generate_random_string(length=4, characters=string.ascii_letters + string.digits):
    """
    Generate a random string of the specified length using the given characters.

    :param length: The length of the random string (default is 12).
    :param characters: The characters to choose from when generating the string
                      (default is uppercase letters, lowercase letters, and digits).
    :return: A random string of the specified length.
    """
    return "".join(random.choice(characters) for _ in range(length))


@hydra.main(config_name="config", config_path="configs")
def main(config: DictConfig):
    assert config.checkpoint, "Please specify a checkpoint to evaluate"
    config.checkpoint = to_absolute_path(config.checkpoint)
    config.test = True

    # set numpy formatting for printing only
    set_np_formatting()

    if config.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        config.sim_device = f"cuda:{rank}"
        config.rl_device = f"cuda:{rank}"
        # sets seed. if seed is -1 will pick a random one
        config.seed = set_seed(config.seed + rank)
    else:
        # use the same device for sim and rl
        config.sim_device = (
            f"cuda:{config.device_id}" if config.device_id >= 0 else "cpu"
        )
        config.rl_device = (
            f"cuda:{config.device_id}" if config.device_id >= 0 else "cpu"
        )
        config.seed = set_seed(config.seed)

    cprint("Start Building the Environment", "green", attrs=["bold"])
    env = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
        virtual_screen_capture=False,
        force_render=True,
    )

    # automatic folder naming
    ckpt_name = config.checkpoint.split("/")[-3]
    config.output_name = f"{ckpt_name}"
    output_dif = os.path.join(
        config.eval_root_dir, # "/home/toru/Developer/Bimanual-Isaac/eval_scripts/jan",
        config.output_name
    )
    os.makedirs(output_dif, exist_ok=True)

    agent = PPO(env, output_dif, full_config=config)
    agent.restore_test(config.checkpoint)
    agent.test(video_length=300, save_trajs=True)


if __name__ == "__main__":
    main()
