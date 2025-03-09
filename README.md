# Twisting Lids Off with Two Hands

[[Project](https://toruowo.github.io/bimanual-twist/)]
[[Paper](https://arxiv.org/abs/2403.02338)]

[Toru Lin*](https://toruowo.github.io/),
[Zhao-Heng Yin*](https://zhaohengyin.github.io/),
[Haozhi Qi](https://haozhi.io/),
[Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)
<br>

## Overview

This repo contains the code and instructions to train and evaluate the bimanual twisting policy as introduced in paper "Twisting Lids Off with Two Hands".

## Installation

1. Download IsaacGym Preview 4 from the [NVIDIA website](https://developer.nvidia.com/isaac-gym)

2. Extract and install IsaacGym:
```bash
cd isaacgym/python
pip install -e .
```

3. Create and setup the environment:
```bash
mamba create -n twist python=3.8
mamba activate twist
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

cd path/to/twist/project/
pip install -e .
pip install hydra-core gym wandb termcolor tensorboardX pytorch3d numpy==1.19.5
```

After installation, the environment can be viewed by running the following command.
```
python ./env_utils/view_env.py num_envs=8 headless=False
```

## Training

```
python ./minimal_stable_PPO/train.py num_envs=8192 train.ppo.minibatch_size=8192 headless=True pipeline=gpu
```

## Evaluating Trained Policies

```
# Visualize the trained policy in Isaac GUI
python ./minimal_stable_PPO/train.py test=True num_envs=4 headless=False checkpoint=/path/to/checkpoint

# Record videos of the trained policy
python ./minimal_stable_PPO/eval.py task.env.enableCameraSensors=True train.ppo.num_video_envs=4 num_envs=4 headless=True eval_root_dir=/path/to/save/videos checkpoint=/path/to/checkpoint
```

## Reference

If you find our paper or this codebase helpful in your research, please consider citing:

```
@article{lin2024twisting,
    author={Lin, Toru and Yin, Zhao-Heng and Qi, Haozhi and Abbeel, Pieter and Malik, Jitendra},
    title={Twisting Lids Off with Two Hands},
    journal={arXiv:2403.02338},
    year={2024}
}
```