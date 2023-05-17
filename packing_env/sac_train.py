#!/usr/bin/env python3
import time
from gym import spaces
from rclpy import logging
from stable_baselines3 import SAC
from packing_env import PackingEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# np.set_printoptions(threshold=np.inf)

class CombinedExtractor(BaseFeaturesExtractor):
    logger = logging.get_logger('CombinedExtractor')
    def __init__(self, observation_space: spaces.Dict, features_dim=128):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            s = subspace.shape
            if key == "box":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                w = ((((s[1] + 2 - 4) // 2 + 1) - 3) + 1) - 3 + 1
                total_concat_size += w * w * s[0] * 8
                extractors[key] = nn.Sequential(
                    nn.Conv2d(s[0], s[0] * 8, kernel_size=4, stride=2, padding=1, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                )
            elif key == 'obj':
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                w = ((((s[1] + 2 - 4) // 2 + 1) - 3) + 1) - 3 + 1
                total_concat_size += w * w * s[0] * 8
                extractors[key] = nn.Sequential(
                    nn.Conv2d(s[0], s[0] * 8, kernel_size=4, stride=2, padding=1, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                )
            elif key == "num":
                # Run through a simple MLP
                extractors[key] = nn.Identity()
                total_concat_size += s[0]


        # self.linear = nn.Sequential(
        #     nn.Linear(total_concat_size, features_dim),
        #     nn.ReLU(),
        #     nn.Linear(features_dim, features_dim),
        #     nn.ReLU(),
        # )

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # return self.linear(th.cat(encoded_tensor_list, dim=1))
        return th.cat(encoded_tensor_list, dim=1)
    
class TBCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record("difficulty", self.locals['env'].get_attr('difficulty', 0)[0])
        self.logger.record("avg_reward", self.locals['env'].get_attr('avg_reward', 0)[0])
        self.logger.record("success_rate", self.locals['env'].get_attr('success_rate', 0)[0])
        return True

def make_env(env_index: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param env_index: index of the subprocess
    """
    def _init():
        env = PackingEnv(env_index=env_index, discrete_actions=False, bullet_gui=(env_index==0))
        env.reset(seed=seed + env_index)
        return env
    set_random_seed(seed)
    return _init

def main():

    # env = PackingEnv(discrete_actions=False)

    # We collect 4 transitions per call to `ènv.step()`
    # and performs 2 gradient steps per call to `ènv.step()`
    # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
    num_cpu = 8
    vec_env = SubprocVecEnv([make_env(env_index=i) for i in range(num_cpu)])
    policy_kwargs= dict(
        features_extractor_class=CombinedExtractor,
        normalize_images=False,
        net_arch=dict(pi=[128, 128, 128], qf=[128, 128, 128]),
        activation_fn=nn.LeakyReLU,
    )
    model = SAC("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, 
                train_freq=1, gradient_steps=2, verbose=1, learning_starts=1000, tensorboard_log='./log/sac_tb_log/')
    print('========================================================`')
    print(model.policy)
    print('========================================================')

    model.learn(total_timesteps=100_000, tb_log_name='0517_1', callback=TBCallback())

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        # if done:
        #     obs = env.reset()
        # print('obs.shape() = {}'.format(obs.shape))
        # env.render()

if __name__ == '__main__':
    main()