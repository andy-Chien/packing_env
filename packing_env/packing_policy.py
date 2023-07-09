#!/usr/bin/env python3
import time
from gym import spaces
from rclpy import logging
from stable_baselines3 import SAC, PPO
from packing_env import PackingEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# np.set_printoptions(threshold=np.inf)


MODEL = PPO
TRAINING_MODEL_NAME = 'bigger_network'
LOADING_MODEL_NAME = 'SAC_model/bigger_network_ath.zip'
LOAD_MODEL = False
DISCRETE_ACTIONS = True
NUM_CPU = 22
ATH_DIFFICULTY = 0.34

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
                # w = (((((((s[1] + 2 - 4) // 2 + 1) - 3) + 1) - 3) + 1) - 3) + 1 - 3 + 1
                total_concat_size += w * w * s[0] * 8
                extractors[key] = nn.Sequential(
                    nn.Conv2d(s[0], s[0] * 8, kernel_size=4, stride=2, padding=1, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    # nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    # nn.LeakyReLU(),
                    nn.Flatten(),
                )
            elif 'obj' in key:
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                # total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
                w = ((((s[1] + 2 - 4) // 2 + 1) - 3) + 1) - 3 + 1
                # w = (((((((s[1] + 2 - 4) // 2 + 1) - 3) + 1) - 3) + 1) - 3) + 1 - 3 + 1
                total_concat_size += w * w * s[0] * 8
                extractors[key] = nn.Sequential(
                    nn.Conv2d(s[0], s[0] * 8, kernel_size=4, stride=2, padding=1, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    nn.LeakyReLU(),
                    # nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    # nn.LeakyReLU(),
                    # nn.Conv2d(s[0] * 8, s[0] * 8, kernel_size=3, stride=1, padding=0, groups=s[0]),
                    # nn.LeakyReLU(),
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
    
class TrainingCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, model, ath_difficulty, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.model_name = type(model).__name__
        self.ath_difficulty = ath_difficulty
        print('self.model_name = {}'.format(self.model_name))

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        difficulty = self.locals['env'].get_attr('difficulty', 0)[0]
        self.logger.record("difficulty", difficulty)
        self.logger.record("avg_reward", self.locals['env'].get_attr('avg_reward', 0)[0])
        self.logger.record("fill_rate", self.locals['env'].get_attr('fill_rate', 0)[0])
        if difficulty > self.ath_difficulty[0] + 0.002:
            name = 'data/training_data/' + self.model_name + '_model/'
            name += TRAINING_MODEL_NAME + '_ath'
            self.ath_difficulty[0] = difficulty
            self.model.save(name)
        return True
    
class PackingPolicy:
    def __init__(self, load=False, model=SAC, discrete_actions=False, num_cpu=6, ath_difficulty=0.32):
        self.model = model
        self.ath_difficulty = [ath_difficulty]

        self.vec_env = SubprocVecEnv([self.make_env(env_index=i, discrete_actions=discrete_actions)
                                      for i in range(num_cpu)])
        policy_kwargs= dict(
            features_extractor_class=CombinedExtractor,
            normalize_images=False,
            net_arch=dict(pi=[512, 512, 512, 256, 256], qf=[512, 512, 512, 256, 256]),
            activation_fn=nn.LeakyReLU,
        )
        if load:
            file_name = 'data/training_data/' + LOADING_MODEL_NAME
            self.model = model.load(file_name, env=self.vec_env)
        elif model == SAC and not load:
            self.model = model("MultiInputPolicy", self.vec_env, policy_kwargs=policy_kwargs, ent_coef='auto_0.2',
                        train_freq=2, verbose=1, learning_starts=1000, learning_rate=1e-4, 
                        tensorboard_log='./data/training_data/sac_log/')
        elif model == PPO and not load:
            self.model = model("MultiInputPolicy", self.vec_env, policy_kwargs=policy_kwargs,
                        verbose=1, learning_rate=1e-4, n_steps=128, batch_size=64,
                        tensorboard_log='./data/training_data/ppo_log/')

        print('========================================================')
        print(self.model.policy)
        print('========================================================')


    def train(self):
        try:
            self.model.learn(total_timesteps=10_000_000, 
                            tb_log_name=TRAINING_MODEL_NAME, 
                            callback=TrainingCallback(self.model, self.ath_difficulty)
                            )
            return True
        except Exception as e:
            print(e)
            return False
    def evaluation(self):
        obs = self.vec_env.reset()
        for _ in range(100):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.vec_env.step(action)
            # if done:
            #     obs = env.reset()
            # print('obs.shape() = {}'.format(obs.shape))
            # env.render()

    def make_env(self, env_index: int, seed: int = 0, discrete_actions=False):
        """
        Utility function for multiprocessed env.

        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param env_index: index of the subprocess
        """
        def _init():
            env = PackingEnv(env_index=env_index, discrete_actions=discrete_actions, bullet_gui=(env_index==0))
            env.reset(seed=seed + env_index)
            return env
        set_random_seed(seed)
        return _init

    def get_ath_difficulty(self):
        return self.ath_difficulty[0]

def main():
    ath_difficulty = ATH_DIFFICULTY
    while True:
        print('============================ construct ============================')
        print('============================ construct ============================')
        print('============================ construct ============================')
        print('============================ construct ============================')
        print('============================ construct ============================')
        policy = PackingPolicy(LOAD_MODEL, MODEL, DISCRETE_ACTIONS, NUM_CPU, ath_difficulty)
        print('============================ start to train ============================')
        print('============================ start to train ============================')
        print('============================ start to train ============================')
        print('============================ start to train ============================')
        print('============================ start to train ============================')
        if policy.train():
            break
        ath_difficulty = policy.get_ath_difficulty() - 0.005
    policy.evaluation()
    

if __name__ == '__main__':
    main()