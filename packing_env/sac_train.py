#!/usr/bin/env python3
import time

from stable_baselines3 import SAC
from packing_env import PackingEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

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
    model = SAC("CnnPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)
    model.learn(total_timesteps=50_000)

    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        # if done:
        #     obs = env.reset()
        print('obs.shape() = {}'.format(obs.shape))
        # env.render()

if __name__ == '__main__':
    main()