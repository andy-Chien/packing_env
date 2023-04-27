#!/usr/bin/env python3
import time

from stable_baselines3 import SAC
from packing_env import PackingEnv

def main():

    env = PackingEnv(discrete_actions=False)

    # We collect 4 transitions per call to `ènv.step()`
    # and performs 2 gradient steps per call to `ènv.step()`
    # if gradient_steps=-1, then we would do 4 gradients steps per call to `ènv.step()`
    model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()