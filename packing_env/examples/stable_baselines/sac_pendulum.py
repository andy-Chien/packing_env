import gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Pendulum-v1", n_envs=4, seed=0)

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