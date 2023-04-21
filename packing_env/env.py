import gym
import numpy as np
from gym import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self,
            num_col: int = 64,
            num_row: int = 64,
            random_start: bool = True,
            discrete_actions: bool = True,
            channel_last: bool = True,
            xy_action_space: int = 64,
            rot_action_space: int = 72,
        ):
        super().__init__()
        # Example for using image as input (channel-first; channel-last also works):
        
        if discrete_actions:
            self.action_space = spaces.MultiDiscrete(
                [xy_action_space, xy_action_space, rot_action_space])
        else:
            self.action_space = spaces.Box(0, 1, (3,))

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, num_col, num_row), dtype=np.uint8)
    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation

    def render(self, mode="human"):
        ...

    def close(self):
        ...