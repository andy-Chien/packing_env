#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


import imp
import pybullet as pb
import time
import pybullet_data
import rospkg
import numpy as np
from packing_env import ModelManager, SimCameraManager

class PackingEnvironmentEnv(py_environment.PyEnvironment):
    def __init__(self, env_config, model_manager, sim_camera_manager):
        self.a_cfg = env_config['action']
        self.o_cfg = env_config['observation']
        self.r_cfg = env_config['reward']
        self.c_cfg = env_config['conditions']
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self.a_cfg['width'], self.a_cfg['high'], self.a_cfg['rotation'],), 
                   dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.o_cfg['width'], self.o_cfg['high'], self.o_cfg['views']), 
                   dtype=np.int32, minimum=0, maximum=1, name='observation')
        self._state = 0
        self._episode_ended = False
        self._model_manager = model_manager
        self._sim_cam_manager = sim_camera_manager

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self._condition_update()
        bound = self._sample_packing_box()
        self._model_list = self._model_manager.sample_models_in_bound(bound, self.c_cfg['fill_rate']['value'])
        state = self._get_state()
        return ts.restart(state)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

    def _condition_update(self):
        pass

    def _sample_packing_box(self):
        pass