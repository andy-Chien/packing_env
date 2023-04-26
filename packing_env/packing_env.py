import gym
import numpy as np
import open3d as o3d
from rclpy import logging
from gym import spaces
from ament_index_python.packages import get_package_share_directory
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter


MODEL_PKG = 'objects_model'
CONFIG_PKG = 'packing_env'



class PackingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    logger = logging.get_logger('packing env')
    metadata = {"render.modes": ["human"]}

    def __init__(self,
            img_width: int = 64,
            random_start: bool = True,
            discrete_actions: bool = True,
            channel_last: bool = True,
            xy_action_space: int = 64,
            rot_action_space: int = 72,
        ):
        self.logger.info('__init__')
        super().__init__()
        model_path = get_package_share_directory(MODEL_PKG)
        camera_config_path = get_package_share_directory(CONFIG_PKG) + '/config/bullet_camera.yaml'
        self.box_bound = [[0.1, 0.1, 0.1], [0.17, 0.17, 0.17]]
        self.start_bound = [[-0.05, -0.05, 1.25], [0.05, 0.05, 1.35]]

        self.img_width = img_width

        self.bh = BulletHandler()
        self.gc = GeometryConverter()

        self.mm = ModelManager(model_path, self.bh)
        self.cm = SimCameraManager(camera_config_path)
        self.camera_list = []
        if self.cm.create_camera('cam_1', 'Bullet_Camera', 'TEST', \
                                        [0, 0.3, 1.35], [0, 0, 1.25], [0, -0.1, 0.3]):
            self.camera_list.append('cam_1')

        if self.cm.create_camera('cam_2', 'Bullet_Camera', 'TEST', \
                                        [0, -0.3, 1.35], [0, 0, 1.25], [0, 0.1, 0.3]):
            self.camera_list.append('cam_2')
        
        if discrete_actions:
            self.action_space = spaces.MultiDiscrete(
                [xy_action_space, xy_action_space, rot_action_space])
        else:
            self.action_space = spaces.Box(0, 1, (3,))

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.img_width, self.img_width), dtype=np.uint8)
        self.logger.info('__init__')

    def step(self, action):
        self.logger.info('step')

        self.mm.set_model_pos(self.model, [0, 0, 0.5])
        self.bh.step_simulation(240, realtime=False)

        done = self.is_done()
        reward = self.compute_reward(done)
        if not done:
            self.prepare_model()
            obs = self.get_observation()
        else:
            obs = None
        info = 'hello'

        self.logger.info('step')
        return obs, reward, done, info

    def is_done(self):
        return self.model_indx >= len(self.model_list)

    def compute_reward(self, done):
        return 1 if done else 0

    def prepare_model(self):
        self.logger.info('prepare_model')
        pos = self.mm.random_pos(self.start_bound)
        quat = self.mm.random_quat()
        self.model = self.model_list[self.model_indx]
        self.mm.load_model(self.model, pos, quat)
        self.model_indx += 1
        self.logger.info('prepare_model')

    def get_observation(self):
        self.logger.info('get_observation')
        cloud_list = []
        for cam in self.camera_list:
            cloud_list.append(self.cm.get_point_cloud(cam))

        merged_cloud = self.gc.merge_cloud(cloud_list)
        self.voxel = self.gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
        pixel_size = 0.002
        tar_center = [0, 0, 1.3]
        self.views = self.gc.get_3_views_from_voxel(self.voxel, pixel_size, self.img_width, tar_center)
        
        self.logger.info('get_observation')
        return self.observation_space.sample()

    def reset(self):
        self.logger.info('reset')
        self.model_indx = 0
        self.model_list = self.mm.sample_models_in_bound(self.box_bound, 0.8)
        self.bh.reset_all()
        self.prepare_model()
        self.logger.info('reset')
        return self.get_observation()

    def render(self, mode="human"):
        self.logger.info('render')
        self.gc.o3d_show(self.voxel)
        for view in self.views:
            self.gc.o3d_show(o3d.geometry.Image(view))
        self.logger.info('render')

    def close(self):
        ...