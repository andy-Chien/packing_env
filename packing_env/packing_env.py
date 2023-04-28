import gym
import time
import numpy as np
import open3d as o3d
import quaternion as qtn
from rclpy import logging
from gym import spaces
from ament_index_python.packages import get_package_share_directory
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter


MODEL_PKG = 'objects_model'
THIS_PKG = 'packing_env'
BOX_BOUND = [[0.15, 0.15, 0.15], [0.3, 0.3, 0.3]]
START_BOUND = [[-0.05, -0.05, 1.25], [0.05, 0.05, 1.35]]

VIEWS_PER_OBJ = 3


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
        camera_config_path = get_package_share_directory(THIS_PKG) + '/config/bullet_camera.yaml'

        self.img_width = img_width
        self.bh = BulletHandler()
        self.gc = GeometryConverter()
        self.mm = ModelManager(model_path, self.bh)
        self.cm = SimCameraManager(camera_config_path)
        self.bh.set_model_path(get_package_share_directory(THIS_PKG) + '/mesh') # for packing box
        self.cm.create_camera('box_cam', 'Bullet_Camera', 'TEST', \
                              [0, 0.3, 1.35], [0, 0, 1.25], [0, -0.1, 0.3])

        self.cm.create_camera('obj_cam', 'Bullet_Camera', 'TEST', \
                              [0, 0, 0.5], [0, 0, 0.05], [0.1, 0, 0])
        
        if discrete_actions:
            self.action_space = spaces.MultiDiscrete(
                [xy_action_space, xy_action_space, rot_action_space])
        else:
            self.action_space = spaces.Box(0, 1, (3,))

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, self.img_width, self.img_width), dtype=np.uint8)
        self.logger.info('__init__')

    def step(self, action):
        self.logger.info('step')

        self.mm.set_model_pos(self.curr_model, [0, 0, 0.35])
        self.bh.step_simulation(120, realtime=False)
        self.bh.set_model_pose(self.box_id, self.box_pos, [0,0,0,1])
        self.bh.step_simulation(120, realtime=False)


        done = self.is_done()
        reward = self._compute_reward(done)
        if not done:
            self.prepare_objects()
            obs = self.get_observation()
        else:
            obs = None
        info = {'info': 'hello'}

        self.logger.info('step')
        return obs, reward, done, info

    def is_done(self):
        return self.model_indx >= len(self.model_list)

    def _compute_reward(self, done):
        return 1 if done else 0

    def prepare_objects(self):
        self.logger.info('prepare_objects')
        pos = self.mm.random_pos(START_BOUND)
        quat = self.mm.random_quat()
        self.curr_model = self.model_list[self.model_indx]
        self.mm.load_model(self.curr_model, pos, quat)
        self.model_indx += 1
        self.logger.info('prepare_objects')

    def prepare_packing_box(self):
        self.logger.info('prepare_packing_box')
        box_size = np.random.uniform(BOX_BOUND[0], BOX_BOUND[1])
        self.box_pos = [-1.2*box_size[0]/2, -1.2*box_size[1]/2, 0.0]
        self.box_id = self.bh.load_stl('packing_box.stl', box_size, self.box_pos, [0, 0, 0, 1])
        self.bh.step_simulation(240, realtime=False)
        self.logger.info('prepare_packing_box, id = {}'.format(self.box_id))
        return box_size
    
    def get_observation(self):
        self.logger.info('get_observation')
        pixel_size = 0.02
        box_cloud = self.cm.get_point_cloud('box_cam')
        box_voxel = self.gc.get_voxel_from_cloud(box_cloud, voxel_size=0.002)
        box_view = self.gc.get_view_from_voxel(box_voxel, pixel_size, self.img_width)
        obj_cloud_list = []
        relative_angle = np.pi / VIEWS_PER_OBJ
        self.logger.info('get_observation 2')
        for _ in range(VIEWS_PER_OBJ):
            obj_cloud_list.append(self.cm.get_point_cloud('obj_cam'))
            self.mm.set_model_relative_euler(self.curr_model, [0, 0, relative_angle])
        self.logger.info('get_observation 3')
        merged_cloud = self.gc.merge_cloud(obj_cloud_list)
        self.voxel = self.gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
        self.logger.info('get_observation 4')
        tar_center = list((np.array(START_BOUND[0]) + np.array(START_BOUND[1])) / 2)
        self.logger.info('get_observation 5')
        self.views = self.gc.get_3_views_from_voxel(self.voxel, pixel_size, self.img_width, tar_center)
        self.logger.info('get_observation 6')
        self.views = np.append(self.views, np.expand_dims(box_view, axis=0), axis=0)
        
        self.logger.info('get_observation')
        return self.views

    def reset(self):
        self.logger.info('reset')
        self.bh.reset_all()
        box_size = self.prepare_packing_box()
        self.model_indx = 0
        self.model_list = self.mm.sample_models_in_bound(box_size, 0.5)
        self.prepare_objects()
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