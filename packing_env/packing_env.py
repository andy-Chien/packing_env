import gym
import time
import numpy as np
import open3d as o3d
import quaternion as qtn
from rclpy import logging
from gym import spaces
from matplotlib import pyplot as plt
from ament_index_python.packages import get_package_share_directory
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter
# np.set_printoptions(threshold=np.inf)

SHOE_IMG = False

MODEL_PKG = 'objects_model'
THIS_PKG = 'packing_env'
BOX_BOUND = [[0.15, 0.15, 0.15], [0.3, 0.3, 0.3]]
START_BOUND = [[0, 0, 1.25], [0, 0, 1.35]]

VIEWS_PER_OBJ = 3
CAPTURE_POS = [0, 0, 1.3]


class PackingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    logger = logging.get_logger('packing env')
    metadata = {"render.modes": ["human"]}

    def __init__(self,
            img_width: int = 64,
            random_start: bool = True,
            discrete_actions: bool = False,
            channel_last: bool = True,
            xy_action_space: int = 64,
            rot_action_space: int = 72,
        ):
        self.logger.info('__init__')
        super().__init__()
        model_path = get_package_share_directory(MODEL_PKG)
        camera_config_path = get_package_share_directory(THIS_PKG) + '/config/bullet_camera.yaml'

        self.img_width = img_width
        self.done = False
        self.bh = BulletHandler()
        self.gc = GeometryConverter()
        self.mm = ModelManager(model_path, self.bh)
        self.cm = SimCameraManager(camera_config_path)
        self.bh.set_model_path(get_package_share_directory(THIS_PKG) + '/mesh') # for packing box
        self.cm.create_camera('obj_cam', 'Bullet_Camera', 'OBJ_CAM', \
                              [0, 0.3, 1.25], [0, 0, 1.35], [0, 0.1, 0.3])

        self.cm.create_camera('box_cam', 'Bullet_Camera', 'BOX_CAM', \
                              [0, 0, 0.5], [0, 0, 0.05], [0.1, 0, 0])
        self.xy_action_space = xy_action_space
        self.rot_action_space = rot_action_space
        if discrete_actions:
            self.action_space = spaces.MultiDiscrete(
                [xy_action_space, xy_action_space, rot_action_space])
        else:
            self.action_space = spaces.Box(-1, 1, (3,))

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, self.img_width, self.img_width), dtype=np.uint8)
        self.logger.info('__init__')

    def step(self, action):
        self.logger.info('step, action = {}'.format(action))
        action = self.decode_action(action)
        self.mm.set_model_pos(self.curr_model, [action[0], action[1], 0.35])
        self.mm.set_model_relative_euler(self.curr_model, [0, 0, action[2]])
        self.bh.step_simulation(240, realtime=False)
        self.bh.set_model_pose(self.box_id, self.box_pos, [0,0,0,1])
        self.bh.step_simulation(60, realtime=False)

        obj_volume = self.mm.get_model_convex_volume(self.curr_model)
        reward = self._compute_reward()
        done = self.is_done()

        obs = None
        while self.obj_in_queue() and obs is None:
            model = self.prepare_objects()
            obs = self.get_observation(model)

        if (not self.obj_in_queue()) or obs is None:
            self.mm.remove_model(model)
            reward = self._compute_reward()
            done = self.is_done()
        else:
            self.curr_model = model

        if not done:
            self.volume_sum += obj_volume
        
        info = {'info': 'hello'}

        self.logger.info('step')
        return obs, reward, done, info
    
    def decode_action(self, action):
        if isinstance(self.action_space, spaces.Box):
            action[0] *= self.box_size[0] / 2
            action[1] *= self.box_size[1] / 2 # 2 because [-1, 1]
            action[2] *= np.pi
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action[0] *= self.box_size[0] / self.xy_action_space
            action[1] *= self.box_size[1] / self.xy_action_space
            action[2] *= np.pi / self.rot_action_space
        return action
    
    def obj_in_queue(self):
        return self.model_indx < len(self.model_list)

    def is_done(self):
        return self.done

    def _compute_reward(self):
        obj_pos, _ = self.mm.get_model_pose(self.curr_model)
        if obj_pos[0] > self.box_size[0] / 2 or \
                obj_pos[0] < -1 * self.box_size[0] / 2 or \
                obj_pos[1] > self.box_size[1] / 2 or \
                obj_pos[1] < -1 * self.box_size[1] / 2 or \
                obj_pos[2] > self.box_size[2]:
            self.done = True
            r = -1 + self.volume_sum / self.box_volume
        
        elif not self.obj_in_queue():
            self.done = True
            r = 1
        else:
            self.done = False
            r = 0
        return r

    def prepare_objects(self):
        self.logger.info('prepare_objects')
        pos = self.mm.random_pos(START_BOUND)
        quat = self.mm.random_quat()
        # quat = [0, 0, 0, 1]
        curr_model = self.model_list[self.model_indx]
        self.mm.load_model(curr_model, pos, quat)
        self.model_indx += 1
        self.logger.info('prepare_objects, pos = {}'.format(pos))
        return curr_model

    def prepare_packing_box(self):
        self.logger.info('prepare_packing_box')
        box_size = np.random.uniform(BOX_BOUND[0], BOX_BOUND[1])
        box_pos = [-1.2*box_size[0]/2, -1.2*box_size[1]/2, 0.0]
        box_id = self.bh.load_stl('packing_box.stl', box_size, box_pos, [0, 0, 0, 1])
        self.logger.info('prepare_packing_box, id = {}'.format(box_id))
        return box_size, box_pos, box_id
    
    def get_observation(self, model):
        self.logger.info('get_observation')
        box_cloud = self.cm.get_point_cloud('box_cam')
        box_cloud = box_cloud.transform(self.cm.get_extrinsic('box_cam'))
        # self.gc.o3d_show(box_cloud)
        box_voxel = self.gc.get_voxel_from_cloud(box_cloud, voxel_size=0.002)
        t0 = time.time()
        tar_center = [0, 0, self.box_size[2]]
        box_view = self.gc.get_view_from_voxel(box_voxel, self.pixel_size, self.img_width, tar_center, self.bound_size, '-z')
        if box_view is None:
            return None
        print(box_view)
        if SHOE_IMG:
            plt.imshow(box_view, cmap='gray', vmin=0, vmax=255)
            plt.show()
        t1 = time.time()
        # time.sleep(2)
        self.logger.info('get_view_from_voxel spend {} s. size = {}'.format(t1-t0, len(box_cloud.points)))
        obj_cloud_list = []
        relative_angle = 2 * np.pi / VIEWS_PER_OBJ
        curr_angle = 0.0
        self.logger.info('get_observation 2')
        for i in range(VIEWS_PER_OBJ):
            cloud = self.cm.get_point_cloud('obj_cam')
            print('point cloud center 1 = {}'.format(cloud.get_center()))
            cloud = cloud.transform(self.cm.get_extrinsic('obj_cam'))
            print('point cloud center 2 = {}'.format(cloud.get_center()))
            if i > 0:
                cloud = self.gc.cloud_rotate_euler(cloud, [0, 0, curr_angle], CAPTURE_POS)
            obj_cloud_list.append(cloud)
            curr_angle -= relative_angle
            self.mm.set_model_relative_euler(model, [0, 0, relative_angle])
            # time.sleep(1)
        self.logger.info('get_observation 3')
        merged_cloud = self.gc.merge_cloud(obj_cloud_list)
        if len(merged_cloud.points) < 100:
            return None
        self.logger.info('merge_cloud size = {}'.format(len(merged_cloud.points)))
        self.logger.info('merge_cloud remove_non_finite_points size = {}'.format(len(merged_cloud.points)))
        self.voxel = self.gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
        self.logger.info('get_observation 4')
        tar_center = list((np.array(START_BOUND[0]) + np.array(START_BOUND[1])) / 2)
        self.logger.info('get_observation 5')
        t0 = time.time()
        self.views = self.gc.get_3_views_from_voxel(self.voxel, self.pixel_size, self.img_width, tar_center, self.bound_size)
        if self.views is None:
            return None
        if SHOE_IMG:
            plt.imshow(self.views[0], cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(self.views[1], cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(self.views[2], cmap='gray', vmin=0, vmax=255)
            plt.show()
        t1 = time.time()
        self.logger.info('get_3_views_from_voxel spend {} s. size = {}'.format(t1-t0, len(merged_cloud.points)))
        self.logger.info('get_observation 6')
        self.views = np.append(self.views, np.expand_dims(box_view, axis=0), axis=0)
        
        self.logger.info('get_observation')
        return self.views

    def reset(self):
        self.logger.info('reset')
        self.bh.reset_all()
        self.box_size, self.box_pos, self.box_id = self.prepare_packing_box()
        self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        self.bound_size = max(self.box_size) + 0.1
        self.pixel_size = self.bound_size / self.img_width
        self.volume_sum = 0
        self.model_indx = 0
        self.model_list = self.mm.sample_models_in_bound(self.box_size, 0.5)
        self.bh.step_simulation(60, realtime=False)
        self.curr_model = self.prepare_objects()
        self.logger.info('reset')
        obs = self.get_observation(self.curr_model)
        if obs is None:
            return self.reset()
        return obs

    def render(self, mode="human"):
        self.logger.info('render')
        self.gc.o3d_show(self.voxel)
        for view in self.views:
            self.gc.o3d_show(o3d.geometry.Image(view))
        self.logger.info('render')

    def close(self):
        ...