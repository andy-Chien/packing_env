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

SHOW_IMG = False

MODEL_PKG = 'objects_model'
THIS_PKG = 'packing_env'
BOX_BOUND = [[0.15, 0.15, 0.15], [0.3, 0.3, 0.3]]
START_BOUND = [[0, 0, 1.25], [0, 0, 1.35]]

VIEWS_PER_OBJ = 3
CAPTURE_POS = [0, 0, 1.3]
NUM_TO_CALC_SUCCESS_RATE = 100


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
            bullet_gui: bool = False,
            env_index: int = 0,
        ):
        super().__init__()
        model_path = get_package_share_directory(MODEL_PKG)
        camera_config_path = get_package_share_directory(THIS_PKG) + '/config/bullet_camera.yaml'
        self.img_width = img_width
        self.env_index = env_index
        self.bh = BulletHandler(bullet_gui)
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

        self.observation_space = spaces.Dict({
            'box': spaces.Box(low=0, high=255, shape=(1, self.img_width, self.img_width), dtype=np.uint8),
            'obj': spaces.Box(low=0, high=255, shape=(3, self.img_width, self.img_width), dtype=np.uint8),
            'num': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.success_buffer = []
        self.reward_buffer = []
        self.box_fill_rate = 0.2
        self.center_xy = [0, 0]
        self.eps_cnt = 0

    def step(self, action):
        action_transed = self.decode_action(action)
        z_to_place, z_in_box = self.compute_place_z(action_transed)
        self.mm.set_model_pos(self.curr_model, [action_transed[0], action_transed[1], z_in_box])
        self.mm.set_model_relative_euler(self.curr_model, [0, 0, action_transed[2]])
        c_points = self.bh.get_closest_points(self.box_id, self.mm.get_model_id(self.curr_model))
        self.logger.info('action = {}, z_to_place = {}, leng of c_points is {}'.format(action, z_to_place, len(c_points)))
        if z_to_place < 0:
            self.logger.info('!!!!!!!!!!!!!!!!!!! FUCK !!!!!!!!!!!!!!!!!!!!')
        self.mm.set_model_pos(self.curr_model, [action_transed[0], action_transed[1], z_to_place])
        self.bh.step_simulation(120, realtime=self.env_index==0)
        self.bh.step_simulation(240, realtime=False)
        self.bh.set_model_pose(self.box_id, self.box_pos, [0,0,0,1])
        self.bh.step_simulation(60, realtime=False)

        obj_volume = self.mm.get_model_convex_volume(self.curr_model)

        done = self._check_success()
        
        if not done:
            self.volume_sum += obj_volume
        
        reward = self._compute_reward(c_points)

        obs = None
        model = None
        while self.obj_in_queue() and obs is None and not done:
            model = self.prepare_objects()
            obs = self.get_observation(model)
            if obs is None:
                self.mm.remove_model(model)


        if model is not None and obs is not None:
            self.curr_model = model
        elif not done:
            done = self._check_success()
            reward = self._compute_reward(c_points)

        self.reward_buffer[-1] += reward

        if done:
            self.eps_cnt += 1
            bfr = self.box_fill_rate
            success_rate = avg_reward = 0.0
            if len(self.success_buffer) == NUM_TO_CALC_SUCCESS_RATE:
                success_rate = sum(self.success_buffer) / NUM_TO_CALC_SUCCESS_RATE
                avg_reward = sum(self.reward_buffer) / NUM_TO_CALC_SUCCESS_RATE
                if success_rate > 0.7 and self.success:
                    self.box_fill_rate = min(self.box_fill_rate * 1.01, 0.9)
                elif success_rate < 0.5 and self.failed:
                    self.box_fill_rate = max(self.box_fill_rate * 0.99, 0.1)
            self.logger.info('-------------------------------------------------------------------')
            self.logger.info('env: {}, eps: {}, success rate = {}, avg reward = {}, fill rate = {}'.format(
                self.env_index, self.eps_cnt, success_rate, avg_reward,  bfr))
            self.logger.info('-------------------------------------------------------------------')
            

        
        info = {'info': 'hello'}

        return obs, reward, done, info
    
    def decode_action(self, action):
        action_transed = [x for x in action]
        if isinstance(self.action_space, spaces.Box):
            action_transed[0] = action_transed[0] * self.box_size[0] / 2 + self.center_xy[0]
            action_transed[1] = action_transed[1] * self.box_size[1] / 2 + self.center_xy[1] # 2 because [-1, 1]
            action_transed[2] *= np.pi
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_transed[0] -= self.xy_action_space / 2
            action_transed[1] -= self.xy_action_space / 2
            action_transed[2] -= self.rot_action_space / 2
            action_transed[0] = action_transed[0] * self.box_size[0] / self.xy_action_space + self.center_xy[0]
            action_transed[1] = action_transed[1] * self.box_size[1] / self.xy_action_space + self.center_xy[1]
            action_transed[2] *= np.pi / self.rot_action_space
        return action_transed
    
    def compute_place_z(self, action):
        index_x = round(-1 * (action[1] - self.center_xy[1]) / self.pixel_size + self.img_width / 2)
        index_y = round(-1 * (action[0] - self.center_xy[0]) / self.pixel_size + self.img_width / 2)
        obj_bottom_view = self.obj_views[2]
        obj_bv_transpose = np.array(obj_bottom_view).transpose()
        x_min, y_min, x_max, y_max = 0, 0, self.img_width, self.img_width
        for i in range(self.img_width):
            if x_min == 0 and any(obj_bottom_view[i]):
                x_min = i
                break
        for i in reversed(range(self.img_width)):
            if x_max == self.img_width and any(obj_bottom_view[i]):
                x_max = i
                break
        for i in range(self.img_width):
            if y_min == 0 and any(obj_bv_transpose[i]):
                y_min = i
                break
        for i in reversed(range(self.img_width)):
            if y_max == self.img_width and any(obj_bv_transpose[i]):
                y_max = i
                break

        x_range = round((x_max - x_min) / 2)
        y_range = round((y_max - y_min) / 2)
        indx_x_min, indx_x_max = max(0, index_x - x_range), min(self.img_width, index_x + x_range)
        indx_y_min, indx_y_max = max(0, index_y - y_range), min(self.img_width, index_y + y_range)
        depth_min = 999
        for i in range(indx_x_min, indx_x_max):
            for j in range(indx_y_min, indx_y_max):
                depth = self.box_view[i][j]
                if depth < depth_min:
                    depth_min = depth
        z = self.box_size[2] - depth_min * self.bound_size / 255
        z_in_box = self.box_size[2] / 2
        # self.logger.info('depth_min = {}, z = {}, x_range = {}, y_range = {}'.format(depth_min, z, x_range, y_range))
        obj_front_view = self.obj_views[0]
        for i in reversed(range(len(obj_front_view))):
            if any(obj_front_view[i]):
                z_adjust = max(0, (i - self.img_width / 2)) * self.pixel_size # assume midle of img is midle of obj
                z += z_adjust + 0.05
                z_in_box += z_adjust
                break
        # self.logger.info('depth_min = {}, z = {}'.format(depth_min, z))
        return z, z_in_box
        
    
    def obj_in_queue(self):
        return len(self.model_list) > 0

    def is_done(self):
        return self.success or self.failed
    
    def _check_success(self):
        obj_pos, _ = self.mm.get_model_pose(self.curr_model)
        if obj_pos[0] > self.box_size[0] / 2 or \
                obj_pos[0] < -1 * self.box_size[0] / 2 or \
                obj_pos[1] > self.box_size[1] / 2 or \
                obj_pos[1] < -1 * self.box_size[1] / 2 or \
                obj_pos[2] > self.box_size[2]:
            self.failed = True
            self.success_buffer[-1] = 0
        elif not self.obj_in_queue():
            self.success = True
            self.success_buffer[-1] = 1
            

        return self.is_done()

    def _compute_reward(self, collision_points=[]):
        if self.failed:
            r = -1 + self.volume_sum / self.box_volume
        elif self.success:
            r = self.volume_sum / self.box_volume
        elif collision_points != []:
            r = (-1 + self.volume_sum / self.box_volume) / 10
        else:
            r = (self.volume_sum / self.box_volume) / 10
        
        return r

    def prepare_objects(self):
        pos = self.mm.random_pos(START_BOUND)
        quat = self.mm.random_quat()
        # quat = [0, 0, 0, 1]
        curr_model = self.model_list.pop()
        self.mm.load_model(curr_model, pos, quat)
        return curr_model

    def prepare_packing_box(self):
        box_size = np.random.uniform(BOX_BOUND[0], BOX_BOUND[1])
        box_pos = [-1.2*box_size[0]/2, -1.2*box_size[1]/2, 0.0]
        box_id = self.bh.load_stl('packing_box_with_cover.obj', box_size, box_pos, [0, 0, 0, 1], static=True)
        return box_size, box_pos, box_id
    
    def get_observation(self, model):
        box_cloud = self.cm.get_point_cloud('box_cam')
        box_cloud = box_cloud.transform(self.cm.get_extrinsic('box_cam'))
        box_voxel = self.gc.get_voxel_from_cloud(box_cloud, voxel_size=0.002)
        tar_center = [self.center_xy[0], self.center_xy[1], self.box_size[2]]
        self.box_view = self.gc.get_view_from_voxel(box_voxel, self.pixel_size, self.img_width, tar_center, self.bound_size, '-z')
        if self.box_view is None:
            return None
        if SHOW_IMG:
            plt.imshow(self.box_view, cmap='gray', vmin=0, vmax=255)
            plt.show()
        obj_cloud_list = []
        relative_angle = 2 * np.pi / VIEWS_PER_OBJ
        curr_angle = 0.0
        for i in range(VIEWS_PER_OBJ):
            cloud = self.cm.get_point_cloud('obj_cam')
            cloud = cloud.transform(self.cm.get_extrinsic('obj_cam'))
            if i > 0:
                cloud = self.gc.cloud_rotate_euler(cloud, [0, 0, curr_angle], CAPTURE_POS)
            obj_cloud_list.append(cloud)
            curr_angle -= relative_angle
            self.mm.set_model_relative_euler(model, [0, 0, relative_angle])
            # time.sleep(1)
        merged_cloud = self.gc.merge_cloud(obj_cloud_list)
        if len(merged_cloud.points) < 100:
            return None
        self.voxel = self.gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
        tar_center = list((np.array(START_BOUND[0]) + np.array(START_BOUND[1])) / 2)
        self.obj_views = self.gc.get_3_views_from_voxel(self.voxel, self.pixel_size, self.img_width, tar_center, self.bound_size)
        if self.obj_views is None:
            return None
        if SHOW_IMG:
            plt.imshow(self.obj_views[0], cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(self.obj_views[1], cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(self.obj_views[2], cmap='gray', vmin=0, vmax=255)
            plt.show()
        # self.obj_views = np.append(self.obj_views, np.expand_dims(self.box_view, axis=0), axis=0)

        obs = {'box': [self.box_view],
               'obj': self.obj_views,
               'num': [1.0]}
        return obs

    def reset(self, seed=0):
        self.success = False
        self.failed = False
        self.bh.reset_all()
        self.mm.reset()
        self.box_size, self.box_pos, self.box_id = self.prepare_packing_box()
        self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        self.bound_size = max(self.box_size) + 0.1
        self.pixel_size = self.bound_size / self.img_width
        self.volume_sum = 0
        self.model_list = self.mm.sample_models_in_bound(self.box_size, self.box_fill_rate)
        if len(self.model_list) < 1:
            return self.reset()
        self.bh.step_simulation(60, realtime=False)
        self.curr_model = self.prepare_objects()
        obs = self.get_observation(self.curr_model)
        if obs is None:
            return self.reset()

        self.reward_buffer.append(0.0)
        if len(self.reward_buffer) > NUM_TO_CALC_SUCCESS_RATE:
            self.reward_buffer.pop(0)

        self.success_buffer.append(0)
        if len(self.success_buffer) > NUM_TO_CALC_SUCCESS_RATE:
            self.success_buffer.pop(0)
        return obs

    def render(self, mode="human"):
        self.gc.o3d_show(self.voxel)
        for view in self.obj_views:
            self.gc.o3d_show(o3d.geometry.Image(view))

    def close(self):
        ...