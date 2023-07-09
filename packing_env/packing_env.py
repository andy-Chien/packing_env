import gym
import time
import numpy as np
import open3d as o3d
import quaternion as qtn
from rclpy import logging
from gym import spaces
from math import ceil, floor
from matplotlib import pyplot as plt
from ament_index_python.packages import get_package_share_directory
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter
# np.set_printoptions(threshold=np.inf)

SHOW_IMG = False

MODEL_PKG = 'objects_model'
THIS_PKG = 'packing_env'
BOX_BOUND = [[0.12, 0.12, 0.12], [0.28, 0.28, 0.28]]
START_BOUND = [[0, 0, 1.25], [0, 0, 1.35]]

VIEWS_PER_OBJ = 3
CAPTURE_POS = [0, 0, 1.3]
NUM_TO_CALC_SUCCESS_RATE = 30
INIT_DIFFICULTY = 0.32


class PackingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    logger = logging.get_logger('packing env')
    metadata = {"render.modes": ["human"]}

    def __init__(self,
            img_width: int = 64,
            random_start: bool = True,
            discrete_actions: bool = False,
            channel_last: bool = True,
            xy_action_space: int = 32,
            rot_action_space: int = 36,
            bullet_gui: bool = False,
            env_index: int = 0,
        ):
        super().__init__()
        self.model_pkg = MODEL_PKG
        self.model_pkg += '_' + str(env_index)
        model_path = get_package_share_directory(self.model_pkg)
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
            self.action_space = spaces.Box(-1, 1, (4,))

        self.observation_space = spaces.Dict({
            'box': spaces.Box(low=0, high=1.0, shape=(1, self.img_width, self.img_width), dtype=np.float32),
            'obj_f': spaces.Box(low=0, high=1.0, shape=(1, self.img_width, self.img_width), dtype=np.float32),
            'obj_s': spaces.Box(low=0, high=1.0, shape=(1, self.img_width, self.img_width), dtype=np.float32),
            'obj_b': spaces.Box(low=0, high=1.0, shape=(1, self.img_width, self.img_width), dtype=np.float32)})
            # 'num': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        # })
        self.box_bound = np.array(BOX_BOUND)
        self.success_buffer = []
        self.reward_buffer = []
        self.difficulty = INIT_DIFFICULTY
        self.center_xy = [0, 0]
        self.eps_cnt = 0
        self.success_rate = 0.0
        self.avg_reward = 0.0
        self.box_scale = 1.0
        self.fill_rate = 0

    def step(self, action):
        action_transed = self.decode_action(np.array(action, dtype=np.float32))
        z_to_place, z_in_box = self.compute_place_z(action_transed)
        self.mm.set_model_pos(self.curr_model, [action_transed[0], action_transed[1], z_in_box])
        self.mm.set_model_relative_euler(self.curr_model, [0, 0, action_transed[2]])
        c_points = self.bh.get_closest_points(self.box_id, self.mm.get_model_id(self.curr_model))
        if z_to_place < 0:
            self.logger.info('!!!!!!!!!!!!!!!!!!! FUCK !!!!!!!!!!!!!!!!!!!!')
            self.logger.info('!!!!!!!!!!!!!!!!!!! FUCK !!!!!!!!!!!!!!!!!!!!')
            self.logger.info('!!!!!!!!!!!!!!!!!!! FUCK !!!!!!!!!!!!!!!!!!!!')
            self.logger.info('!!!!!!!!!!!!!!!!!!! FUCK !!!!!!!!!!!!!!!!!!!!')
        self.mm.set_model_pos(self.curr_model, [action_transed[0], action_transed[1], z_to_place])
        check_z_c = self.bh.get_closest_points(self.box_id, self.mm.get_model_id(self.curr_model))
        if len(check_z_c) > 1:
            self.mm.set_model_pos(self.curr_model, [action_transed[0], action_transed[1], z_to_place + 0.05])

        rot_z = 0.0
        if 'random_box' in self.curr_model:
            rot_z = self.mm.get_model_rot_z(self.curr_model)
        self.bh.step_simulation(60, realtime=self.env_index==0)
        obj_pos_begin, obj_ori_begin = self.mm.get_model_pose(self.curr_model)

        self.bh.step_simulation(240, realtime=False)
        obj_pos_after, obj_ori_after = self.mm.get_model_pose(self.curr_model)
        self.bh.set_model_pose(self.box_id, self.box_pos, [0,0,0,1])
        self.bh.step_simulation(60, realtime=False)

        
        self.volume_sum += self.mm.get_model_convex_volume(self.curr_model)

        obs = None
        model = None
        while self.obj_in_queue() and obs is None:
            model = self.prepare_objects()
            obs = self.get_observation(model)
            if obs is None:
                self.mm.remove_model(model)
        min_value = 0.0
        particle_var = None
        if obs is not None:
            iw = self.img_width
            ps = self.pixel_size
            hbs = self.box_size / 2
            idx_x_min, idx_x_max = ceil(iw / 2 - hbs[0] / ps + 3), floor(iw / 2 + hbs[0] / ps - 2)
            idx_y_min, idx_y_max = ceil(iw / 2 - hbs[1] / ps + 3), floor(iw / 2 + hbs[1] / ps - 2)
            box_view_int = (self.box_view[
                idx_x_min : idx_x_max , idx_y_min : idx_y_max] * self.bound_size / ps).round().astype(np.int32)

            min_value = np.min(box_view_int)
            max_value = np.max(box_view_int)
            voxel_list = []
            # voxel_sum = np.array([0.0, 0.0, 0.0])
            for idx_x in range(box_view_int.shape[0]):
                for idx_y in range(box_view_int.shape[1]):
                    num_z = box_view_int[idx_x, idx_y] - min_value
                    for idx_z in range(num_z):
                        voxel_list.append(np.array([idx_x, idx_y, idx_z]))
            #             voxel_sum += voxel_list[-1]
            # voxel_origin = (voxel_sum / len(voxel_list)).round().astype(np.int32)
            # voxel_sum = np.array([0.0, 0.0, 0.0])
            # for voxel in voxel_list:
            #     voxel_sum += np.absolute(voxel_origin - voxel)
            the_size = np.array(
                [max(idx_x_max - idx_x_min, 1), 
                 max(idx_y_max - idx_y_min, 1),
                 max(max_value - min_value, 1)],
                dtype=np.float32
            )
            # voxel_avg_dis = voxel_sum / len(voxel_list)
            # voxel_avg_dis /= (the_size / 2)
            if len(voxel_list) <= 1:
                particle_var = np.array([0.0, 0.0, 0.0])
            else:
                cube_size = max(floor(len(voxel_list)**(1/3)), 2)
                cube_voxel_var = np.var(np.array([range(cube_size)] * 3), axis=1)
                particle_var = np.array([max(x, 0.0) for x in (np.var(np.array(voxel_list), axis=0) - cube_voxel_var)])
                # print("particle_var = {}, min_value = {}, max_value = {}".format(particle_var, min_value, max_value))
                particle_var /= np.array([max(x, 1.0) for x in ((the_size / 2)**2 - cube_voxel_var)])

        done = self._check_success(obj_pos_begin, min_value)

        fill_rate = self.volume_sum / self.box_volume
        pos_dif = np.array([a - b for a, b in zip(obj_pos_after[:2], action_transed[:2])])
        reward = self._compute_reward(c_points, particle_var, fill_rate, pos_dif, rot_z)


        if self.env_index == 0 and not done:
            self.logger.info('fill_rate = {}, action_transed = {}, z_to_place = {}, self.box_bound {}, particle_var =  {}, reward = {},'.format(
                fill_rate, action_transed, z_to_place, self.box_bound, particle_var, reward))
            if len(c_points) == 0 and c_points != []:
                self.logger.info('c_points = {}'.format(c_points))

        if model is not None and obs is not None:
            self.curr_model = model
        else:
            done = True

        self.reward_buffer[-1] += reward

        if done:
            self.eps_cnt += 1
            dft = self.difficulty
            success_rate = avg_reward = 0.0
            if len(self.success_buffer) == NUM_TO_CALC_SUCCESS_RATE:
                success_rate = sum(self.success_buffer) / NUM_TO_CALC_SUCCESS_RATE
                avg_reward = sum(self.reward_buffer) / NUM_TO_CALC_SUCCESS_RATE
                if fill_rate > self.difficulty / 0.9:
                    self.difficulty = min(self.difficulty + 0.001, 0.9)
                elif fill_rate < self.difficulty * 0.9 and self.failed:
                    self.difficulty = max(self.difficulty - 0.001, INIT_DIFFICULTY)
            self.logger.info('-------------------------------------------------------------------')
            self.logger.info('env: {}, eps: {}, fill rate = {}, avg reward = {}, difficulty = {}'.format(
                self.env_index, self.eps_cnt, fill_rate, avg_reward, dft))
            self.logger.info('-------------------------------------------------------------------')
            self.success_rate, self.avg_reward, self.fill_rate = success_rate, avg_reward, fill_rate

        info = {'info': 'hello'}

        return obs, reward, done, info
    
    def decode_action(self, action):
        action_transed = [0.0] * 3 # [x*abs(x) for x in action]
        if isinstance(self.action_space, spaces.Box):
            action_transed[0] = action[0] * self.box_size[0] / 2 + self.center_xy[0]
            action_transed[1] = action[1] * self.box_size[1] / 2 + self.center_xy[1] # 2 because [-1, 1]
            ql = (action[2]**2 + action[3]**2)**0.5
            qw = action[2] / ql if ql > 0.0001 else 1.
            qz = action[3] / ql if ql > 0.0001 else 0.
            action_transed[2] = qtn.as_rotation_vector(qtn.quaternion(qw, 0., 0., qz))[2]
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action[0] -= self.xy_action_space / 2
            action[1] -= self.xy_action_space / 2
            action[2] -= self.rot_action_space / 2
            action_transed[0] = action[0] * self.box_size[0] / self.xy_action_space + self.center_xy[0]
            action_transed[1] = action[1] * self.box_size[1] / self.xy_action_space + self.center_xy[1]
            action_transed[2] = action[2] * 2 * np.pi / self.rot_action_space
        return action_transed
    
    def compute_place_z(self, action):
        iw = self.img_width
        ps = self.pixel_size
        rotated_cloud = self.gc.geometry_rotate_euler(self.obj_cloud, [0, 0, action[2]], CAPTURE_POS)
        rotated_voxel = self.gc.get_voxel_from_cloud(rotated_cloud, voxel_size=0.008)
        tar_center = list((np.array(START_BOUND[0]) + np.array(START_BOUND[1])) / 2)
        tar_center = [tar_center[0], tar_center[1], tar_center[2] - self.bound_size / 2]
        obj_bottom_view = self.gc.get_view_from_voxel(
            rotated_voxel, ps, iw, tar_center, self.bound_size, 'z', y_rev=True)
        if SHOW_IMG:
            # self.gc.o3d_show(self.obj_cloud)
            # self.gc.o3d_show(rotated_cloud)
            # self.gc.o3d_show(rotated_voxel)
            plt.imshow(obj_bottom_view, cmap='gray', vmin=0, vmax=1.0)
            plt.show()
        
        obj_bv_transpose = np.array(obj_bottom_view).transpose()

        x_min, y_min, x_max, y_max = 0, 0, iw, iw
        for i in range(iw):
            if any(obj_bottom_view[i]):
                x_min = i
                break
        for i in reversed(range(iw)):
            if any(obj_bottom_view[i]):
                x_max = i
                break
        for i in range(iw):
            if any(obj_bv_transpose[i]):
                y_min = i
                break
        for i in reversed(range(iw)):
            if any(obj_bv_transpose[i]):
                y_max = i
                break

        x_range = round((x_max - x_min) / 2)
        y_range = round((y_max - y_min) / 2)
        x_range_real = x_range * ps
        y_range_real = y_range * ps

        hbs = self.box_size / 2
        dis_to_x_wall = min(abs(action[0] - hbs[0]), abs(action[0] + hbs[0]))
        dis_to_y_wall = min(abs(action[1] - hbs[1]), abs(action[1] + hbs[1]))
        obj_in_x_wall = min(max(x_range_real - dis_to_x_wall, 0), x_range_real)
        obj_in_y_wall = min(max(y_range_real - dis_to_y_wall, 0), y_range_real)
        self.obj_in_wall = obj_in_x_wall + obj_in_y_wall
        action[0] -= np.sign(action[0]) * obj_in_x_wall
        action[1] -= np.sign(action[1]) * obj_in_y_wall

        translated_cloud = self.gc.geometry_translate(rotated_cloud, [action[0], action[1], 0])
        translated_voxel = self.gc.get_voxel_from_cloud(translated_cloud, voxel_size=0.008)
        translated_obj_bottom_view = self.gc.get_view_from_voxel(
            translated_voxel, ps, iw, tar_center, self.bound_size, 'z', y_rev=True)
        if SHOW_IMG:
            plt.imshow(translated_obj_bottom_view, cmap='gray', vmin=0, vmax=1.0)
            plt.show()

        index_x = round(-1 * (action[0] - self.center_xy[0]) / ps + iw / 2)
        index_y = round(-1 * (action[1] - self.center_xy[1]) / ps + iw / 2)
        box_indx_x_min, box_indx_x_max = ceil(iw / 2 - hbs[0] / ps + 3), floor(iw / 2 + hbs[0] / ps - 3)
        box_indx_y_min, box_indx_y_max = ceil(iw / 2 - hbs[1] / ps + 3), floor(iw / 2 + hbs[1] / ps - 3)
        indx_x_min, indx_x_max = max(box_indx_x_min, index_x - x_range), min(box_indx_x_max, index_x + x_range)
        indx_y_min, indx_y_max = max(box_indx_y_min, index_y - y_range), min(box_indx_y_max, index_y + y_range)


        depth_min = self.box_size[2] / self.bound_size
        for i in range(indx_x_min, indx_x_max):
            depth_x = self.box_view[i]
            for j in range(indx_y_min, indx_y_max):
                depth = depth_x[j]
                if depth > 0.001 and depth < depth_min:
                    depth_min = depth
        z = self.box_size[2] - depth_min * self.bound_size
        z_in_box = self.box_size[2] / 2
        # self.logger.info('depth_min = {}, z = {}, x_range = {}, y_range = {}'.format(depth_min, z, x_range, y_range))
        obj_front_view = self.obj_views[0]
        for i in reversed(range(len(obj_front_view))):
            if any(obj_front_view[i]):
                z_adjust = max(-1 * z_in_box, (i - iw / 2)) * ps # assume midle of img is midle of obj
                z += z_adjust + 0.01
                z_in_box += z_adjust
                break
        # self.logger.info('depth_min = {}, z = {}'.format(depth_min, z))
        
        return z, z_in_box
        
    
    def obj_in_queue(self):
        return len(self.model_list) > 0

    def is_done(self):
        return self.success or self.failed
    
    def _check_success(self, obj_pos, min_z):
        if min_z < -5 or \
                obj_pos[0] > self.box_size[0] / 2 or \
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

    def _compute_reward(self, collision_points=[], particle_var=None, fill_rate=0, pos_dif=0, rot_z=0):
        factor = 0
        if self.failed:
            r = 3 * (-1 + fill_rate)
        elif len(collision_points) > 1 and self.obj_in_wall > 0.01:
            factor = min((self.obj_in_wall - 0.01) / 0.2, 1.0) / 10
            r = (-1 + fill_rate) * factor
        else:
            r = fill_rate

        # # =========== For SAC 0.1 ============
        # if particle_var is not None:
        #     avg_var =  np.average(particle_var, weights=np.array([1, 1, 1]))
        #     var_factor = avg_var - 0.1
        #     if var_factor < 0:
        #         r -= fill_rate * var_factor
        #     else:
        #         r -= (1 - fill_rate) * var_factor
        # pos_factor = ((((np.linalg.norm(pos_dif)) / self.bound_size)**2) - 0.05)
        # if pos_factor < 0:
        #     r -= fill_rate * pos_factor
        # else:
        #     r -= (1 - fill_rate) * pos_factor
        # print("pos dif rate = {}".format((np.linalg.norm(pos_dif) / self.bound_size)**2))
        # # =========== END For SAC 0.1 ============


        # # =========== For PPO 0.1 ============
        # abs_z = abs(rot_z)
        # min_ang = min(abs_z, abs(abs_z - np.pi/2), abs(abs_z - np.pi))
        # ang_factor = (min_ang - 0.2) * 0.5
        # if ang_factor < 0:
        #     r -= fill_rate * ang_factor
        # else:
        #     r -= (1 - fill_rate) * ang_factor
        # # =========== END For PPO 0.1 ============

        # print("factor = {}, avg_var = {}, pos dif rate = {}, min_ang = {}".format(factor, avg_var, np.linalg.norm(pos_dif) / self.bound_size, min_ang))

        return r

    def prepare_objects(self):
        pos = self.mm.random_pos(START_BOUND)
        quat = self.mm.random_quat(range=self.difficulty**5)
        # quat = [0, 0, 0, 1]
        curr_model = self.model_list.pop()
        self.mm.load_model(curr_model, pos, quat)
        return curr_model

    def prepare_packing_box(self, box_size):
        box_pos = [-1.2*box_size[0]/2, -1.2*box_size[1]/2, 0.0]  # 1.2 is the original mesh origin
        box_id = self.bh.load_stl('packing_box_with_cover.obj', box_size, box_pos, [0, 0, 0, 1], static=True)
        return box_pos, box_id
    
    def random_box_size(self):
        self.box_bound *= self.box_scale
        box_bound_factor = [(1 - self.difficulty) * (b - a) / 2 for a, b in zip(self.box_bound[0], self.box_bound[1])]
        bb_0 = [a + b for a, b in zip(self.box_bound[0], box_bound_factor)]
        bb_1 = [a - b for a, b in zip(self.box_bound[1], box_bound_factor)]
        box_size = np.random.uniform(bb_0, bb_1)
        return box_size
    
    def get_observation(self, model, reset=False):
        box_cloud = self.cm.get_point_cloud('box_cam')
        box_cloud = box_cloud.transform(self.cm.get_extrinsic('box_cam'))
        box_voxel = self.gc.get_voxel_from_cloud(box_cloud, voxel_size=0.003)
        tar_center = [self.center_xy[0], self.center_xy[1], self.box_size[2] * 1.1]
        old_box_view = self.box_view.copy() if not reset else None
        self.box_view = self.gc.get_view_from_voxel(box_voxel, self.pixel_size, self.img_width, tar_center, self.bound_size, '-z')
        if self.box_view is None:
            return None

        if not reset:
            for r1, r2 in zip(old_box_view, self.box_view):
                if np.any(r1):
                    for i in range(len(r1)):
                        if r2[i] == 0:
                            r2[i] = r1[i]

        if SHOW_IMG:
            plt.imshow(self.box_view, cmap='gray', vmin=0, vmax=1.0)
            plt.show()
        obj_cloud_list = []
        relative_angle = 2 * np.pi / VIEWS_PER_OBJ
        curr_angle = 0.0
        for i in range(VIEWS_PER_OBJ):
            cloud = self.cm.get_point_cloud('obj_cam')
            cloud = cloud.transform(self.cm.get_extrinsic('obj_cam'))
            if i > 0:
                cloud = self.gc.geometry_rotate_euler(cloud, [0, 0, curr_angle], CAPTURE_POS)
            obj_cloud_list.append(cloud)
            curr_angle -= relative_angle
            self.mm.set_model_relative_euler(model, [0, 0, relative_angle])
            # time.sleep(1)
        self.obj_cloud = self.gc.merge_cloud(obj_cloud_list)
        if len(self.obj_cloud.points) < 100:
            return None
        self.obj_voxel = self.gc.get_voxel_from_cloud(self.obj_cloud, voxel_size=0.003)
        tar_center = list((np.array(START_BOUND[0]) + np.array(START_BOUND[1])) / 2)
        self.obj_views = self.gc.get_3_views_from_voxel(self.obj_voxel, self.pixel_size, self.img_width, tar_center, self.bound_size)
        if self.obj_views is None:
            return None
        if SHOW_IMG:
            plt.imshow(self.obj_views[0], cmap='gray', vmin=0, vmax=1.0)
            plt.show()
            plt.imshow(self.obj_views[1], cmap='gray', vmin=0, vmax=1.0)
            plt.show()
            plt.imshow(self.obj_views[2], cmap='gray', vmin=0, vmax=1.0)
            plt.show()
        # self.obj_views = np.append(self.obj_views, np.expand_dims(self.box_view, axis=0), axis=0)

        obs = {'box': [self.box_view],
               'obj_f': [self.obj_views[0]],
               'obj_s': [self.obj_views[1]],
               'obj_b': [self.obj_views[2]]}
            #    'num': [1.0]}    
        return obs

    def reset(self, seed=0):
        self.success = False
        self.failed = False
        self.bh.reset_all()
        self.mm.reset()
        self.model_list = []
        while len(self.model_list) < 1:
            self.box_size = self.random_box_size()
            self.box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
            self.bound_size = max(max(self.box_size) + 0.01, max(self.box_size) * 1.1)
            self.pixel_size = self.bound_size / self.img_width
            self.volume_sum = 0
            self.model_list, self.box_scale = self.mm.sample_models_in_bound(
                self.box_size, fill_rate=self.difficulty, max_length_rate=(self.difficulty/2)**0.33)

        self.box_pos, self.box_id = self.prepare_packing_box(self.box_size)
        self.bh.step_simulation(60, realtime=False)
        self.curr_model = self.prepare_objects()
        obs = self.get_observation(self.curr_model, reset=True)
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
        self.gc.o3d_show(self.obj_voxel)
        for view in self.obj_views:
            self.gc.o3d_show(o3d.geometry.Image(view))

    def close(self):
        ...