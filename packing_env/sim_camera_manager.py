from cmath import sqrt
from multiprocessing.managers import ValueProxy
import os
import numpy as np
import pybullet as pb
import open3d as o3d
from random import choice as random_choice
from yaml import safe_load as yaml_load
from rclpy import logging

class SimCameraManager():
    def __init__(self, config_path):
        self.camera_data = self.__config_loader(config_path)
        self.active_cameras = dict()
        self.logger = logging.get_logger('sim_camera_manager')
        print(self.camera_data)

    def __config_loader(self, path):
        with open(path, 'r') as stream:
            data = yaml_load(stream)
        return data

    def create_camera(self, name, model, type, pos, target_pos, up_vec):
        if model in self.camera_data.keys() and type in self.camera_data[model].keys():
            cam = dict(self.camera_data[model][type])
            aspect = cam['width'] / cam['height']
            cam['view_matrix']  = pb.computeViewMatrix(pos, target_pos, up_vec)
            cam['projection_matrix'] = pb.computeProjectionMatrixFOV(cam['fov'], aspect, \
                                                                    cam['near_plane'], cam['far_plane'])
            cam['intrinsic'] = o3d.camera.PinholeCameraIntrinsic(width=cam['width'], height=cam['height'], \
                                                    fx=cam['fx'], fy=cam['fy'], cx=cam['cx'], cy=cam['cy'])
            cam['extrinsic'] = self._compute_extrinsic(pos, target_pos, up_vec)
            print('====================cam[extrinsic]========================')
            print(cam['extrinsic'])
            print('====================cam[extrinsic]========================')
            self.active_cameras[name] = cam
            return True
        else:
            self.logger.error('[SimCameraManager]: Camera model or type is wrong!')
            print('{} in {} is {}'.format(model, self.camera_data.keys(), model in self.camera_data.keys()))
            return False

    def get_image(self, name):
        if name in self.active_cameras.keys():
            cam = self.active_cameras[name]
            images = pb.getCameraImage(cam['width'], cam['height'], cam['view_matrix'], \
                                    cam['projection_matrix'], renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            rgb_image = images[2]
            near, far = cam['near_plane'], cam['far_plane']
            depth_image = far * near / (far - (far - near) * images[3])
            return (rgb_image, depth_image)
        else:
            self.logger.error('[SimCameraManager]: Camera does not exist!')
            return None

    def get_point_cloud(self, name):
        images = self.get_image(name)
        if images:
            cloud = self._depth_to_cloud(self.active_cameras[name], images[1])
            return cloud
        else:
            self.logger.error('[SimCameraManager]: Get image fail!')
            return None
    
    def get_voxel_from_cloud(self, name):
        cloud = self.get_point_cloud(name)
        if cloud:
            voxel = self._cloud_to_voxel(cloud)
            return voxel
        else:
            self.logger.error('[SimCameraManager]: Get cloud fail!')
            return None
    
    def get_voxel_from_depth(self, name):
        images = self.get_image(name)
        if images:
            voxel = self._depth_to_voxel(self.active_cameras[name], images[1])
            return voxel
        else:
            self.logger.error('[SimCameraManager]: Get image fail!')
            return None
        
    def get_extrinsic(self, name):
        return self.active_cameras[name]['extrinsic']

    def _depth_to_cloud(self, cam, depth_img):
        o3d_depth_img = o3d.geometry.Image(np.array(depth_img * 1000, dtype=np.uint16))
        depth_trunc = cam['far_plane'] - 0.01
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth_img, cam['intrinsic'], depth_scale=1000.0,
            depth_trunc=depth_trunc, stride=1
        )
        return cloud

    def _cloud_to_voxel(self, cloud):
        voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, 0.003)
        return voxel

    def _depth_to_voxel(self, cam, depth_img):
        o3d_depth_img = o3d.geometry.Image(depth_img)
        cam_param = o3d.camera.PinholeCameraParameters()
        cam_param.extrinsic = cam['extrinsic'] 
        cam_param.intrinsic = cam['intrinsic']
        o = np.array([0,0,0], dtype=np.float64)
        c = np.array([0,0,0], dtype=np.float64)
        voxel = o3d.geometry.VoxelGrid.create_dense(origin=o, color=c, voxel_size=0.01, width=0.5, height=0.5, depth=0.5)
        voxel.carve_depth_map(o3d_depth_img, cam_param)
        return voxel

    def _compute_extrinsic(self, pos, target_pos, up_vec):
        w2c = np.identity(4)
        w2c[:3, 3] = pos
        vec_z = np.array(target_pos) - np.array(pos)
        w2c[:3, 2] = vec_z / np.linalg.norm(vec_z)
        vec_y = -1 * np.array(up_vec)
        # vec_y = vec - np.dot(vec, vec_z) * vec_z
        w2c[:3, 1] = vec_y / np.linalg.norm(vec_y)
        vec_x = np.cross(vec_y, vec_z)
        w2c[:3, 0] = vec_x / np.linalg.norm(vec_x)
        return w2c # np.linalg.inv(w2c)