import os
import numpy as np
import pybullet as pb
from random import choice as random_choice
from yaml import safe_load as yaml_load
from rospy import logerr

class SimCameraManager():
    def __init__(self, config_path):
        self.camera_data = self.__config_loader(config_path)
        self.active_cameras = dict()
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
            self.active_cameras[name] = cam
            return True
        else:
            logerr('[create_camera]: Camera model or type is wrong!')
            print('{} in {} is {}'.format(model, self.camera_data.keys(), model in self.camera_data.keys()))
            return False

    def get_image(self, name):
        if name in self.active_cameras.keys():
            cam = self.active_cameras[name]
            images = pb.getCameraImage(cam['width'], cam['height'], cam['view_matrix'], \
                                    cam['projection_matrix'], renderer=pb.ER_BULLET_HARDWARE_OPENGL)
            rgb_image = images[2]
            depth_image = images[3]
            return (rgb_image, depth_image)
        else:
            logerr('[get_image]: Camera does not exist!')
            return None