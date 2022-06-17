#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import pybullet as pb
import time
import pybullet_data
import rospkg
import numpy as np
import open3d as o3d
from packing_env import BulletHandler
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter

model_pkg_name = 'objects_model'
this_pkg_name = 'packing_env'
rospack = rospkg.RosPack()
model_path = rospack.get_path(model_pkg_name)
camera_config_path = rospack.get_path(this_pkg_name) + '/config/bullet_camera.yaml'

box_bound = [[0.1, 0.1, 0.1], [0.17, 0.17, 0.17]]
start_bound = [[-0.05, -0.05, 1.25], [0.05, 0.05, 1.35]]

bh = BulletHandler()
gc = GeometryConverter()

model_manager = ModelManager(model_path, bh)
camera_manager = SimCameraManager(camera_config_path)
camera_list = []
if camera_manager.create_camera('cam_1', 'Bullet_Camera', 'TEST', \
                                [0, 0.3, 1.35], [0, 0, 1.25], [0, -0.1, 0.3]):
    camera_list.append('cam_1')

if camera_manager.create_camera('cam_2', 'Bullet_Camera', 'TEST', \
                                [0, -0.3, 1.35], [0, 0, 1.25], [0, 0.1, 0.3]):
    camera_list.append('cam_2')

for eps in range(10):
    model_list = model_manager.sample_models_in_bound(box_bound, 0.8)
    for model in model_list:
        pos = model_manager.random_pos(start_bound)
        quat = model_manager.random_quat()
        model_manager.load_model(model, pos, quat)
        cloud_list = []
        for cam in camera_list:
            # t0 = time.time()
            cloud_list.append(camera_manager.get_point_cloud(cam))
            # t1 = time.time()
            # time.sleep(0.3)
        merged_cloud = gc.merge_cloud(cloud_list)
        voxel = gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
        gc.o3d_show(voxel)
        model_manager.set_model_pos(model, [0,0,0.5])
        bh.step_simulation(100, realtime=False)

    bh.reset_all()

bh.step_simulation(10000, realtime=True)