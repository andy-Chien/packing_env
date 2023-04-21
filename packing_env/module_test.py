#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import pybullet as pb
import time
# import pybullet_data
import numpy as np
import open3d as o3d
from ament_index_python.packages import get_package_share_directory
from packing_env import ModelManager, SimCameraManager, BulletHandler, GeometryConverter


def main():
    model_pkg_name = 'objects_model'
    this_pkg_name = 'packing_env'
    model_path = get_package_share_directory(model_pkg_name)
    camera_config_path = get_package_share_directory(this_pkg_name) + '/config/bullet_camera.yaml'

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
            t0 = time.time()
            pos = model_manager.random_pos(start_bound)
            quat = model_manager.random_quat()
            model_manager.load_model(model, pos, quat)
            cloud_list = []
            for cam in camera_list:
                cloud_list.append(camera_manager.get_point_cloud(cam))
                # time.sleep(0.3)
            merged_cloud = gc.merge_cloud(cloud_list)
            voxel = gc.get_voxel_from_cloud(merged_cloud, voxel_size=0.002)
            # gc.o3d_show(voxel)
            pixel_size = 0.002
            width = 128
            tar_center = [0, 0, 1.3]
            views = gc.get_3_views_from_voxel(voxel, pixel_size, width, tar_center)
            # gc.o3d_show(o3d.geometry.Image(views[0]))
            # gc.o3d_show(o3d.geometry.Image(views[1]))
            # gc.o3d_show(o3d.geometry.Image(views[2]))
            model_manager.set_model_pos(model, [0,0,0.5])
            bh.step_simulation(240, realtime=False)
            t1 = time.time()
            print('time = {}'.format(t1-t0))

        bh.reset_all()

    bh.step_simulation(10000, realtime=True)

if __name__ == '__main__':
    main()