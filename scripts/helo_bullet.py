#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import pybullet as pb
import time
import pybullet_data
import rospkg
import numpy as np
from packing_env import ModelManager, SimCameraManager

model_pkg_name = 'objects_model'
this_pkg_name = 'packing_env'
rospack = rospkg.RosPack()
model_path = rospack.get_path(model_pkg_name)
camera_config_path = rospack.get_path(this_pkg_name) + '/config/bullet_camera.yaml'

box_bound = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
start_bound = [[-0.15, -0.15, 0.25], [0.15, 0.15, 0.35]]


physics_client = pb.connect(pb.GUI)#or p.DIRECT for non-graphical version
pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pb.setGravity(0,0,-10)
planeId = pb.loadURDF("plane.urdf")

model_manager = ModelManager(model_path)
model_list = model_manager.sample_models_in_bound(box_bound, 0.8)
for model in model_list:
    model_manager.load_model(model, start_bound)
    # for _ in range(20):
    #     pb.stepSimulation()

camera_manager = SimCameraManager(camera_config_path)
camera_list = []
if camera_manager.create_camera('cam_1', 'Azure_Kinect', 'NFOV_UNBINNED', \
                                [0, 0.3, 0.35], [0, 0, 0.25], [0, -0.1, 0.3]):
    camera_list.append('cam_1')

if camera_manager.create_camera('cam_2', 'Azure_Kinect', 'NFOV_UNBINNED', \
                                [0, -0.3, 0.35], [0, 0, 0.25], [0, 0.1, 0.3]):
    camera_list.append('cam_2')

for cam in camera_list:
    images = camera_manager.get_image(cam)
    for i in range (240):
        # pb.stepSimulation()
        time.sleep(1./240.)

for i in range (10000):
    pb.stepSimulation()
    time.sleep(1./240.)
# cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
pb.disconnect()

