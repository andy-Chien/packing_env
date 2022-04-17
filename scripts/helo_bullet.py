#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pybullet as pb
import time
import pybullet_data
import rospkg

model_pkg_name = 'objects_model'
rospack = rospkg.RosPack()
model_path = rospack.get_path(model_pkg_name)

physicsClient = pb.connect(pb.GUI)#or p.DIRECT for non-graphical version
pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pb.setGravity(0,0,-10)
planeId = pb.loadURDF("plane.urdf")
pb.setAdditionalSearchPath(model_path + '/urdf/')
startPos = [0,0,1]
startOrientation = pb.getQuaternionFromEuler([0,0,0])
boxId = pb.loadURDF("Y8439_Tracbota.urdf",startPos, startOrientation)
pb.loadURDF("champion_copper_plus_spark_plug.urdf",startPos, startOrientation)
pb.loadURDF("crayola_64_ct.urdf",startPos, startOrientation)
pb.loadURDF("dove_beauty_bar.urdf",startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    pb.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = pb.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
pb.disconnect()

