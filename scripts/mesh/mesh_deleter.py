# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
from ast import arg
from importlib import import_module
from time import time, sleep
import rospkg
import os
import numpy as np
import threading
import queue


THREADS = 1
SOURCE_PATH = '/mesh/some_failure/' # berkeley_dateset
TARGET_PATH = '/mesh/downsampling/'
PKG_NAME = 'objects_model'

if __name__ == '__main__':
    rospack = rospkg.RosPack()
    package_path = rospack.get_path(PKG_NAME)
    mesh_source_dir = package_path + SOURCE_PATH
    mesh_target_dir = package_path + TARGET_PATH
    
    mesh_files = [_ for _ in os.listdir(mesh_source_dir) if _.endswith(r".obj")]
    
    for no, mesh_file in enumerate(mesh_files):
       file = mesh_target_dir + mesh_file
       if os.path.exists(file):
           os.remove(file)
           print('{} has been removed!'.format(mesh_file))
           
        
    # name, xyz, rpy, mass, mesh_file, color_name, rgba