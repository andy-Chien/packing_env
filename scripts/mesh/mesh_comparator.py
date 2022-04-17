# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
from ast import arg
from importlib import import_module
import re
from time import time, sleep
import rospkg
import os
import numpy as np
import threading
import queue
from mesh_helper import MeshHelper
import tty, sys, termios


THREADS = 1
SOURCE_PATH = '/mesh/downsampling/' # berkeley_dateset
TARGET_PATH = '/mesh/tmp/'
PKG_NAME = 'objects_model'



if __name__ == '__main__':
    filedescriptors = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    rospack = rospkg.RosPack()
    package_path = rospack.get_path(PKG_NAME)
    mesh_source_dir = package_path + SOURCE_PATH
    mesh_target_dir = package_path + TARGET_PATH
    
    mesh_files = [_ for _ in os.listdir(mesh_source_dir) if _.endswith(r".obj")]
    count = [0, 0]
    lock = threading.Lock()
    threads = []
    removed_files = [_ for _ in os.listdir(mesh_target_dir) if _.endswith(r".obj")]
    mesh_queue = queue.Queue()
    for no, mesh_file_1 in enumerate(mesh_files):
        print(no)
        if no < 1184:
            continue
        cc = False
        for removed_file in removed_files:
            if mesh_file_1 == removed_file:
                cc = True
        if cc:
            continue
        file_path_1 = mesh_source_dir + mesh_file_1
        mesh_1 = MeshHelper(file_path_1)
        vs_1 = mesh_1.get_vertices()
        for mesh_file_2 in mesh_files[no+1:]:
            cc = False
            if mesh_file_2 == mesh_file_1:
                continue
            for removed_file in removed_files:
                if mesh_file_2 == removed_file:
                    cc = True
            if cc:
                continue
            file_path_2 = mesh_source_dir + mesh_file_2
            mesh_2 = MeshHelper(file_path_2)
            vs_2 = mesh_2.get_vertices()
            removed = True
            if vs_1.size == vs_2.size:
                length = len(vs_1) if len(vs_1) < 20 else 20
                for i in range(length):
                    if np.linalg.norm(vs_1[i] - vs_2[i]) > 0.0001:
                        removed = False
                        break
            else:
                removed = False
            if removed:
                # mesh_2.save_file(mesh_target_dir + mesh_file_2)

                with open(mesh_target_dir + mesh_file_2, "w+") as f:
                    f.write('')
                removed_files.append(mesh_file_2)

        
