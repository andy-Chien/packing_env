# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
from ast import arg
from importlib import import_module
from time import time, sleep
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
import threading
import queue
from mesh_helper import MeshHelper
import tty, sys, termios


THREADS = 6
SOURCE_PATH = '/mesh/downsampling/' # berkeley_dateset
TARGET_PATH = '/mesh/tmp/'
PKG_NAME = 'objects_model'

if __name__ == '__main__':
    filedescriptors = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    package_path = get_package_share_directory(PKG_NAME)
    mesh_source_dir = package_path + SOURCE_PATH
    mesh_target_dir = package_path + TARGET_PATH
    
    mesh_files = [_ for _ in os.listdir(mesh_source_dir) if _.endswith(r".obj")]
    count = [0, 0]
    lock = threading.Lock()
    threads = []
    mesh_queue = queue.Queue()
    for no, mesh_file in enumerate(mesh_files):
        print(mesh_file)
        file_path = mesh_source_dir + mesh_file
        mesh = MeshHelper(file_path)
        stop_threads = False
        thread = threading.Thread(target=mesh.show, args=(lambda : stop_threads, ))
        thread.start()
        while True:
            x=sys.stdin.read(1)[0]
            if x == "f":
                mesh.save_file(mesh_target_dir + mesh_file)
                print("failure mesh")
                break
            elif  x == "v":
                print("valid mesh")
                break
        stop_threads = True
        thread.join()
        
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN,filedescriptors)
    # name, xyz, rpy, mass, mesh_file, color_name, rgba