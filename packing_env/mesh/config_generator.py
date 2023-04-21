# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
from ast import arg
from importlib import import_module
from operator import imod
from time import time, sleep
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
import threading
import queue
import random
from mesh_helper import MeshHelper

class ConfigGenerateThread(threading.Thread):
    def __init__(self, q, lock, count, pkg_name, source_path, target_path, target_size, target_num):
        super().__init__()
        self.queue = q
        self.lock = lock
        self.count = count
        self.pkg_name = pkg_name
        self.package_path = get_package_share_directory(pkg_name)
        self.mesh_source_dir = self.package_path + source_path
        self.config_target_dir = self.package_path + target_path
        self.target_size = target_size
        self.target_num = target_num
        self.failed = []

    def run(self):
        while self.queue.qsize() > 0:
            [number, mesh_file] = self.queue.get()
            self.mesh_config_generate(number, mesh_file)

        for obj in self.failed:
            print(obj)

    def mesh_config_generate(self, number, mesh_file):
        file_path = self.mesh_source_dir + mesh_file
        mesh = MeshHelper(file_path)
        convex_hull_mesh = MeshHelper(mesh.get_convex_hull())

        print('Get {}\'s valume, is number {}'.format(os.path.splitext(mesh_file)[0], number))
        if not mesh.mesh.is_watertight() or not convex_hull_mesh.mesh.is_watertight():
            self.failed.append(os.path.splitext(mesh_file)[0])
            return
        origin_volume = mesh.get_volume()
        convex_volume = convex_hull_mesh.get_volume()
        max_length = mesh.get_max_length()
        urdf_file = '/urdf/' + os.path.splitext(mesh_file)[0] + '.urdf'
        config = ''.join([
                'object_model:',
                '\n    %s:' % os.path.splitext(mesh_file)[0],
                '\n        origin_volume: %s' % origin_volume,
                '\n        convex_volume: %s' % convex_volume,
                '\n        max_length: %s' % max_length,
                '\n        urdf_file: %s' % urdf_file,
                '\n'
                ])
        file_name = self.config_target_dir + os.path.splitext(mesh_file)[0] + '.yaml'
        with open(file_name, "w+") as f:
            f.write(config)



THREADS = 6
SOURCE_PATH = '/mesh/'
TARGET_PATH = '/config/'
PKG_NAME = 'objects_model'

TARGET_SIZE = [0.06, 0.24]
TARGET_NUM = 3

if __name__ == '__main__':
    package_path = get_package_share_directory(PKG_NAME)
    mesh_path = package_path + SOURCE_PATH
    mesh_write_path = package_path + TARGET_PATH
    
    mesh_files = [_ for _ in os.listdir(mesh_path) if _.endswith(r".obj")]
    count = [0, 0]
    lock = threading.Lock()
    threads = []
    mesh_queue = queue.Queue()
    for no, file in enumerate(mesh_files):
        mesh_queue.put([no, file])

    for i in range(THREADS):
        threads.append(ConfigGenerateThread(mesh_queue, lock, count, PKG_NAME, SOURCE_PATH, TARGET_PATH, TARGET_SIZE, TARGET_NUM))
        threads[-1].start()

    for thread in threads:
        thread.join()
    # name, xyz, rpy, mass, mesh_file, color_name, rgba