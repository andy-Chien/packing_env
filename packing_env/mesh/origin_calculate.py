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
from mesh.mesh_helper import MeshHelper


class OriginCalculateThread(threading.Thread):
    def __init__(self, q, lock, count, pkg_name, source_path, target_path, target_size, target_num):
        super().__init__()
        self.queue = q
        self.lock = lock
        self.count = count
        self.pkg_name = pkg_name
        self.package_path = get_package_share_directory(pkg_name)
        self.mesh_source_dir = self.package_path + source_path
        self.mesh_target_dir = self.package_path + target_path
        self.mesh_dir_for_urdf = 'package://' + self.pkg_name + target_path
        self.target_size = target_size
        self.target_num = target_num

    def compute_random_range(self):
        size_range = (self.target_size[1] - self.target_size[0])
        if size_range < 0:
            size_range *= -1
            self.target_size[0], self.target_size[1] = self.target_size[1], self.target_size[0]
        self.random_ranges = [size_range * i / self.target_num + self.target_size[0] for i in range(self.target_num + 1)]

    def run(self):
        while self.queue.qsize() > 0:
            [number, mesh_file] = self.queue.get()
            self.mesh_urdf_generate(number, mesh_file)

    def mesh_urdf_generate(self, number, mesh_file):
        file_path = self.mesh_source_dir + mesh_file
        mesh = MeshHelper(file_path)
        mesh.move_origin_to_center()
        # mesh.down_sampling()
        # mesh_length = mesh.get_max_length()
        # self.compute_random_range()
        
        # for scale_indx in range(self.target_num):
        #     target_length = np.random.uniform(low=self.random_ranges[scale_indx], high=self.random_ranges[scale_indx+1])
        #     scaled_mesh = MeshHelper(mesh.scale_mesh(target_length / mesh_length))
        #     if abs(target_length - scaled_mesh.get_max_length()) > 0.002:
        #         print('!!!!!!!!!!!!!! Mesh scale fault !!!!!!!!!!!!!!!')
        #         return
        #     scaled_mesh_file = os.path.splitext(mesh_file)[0] + '_' + str(scale_indx) + os.path.splitext(mesh_file)[1]
        #     self.urdf_generate(scaled_mesh, number, scaled_mesh_file)
        self.urdf_generate(mesh, number, mesh_file)

    def urdf_generate(self, mesh, number, mesh_file):
        if mesh.calculate_mass_center_and_volume():
            # if mesh.volume < 0:
            #     mesh.inverted_surface()
            #     if not mesh.calculate_mass_center_and_volume():
            #         print('After inverted surface, mesh file {} is not watertight or not orientable, {} files are failure'.format(mesh_file[:10], count[1]))
            #         return

            mesh.save_file(self.mesh_target_dir + mesh_file)

            rand_array = np.random.rand(4)
            rand_array[3] = 0.9 + rand_array[3] / 10
            rgba = str(rand_array).replace('[', '').replace(']', '')
            origin = str(mesh.origin).replace('[', '').replace(']', '')
            robot_name = os.path.splitext(mesh_file)[0]

            self.lock.acquire()
            count[0]+=1
            print('Successfully saved ================ {}th ================ file: {}, -------- {} ------- files are saved'.format(number, robot_name[:10], count[0]))
            self.lock.release()
        else:
            self.lock.acquire()
            count[1] += 1
            print('Mesh file {} is not watertight or not orientable, {} files are failure'.format(mesh_file[:10], count[1]))
            self.lock.release()


THREADS = 6
SOURCE_PATH = '/mesh/downsampling/' # berkeley_dateset
TARGET_PATH = '/mesh/scaled/'
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
        threads.append(OriginCalculateThread(mesh_queue, lock, count, PKG_NAME, SOURCE_PATH, TARGET_PATH, TARGET_SIZE, TARGET_NUM))
        threads[-1].start()

    for thread in threads:
        thread.join()
    # name, xyz, rpy, mass, mesh_file, color_name, rgba