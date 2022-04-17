# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
from ast import arg
from importlib import import_module
from operator import imod
from time import time, sleep
import rospkg
import os
import numpy as np
import threading
import queue
import random
from mesh.mesh_helper import MeshHelper

class File:
    """Create XML file for ROS urdf model.
    Args:
        'filename': name of file to be saved to. Should end with .urdf.xacro.
        'elements': individual elements, e.g. link, xacro property, joint
    Raises:
        No Error yet
    """
    def __init__(self, *elements, name, filename='example.urdf.xacro'):
        self.elements = elements
        self.xml = []
        self.filename = filename
        self.name = name

    def __complete_xml(self):
        self.elements = list(self.elements)
        self.elements.insert(0, '<?xml version="1.0"?>')
        self.elements.insert(1, '\n<robot name="' + self.name + '">')
        self.elements.append('</robot>')

    def set_elements(self, *elements):
        self.elements = elements

    def save(self):
        self.__complete_xml()
        with open(self.filename, "w+") as f:
            for item in self.elements:
                f.write(item if type(item) is str else item.element)

    class Property:
        def __init__(self, name, value):
            self.name = name
            self.value = value
            self.element = '<xacro:property name="%s" value="%s"/>\n' % (self.name, self.value)

    class Link:
        """Create 'link' urdf element.
        Args:
            rgba: red - green - blue - alpha.
            size: size of object.
        Raises:
            No Error yet
        """
        def __init__(self, name, xyz, rpy, mass, mesh_file, color_name, rgba):
            self.name = name
            self.xyz = xyz
            self.rpy = rpy
            self.mass = mass
            self.mesh_file = mesh_file
            self.color_name = color_name
            self.rgba = rgba
            self.element = self.xmlify()
            
        # Formatting options, see http://tinyurl.com/y9vohsxz

        # Manual xml string, as backup: '#'\n\n<link name="test1">\n\t<visual>\n\t\t<geometry>\n\t\t\t<box size="0.6 0.3 0.2"/>\n\t\t</geometry>
        # \n\t\t<material name="green">\n\t\t\t<color rgba="0.1 0.3 0.2"/>\n\t\t</material>\n\t</visual>\n</link>''

        # ISSUE: String after closing brackets of material need to be indented one space more. wtf
        def xmlify(self):
            xml = '\n'.join([
                    '\n  <link name="%s">' % self.name,
                    '    <contact>',
                    '      <lateral_friction value="1.0"/>',
                    '      <rolling_friction value="0.0"/>',
                    '      <inertia_scaling value="3.0"/>',
                    '      <contact_cfm value="0.0"/>',
                    '      <contact_erp value="1.0"/>',
                    '    </contact>',
                    '    <inertial>',
                    '      <origin xyz="%s" rpy="%s"/>' % (self.xyz, self.rpy),
                    '      <mass value="%s"/>' % self.mass,
                    '      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="0"/>',
                    '    </inertial>',
                    '    <visual>',
                    '      <geometry>',
                    '        <mesh filename="%s"/>' % (self.mesh_file),
                    '      </geometry>',
                    '      <material name="%s">' % self.color_name,
                    '         <color rgba="%s"/>' % self.rgba,
                    '      </material>',
                    '    </visual>',
                    '    <collision>',
                    '      <geometry>',
                    '        <mesh filename="%s"/>' % (self.mesh_file),
                    '      </geometry>',
                    '    </collision>',
                    '  </link>\n'
                ])

            return xml

    class Joint:
        def __init__(self, name, typ, parent, child, xyz, rpy='0 0 0'):
            self.name = name
            self.type = typ
            self.parent = parent
            self.child = child
            self.xyz = xyz
            self.rpy = rpy
            self.element = self.xmlify()

        def xmlify(self):
            xml = '\n'.join([
                '\n  <joint name="%s">' % self.name,
                '    <parent link="%s"/>' % self.parent,
                '    <child link="%s"/>' % self.child,
                '    <origin xyz="%s" rpy="%s"/>' % (self.xyz, self.rpy),
                '  </joint>\n'
                ])

            return xml

class URDFGenerateThread(threading.Thread):
    def __init__(self, q, lock, count, pkg_name, source_path, target_path, target_size, target_num):
        super().__init__()
        self.queue = q
        self.lock = lock
        self.count = count
        self.pkg_name = pkg_name
        self.package_path = rospack.get_path(pkg_name)
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
        # mesh.down_sampling()
        mesh_length = mesh.get_max_length()
        self.compute_random_range()
        
        for scale_indx in range(self.target_num):
            target_length = np.random.uniform(low=self.random_ranges[scale_indx], high=self.random_ranges[scale_indx+1])
            scaled_mesh = MeshHelper(mesh.scale_mesh(target_length / mesh_length))
            if abs(target_length - scaled_mesh.get_max_length()) > 0.002:
                print('!!!!!!!!!!!!!! Mesh scale fault !!!!!!!!!!!!!!!')
                return
            scaled_mesh_file = os.path.splitext(mesh_file)[0] + '_' + str(scale_indx) + os.path.splitext(mesh_file)[1]
            self.urdf_generate(scaled_mesh, number, scaled_mesh_file)

    def urdf_generate(self, mesh, number, mesh_file):
        mesh_path_for_urdf = self.mesh_dir_for_urdf + mesh_file
        urdf_path = self.package_path + '/urdf/'
        if mesh.calculate_mass_center_and_volume():
            if mesh.volume < 0:
                mesh.inverted_surface()
                if not mesh.calculate_mass_center_and_volume():
                    print('After inverted surface, mesh file {} is not watertight or not orientable, {} files are failure'.format(mesh_file[:10], count[1]))
                    return

            mesh.save_file(self.mesh_target_dir + mesh_file)

            rand_array = np.random.rand(4)
            rand_array[3] = 0.9 + rand_array[3] / 10
            rgba = str(rand_array).replace('[', '').replace(']', '')
            origin = str(mesh.origin).replace('[', '').replace(']', '')
            robot_name = os.path.splitext(mesh_file)[0]

            uref_file = File(
                        File.Link('base_link', origin, '0 0 0', str(mesh.mass), mesh_path_for_urdf, 'rand', rgba),
                        name=robot_name,
                        filename = urdf_path + robot_name + '.urdf'
                )
            uref_file.save()
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
    rospack = rospkg.RosPack()
    package_path = rospack.get_path(PKG_NAME)
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
        threads.append(URDFGenerateThread(mesh_queue, lock, count, PKG_NAME, SOURCE_PATH, TARGET_PATH, TARGET_SIZE, TARGET_NUM))
        threads[-1].start()

    for thread in threads:
        thread.join()
    # name, xyz, rpy, mass, mesh_file, color_name, rgba