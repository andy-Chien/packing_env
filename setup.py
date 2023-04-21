from setuptools import setup
import os
from glob import glob


package_name = 'packing_env'
bullet_example = 'packing_env/examples/bullet'
sb3_example = 'packing_env/examples/stable_baselines'
mesh_utils = 'packing_env/mesh'
urdf_utils = 'packing_env/urdf'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, bullet_example, sb3_example, mesh_utils, ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'mesh'), glob('mesh/*.obj')),
    ],
    install_requires=['setuptools',
                      'numpy-quaternion',
                      'numba',
                      ],
    zip_safe=True,
    maintainer='Andy Chien',
    maintainer_email='r960411@gmail.com',
    description='The packing_env package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'module_test = packing_env.module_test:main',
        ],
    },
)
