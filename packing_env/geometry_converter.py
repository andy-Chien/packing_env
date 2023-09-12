from turtle import rt
import open3d as o3d
import numpy as np
from rclpy import logging
import time

AVAILABLE_TYPE = {'o3d_cloud', 'o3d_voxel', 'ros_cloud'}

class GeometryConverter(object):
    logger = logging.get_logger('GeometryConverter')

    def __init__(self):
        pass
    def _construct_data(self, data):
        if isinstance(data, o3d.geometry.PointCloud):
            self.o3d_cloud = data
        if isinstance(data, o3d.geometry.PointCloud):
            self.o3d_cloud = data


    def get_voxel_from_cloud(self, cloud, voxel_size=0.01):
        if isinstance(cloud, o3d.geometry.PointCloud):
            voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, voxel_size)
            return voxel
        else:
            self.logger.warn('get_voxel_from_cloud only support o3d cloud')
            return None

    def get_view_from_voxel(
            self, voxel, pixel_size, width, tar_center=[0,0,0], far_flat=1.0, axis='z', x_rev=False, y_rev=False):
        axis_map = {'x': 0, 'y':1, 'z':2, '-x': 0, '-y':1, '-z':2}
        # max_b, min_b = voxel.get_max_bound(), voxel.get_min_bound()
        # vl, img_l = np.linalg.norm(max_b - min_b), pow(2 * pow(pixel_size * width, 2), 0.5)
        # if vl > img_l:
        #     self.logger.warn('Length of object is longer than target view')
        vs, vo, tc = voxel.voxel_size, voxel.origin, np.array(tar_center, dtype=np.float32)
        voxel_index_list = np.asarray([v.grid_index for v in voxel.get_voxels()], dtype=np.float32)
        if len(voxel_index_list) < 1:
            self.logger.warn('There are no points in voxel')
            return None
        tar_size_voxel = (voxel_index_list * vs + vo - tc).astype(np.float32)
        view = np.zeros([width, width], dtype=np.float32)
        axis_indx = axis_map[axis]
        ax = -1 if '-' in axis else 1

        x_rev_factor = -1 if x_rev else 1
        y_rev_factor = -1 if y_rev else 1

        if axis == 'x':
            x_indx = lambda v: round((x_rev_factor * -1*v[2] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * -1*v[1] / pixel_size + width / 2))
        elif axis == '-x':
            x_indx = lambda v: round((x_rev_factor * -1*v[2] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * v[1] / pixel_size + width / 2))
        elif axis == 'y':
            x_indx = lambda v: round((x_rev_factor * -1*v[2] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * v[0] / pixel_size + width / 2))
        elif axis == '-y':
            x_indx = lambda v: round((x_rev_factor * -1*v[2] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * -1*v[0] / pixel_size + width / 2))
        elif axis == 'z':
            x_indx = lambda v: round((x_rev_factor * -1*v[0] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * v[1] / pixel_size + width / 2))
        elif axis == '-z':
            x_indx = lambda v: round((x_rev_factor * -1*v[0] / pixel_size + width / 2))
            y_indx = lambda v: round((y_rev_factor * -1*v[1] / pixel_size + width / 2))

        for v in tar_size_voxel:
            x_idx, y_idx = x_indx(v), y_indx(v)
            if not all([0 <= x < width for x in [x_idx, y_idx]]):
                continue
            # val_new = max(0, (min(ax * v[axis_indx], far_flat) / far_flat))
            val_new = min(ax * v[axis_indx], far_flat) / far_flat
            val = view[x_idx, y_idx]
            if val_new < val or val <= 0.00001:
                view[x_idx, y_idx] = val_new
        return view

    def get_3_views_from_voxel(self, voxel, pixel_size, width, tar_center=[0, 0, 0], far_flat=1.0, axis=['x', 'y', 'z'],
                               x_rev_list=[False, False, False], y_rev_list=[False, False, False]):
        views = []
        for ax, x_rev, y_rev in zip(axis, x_rev_list, y_rev_list):
            if ax == 'x':
                center = [tar_center[0] - far_flat / 2, tar_center[1], tar_center[2]]
            elif ax == '-x':
                center = [tar_center[0] + far_flat / 2, tar_center[1], tar_center[2]]
            elif ax == 'y':
                center = [tar_center[0], tar_center[1] - far_flat / 2, tar_center[2]]
            elif ax == '-y':
                center = [tar_center[0], tar_center[1] + far_flat / 2, tar_center[2]]
            elif ax == 'z':
                center = [tar_center[0], tar_center[1], tar_center[2] - far_flat / 2]
            elif ax == '-z':
                center = [tar_center[0], tar_center[1], tar_center[2] + far_flat / 2]

            view = self.get_view_from_voxel(voxel, pixel_size, width, center, far_flat, ax, x_rev, y_rev)
            if view is None:
                return None
            views.append(view)
        return np.asanyarray(views)

    def merge_cloud(self, cloud_list):
        merged_points = np.concatenate([np.asarray(cloud.points) for cloud in cloud_list])
        o3d_points = o3d.pybind.utility.Vector3dVector(merged_points)
        merged_cloud = o3d.geometry.PointCloud(o3d_points)
        return merged_cloud

    def geometry_rotate_euler(self, obj, euler, center=None):
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(np.asarray(np.array(euler).reshape(3, 1)))
        if center is not None:
            return obj.rotate(rot_mat, np.reshape(center, (3,1)))
        return obj.rotate(rot_mat)
    
    def geometry_translate(self, obj, pos, relative=True):
        return obj.translate(np.reshape(pos, (3,1)), relative)

    def o3d_show(self, o3d_obj):
        o3d.visualization.draw_geometries([o3d_obj], window_name='Open3D', width=1920, height=1080, \
            left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)