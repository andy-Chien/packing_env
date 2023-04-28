from turtle import rt
import open3d as o3d
import numpy as np

AVAILABLE_TYPE = {'o3d_cloud', 'o3d_voxel', 'ros_cloud'}

class GeometryConverter:
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
            print('[GeometryConverter]: get_voxel_from_cloud only support o3d cloud')
            return None

    def get_view_from_voxel(self, voxel, pixel_size, width, tar_center=[0,0,0], axis='z'):
        axis_map = {'x': 0, 'y':1, 'z':2, '-x': 0, '-y':1, '-z':2}
        max_b, min_b = voxel.get_max_bound(), voxel.get_min_bound()
        vl, img_l = np.linalg.norm(max_b - min_b), pow(2 * pow(pixel_size * width, 2), 0.5)
        if vl > img_l:
            print('[GeometryConverter]: Length of object is longer than target view')
            return
        vs, vo, tc = voxel.voxel_size, voxel.origin, np.array(tar_center, dtype=np.float32)
        trans = -1 * tc + vo  + vs / 2
        voxel_index_list = np.asarray([v.grid_index for v in voxel.get_voxels()], dtype=np.float32)
        if len(voxel_index_list) < 1:
            print('[GeometryConverter]: There are no points in voxel')
            return
        tar_size_voxel = ((voxel_index_list * vs + trans) / pixel_size).astype(np.int32)
        view = np.zeros([width, width], dtype=np.uint8)
        axis_indx = axis_map[axis]
        ax = -1 if '-' in axis else 1
        for v in tar_size_voxel:
            indx = [int(x + width / 2) for x in np.delete(v, axis_indx)]
            if not all([0 <= x < width for x in indx]):
                continue
            val_new = max(1, min(255, ax * (ax * width / 2 - v[axis_indx])))
            val = view[indx[1], indx[0]]
            if val_new < val or val == 0:
                view[indx[1], indx[0]] = val_new
        return view

    def get_3_views_from_voxel(self, voxel, pixel_size, width, tar_center=[0, 0, 0], axis=[1, 1, 1]):
        max_b, min_b = voxel.get_max_bound(), voxel.get_min_bound()
        vl, img_l = np.linalg.norm(max_b - min_b), pow(2 * pow(pixel_size * width, 2), 0.5)
        if vl > img_l:
            print('[GeometryConverter]: Length of object is longer than target view')
            return
        vs, vo, tc = voxel.voxel_size, voxel.origin, np.array(tar_center, dtype=np.float32)
        trans = -1 * tc + vo  + vs / 2
        voxel_index_list = np.asarray([v.grid_index for v in voxel.get_voxels()], dtype=np.float32)
        if len(voxel_index_list) < 1:
            print('[GeometryConverter]: There are no points in voxel')
            return
        tar_size_voxel = ((voxel_index_list * vs + trans) / pixel_size).astype(np.int32)
        view = []
        for i, ax in enumerate([1 if x > 0 else -1 for x in axis]):
            img = np.zeros([width, width], dtype=np.uint8)
            for v in tar_size_voxel:
                indx = [int(x + width / 2) for x in np.delete(v, i)]
                if not all([0 <= x < width for x in indx]):
                    continue
                val_new = max(1, min(255, ax * (ax * width / 2 - v[i])))
                val = img[indx[1], indx[0]]
                if val_new < val or val == 0:
                    img[indx[1], indx[0]] = val_new
            view.append(img)
        return view

    def merge_cloud(self, cloud_list):
        merged_points = np.concatenate([np.asarray(cloud.points) for cloud in cloud_list])
        o3d_points = o3d.cuda.pybind.utility.Vector3dVector(merged_points)
        merged_cloud = o3d.geometry.PointCloud(o3d_points)
        return merged_cloud

    def o3d_show(self, o3d_obj):
        o3d.visualization.draw_geometries([o3d_obj], window_name='Open3D', width=1920, height=1080, \
            left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)