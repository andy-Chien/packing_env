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

    def merge_cloud(self, cloud_list):
        merged_points = np.concatenate([np.asarray(cloud.points) for cloud in cloud_list])
        o3d_points = o3d.cuda.pybind.utility.Vector3dVector(merged_points)
        merged_cloud = o3d.geometry.PointCloud(o3d_points)
        return merged_cloud

    def o3d_show(self, o3d_obj):
        o3d.visualization.draw_geometries([o3d_obj], window_name='Open3D', width=1920, height=1080, \
            left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)