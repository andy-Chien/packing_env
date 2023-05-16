import numpy as np
import open3d as o3d
import time
from numba import jit

class MeshHelper:
    def __init__(self, mesh):
        if isinstance(mesh, str):
            self.mesh = o3d.io.read_triangle_mesh(mesh)
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            self.mesh = mesh
        self.origin, self.volume, self.mass = None, None, None
    
    @jit
    def __calculate_mass_center_and_volume(self):
        volume = [0.]
        centroid = np.zeros(3)
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        for triangle in triangles:
            a = vertices[triangle[0]]
            b = vertices[triangle[1]]
            c = vertices[triangle[2]]
            tetrahedron_volume = np.dot(a, np.cross(b, c)) / 6.0 # signed volume of tetrahedron

            centroid += tetrahedron_volume * (a + b + c) / 4.
            volume[0] += tetrahedron_volume
        return centroid / volume[0], volume[0]

    def calculate_mass_center_and_volume(self):
        if self.mesh.is_orientable() and self.mesh.is_watertight():
            vv = self.mesh.get_volume()
            self.origin, self.volume = self.__calculate_mass_center_and_volume()
            # print("vv = {}, volume = {}".format(vv, self.volume))
            self.mass = self.volume * 1000 # m3 to kg
            return True
        else:
            if not self.mesh.is_orientable():
                print('not self.mesh.is_orientable()')
            if not self.mesh.is_watertight():
                print('not self.mesh.is_watertight()')
            return False

    def check_surface(self):
        triangles = np.asarray(self.mesh.triangles)
        vertex_normals = np.asarray(self.mesh.vertex_normals)
        triangle_normals = np.asarray(self.mesh.triangle_normals)
        for triangle, triangle_normal in zip(triangles, triangle_normals):
            a = vertex_normals[triangle[0]]
            b = vertex_normals[triangle[1]]
            c = vertex_normals[triangle[2]]
            if np.dot(triangle_normal, a + b + c) < 0:
                triangle[0], triangle[2] = triangle[2], triangle[0]

    def get_vertices(self):
        return np.asarray(self.mesh.vertices)

    def get_triangles(self):
        return np.asarray(self.mesh.triangles)

    def compute_normals(self):
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()
        print('size of self.mesh.triangle_normals is {}, triangles is {}'.format(len(self.mesh.triangle_normals), len(self.mesh.triangles)))
        print('size of self.mesh.vertex_normals is {}, vertices is {}'.format(len(self.mesh.vertex_normals), len(self.mesh.vertices)))
        print(np.asarray(self.mesh.vertex_normals))
        print(np.asarray(self.mesh.triangle_normals))

    def down_sampling(self, target_number_of_triangles=3000):
        self.mesh = self.mesh.simplify_quadric_decimation(target_number_of_triangles)

    @jit
    def inverted_surface(self):
        triangles = np.asarray(self.mesh.triangles)
        for triangle in triangles:
            triangle[0], triangle[2] = triangle[2], triangle[0]

    def save_file(self, file_name):
        o3d.io.write_triangle_mesh(file_name, self.mesh)
    
    @staticmethod
    def save_mesh_file(mesh, file_name):
        o3d.io.write_triangle_mesh(file_name, mesh)

    def get_convex_hull(self):
        return o3d.geometry.TriangleMesh(self.mesh).compute_convex_hull()[0]

    def get_volume(self):
        return self.mesh.get_volume()

    def show(self, stop=False):
        # o3d.visualization.draw(self.mesh, raw_mode=True, non_blocking_and_return_uid=True)
        # o3d.visualization.draw_geometries([self.mesh])
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480)
        vis.add_geometry(self.mesh)
        while True:
            vis.poll_events()
            vis.update_renderer()
            if stop():
                break
        vis.destroy_window()

    def get_max_length(self):
        bounding_box = self.mesh.get_minimal_oriented_bounding_box()
        center = bounding_box.get_center()
        max_bound = bounding_box.get_max_bound()
        return 2 * np.linalg.norm((max_bound - center))

    def scale_mesh(self, scale):
        center = self.mesh.get_center()
        return o3d.geometry.TriangleMesh(self.mesh).scale(scale, center)

    def move_origin_to_center(self):
        bounding_box = self.mesh.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        self.mesh = self.mesh.translate(center, False)

    def move_origin_to_center_and_rotate(self):
        self.move_origin_to_center()
        bounding_box = self.mesh.get_minimal_oriented_bounding_box()
        self.mesh.rotate(np.transpose(bounding_box.R))

    def compute_bounding_box_inertia(self):
        if not self.mass:
            return None
        bounding_box = self.mesh.get_axis_aligned_bounding_box()
        [a, b, c] = bounding_box.get_extent()
        m =  self.mass
        xx = m * (b*b + c*c) / 12
        yy = m * (a*a + c*c) / 12
        zz = m * (a*a + b*b) / 12
        return [xx, yy, zz]




if __name__ == '__main__':
    print("Testing IO for meshes ...")
    file = '/home/andy/Downloads/dex_net_meshes/meshes/dexnet_1.0_raw_meshes/meshes/amazon_picking_challenge/crayola_64_ct.obj'
    start = time.time()
    mesh = MeshHelper(file)
    end = time.time()
    print('origin = {}, volume = {}, time = {}'.format(mesh.origin, mesh.volume, end - start))
    o3d.visualization.draw(mesh.mesh, raw_mode=True)