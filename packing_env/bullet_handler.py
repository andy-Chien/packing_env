import os
import time
import numpy as np
import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client as bc 

class BulletHandler:
    def __init__(self, gui=True):
        if gui:
            self.pb = bc.BulletClient(connection_mode=pb.GUI)
        else:
            self.pb = bc.BulletClient(connection_mode=pb.DIRECT)
        self.model_path = set()
        #TODO: current_path seems useless
        self.current_path = None
        self._setup()

    def _setup(self):
        self.pb.setGravity(0,0,-10)
        self.pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.pb.loadURDF('plane.urdf')
        if self.current_path:
            self.pb.setAdditionalSearchPath(self.current_path)

    def reset_all(self):
        self.pb.resetSimulation()
        self._setup()

    def set_model_path(self, path):
        self.model_path.add(path)
        self.current_path = path 
        self.pb.setAdditionalSearchPath(path)
        # print('[BulletHandler]: !!!!!!!!!!!!!!!!!!!path = {}'.format(path))
    
    def load_urdf(self, path, pos, quat):
        try:
            # print('[BulletHandler]: path = {}'.format(path))
            return self.pb.loadURDF(path, pos, quat)
        except:
            for p in self.model_path:
                try:
                    self.set_model_path(p)
                    # print('[BulletHandler]: Model path swich to {}'.format(p))
                    model_id = self.pb.loadURDF(path, pos, quat)
                    return model_id
                except:
                    pass
                    # print('[BulletHandler]: Try another path')
        print('[BulletHandler]: Model path not exist')
        return -1
    
    def load_stl(self, file, scale, pos, quat, mass=10):
        def load_stl():
            coll_id = self.pb.createCollisionShape(shapeType=pb.GEOM_MESH, flags=pb.GEOM_FORCE_CONCAVE_TRIMESH, \
                    meshScale=scale, fileName=file)
            vis_id = self.pb.createVisualShape(
                shapeType=pb.GEOM_MESH, meshScale=scale, fileName=file, 
                rgbaColor=[0.92, 0.66, 0.33, 0.5], specularColor=[0.4, 0.4, 0.4])
        
            model_id = self.pb.createMultiBody(baseMass=mass, 
                                          baseCollisionShapeIndex=coll_id, 
                                          baseVisualShapeIndex=vis_id, 
                                          basePosition=pos, 
                                          baseOrientation=quat)
            return model_id
        try:
            # print('[BulletHandler]: loading stl file = {}'.format(file))
            return load_stl()
        except:
            for p in self.model_path:
                try:
                    self.set_model_path(p)
                    # print('[BulletHandler]: Model path swich to {}'.format(p))
                    model_id = load_stl()
                    return model_id
                except:
                    pass
                    # print('[BulletHandler]: Try another path')
            print('[BulletHandler]: STL load failed')
            return -1
        
    def create_box(self, box_size, pos, quat, mass=1):
        half_extents = np.array(box_size) / 2
        coll_id = self.pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=half_extents)
        vis_id = self.pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=half_extents)
        model_id = self.pb.createMultiBody(mass, coll_id, vis_id, pos, quat)
        return model_id

    def set_model_pose(self, model_id, pos, quat):
        self.pb.resetBasePositionAndOrientation(model_id, pos, quat)

    def get_model_pose(self, model_id):
        return self.pb.getBasePositionAndOrientation(model_id)
    
    def remove_model(self, model_id):
        return self.pb.removeBody(model_id)

    def step_simulation(self, setp_cnt, realtime=False):
        for i in range (setp_cnt):
            self.pb.stepSimulation()
            if realtime:
                time.sleep(1./240.)