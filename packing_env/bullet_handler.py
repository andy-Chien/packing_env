import os
import time
import numpy as np
import pybullet as pb
import pybullet_data

class BulletHandler:
    def __init__(self):
        self.physics_client = pb.connect(pb.GUI)
        self.model_path = set()
        self.current_path = None
        self._setup()

    def _setup(self):
        pb.setGravity(0,0,-10)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF('plane.urdf')
        if self.current_path:
            pb.setAdditionalSearchPath(self.current_path)

    def reset_all(self):
        pb.resetSimulation()
        self._setup()

    def set_model_path(self, path):
        self.model_path.add(path)
        self.current_path = path
        pb.setAdditionalSearchPath(path)
        print('[BulletHandler]: !!!!!!!!!!!!!!!!!!!path = {}'.format(path))
    
    def load_urdf(self, path, pos, quat):
        try:
            print('[BulletHandler]: path = {}'.format(path))
            return pb.loadURDF(path, pos, quat)
        except:
            for p in self.model_path:
                try:
                    self.set_model_path(p)
                    model_id = pb.loadURDF(path, pos, quat)
                    print('[BulletHandler]: Model path swich to {}'.format(p))
                    return model_id
                except:
                    print('[BulletHandler]: Try another path')
        print('[BulletHandler]: Model path not exist')
        return -1

    def set_model_pose(self, model_id, pos, quat):
        pb.resetBasePositionAndOrientation(model_id, pos, quat)

    def step_simulation(self, setp_cnt, realtime=False):
        for i in range (setp_cnt):
            pb.stepSimulation()
            if realtime:
                time.sleep(1./240.)