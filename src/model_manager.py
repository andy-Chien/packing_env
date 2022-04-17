import os
import numpy as np
import pybullet as pb

class ModelManager():
    def __init__(self, model_path):
        self.urdf_path = model_path + '/urdf/'
        pb.setAdditionalSearchPath(self.urdf_path)
        self.urdf_list = [_ for _ in os.listdir(self.urdf_path) if _.endswith(r".urdf")]
        

    def sample_models_in_bound(self, bound):
        self.sampled_models_list = []
        

    def load_sampled_models(self, start_bound):
        for modes in self.sampled_models_list:
            start_pos = [np.random.uniform(low=start_bound[0][i], high=start_bound[1][i]) for i in range(3)]
            start_quat = pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))
            self.loaded_list.append(pb.loadURDF(modes, start_pos, start_quat))

