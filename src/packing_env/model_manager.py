import os
import numpy as np
import pybullet as pb
from random import choice as random_choice
from yaml import safe_load as yaml_load


class ModelManager():
    def __init__(self, model_path):
        self.model_path = model_path
        pb.setAdditionalSearchPath(model_path + '/urdf/')
        self.model_list = [os.path.splitext(file)[0] for file in os.listdir(model_path + '/config/') if file.endswith(r".yaml")]
        self.models = dict()
        for model_name in self.model_list:
            model = self.__model_config_loader(model_path + '/config/' + model_name + '.yaml')
            self.models = {**self.models, **model['object_model']}
        self.loaded_list = []
        
    def __model_config_loader(self, path):
        with open(path, 'r') as stream:
            data = yaml_load(stream)
        return data


    def sample_models_in_bound(self, bound, fill_rate):
        volume_sum = 0
        self.sampled_models_list = []
        min_bound = np.array(bound[0])
        max_bound = np.array(bound[1])
        bound_size = np.absolute(max_bound - min_bound)
        max_length = np.amax(bound_size)
        bound_volume = np.prod(bound_size)
        print('bound_volume = {}'.format(bound_volume))
        while volume_sum < bound_volume * fill_rate:
            model = random_choice(self.model_list)
            if self.models[model]['max_length'] < max_length \
                and volume_sum + self.models[model]['convex_volume'] < bound_volume:
                volume_sum += self.models[model]['convex_volume']
                self.sampled_models_list.append(model)
        return self.sampled_models_list
        
    def load_sampled_models(self, start_bound):
        for model in self.sampled_models_list:
            start_pos = np.random.uniform(low=start_bound[0], high=start_bound[1])
            start_quat = pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))
            self.model_path + self.models[model]['urdf_file']
            self.loaded_list.append(pb.loadURDF(model + '.urdf', start_pos, start_quat))

    def load_model(self, model, start_bound):
        start_pos = np.random.uniform(low=start_bound[0], high=start_bound[1])
        start_quat = pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))
        self.model_path + self.models[model]['urdf_file']
        self.loaded_list.append(pb.loadURDF(model + '.urdf', start_pos, start_quat))

