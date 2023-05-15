import os
import numpy as np
import pybullet as pb
import quaternion as qtn
from random import choice as random_choice
from yaml import safe_load as yaml_load
# from packing_env import BulletHandler


class ModelManager():
    def __init__(self, model_path, bullet_handler):
        self.bh = bullet_handler
        self.model_path = model_path
        # pb.setAdditionalSearchPath(model_path + '/urdf/')
        self.bh.set_model_path(model_path + '/urdf')
        self.model_list = [os.path.splitext(file)[0] for file in os.listdir(model_path + '/config/') if file.endswith(r".yaml")]
        self.models = dict()
        for model_name in self.model_list:
            model = self.__model_config_loader(model_path + '/config/' + model_name + '.yaml')
            self.models = {**self.models, **model['object_model']}
        self.loaded_models = dict()
        print('self.models len = {}, model_list len = {}'.format(len(self.models), len(self.model_list)))
        
    def __model_config_loader(self, path):
        with open(path, 'r') as stream:
            data = yaml_load(stream)
        return data


    def sample_models_in_bound(self, box_size, fill_rate, min_size_rate=0.03, excess_tolerace=1.2, generate_box=True):
        volume_sum = 0
        box_obj_cnt = 0
        failed_cnt = 0
        self.sampled_models_list = []
        # min_bound = np.array(bound[0])
        # max_bound = np.array(bound[1])
        # bound_size = np.absolute(max_bound - min_bound)
        max_length = np.amax(box_size[:2])
        bound_volume = np.prod(box_size)
        print('bound_volume = {}'.format(bound_volume))
        while volume_sum < bound_volume * fill_rate and failed_cnt < 100:
            et = 999 if self.sampled_models_list is [] else excess_tolerace
            if random_choice([True, False]) or not generate_box:
                for _ in range(10):
                    model = random_choice(self.model_list)
                    if self.models[model]['max_length'] < max_length \
                            and volume_sum + self.models[model]['convex_volume'] \
                            < min(bound_volume * fill_rate * et, bound_volume) \
                            and self.models[model]['convex_volume'] > bound_volume * min_size_rate:
                        volume_sum += self.models[model]['convex_volume']
                        self.sampled_models_list.append(model)
                        failed_cnt = 0
                        break
                    else:
                        failed_cnt += 1
            else:
                min_len = ((bound_volume * min_size_rate) ** (1 / 3)) / 2 
                box_obj_size = np.random.uniform([min_len, min_len, min_len], box_size)
                box_obj_vol = np.prod(box_obj_size)
                box_obj_max_len = np.amax(box_obj_size)
                if box_obj_max_len < max_length \
                        and volume_sum + box_obj_vol \
                        < min(bound_volume * fill_rate * et, bound_volume) \
                        and box_obj_vol > bound_volume * min_size_rate:
                    name = 'a_random_box_obj_' + str(box_obj_cnt)
                    box_obj_cnt += 1
                    self.models[name] = dict()
                    self.models[name]['max_length'] = box_obj_max_len
                    self.models[name]['origin_volume'] = box_obj_vol
                    self.models[name]['convex_volume'] = box_obj_vol
                    self.models[name]['box_size'] = box_obj_size
                    volume_sum += box_obj_vol
                    self.sampled_models_list.append(name)
                    failed_cnt = 0
                else:
                    failed_cnt += 1

        return self.sampled_models_list
    
    def reset(self):
        self.loaded_models = dict()
        
    def load_sampled_models(self, start_bound):
        for model in self.sampled_models_list:
            start_pos = np.random.uniform(low=start_bound[0], high=start_bound[1])
            start_quat = pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))
            self.load_model(model, start_pos, start_quat)
            # self.model_path + self.models[model]['urdf_file']
            # self.loaded_models[model] = pb.loadURDF(model + '.urdf', start_pos, start_quat)
            # self.loaded_models[model] = self.bh.load_urdf(model + '.urdf', start_pos, start_quat)

    def load_model(self, model, start_bound):
        start_pos = np.random.uniform(low=start_bound[0], high=start_bound[1])
        start_quat = pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))
        self.load_model(model, start_pos, start_quat)
        # self.model_path + self.models[model]['urdf_file']
        # self.loaded_models[model] = pb.loadURDF(model + '.urdf', start_pos, start_quat)
        # self.loaded_models[model] = self.bh.load_urdf(model + '.urdf', start_pos, start_quat)

    def load_model(self, model, pos, quat):
        if 'a_random_box_obj_' in model:
            box = self.models[model]
            # self.loaded_models[model] = self.bh.create_box(box['box_size'], pos, quat, box['origin_volume'] * 1000)
            box_size = box['box_size']
            color = np.random.uniform(low=0, high=1, size=3).tolist()
            self.loaded_models[model] = self.bh.load_stl(
                'obj_box.obj', box_size, pos, quat, mass=box['origin_volume'] * 1000, color=color)
        else:
            self.loaded_models[model] = self.bh.load_urdf(model + '.urdf', pos, quat)
        self.set_model_pose(model, pos, quat)

    def random_pos(self, bound):
        return np.random.uniform(low=bound[0], high=bound[1])

    def random_quat(self):
        return pb.getQuaternionFromEuler(np.random.uniform(low=-1*np.pi, high=np.pi, size=3))

    def set_model_pos(self, model, pos):
        # pb.resetBasePositionAndOrientation(self.loaded_models[model], pos, [0,0,0,1])
        model_id = self.loaded_models[model]
        _, curr_quat = self.bh.get_model_pose(model_id)
        self.bh.set_model_pose(self.loaded_models[model], pos, curr_quat)

    def set_model_pose(self, model, pos, quat):
        # pb.resetBasePositionAndOrientation(self.loaded_models[model], pos, quat)
        self.bh.set_model_pose(self.loaded_models[model], pos, quat)

    def set_model_pos_rz(self, model, pos, rz):
        quat = pb.getQuaternionFromEuler([0, 0, rz])
        self.bh.set_model_pose(self.loaded_models[model], pos, quat)

    def set_model_relative_euler(self, model, euler_angle):
        quat = pb.getQuaternionFromEuler(euler_angle)
        self.set_model_relative_pose(model, [0,0,0], quat)

    def set_model_relative_pose(self, model, pos, quat):
        model_id = self.loaded_models[model]
        curr_pos, curr_quat = self.bh.get_model_pose(model_id)
        new_pos = [x + y for x, y in zip(curr_pos, pos)]
        q0 = np.quaternion(curr_quat[3], curr_quat[0], curr_quat[1], curr_quat[2])
        q1 = np.quaternion(quat[3], quat[0], quat[1], quat[2])
        q = list(qtn.as_float_array(q1 * q0))
        new_quat = [q[1], q[2], q[3], q[0]]
        self.bh.set_model_pose(model_id, new_pos, new_quat)

    def get_model_pose(self, model):
        return self.bh.get_model_pose(self.loaded_models[model])
    
    def get_model_id(self, model):
        return self.loaded_models[model]
    
    def remove_model(self, model):
        return self.bh.remove_model(self.loaded_models[model])
    
    def get_model_convex_volume(self, model):
        return self.models[model]['convex_volume']

