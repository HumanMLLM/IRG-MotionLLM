import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import json
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    
    adapted_batch = {
        "system_prompt": [info['system_prompt'] for info in batch],
        "instruction": [info['instruction'] for info in batch],
        "response": [info['response'] for info in batch],
        "breakdown_response": [info['breakdown_response'] for info in batch],
        "motion_info": [info['motion_data'] for info in batch],
        "texts": [info['texts'] for info in batch],
        "task": [info['task'] for info in batch],
    }
    return adapted_batch


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, type, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, is_debug=False, base_datasets=['motion_chain_conv'], generation_mode='direct', gt_forcing=False, ignore_incorrect=False, data_json_files=None):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.type = type
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.is_debug = is_debug
        self.caption_select = 'random'
        self.max_motion_length=256
        self.base_datasets = base_datasets
        self.annots = []
        self.generation_mode = generation_mode
        self.gen_round = -1
        self.gen_w_planning = False
        self.gt_forcing = gt_forcing
        self.ignore_incorrect = ignore_incorrect
        self.data_json_files = data_json_files.split(':')
        self.get_meta_data()

    def get_meta_data(self):
        # import pdb; pdb.set_trace()
        self.all_insts = []
        for f in self.data_json_files:
            self.all_insts += json.load(open(f, 'r'))['annotations']
        self.motion_meta_data = {'humanml3d': {}}
        self.motion_meta_data['humanml3d']['data_root'] = '/YOUR/DATASET/TO/HumanML3D_guo'
        self.motion_meta_data['humanml3d']['motion_dir'] = pjoin(self.motion_meta_data['humanml3d']['data_root'], 'new_joint_vecs')
        self.motion_meta_data['humanml3d']['text_dir'] = pjoin(self.motion_meta_data['humanml3d']['data_root'], 'texts')
        self.motion_meta_data['humanml3d']['joints_num'] = 22
        self.motion_meta_data['humanml3d']['radius'] = 4
        self.motion_meta_data['humanml3d']['motion_fps'] = 20
        self.motion_meta_data['humanml3d']['max_motion_length'] = self.max_motion_length
        self.motion_meta_data['humanml3d']['dim_pose'] = 263
        self.motion_meta_data['humanml3d']['kinematic_chain'] = paramUtil.t2m_kinematic_chain
        self.motion_meta_data['humanml3d']['meta_dir'] = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        self.motion_meta_data['humanml3d']['mean'] = np.load(pjoin(self.motion_meta_data['humanml3d']['meta_dir'], 'mean.npy'))
        self.motion_meta_data['humanml3d']['std'] = np.load(pjoin(self.motion_meta_data['humanml3d']['meta_dir'], 'std.npy'))

        self.prompt_templates = json.load(open('dataset/prompt_template-2.json', 'r'))
        return

    def build_context(self, task, conversations):
        system_prompt = conversations[0]["value"]
        user_prompt = conversations[1]["value"]
        answer = conversations[2]["value"]

        for key, template in self.prompt_templates.items():
            system_prompt = system_prompt.replace(key, random.choice(template))
            user_prompt = user_prompt.replace(key, random.choice(template))
        return system_prompt, user_prompt, answer, None
    def __len__(self):
        return len(self.all_insts)

    def load_motion(self, motion_info):
        if motion_info is None:
            default_null_motion = np.zeros((0, 263))
            return default_null_motion, 0, 0, ''
        motion_id = motion_info['motion_name']
        dataset_name = 'humanml3d'
        # start & end time
        f_tag = float(motion_info['f_tag'])
        to_tag = float(motion_info['to_tag'])

        mean = self.motion_meta_data[dataset_name]['mean']
        std = self.motion_meta_data[dataset_name]['std']
        motion_dir = self.motion_meta_data[dataset_name]['motion_dir']
        motion_fps = self.motion_meta_data[dataset_name]['motion_fps']

        motion_id = [p for p in motion_id.split('_') if len(p) in [6,7]][0]
        motion = np.load(pjoin(motion_dir, motion_id + '.npy'))
        if not (f_tag == 0.0 and to_tag == 0.0):
            if f_tag < to_tag:      # TODO: check 一下这个问题
                motion = motion[int(f_tag * motion_fps):int(to_tag * motion_fps)]
        

        # 我们这里只考虑一种数据增强
        m_length = len(motion)
        m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]
        motion_second = motion.shape[0] / motion_fps    # motion的秒数

        # Z Normalization
        motion = self.forward_transform(motion, mean, std)


        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return motion, motion_second, m_length, motion_id

    def inv_transform(self, data, mean, std):
        return data * std + mean

    def forward_transform(self, data, mean, std):
        return (data - mean) / std
    def __getitem__(self, index):
        annot = self.all_insts[index]
        task=annot['task']
        if "candidate_motion_text" in annot:
            try:
                motion_text_info = random.choice(annot['candidate_motion_text'])
            except:
                print(annot)
                print(annot['candidate_motion_text'])
                assert False
            del annot['candidate_motion_text']
            for k, v in motion_text_info.items():
                annot[k] = v

        motions = annot['motions']
        motions_npy = {}
        for motion_key, motion_info in motions.items():
            motion, motion_second, m_length, motion_name = self.load_motion(motion_info)
            motions_npy[motion_key] = {
                "motion": motion,
                "motion_second": motion_second,
                "m_length": m_length,
                "motion_name": motion_name,
            }

        texts = annot['text'] if 'text' in annot else annot['texts']
        try:
            texts_ = {k: random.choice(v) if isinstance(v, list) else v for k, v in texts.items()}
        except:
            print(annot)
            print(texts)
            assert False
        texts = texts_

        system_prompt, instruction, response, breakdown_responses = self.build_context(task, annot['conversations'])

        info = {}
        info['system_prompt'] = system_prompt
        info['instruction'] = instruction
        info['response'] = response
        info['breakdown_response'] = breakdown_responses
        info['motion_data'] = motions_npy
        info['texts'] = texts
        info['task'] = task
        return info
