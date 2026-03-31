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
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, type, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, is_debug=False):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.type = type
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.w_vectorizer = w_vectorizer
        self.is_debug = is_debug
        if dataset_name == 't2m':
            self.data_root = '/mnt/data1/yuanming/datasets/HumanML3D_guo'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = '../KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        if type == 'test':
            split_file = pjoin(self.data_root, 'test.txt')
        elif type == 'val':
            split_file = pjoin(self.data_root, 'val.txt')
        elif type == 'train':
            split_file = pjoin(self.data_root, 'train.txt')
        else:
            raise ValueError('Invalid type')

        min_motion_len = 40 if self.dataset_name =='t2m' else 24
        # min_motion_len = 64

        joints_num = self.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for idx, name in enumerate(tqdm(id_list)):
            if self.is_debug:
                if idx > 200:
                    break
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict],
                                                       'f_tag': f_tag,
                                                       'to_tag': to_tag}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data,
                                       'f_tag': f_tag,
                                        'to_tag': to_tag}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                # print(e)
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def save_data_dict(self, path):
        saved_dict = {}
        for motion_name, info in self.data_dict.items():
            # info['motion'] = info['motion'].cpu().numpy()
            saved_dict[motion_name] = {k:v for k, v in info.items() if k != 'motion'}
        json.dump(saved_dict, open(path, 'w'), indent=4)  # self.save_data_dict('/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/dataset/test_data_dict.json')

    def load_single_data(self, data):
        # 这个部分跟__getitem__()基本一样，用这个函数来load negative sample
        # Load the current data
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        all_captions = [info['caption'] for info in text_list]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        while len(all_captions) < 3:
            all_captions.append(all_captions[-1])
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, all_captions

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        
        # Load the current data
        # data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        all_captions = [info['caption'] for info in text_list]
        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        while len(all_captions) < 3:
            all_captions.append(all_captions[-1])
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        # get negative sample
        neg_idx = random.choice([i for i in range(self.__len__())])
        neg_name = self.name_list[neg_idx]
        neg_data = self.data_dict[neg_name]
        
        _, _, _, _, neg_motion, neg_m_length, _, neg_name, _ = self.load_single_data(neg_data)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), name, all_captions, neg_motion, neg_m_length, neg_name



'''For use of training multi-task model with both T2M and M2T'''
class MultiTaskMotionDataset(data.Dataset):
    def __init__(self, dataset_name, type, w_vectorizer, feat_bias = 5, max_text_len = 20, unit_length = 4, task_ratio=0.5):
        """
        Multi-task dataset for training both T2M and M2T with shared LoRA parameters
        task_ratio: ratio of T2M tasks vs M2T tasks (0.5 means equal split)
        """
        self.t2m_dataset = Text2MotionDataset(dataset_name, type, w_vectorizer, feat_bias, max_text_len, unit_length)
        self.task_ratio = task_ratio
        
    def __len__(self):
        return len(self.t2m_dataset)
    
    def __getitem__(self, item):
        # Get item from the base dataset
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, name = self.t2m_dataset[item]
        
        # Randomly assign task type
        if random.random() < self.task_ratio:
            # T2M task - text to motion generation
            task_type = 0
        else:
            # M2T task - motion to text generation
            task_type = 1
            
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens, name, task_type


def DATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, is_test, w_vectorizer, unit_length=unit_length),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return val_loader


def MultiTaskDATALoader(dataset_name, is_test,
                        batch_size, w_vectorizer,
                        num_workers = 8, unit_length = 4, task_ratio=0.5):
    """Data loader for multi-task training"""
    
    loader = torch.utils.data.DataLoader(MultiTaskMotionDataset(dataset_name, is_test, w_vectorizer, 
                                                                unit_length=unit_length, task_ratio=task_ratio),
                                         batch_size,
                                         shuffle = True,
                                         num_workers=num_workers,
                                         collate_fn=multi_task_collate_fn,
                                         drop_last = True)
    return loader


def multi_task_collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x