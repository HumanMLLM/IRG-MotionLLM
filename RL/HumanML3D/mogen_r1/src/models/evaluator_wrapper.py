
import torch
from os.path import join as pjoin
import numpy as np
import torch.nn.functional as F
from models.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from utils.word_vectorizer import POS_enumerator
from utils.word_vectorizer import WordVectorizer
from models.tmr.collate import collate_x_dict
import os
import random
def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc

import spacy
nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

class EvaluatorModelWrapper(object):
    def __init__(self, opt, reward_mode='guo_tm_distance'):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

        self.w_vectorizer = WordVectorizer('./glove', 'our_vab')

        self.mean = np.load(pjoin('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta', 'mean.npy'))
        self.std = np.load(pjoin('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta', 'std.npy'))

        self.reward_mode = reward_mode

        self.ranking_alpha = 1
        self.ranking_beta = 1
        self.ranking_reward_0 = 0.5  # 这是一个预定义的，当模型只生成一次的reward阈值

        self.eps = 1e-7

    def load_to_device(self, load_device):
        self.device = load_device
        self.text_encoder.to(load_device)
        self.motion_encoder.to(load_device)
        self.movement_encoder.to(load_device)
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        for maname, param in self.motion_encoder.named_parameters():
            param.requires_grad = False
        for maname, param in self.movement_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
        return self
    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            # '''Text Encoding'''
            # text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            '''Text Encoding'''
            # Sort cap_lens in descending order
            sorted_cap_lens, indices = torch.sort(cap_lens, descending=True)
            sorted_word_embs = word_embs[indices]
            sorted_pos_ohot = pos_ohot[indices]
            
            # Encode sorted inputs
            text_embedding_sorted = self.text_encoder(sorted_word_embs, sorted_pos_ohot, sorted_cap_lens)
            
            # Restore original order
            _, reverse_indices = torch.sort(indices)
            text_embedding = text_embedding_sorted[reverse_indices]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            # align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            # motions = motions[align_idx]
            # m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

    def MM_similarity_reward(self, motions_x, m_lens_x, motions_y, m_lens_y):
        with torch.no_grad():
            # Encode both motion and text
            latent_motion_x = self.get_motion_embeddings(motions_x, m_lens_x)   # B, 256
            latent_motion_y = self.get_motion_embeddings(motions_y, m_lens_y)   # B, 256

            motion_embedding_x = latent_motion_x
            motion_embedding_y = latent_motion_y

            # TODO: cosine similarity
            similarities = F.cosine_similarity(motion_embedding_x, motion_embedding_y, dim=1)

            distance = torch.norm(motion_embedding_x - motion_embedding_y, dim=-1)
        return (0 - distance), (1 + similarities) / 2   # normalize to [0, 1]

    def TM_similarity_reward(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            text_embedding, motion_embedding = self.get_co_embeddings(word_embs, pos_ohot, cap_lens, motions, m_lens)

            # TODO: cosine similarity
            similarities = F.cosine_similarity(text_embedding, motion_embedding, dim=1)
            distance = torch.norm(text_embedding - motion_embedding, dim=-1)
        return (0 - distance), (1 + similarities) / 2

    def normalize_motion(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def ranking_reward_hard(self, all_think_tm_sim):
        reward_list = []
        for think_tm_sim in all_think_tm_sim:
            if len(think_tm_sim) == 1:
                reward_list.append(torch.tensor(self.ranking_reward_0).to(self.device, dtype=float)) 
                continue
            gammas = []

            for t in range(1, len(think_tm_sim)):
                score_prev = think_tm_sim[:t]
                score_t = think_tm_sim[t]

                max_score_prev = max(score_prev)
                # score_weights = [self.ranking_alpha**i for i in range(len(score_prev))][::-1]
                # score_weights = torch.tensor(score_weights).to(self.device)
                gamma_t = max_score_prev <= score_t + 0.01   # 这里加0.01 是为了提升一点容忍度，以防score差距太小
                gammas.append(gamma_t.to(dtype=int))
            gamma_weights = [self.ranking_beta**i for i in range(len(gammas))][::-1]
            reward = sum([w * g for w, g in zip(gamma_weights, gammas)]) / (sum(gamma_weights) + self.eps)
            reward_list.append(reward) 
        rewards = torch.stack(reward_list)
        return rewards

    def ranking_reward_soft(self, all_think_tm_sim):
        reward_list = []
        for think_tm_sim in all_think_tm_sim:
            if len(think_tm_sim) == 1:
                reward_list.append(torch.tensor(self.ranking_reward_0).to(self.device, dtype=float)) 
                continue
            gammas = []

            for t in range(1, len(think_tm_sim)):
                score_prev = think_tm_sim[:t]
                score_t = think_tm_sim[t]
                score_weights = [self.ranking_alpha**i for i in range(len(score_prev))][::-1]
                score_weights = torch.tensor(score_weights).to(self.device)
                gamma_t = torch.sum(score_prev * score_weights) / (torch.sum(score_weights) + self.eps) <= score_t + 0.01   # 这里加0.01 是为了提升一点容忍度，以防score差距太小
                gammas.append(gamma_t.to(dtype=int))
            gamma_weights = [self.ranking_beta**i for i in range(len(gammas))][::-1]
            reward = sum([w * g for w, g in zip(gamma_weights, gammas)]) / (sum(gamma_weights) + self.eps)
            reward_list.append(reward) 
        rewards = torch.stack(reward_list)
        return rewards

    def process_reward_refinement_single_step(self, completions, **kwargs):
        ref_captions = [cap for cap in kwargs['reference_caption']]
        all_pos_ohot = []
        all_word_embeddings = []
        all_sent_lens = []
        reward_dict = {}
        for caption in ref_captions:
            word_list, pose_list = process_text(caption)
            tokens = ['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))]
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
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
            all_pos_ohot.append(torch.from_numpy(pos_one_hots).unsqueeze(0))
            all_word_embeddings.append(torch.from_numpy(word_embeddings).unsqueeze(0))
            all_sent_lens.append(sent_len)
        all_pos_ohot = torch.cat(all_pos_ohot, dim=0).to(self.device)
        all_word_embeddings = torch.cat(all_word_embeddings, dim=0).to(self.device)
        all_sent_lens = torch.tensor(all_sent_lens).to(self.device)


        initial_motions = [motion_info["init_motion_263"][0] for motion_info in kwargs['refinement_motion_pairs']]
        initial_motion_dict_list = [{"x": motion.float(), "length": len(motion)} for motion in initial_motions]
        initial_motion_dict = collate_x_dict(initial_motion_dict_list)
        initial_m_lens = torch.tensor(initial_motion_dict['length']).to(self.device)
        initial_m_masked = initial_motion_dict['x']

        # import pdb; pdb.set_trace()
        w_refined_motions = [motion_info["refinement_motion_263"] != [] for motion_info in kwargs['refinement_motion_pairs']]
        refined_motions = [motion_info["refinement_motion_263"][0] if motion_info["refinement_motion_263"] != [] else torch.zeros(196, 263).to(device=self.device, dtype=torch.bfloat16) for motion_info in kwargs['refinement_motion_pairs']]
        refined_motion_dict_list = [{"x": motion.float(), "length": len(motion)} for motion in refined_motions]
        refined_motion_dict = collate_x_dict(refined_motion_dict_list)
        refined_m_lens = torch.tensor(refined_motion_dict['length']).to(self.device)
        refined_m_masked = refined_motion_dict['x']

        # 获取GT motion
        gt_motions = []
        gt_m_lens = []
        if '<Goal_Motion>' in kwargs['motions'][0] and kwargs['motions'][0]['<Goal_Motion>'] != None:
            for gt_motion_info in  kwargs['motions']:
                meta_path = gt_motion_info['<Goal_Motion>']
                motion_root = '/'.join(meta_path.split('/')[:-1])
                motion_meta_name = meta_path.split('/')[-1]
                
                motion_name = [part for part in motion_meta_name.split('_') if len(part) in [6,7]][0]
                motion_path = os.path.join(motion_root, motion_name+'.npy')

                max_motion_length = 196
                unit_length = 4
                motion_data = np.load(motion_path)
                m_length = len(motion_data)
                m_length = (m_length // unit_length) * unit_length
                

                # idx = random.randint(0, len(motion_data) - m_length)
                idx = 0
                motion_data = motion_data[idx:idx+m_length]

                # "Z Normalization"
                motion_data = (motion_data - self.mean) / self.std

                if m_length < max_motion_length:
                    motion_data = np.concatenate([motion_data,
                                            np.zeros((max_motion_length - m_length, motion_data.shape[1]))
                                            ], axis=0)
                gt_motions.append(motion_data)
                gt_m_lens.append(m_length)
            gt_motions = torch.from_numpy(np.stack(gt_motions, axis=0)).float().to(self.device)
            gt_m_lens = torch.tensor(gt_m_lens).to(self.device)

            mm_dis_initial, mm_sim_initial = self.MM_similarity_reward(motions_x=gt_motions, m_lens_x=gt_m_lens, motions_y=initial_m_masked, m_lens_y=initial_m_lens)
            mm_dis_refined, mm_sim_refined = self.MM_similarity_reward(motions_x=gt_motions, m_lens_x=gt_m_lens, motions_y=refined_m_masked, m_lens_y=refined_m_lens)
            mm_dis_improvement_reward = (mm_dis_refined > mm_dis_initial).to(dtype=int)
            mm_sim_improvement_reward = (mm_sim_refined > mm_sim_initial).to(dtype=int)

            if 'mm' in self.reward_mode:
                if 'dis' in self.reward_mode:
                    reward_dict["mm_improvement"] = mm_dis_improvement_reward
                    reward_dict["mm_alignment"] = mm_dis_refined
                elif 'sim' in self.reward_mode:
                    reward_dict["mm_improvement"] = mm_sim_improvement_reward
                    reward_dict["mm_alignment"] = mm_sim_refined
        else:
            if 'mm' in self.reward_mode:
                mm_dis_refined = torch.zeros(len(refined_m_masked)).to(self.device)
                mm_sim_refined = torch.zeros(len(refined_m_masked)).to(self.device)
                mm_dis_initial = torch.zeros(len(refined_m_masked)).to(self.device)
                mm_sim_initial = torch.zeros(len(refined_m_masked)).to(self.device)
                mm_dis_improvement_reward = (mm_dis_refined >= mm_dis_initial).to(dtype=int)
                mm_sim_improvement_reward = (mm_sim_refined >= mm_sim_initial).to(dtype=int)
                if 'dis' in self.reward_mode:
                    reward_dict["mm_improvement"] = mm_dis_improvement_reward
                    reward_dict["mm_alignment"] = mm_dis_refined
                elif 'sim' in self.reward_mode:
                    reward_dict["mm_improvement"] = mm_sim_improvement_reward
                    reward_dict["mm_alignment"] = mm_sim_refined

        tm_dis_refined, tm_sim_refined = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=refined_m_masked, m_lens=refined_m_lens)
        tm_dis_initial, tm_sim_initial = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=initial_m_masked, m_lens=initial_m_lens)
        # import pdb; pdb.set_trace()
        tm_dis_improvement_reward = (tm_dis_refined >= tm_dis_initial).to(dtype=int)
        tm_sim_improvement_reward = (tm_sim_refined >= tm_sim_initial).to(dtype=int)
        # import pdb; pdb.set_trace()

        # 这里做一个baseline 策略
        for mask_i, mask in enumerate(w_refined_motions):
            if not mask:
                tm_dis_improvement_reward[mask_i] = 0
                tm_sim_improvement_reward[mask_i] = 0
                mm_dis_improvement_reward[mask_i] = 0
                mm_sim_improvement_reward[mask_i] = 0
            
                tm_sim_refined[mask_i] = tm_sim_initial[mask_i]
                tm_dis_refined[mask_i] = tm_dis_initial[mask_i]
                mm_sim_refined[mask_i] = mm_sim_initial[mask_i]
                mm_dis_refined[mask_i] = mm_dis_initial[mask_i]


        if 'tm' in self.reward_mode:
            if 'dis' in self.reward_mode:
                reward_dict["tm_improvement"] = tm_dis_improvement_reward
                reward_dict["tm_alignment"] = tm_dis_refined
            elif 'sim' in self.reward_mode:
                reward_dict["tm_improvement"] = tm_sim_improvement_reward
                reward_dict["tm_alignment"] = tm_sim_refined
        return reward_dict

    def process_reward(self, completions, **kwargs):
        if 'refinement_single_step' in self.reward_mode:
            return self.process_reward_refinement_single_step(completions, **kwargs)
        ref_captions = [cap for cap in kwargs['reference_caption']]
        
        assert len(completions) == len(ref_captions)
        all_think_tm_sim = []
        all_think_tm_dis = []
        for comp, ref_cap in zip(completions, ref_captions):
            num_gen = len(comp["think_motion_denormed_263"])
            if num_gen == 0:
                all_think_tm_sim.append(torch.tensor([0.0]).to(self.device))
                all_think_tm_dis.append(torch.tensor([-10.0]).to(self.device))
                continue
            
            repeated_ref_cap = [ref_cap] * num_gen

            # 处理caption
            all_pos_ohot = []
            all_word_embeddings = []
            all_sent_lens = []
            for caption in repeated_ref_cap:
                word_list, pose_list = process_text(caption)
                tokens = ['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))]
                if len(tokens) < self.opt.max_text_len:
                    # pad with "unk"
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)
                    tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
                else:
                    # crop
                    tokens = tokens[:self.opt.max_text_len]
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
                all_pos_ohot.append(torch.from_numpy(pos_one_hots).unsqueeze(0))
                all_word_embeddings.append(torch.from_numpy(word_embeddings).unsqueeze(0))
                all_sent_lens.append(sent_len)
            all_pos_ohot = torch.cat(all_pos_ohot, dim=0).to(self.device)
            all_word_embeddings = torch.cat(all_word_embeddings, dim=0).to(self.device)
            all_sent_lens = torch.tensor(all_sent_lens).to(self.device)


            think_motions = [tnk_motion[0] for tnk_motion in comp["think_motion_263"]]
            think_motion_dict_list = [{"x": motion.float(), "length": len(motion)} for motion in think_motions]
            think_motion_dict = collate_x_dict(think_motion_dict_list)
            think_m_lens = torch.tensor(think_motion_dict['length']).to(self.device)
            think_m_masked = think_motion_dict['x']

            think_tm_dis, think_tm_sim = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=think_m_masked, m_lens=think_m_lens)
            
            all_think_tm_sim.append(think_tm_sim)
            all_think_tm_dis.append(think_tm_dis)

        if self.reward_mode == 'guo_process_motion_best_similarity':
            all_think_tm_sim_list = []
            for think_tm_sim in all_think_tm_sim:
                think_tm_sim.sort()
                all_think_tm_sim_list.append(think_tm_sim[-1]) 
            all_think_tm_sim = torch.stack(all_think_tm_sim_list)
            reward = all_think_tm_sim
        elif self.reward_mode == 'guo_process_motion_best_distance':
            all_think_tm_dis_list = []
            for think_tm_dis in all_think_tm_dis:
                think_tm_dis.sort()
                all_think_tm_dis_list.append(think_tm_dis[-1])          # 因为这里算的distance 做了负值反转，所以可以直接取最后一个 
            all_think_tm_dis = torch.stack(all_think_tm_dis_list)
            reward = all_think_tm_dis
        elif self.reward_mode == 'guo_process_motion_avg_similarity':
            all_think_tm_sim_list = []
            for think_tm_sim in all_think_tm_sim:
                all_think_tm_sim_list.append(think_tm_sim.mean()) 
            all_think_tm_sim = torch.stack(all_think_tm_sim_list)
            reward = all_think_tm_sim
        elif self.reward_mode == 'guo_process_motion_avg_distance':
            all_think_tm_dis_list = []
            for think_tm_dis in all_think_tm_dis:
                all_think_tm_dis_list.append(think_tm_dis.mean()) 
            all_think_tm_dis = torch.stack(all_think_tm_dis_list)
            reward = all_think_tm_dis
        elif self.reward_mode == 'guo_process_motion_last_similarity':
            all_think_tm_sim_list = []
            for think_tm_sim in all_think_tm_sim:
                # think_tm_sim.sort()
                all_think_tm_sim_list.append(think_tm_sim[-1]) 
            all_think_tm_sim = torch.stack(all_think_tm_sim_list)
            reward = all_think_tm_sim
        elif self.reward_mode == 'guo_process_motion_last_distance':
            all_think_tm_dis_list = []
            for think_tm_dis in all_think_tm_dis:
                all_think_tm_dis_list.append(think_tm_dis[-1]) 
            all_think_tm_dis = torch.stack(all_think_tm_dis_list)
            reward = all_think_tm_dis
        elif self.reward_mode == 'guo_process_motion_last_distance_best':
            all_think_tm_dis_list = []
            for think_tm_dis in all_think_tm_dis:
                if len(think_tm_dis) == 1:
                    all_think_tm_dis_list.append(0.5)   # 如果只有一个，那么用0.5作为奖励
                else:
                    all_think_tm_dis_list.append(int(think_tm_dis[-1] > max(think_tm_dis[:-1])))    # 最后一个生成结果要严格好于前面的所有生成结果
            reward = torch.tensor(all_think_tm_dis_list)
        elif self.reward_mode == 'guo_process_motion_first_similarity':
            all_think_tm_sim_list = []
            for think_tm_sim in all_think_tm_sim:
                # think_tm_sim.sort()
                all_think_tm_sim_list.append(think_tm_sim[0]) 
            all_think_tm_sim = torch.stack(all_think_tm_sim_list)
            reward = all_think_tm_sim
        elif self.reward_mode in ['guo_process_init_gen_distance', 'guo_process_motion_first_distance']:
            all_think_tm_dis_list = []
            for think_tm_dis in all_think_tm_dis:
                all_think_tm_dis_list.append(think_tm_dis[0]) 
            all_think_tm_dis = torch.stack(all_think_tm_dis_list)
            reward = all_think_tm_dis
        elif self.reward_mode == 'guo_process_motion_similarity_rank':
            reward = self.ranking_reward_hard(all_think_tm_sim)

        return reward

    def sub_trace_reward(self, completions, **kwargs):
        # 获取中间的生成结果
        if self.reward_mode in ['guo_sub_trace_mm_distance','guo_sub_trace_tm_distance','guo_sub_trace_mm_similarity','guo_sub_trace_tm_similarity']:
            combined_think_mo_pos = []
            combined_think_motions = []
            for c_i, comp in enumerate(completions):
                for pos in comp['think_motion_pos']:
                    combined_think_mo_pos.append([c_i] + pos)
                for mo in comp["think_motion_263"]:
                    combined_think_motions.append(mo[0])
            assert len(combined_think_mo_pos) == len(combined_think_motions)
            
            # 计算每个中间生成结果的reward
            combined_think_motion_dict_list = [{"x": motion.float(), "length": len(motion)} for motion in combined_think_motions]
            combined_think_motion_dict = collate_x_dict(combined_think_motion_dict_list)
            combined_think_m_lens = torch.tensor(combined_think_motion_dict['length']).to(self.device)
            combined_think_m_masked = combined_think_motion_dict['x']

            ref_captions = [cap for cap in kwargs['reference_caption']]
            ref_captions_ = [ref_captions[pos[0]] for pos in combined_think_mo_pos]
            ref_captions = ref_captions_
            assert len(ref_captions) == len(combined_think_mo_pos)
            all_pos_ohot = []
            all_word_embeddings = []
            all_sent_lens = []
            for caption in ref_captions:
                word_list, pose_list = process_text(caption)
                tokens = ['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))]
                if len(tokens) < self.opt.max_text_len:
                    # pad with "unk"
                    tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                    sent_len = len(tokens)
                    tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
                else:
                    # crop
                    tokens = tokens[:self.opt.max_text_len]
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
                all_pos_ohot.append(torch.from_numpy(pos_one_hots).unsqueeze(0))
                all_word_embeddings.append(torch.from_numpy(word_embeddings).unsqueeze(0))
                all_sent_lens.append(sent_len)
            all_pos_ohot = torch.cat(all_pos_ohot, dim=0).to(self.device)
            all_word_embeddings = torch.cat(all_word_embeddings, dim=0).to(self.device)
            all_sent_lens = torch.tensor(all_sent_lens).to(self.device)

            tm_dis_reward, tm_sim_reward = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=combined_think_m_masked, m_lens=combined_think_m_lens)
            if self.reward_mode == 'guo_sub_trace_tm_distance':
                reward = tm_dis_reward
            elif self.reward_mode == 'guo_sub_trace_tm_similarity':
                reward = tm_sim_reward
            return reward, combined_think_mo_pos

        return


    def __call__(self, completions, **kwargs):
        if 'sub_trace' in self.reward_mode:
            # sub trace reward
            return self.sub_trace_reward(completions, **kwargs)
        if 'process' in self.reward_mode:
            # process reward
            return self.process_reward(completions, **kwargs)
        # 将这里的call 定义为计算T2M reward

        ref_captions = [cap for cap in kwargs['reference_caption']]
        all_pos_ohot = []
        all_word_embeddings = []
        all_sent_lens = []
        for caption in ref_captions:
            word_list, pose_list = process_text(caption)
            tokens = ['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))]
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
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
            all_pos_ohot.append(torch.from_numpy(pos_one_hots).unsqueeze(0))
            all_word_embeddings.append(torch.from_numpy(word_embeddings).unsqueeze(0))
            all_sent_lens.append(sent_len)
        all_pos_ohot = torch.cat(all_pos_ohot, dim=0).to(self.device)
        all_word_embeddings = torch.cat(all_word_embeddings, dim=0).to(self.device)
        all_sent_lens = torch.tensor(all_sent_lens).to(self.device)

        answer_motions = [comp["answer_motion_263"][0] for comp in completions]
        # answer_motions = torch.cat(answer_motions, dim=0).to(self.device)
        answer_motion_dict_list = [{"x": motion.float(), "length": len(motion)} for motion in answer_motions]
        answer_motion_dict = collate_x_dict(answer_motion_dict_list)
        answer_m_lens = torch.tensor(answer_motion_dict['length']).to(self.device)
        answer_m_masked = answer_motion_dict['x']

        # 获取GT motion
        gt_motions = []
        gt_m_lens = []
        # import pdb; pdb.set_trace()
        if '<Goal_Motion>' in kwargs['motions'][0] and kwargs['motions'][0]['<Goal_Motion>'] != None:
            for gt_motion_info in  kwargs['motions']:
                meta_path = gt_motion_info['<Goal_Motion>']
                motion_root = '/'.join(meta_path.split('/')[:-1])
                motion_meta_name = meta_path.split('/')[-1]
                
                motion_name = [part for part in motion_meta_name.split('_') if len(part) in [6,7]][0]
                motion_path = os.path.join(motion_root, motion_name+'.npy')

                max_motion_length = 196
                unit_length = 4
                motion_data = np.load(motion_path)
                m_length = len(motion_data)
                m_length = (m_length // unit_length) * unit_length
                

                idx = random.randint(0, len(motion_data) - m_length)
                motion_data = motion_data[idx:idx+m_length]

                # "Z Normalization"
                motion_data = (motion_data - self.mean) / self.std

                if m_length < max_motion_length:
                    motion_data = np.concatenate([motion_data,
                                            np.zeros((max_motion_length - m_length, motion_data.shape[1]))
                                            ], axis=0)
                gt_motions.append(motion_data)
                gt_m_lens.append(m_length)
            gt_motions = torch.from_numpy(np.stack(gt_motions, axis=0)).float().to(self.device)
            gt_m_lens = torch.tensor(gt_m_lens).to(self.device)

            mm_dis_reward, mm_sim_reward = self.MM_similarity_reward(motions_x=gt_motions, m_lens_x=gt_m_lens, motions_y=answer_m_masked, m_lens_y=answer_m_lens)
        else:
            mm_dis_reward = torch.zeros(len(answer_m_masked)).to(self.device)
            mm_sim_reward = torch.zeros(len(answer_m_masked)).to(self.device)

        tm_dis_reward, tm_sim_reward = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=answer_m_masked, m_lens=answer_m_lens)
        # tm_dis_reward_real, tm_sim_reward_real = self.TM_similarity_reward(word_embs=all_word_embeddings, pos_ohot=all_pos_ohot, cap_lens=all_sent_lens, motions=gt_motions, m_lens=gt_m_lens)
        
        # import pdb; pdb.set_trace()
        if self.reward_mode == 'guo_tm_distance':
            return tm_dis_reward
        elif self.reward_mode == 'guo_mm_distance':
            return mm_dis_reward
        elif self.reward_mode == 'guo_tm_similarity':
            return tm_sim_reward
        elif self.reward_mode == 'guo_mm_similarity':
            return mm_sim_reward
        # return tm_dis_reward, mm_dis_reward

