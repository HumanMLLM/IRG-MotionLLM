from options.option_llm import get_args_parser
from models.mllm_single_lora import MotionLLM
import torch
from dataset import dataset_TM_eval
from torch.utils.data.distributed import DistributedSampler
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset.data_processor import TextProcessor

from options.get_eval_option import get_opt
import numpy as np
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import tempfile
import json
import sys
from datetime import timedelta
from tqdm import tqdm
from utils.metrics import *
from models.multi_round_inference import multi_round_inference_engine
import random

# os.environ['MASTER_PORT'] = '12355'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    # Use NCCL backend for GPU training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=1800))

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=3600))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def create_distributed_dataloader(dataset_name, split, batch_size, w_vectorizer, unit_length, rank, world_size, is_debug=False, return_neg_motion=False):
    """创建分布式数据加载器"""
    dataset = dataset_TM_eval.Text2MotionDataset(dataset_name, split, w_vectorizer, unit_length=unit_length, is_debug=is_debug, return_neg_motion=return_neg_motion)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_TM_eval.collate_fn,
        drop_last=True
    )
    
    return dataloader, sampler

def distributed_evaluation(val_loader, model, eval_wrapper, device, draw=False, savenpy=False, savetxt=False, world_size=1, res_dir=None, batch_inference=False, rank=0, repeat=0):
    """分布式评估函数，所有GPU都参与评估以避免timeout"""
    model.eval()
    fid, div, top1, top2, top3, matching, multi, motion_annotation_np, motion_pred_np = evaluation_test(
        res_dir, val_loader, model, 
        eval_wrapper=eval_wrapper, draw=draw, savenpy=savenpy, savetxt=savetxt,  with_all_caps=False, return_motion_embeddings=True, world_size=world_size,sync_all_results=False, return_dict=True, batch_inference=batch_inference
    )
    # motion_npy_root = os.path.join(res_dir, 'motion_npy')
    # os.makedirs(motion_npy_root, exist_ok=True)
    # model.train()
    return fid, div, top1, top2, top3, matching, multi

# def distributed_evaluation_m2t(val_loader, model, device, draw=False, savenpy=False):
#     """分布式M2T评估函数，所有GPU都参与评估以避免timeout"""
#     model.eval()
#     from utils.evaluation_m2t import evaluation_test_m2t
#     bleu1, bleu4, bert_score, _, _, _, _ = evaluation_test_m2t(
#         None, val_loader, model, device, draw=draw, savenpy=savenpy
#     )
#     model.train()
#     return bleu1, bleu4, bert_score

def seed_everythinig(seed=0):
    """设置所有相关的随机种子以保证可复现性
    
    参数:
        seed (int): 随机种子值，默认为42
    """
    # 1. Python内置随机库
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. PyTorch
    torch.manual_seed(seed)
    # 4. 如果使用CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # 关闭benchmark，保证可复现性
    # 5. 设置Python哈希种子（在某些环境中需要）
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 对于某些CUDA操作需要


@torch.no_grad()
def eval_t2m():
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()

    args = get_args_parser()
    # seed everything
    seed_everythinig(seed=42)
    # 调整设备设置
    args.device = f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)

    # Check if multiple GPUs are available
    ngpus_per_node = torch.cuda.device_count()
    print(f"Available GPUs: {ngpus_per_node}")
    model = MotionLLM(args)
    model.need_normalize = False
    model.training_task = args.eval_task    # 设置training task
    model.with_cot_forward = True
    model.generation_mode = args.generation_mode
    model.prompt_w_response = args.prompt_w_response

    if args.merge_lora:
        model.activate_full_model()     # 自带模型合并
    model = model.to(args.device)

    

    # 包装为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    llm_ckpt_path = args.llm_ckpt
    ckpt_epoch = -1
    no_ckpt_tag = '-no_ckpt'
    # import pdb; pdb.set_trace()
    if llm_ckpt_path != "":
        if hasattr(model, 'module'):
            resume_info, loaded_names, not_loaded_names = model.module.load_model(llm_ckpt_path, verbose=True)
            ckpt_epoch = resume_info['epoch'] if 'epoch' in resume_info else -1
        else:
            resume_info, loaded_names, not_loaded_names = model.load_model(llm_ckpt_path, verbose=True)
            ckpt_epoch = resume_info['epoch'] if 'epoch' in resume_info else -1

        if rank == 0:    
            print('=' * 50)
            print(f"Loaded {len(loaded_names)} parameters")
            print('-' * 50)
            for name in loaded_names:
                print(' - ' + name)
            print('=' * 50)
            print(f"Not loaded {len(not_loaded_names)} parameters")
            print('-' * 50)
            for name in not_loaded_names:
                print(' - ' + name)
            print('=' * 50)
            
        no_ckpt_tag = ''
        
    # import pdb; pdb.set_trace()
    model.eval()
    # Path to checkpoint
    # llm_ckpt_path = 'experiments/exp-1-1-Gemma_2_2b_it-lora-t2m_resume_1/motionllm_t2m_best.pth'

    text_processor = TextProcessor(args)

    # Determine checkpoint directory for saving results
    if args.debug:
        debug_tag =  '-debug'
    else:
        debug_tag = ''
    if args.eval_tag:
        eval_tag = f"-{args.eval_tag}"
    else:
        eval_tag = ''
    if args.do_sample:
        do_sample_tag = '-do_sample'
    else:
        do_sample_tag = ''
    checkpoint_dir = os.path.dirname(llm_ckpt_path)
    results_dir = os.path.join(checkpoint_dir, f"evaluation-{llm_ckpt_path.split('/')[-1].split('.')[0]}_epoch{ckpt_epoch}-{args.eval_set}-{args.eval_task}" + do_sample_tag + eval_tag + no_ckpt_tag + debug_tag)
    out_dir = results_dir

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        # save running scripts
        cmd_file = os.path.join(results_dir, "executed_command.sh")
        with open(cmd_file, "w") as f:
            f.write(" ".join(sys.argv))
    savenpy = True
    savetxt = True
    # 同步
    if world_size > 1:
        dist.barrier()
    # 创建分布式验证数据加载器（所有进程都参与评估）
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'

    val_loader, _ = create_distributed_dataloader(
        args.dataname, args.eval_set, args.eval_batch_size, w_vectorizer, 
        unit_length=2**args.down_t, rank=rank, world_size=world_size, is_debug=args.debug, return_neg_motion=args.dataset_return_neg_motion
    )

    # 同步
    if world_size > 1:
        dist.barrier()

    # Number of repetitions
    repeat_time = args.eval_repeat_times
    all_eval_results = {'FID':[], 'Top1':[], 'Top2':[],'Top3':[], 'Diversity':[], 'Matching':[], 'MultiModality':[]}
    # if ngpus_per_node > 1:
    print(f"Using {ngpus_per_node} GPUs for parallel evaluation")
    
    # For each repeat, we distribute the test data across GPUs
    # all_results = []
    all_repeat_results = []
    for i in range(repeat_time):
        print(f"Running evaluation repeat {i+1}/{repeat_time}")
        
        eval_model = model.module if hasattr(model, 'module') else model
        
        if i == 0:
            save_npy = True
            save_txt = True
        # fid, div, top1, top2, top3, matching, multi = distributed_evaluation(
        #     val_loader, eval_model, eval_wrapper, args.device, draw=False, savenpy=save_npy, savetxt=save_txt, world_size=world_size, res_dir=results_dir, batch_inference=args.batch_inference, rank=rank)

        device = args.device
        print(f"Rank {rank}: Processing {len(val_loader)} batches")

        

        if out_dir is not None:
            os.makedirs(os.path.join(out_dir, 'motion_xyz'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'txt_io'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'motion_263_denormed'), exist_ok=True)
        
        nb_sample = 0
        
        draw_org = []
        draw_pred = []
        draw_text = []
        draw_text_pred = []
        draw_name = []

        # 收集所有进程的 et_pred, em_pred, et, em
        et_pred_list = []
        em_pred_list = []
        et_list = []
        em_list = []

        motion_annotation_list = []
        motion_pred_list = []
        # motion_multimodality = []
        R_precision_real = 0
        R_precision = 0
        matching_score_real = 0
        matching_score_pred = 0

        eval_model = model.module if hasattr(model, 'module') else model
        eval_model.eval()

        if eval_model.w_lora:
            eval_model.llm.set_adapter('shared')
        for eval_batch_idx, batch in enumerate(tqdm(val_loader, leave=False)):
            if args.dataset_return_neg_motion:
                word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name, neg_pose, neg_m_length = batch
            else:
                word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            # motion_multimodality_batch = []
            # for i in range(1):  # Multimodality loop, kept as 1 per your code
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(args.device)
            pred_len = torch.ones(bs).long()

            input_dict = {'texts': [{'<Goal_Caption>': caption[b]} for b in range(bs)]}
            if args.dataset_return_neg_motion:
                input_dict['motions'] = [{} for _ in range(bs)]
                for neg_b, (neg_pose_b, neg_m_length_b) in enumerate(zip(neg_pose, neg_m_length)):
                    motion_data = neg_pose_b.unsqueeze(0)
                    motion_len = neg_m_length_b
                    if motion_len == 0:
                        tokens = torch.Tensor([]).to(args.device)
                    else:
                        if hasattr(model, 'module'):
                            tokens = model.module.net.encode(motion_data[:, :motion_len, :].to(args.device)).squeeze(0)
                        else:
                            tokens = model.net.encode(motion_data[:, :motion_len, :].to(args.device)).squeeze(0)

                        for j in range(tokens.shape[0]):
                            if hasattr(model, 'module'):
                                tokens[j] = model.module.motion_token_indices[tokens[j]]
                            else:
                                tokens[j] = model.motion_token_indices[tokens[j]]
                    input_dict['motions'][neg_b]['<Neg_Motion>'] = tokens
                pass
                # import pdb; pdb.set_trace()

            results = multi_round_inference_engine(eval_model, text_processor, task_type=args.eval_task, input_info=input_dict, do_sample_=args.do_sample, device=args.device, max_refine_round=args.max_refine_round, verbose=(eval_batch_idx < 4 and rank == 0))
            # import pdb; pdb.set_trace()

            for k, result in enumerate(results):
                # result = results[mini_b_k]
                index_motion = result['motion_tokens']
                txt_io_k = {
                    'input_text': result['input_text'],
                    'best_beam_text': result['best_beam_text'],
                    'extracted_motion': str(index_motion),
                    'num_gens': result['num_gens'] if 'num_gens' in result else None,
                }
                
                if len(index_motion) == 0:
                    print('Got empty motion')
                    index_motion = torch.LongTensor([0]).to(device)
                if not (index_motion >= 0).all():
                    print(index_motion)
                    assert False
                if (index_motion > 511).any():
                    print(index_motion)
                    assert False 
                try:
                    if hasattr(model, 'module'):
                        pred_pose = model.module.net.forward_decoder(index_motion)
                    else:
                        pred_pose = model.net.forward_decoder(index_motion)
                except:
                    print(index_motion)
                    assert False

                cur_len = pred_pose.shape[1]
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                # if i == 0 and (savenpy):
                #     pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(device), num_joints)

                #     np.save(os.path.join(out_dir, 'motion_xyz', name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())
                #     np.save(os.path.join(out_dir, 'motion_263_denormed', name[k]+'_pred.npy'), pred_denorm)
                if i == 0 and savetxt:
                    json.dump(txt_io_k, open(os.path.join(out_dir, 'txt_io', name[k]+'.json'), 'w'), indent=4)
                # txt_io_b.append(txt_io_k)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            # motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            pose = pose.to(device).float()
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            
            # 收集每个 batch 的 embeddings
            et_pred_list.append(et_pred)
            em_pred_list.append(em_pred)
            et_list.append(et)
            em_list.append(em)

            nb_sample += bs

        # 同步所有进程
        if world_size > 1:
            dist.barrier()   

        # 合并所有 batch 的 embeddings
        et_pred_tensor = torch.cat(et_pred_list, dim=0)
        em_pred_tensor = torch.cat(em_pred_list, dim=0)
        et_tensor = torch.cat(et_list, dim=0)
        em_tensor = torch.cat(em_list, dim=0)

        # 在分布式环境中收集所有进程的 embeddings
        gathered_et_pred = [torch.zeros_like(et_pred_tensor) for _ in range(world_size)]
        gathered_em_pred = [torch.zeros_like(em_pred_tensor) for _ in range(world_size)]
        gathered_et = [torch.zeros_like(et_tensor) for _ in range(world_size)]
        gathered_em = [torch.zeros_like(em_tensor) for _ in range(world_size)]

        dist.all_gather(gathered_et_pred, et_pred_tensor)
        dist.all_gather(gathered_em_pred, em_pred_tensor)
        dist.all_gather(gathered_et, et_tensor)
        dist.all_gather(gathered_em, em_tensor)

        # 同步 nb_sample
        nb_sample_tensor = torch.tensor(nb_sample, dtype=torch.long, device=device)
        dist.all_reduce(nb_sample_tensor, op=dist.ReduceOp.SUM)
        nb_sample = nb_sample_tensor.item()

        eval_results = {}
        if rank == 0:
            print(f"Got {nb_sample} samples, start calculating metrics...")

            # 合并所有进程的 embeddings
            et_pred_np = torch.cat(gathered_et_pred, dim=0).cpu().numpy()
            em_pred_np = torch.cat(gathered_em_pred, dim=0).cpu().numpy()
            et_np = torch.cat(gathered_et, dim=0).cpu().numpy()
            em_np = torch.cat(gathered_em, dim=0).cpu().numpy()

            R_precision_real = 0
            R_precision = 0
            matching_score_real = 0
            matching_score_pred = 0

            mini_batch_size = 32        # 这是T2M领域固定算Top-K的batch size
            exact_cal_instances = 0
            for i in range(0, nb_sample, mini_batch_size):
                et_pred_np_batch = et_pred_np[i:i+mini_batch_size]
                em_pred_np_batch = em_pred_np[i:i+mini_batch_size]
                et_np_batch = et_np[i:i+mini_batch_size]
                em_np_batch = em_np[i:i+mini_batch_size]

                if et_np_batch.shape[0] < mini_batch_size:
                    continue

                # 计算 R_precision 和 matching_score
                temp_R_precision_real, temp_matching_score_real = calculate_R_precision(et_np_batch, em_np_batch, top_k=3, sum_all=True)
                temp_R_precision, temp_matching_score_pred = calculate_R_precision(et_pred_np_batch, em_pred_np_batch, top_k=3, sum_all=True)

                R_precision_real += temp_R_precision_real
                R_precision += temp_R_precision
                matching_score_real += temp_matching_score_real
                matching_score_pred += temp_matching_score_pred

                exact_cal_instances += mini_batch_size

            # 归一化
            R_precision_real = R_precision_real / exact_cal_instances
            R_precision = R_precision / exact_cal_instances
            matching_score_real = matching_score_real / exact_cal_instances
            matching_score_pred = matching_score_pred / exact_cal_instances

            # 计算 FID 和 diversity
            motion_annotation_np = em_np  # 使用 em 作为 motion_annotation
            motion_pred_np = em_pred_np  # 使用 em_pred 作为 motion_pred
            gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
            mu, cov = calculate_activation_statistics(motion_pred_np)
            diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
            diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

            # 填充返回结果
            eval_results = {
                'eval_fid': fid,
                'eval_diversity': diversity,
                'eval_diversity_real': diversity_real,
                'eval_R_precision_top1': R_precision[0],
                'eval_R_precision_top2': R_precision[1],
                'eval_R_precision_top3': R_precision[2],
                'eval_R_precision_real_top1': R_precision_real[0],
                'eval_R_precision_real_top2': R_precision_real[1],
                'eval_R_precision_real_top3': R_precision_real[2],
                'eval_matching_score_pred': matching_score_pred,
                'eval_matching_score_real': matching_score_real,
            }

            for key, value in eval_results.items():
                print(f'{key}: {value:.4f}')


        # 同步所有进程
        if world_size > 1:
            dist.barrier()
        
        # 广播 results 到所有进程
        results_tensor = torch.tensor([float(eval_results.get(k, 0.0)) for k in [
            'eval_fid', 'eval_diversity', 'eval_diversity_real', 'eval_R_precision_top1', 'eval_R_precision_top2',
            'eval_R_precision_top3','eval_R_precision_real_top1', 'eval_R_precision_real_top2',
            'eval_R_precision_real_top3', 'eval_matching_score_pred', 'eval_matching_score_real'
        ]], dtype=torch.float, device=device)
        dist.broadcast(results_tensor, src=0)  # 主进程广播结果
        if rank != 0:
            eval_results = {
                'eval_fid': results_tensor[0].item(),
                'eval_diversity': results_tensor[1].item(),
                'eval_diversity_real': results_tensor[2].item(),
                'eval_R_precision_top1': results_tensor[3].item(),
                'eval_R_precision_top2': results_tensor[4].item(),
                'eval_R_precision_top3': results_tensor[5].item(),
                'eval_R_precision_real_top1': results_tensor[6].item(),
                'eval_R_precision_real_top2': results_tensor[7].item(),
                'eval_R_precision_real_top3': results_tensor[8].item(),
                'eval_matching_score_pred': results_tensor[9].item(),
                'eval_matching_score_real': results_tensor[10].item(),
            }
        all_repeat_results.append({k:float(v) for k,v in eval_results.items()})

        if rank == 0:
            with open(os.path.join(out_dir, 'all_eval_results_t2m.json'), 'w') as f:
                json.dump({f'Repeat-{r}': all_repeat_results[r] for r in range(len(all_repeat_results))}, f, indent=4)

    # 同步所有进程
    if world_size > 1:
        dist.barrier()

    print('==' * 50)
    print('Avg Results')
    if rank == 0:
        all_metric_names = [
            'eval_fid', 'eval_diversity', 'eval_diversity_real', 'eval_R_precision_top1', 'eval_R_precision_top2',
            'eval_R_precision_top3','eval_R_precision_real_top1', 'eval_R_precision_real_top2',
            'eval_R_precision_real_top3', 'eval_matching_score_pred', 'eval_matching_score_real'
        ]
        for metric_name in all_metric_names:
            metric_values = [result[metric_name] for result in all_repeat_results]
            mean_value = sum(metric_values) / len(metric_values)
            std_value = np.std(metric_values)
            print(f"- {metric_name}: {mean_value:.4f} ± {std_value:.4f}")
    print('==' * 50)


    # 清理分布式环境
    cleanup_distributed()
            # Cre
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    eval_t2m()
