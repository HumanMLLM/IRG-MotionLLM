import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from dataset import dataset_TM_eval
from dataset import dataset_unified_sft
from dataset.data_processor import TextProcessor
from datetime import timedelta
import os
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from models.mllm_single_lora import MotionLLM
from options.option_train import get_args_parser
import logging
import json
import sys
from tqdm import tqdm
import time
import shutil
import random 
from torch.utils.data.distributed import DistributedSampler
from utils.metrics import *

torch.set_printoptions(threshold=10000)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with 'pip install tensorboard' for logging.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

class TrainingVisualizer:
    """Training process visualization tool"""
    def __init__(self, log_dir, rank):
        self.log_dir = log_dir
        self.rank = rank
        self.writer = None
        
        # Only initialize TensorBoard on rank 0
        if self.rank == 0 and TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
                print("TensorBoard logging initialized")
            except Exception as e:
                print(f"TensorBoard initialization failed: {e}")
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalar values to TensorBoard"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_embedding_stats(self, model, step, nb_text_tokens):
        """Log statistics of the new token embeddings to TensorBoard"""
        if self.writer and hasattr(model, 'llm'):
            # 获取embedding层
            if hasattr(model.llm, 'model'):
                # 对于某些模型结构
                if hasattr(model.llm.model, 'embed_tokens'):
                    embeddings = model.llm.model.embed_tokens.weight
                else:
                    embeddings = model.llm.model.get_input_embeddings().weight
            else:
                # 对于其他模型结构
                embeddings = model.llm.get_input_embeddings().weight
            
            # 获取新增的motion token embeddings
            new_embeddings = embeddings[nb_text_tokens:]
            
            # 计算统计信息
            embedding_mean = torch.mean(new_embeddings)
            embedding_std = torch.std(new_embeddings)
            embedding_min = torch.min(new_embeddings)
            embedding_max = torch.max(new_embeddings)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Embeddings/Motion_Embedding_Mean', embedding_mean, step)
            self.writer.add_scalar('Embeddings/Motion_Embedding_Std', embedding_std, step)
            self.writer.add_scalar('Embeddings/Motion_Embedding_Min', embedding_min, step)
            self.writer.add_scalar('Embeddings/Motion_Embedding_Max', embedding_max, step)
            
            # 记录梯度信息（如果可用）
            if new_embeddings.grad is not None:
                grad_mean = torch.mean(new_embeddings.grad)
                grad_std = torch.std(new_embeddings.grad)
                self.writer.add_scalar('Embedding_Gradients/Motion_Embedding_Grad_Mean', grad_mean, step)
                self.writer.add_scalar('Embedding_Gradients/Motion_Embedding_Grad_Std', grad_std, step)
    
    def close(self):
        """Close the TensorBoard writer"""
        if self.writer:
            self.writer.close()

def get_logger(out_dir, rank):
    """只在主进程创建logger"""
    if rank != 0:
        return None
        
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def log_info(logger, message):
    """安全的日志记录"""
    if logger is not None:
        logger.info(message)

def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def check_parameter_freeze_status(model, logger, out_dir, verbose=False):

    """检查模型参数的冻结状态"""
    total_params = 0
    total_embed_params = 0
    trainable_params = 0
    lora_params = 0
    base_model_params = 0
    embedding_params = 0
    
    save_file = open(os.path.join(out_dir, "trainable_params_check.txt"), 'w')
    
    if verbose:
        log_info(logger, "=== Parameter Freeze Status Check ===")
        save_file.write("=== Parameter Freeze Status Check ===\n")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            log_info(logger, f"- {name} | Trainable: √")
            save_file.write(f"- {name} | Trainable: √\n")
        else:
            log_info(logger, f"- {name} | Trainable: x")
            save_file.write(f"- {name} | Trainable: x\n")
    save_file.write("=" * 50 + "\n")

def create_distributed_unified_dataloader(dataset_name, split, batch_size, w_vectorizer, unit_length, rank, world_size, is_debug=False, args=None):
    """创建分布式数据加载器"""
    dataset = dataset_unified_sft.Text2MotionDataset(dataset_name, split, w_vectorizer, unit_length=unit_length, is_debug=is_debug, base_datasets=args.base_datasets, generation_mode=args.generation_mode, gt_forcing=args.gt_forcing, ignore_incorrect=args.ignore_incorrect,data_json_files=args.data_json_files)
    
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
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset_unified_sft.collate_fn
    )
    
    return dataloader, sampler

def backup_codebase(out_dir, rank):
    """Backup codebase files and executed command to experiment directory"""
    if rank != 0:
        return
        
    backup_dir = os.path.join(out_dir, "code_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # List of directories and files to backup
    backup_items = [
        "train_unified_sft.py",
        "dataset",
        "utils",
        "models", 
        "options"
    ]
    
    # Copy each item to backup directory
    for item in backup_items:
        src_path = os.path.join(project_root, item)
        dst_path = os.path.join(backup_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
    
    # Save the executed command
    cmd_file = os.path.join(backup_dir, "executed_command.txt")
    with open(cmd_file, "w") as f:
        f.write(" ".join(sys.argv))
        f.write("\n\n")
        f.write("Command executed at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"Codebase backed up to {backup_dir}")

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

def create_distributed_eval_dataloader(dataset_name, split, batch_size, w_vectorizer, unit_length, rank, world_size, is_debug=False, return_all_captions=False):
    """创建分布式数据加载器"""
    dataset = dataset_TM_eval.Text2MotionDataset(dataset_name, split, w_vectorizer, unit_length=unit_length, is_debug=is_debug, return_all_captions=return_all_captions)
    
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
        collate_fn=dataset_TM_eval.collate_fn
    )
    
    return dataloader, sampler


def main():
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    args = get_args_parser()
    
    # seed everything
    seed_everythinig(args.seed)
    # 设置日志记录（仅主进程）
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    logger = get_logger(args.out_dir, rank)
    log_info(logger, json.dumps(vars(args), indent=4, sort_keys=True))

    # 备份代码库和执行命令（仅主进程）
    backup_codebase(args.out_dir, rank)
    log_info(logger, "Codebase backup completed")

    # 初始化可视化工具（仅主进程）
    visualizer = TrainingVisualizer(args.out_dir, rank)
    log_info(logger, "Training visualizer initialized" if rank == 0 and TENSORBOARD_AVAILABLE else "TensorBoard not available or not on main process")

    # 等待主进程创建目录
    if world_size > 1:
        dist.barrier()

    # 调整设备设置
    args.device = f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)

    model = MotionLLM(args)
    
    resume_epoch = -1
    if not args.resume_llm is None:
        resume_info, loaded_names, not_loaded_names = model.load_model(args.resume_llm, verbose=True)
        # if 'epoch' in resume_info:
        #     resume_epoch = resume_info['epoch']
        # else:
        #     resume_epoch = args.resume_epoch
        print(f"Resuming from checkpoint {args.resume_llm}...")
        print(f"Resuming from epoch {resume_epoch}...")
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

    if args.activate_token_embeds:
        # 激活token embed, 让新加入的token可以训练
        print('Activate all token embeds')
        model.activate_token_embeds()
    if args.activate_new_token_embeds:
        print('Activate new token embeds')
        model.activate_new_token_embeds()
    if args.full_tuning:
        print('Activate full model')
        model.activate_full_model()
    
    model.need_normalize = False
    model.training_task = args.training_task    # 设置training task
    model.with_cot_forward = True
    model.generation_mode = args.generation_mode
    model.gt_forcing = args.gt_forcing
    model.ignore_incorrect = args.ignore_incorrect

    model = model.to(args.device)

    if world_size > 1:
        dist.barrier()

    # 包装为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # 调整学习率
        if not args.wo_scale_lr:
            args.learning_rate = args.learning_rate * world_size
    # import pdb; pdb.set_trace()
    model.train()
    check_parameter_freeze_status(model, logger, out_dir=args.out_dir, verbose=False)

    if world_size > 1:
        dist.barrier()

    # 加载数据集
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    # 创建分布式数据加载器
    train_loader, train_sampler = create_distributed_unified_dataloader(
        args.dataname, "train", args.batch_size, w_vectorizer, 
        unit_length=2**args.down_t, rank=rank, world_size=world_size, is_debug=args.debug, args=args
    )

    # NOTE: Build the validation dataloader
    # 创建分布式验证数据加载器（所有进程都参与评估）
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        
    val_loader, _ = create_distributed_eval_dataloader(
        args.dataname, "val", 2, w_vectorizer, 
        unit_length=2**args.down_t, rank=rank, world_size=world_size, is_debug=args.debug, return_all_captions=False
    )

    text_processor = TextProcessor(args)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None  # TODO: add scheduler

    # check trainable params

    # 同步模型权重
    if world_size > 1:
        dist.barrier()

    best_fid = 1000
    best_top1 = 0
    best_bleu4 = 0
    log_info(logger, f"Starting unified training for {args.epochs_unified} epochs...")
    f_inst_example = open(os.path.join(args.out_dir, 'training_samples.txt'), 'w')
    inst_example_recorded = False
    for epoch in range(args.epochs_unified):
        # 设置epoch用于分布式采样
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if resume_epoch > 0 and epoch <= resume_epoch:
            continue
        if hasattr(model, 'module'):
            model.module.llm.train()
            assert model.module.llm.training
        else:
            model.llm.train()
            assert model.llm.training
        if not args.skip_training:
            batch_losses = []
            batch_accs = []
            batch_motion_acc = []
            batch_text_acc = []

            # 创建进度条（仅主进程）
            if rank == 0:
                epoch_iterator = tqdm(
                    train_loader, 
                    desc=f"Epoch {epoch+1}/{args.epochs_unified}",
                    unit="batch",
                    disable=False
                )
            else:
                epoch_iterator = train_loader
            epoch_start_time = time.time()
            for batch_idx, batch in enumerate(epoch_iterator):
                # tokenize batch size
                # import pdb; pdb.set_trace()
                batch_size = len(batch['system_prompt'])
                
                motions = batch['motion_info']
                for b in range(batch_size):
                    # 处理motions
                    motion_info_b = batch['motion_info'][b]

                    for motion_key, motion_info in motion_info_b.items():
                        motion_data = motion_info['motion']
                        motion_data = torch.from_numpy(motion_data).unsqueeze(0)
                        motion_len = motion_info['m_length']
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
                        batch['motion_info'][b][motion_key]['motion_tokens'] = tokens
                input_ids, target_ids, attention_mask = text_processor.process_batch_data(batch, model.module.tokenizer if hasattr(model, 'module') else model.tokenizer)
                # import pdb; pdb.set_trace()
                if batch_idx < 4 and rank == 0:
                    print('=' * 50)
                    print(input_ids[0].shape)
                    print('-' * 50)
                    print(input_ids[0])
                    print('-' * 50)
                    print(target_ids[0])
                    print('-' * 50)
                    print(attention_mask[0])
                    print('-' * 50)
                    tknz = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer
                    print(tknz.batch_decode(input_ids)[0])
                    print('=' * 50)
                    if not inst_example_recorded:
                        f_inst_example.write('=' * 50)
                        f_inst_example.write('\n')
                        f_inst_example.write(str(input_ids.shape))
                        f_inst_example.write('\n')
                        f_inst_example.write('-' * 50)
                        f_inst_example.write('\n')
                        f_inst_example.write(str(input_ids[0]))
                        f_inst_example.write('\n')
                        f_inst_example.write('-' * 50)
                        f_inst_example.write('\n')
                        f_inst_example.write(str(target_ids[0]))
                        f_inst_example.write('\n')
                        f_inst_example.write('-' * 50)
                        f_inst_example.write('\n')
                        f_inst_example.write(str(attention_mask[0]))
                        f_inst_example.write('\n')
                        f_inst_example.write('-' * 50)
                        f_inst_example.write('\n')
                        tknz = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer
                        f_inst_example.write(str(tknz.batch_decode(input_ids)[0]))
                        f_inst_example.write('\n')
                        f_inst_example.write('=' * 50)
                        f_inst_example.write('\n')
                        f_inst_example.write('\n\n')
                elif rank == 0 and batch_idx >= 4:
                    inst_example_recorded = True
                    f_inst_example.close()
                
                input_data = {
                    "input_ids": input_ids.to(args.device),
                    "targets": target_ids.to(args.device),
                    "attention_mask": attention_mask.to(args.device)
                }
                optimizer.zero_grad()
                loss, gen_acc, motion_gen_acc, text_gen_acc, output, labels = model(data=input_data, return_detailed_acc=True)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                batch_accs.append(gen_acc)
                batch_motion_acc.append(motion_gen_acc)
                batch_text_acc.append(text_gen_acc)

                if rank == 0:
                    current_loss = loss.item()
                    current_acc = gen_acc
                    current_motion_acc = motion_gen_acc
                    current_text_acc = text_gen_acc
                    epoch_iterator.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.4f}',
                        'Motion Acc': f'{current_motion_acc:.4f}',
                        'Text Acc': f'{current_text_acc:.4f}',
                        'Batch': f'{batch_idx+1}/{len(train_loader)}'
                    })

                if not args.debug:
                    current_step = epoch * len(train_loader) + batch_idx
                    if rank == 0 and args.save_step_frequency > 0 and (current_step + 1) % args.save_step_frequency == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        resume_info = {
                            'epoch': epoch,
                            'step': current_step,
                        }
                        # 根据save_step_frequency参数保存checkpoint
                        model_to_save.save_model_for_resume(
                            os.path.join(args.out_dir, f'motionllm_unified_sft_epoch{epoch}_step{current_step}.pth'), 
                            resume_info
                        )

            epoch_time = time.time() - epoch_start_time

            # 同步所有进程
            if world_size > 1:
                dist.barrier()

            # 收集所有进程的loss和accuracy
            if world_size > 1:
                avg_loss = torch.tensor(np.mean(batch_losses)).to(args.device)
                avg_acc = torch.tensor(np.mean(batch_accs)).to(args.device)
                avg_motion_acc = torch.tensor(np.mean(batch_motion_acc)).to(args.device)
                avg_text_acc = torch.tensor(np.mean(batch_text_acc)).to(args.device)
                
                if args.debug:
                    # 加入这个为了观察是否是多卡训练出现了问题
                    print(f'- Rank {rank}: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_motion_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_text_acc, op=dist.ReduceOp.SUM)
                
                avg_loss /= world_size
                avg_acc /= world_size
                avg_motion_acc /= world_size
                avg_text_acc /= world_size
                
                avg_loss = avg_loss.item()
                avg_acc = avg_acc.item()
                avg_motion_acc = avg_motion_acc.item()
                avg_text_acc = avg_text_acc.item()
            else:
                avg_loss = np.mean(batch_losses)
                avg_acc = np.mean(batch_accs)
                avg_motion_acc = np.mean(batch_motion_acc)
                avg_text_acc = np.mean(batch_text_acc)

            # 计算训练速度
            samples_per_sec = len(train_loader.dataset) / epoch_time if rank == 0 else 0
                
            log_info(logger, f'Epoch [{epoch+1}/{args.epochs_unified}] - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, Time: {epoch_time:.2f}s, Speed: {samples_per_sec:.2f} samples/s')

            if rank == 0:
                visualizer.log_scalar('Training/Loss', avg_loss, epoch)
                visualizer.log_scalar('Training/Accuracy', avg_acc, epoch)
                visualizer.log_scalar('Training/Motion_Accuracy', avg_motion_acc, epoch)
                visualizer.log_scalar('Training/Text_Accuracy', avg_text_acc, epoch)
                visualizer.log_scalar('Training/Epoch_Time', epoch_time, epoch)
                visualizer.log_scalar('Training/Samples_per_Second', samples_per_sec, epoch)
            
            # 保存模型（仅主进程）
            if not args.debug:
                if rank == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    resume_info = {
                        'epoch': epoch,
                    }
                    model_to_save.save_model_for_resume(os.path.join(args.out_dir, f'motionllm_unified_sft_latest.pth'), resume_info)
                    
                    # 根据save_frequency参数保存checkpoint
                    if args.save_frequency > 0 and (epoch + 1) % args.save_frequency == 0:
                        model_to_save.save_model_for_resume(
                            os.path.join(args.out_dir, f'motionllm_unified_sft_epoch{epoch}.pth'), 
                            resume_info
                        )
        else:
            batch_losses = []
            batch_accs = []
            batch_motion_acc = []
            batch_text_acc = []

            # 创建进度条（仅主进程）
            if rank == 0:
                epoch_iterator = tqdm(
                    train_loader, 
                    desc=f"Epoch {epoch+1}/{args.epochs_unified}",
                    unit="batch",
                    disable=False
                )
            else:
                epoch_iterator = train_loader
            epoch_start_time = time.time()
            for batch_idx, batch in enumerate(epoch_iterator):
                # tokenize batch size
                # import pdb; pdb.set_trace()
                batch_size = len(batch['system_prompt'])

        # 同步所有进程
        if world_size > 1:
            dist.barrier()


    # 清理分布式环境
    cleanup_distributed()
    
    # 关闭TensorBoard写入器（仅主进程）
    if rank == 0:
        visualizer.close()
if __name__ == "__main__":
    main()
