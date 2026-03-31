from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch
import numpy as np
import models.vqvae as vqvae
from transformers.configuration_utils import PretrainedConfig

# import uuid
from typing import Optional, Dict, Any, List
from transformers import PretrainedConfig, AutoConfig


class MotionLLMConfig(PretrainedConfig):
    model_type = "motion_llm"
    
    def __init__(
        self,
        llm_backbone: str = "/mnt/data1/yuanming/pretrained_models/qwen2.5-3B-Instruct",
        llm_checkpoint: str = "/mnt/data1/yuanming/pretrained_models/qwen2.5-3B-Instruct",
        w_flash_attention: bool = False,
        nb_code: int = 1024,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        vq_act: str = "relu",
        vq_norm: Optional[str] = None,
        nb_joints: int = 22,
        dataname: str = "t2m",
        vq_path: str = "/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/ckpt/vqvae.pth",
        mean_path: str = "/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/ckpt/mean.npy",
        lora_r_t2m: int = 8,
        lora_alpha_t2m: int = 16,
        lora_r_m2t: int = 8,
        lora_alpha_m2t: int = 16,
        lora_dropout: float = 0.01,
        lora_target_modules: List[str] = [
            'o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'
        ],
        first_lora: str = "shared",
        need_normalize: bool = True,
        eval_task: str = "direct",
        context_style: str = "default",
        prompt_w_response: bool = False,
        generation_mode: str = "direct",
        device: str = "cuda",
        version: str = "1.0.0",
        full_finetuned: bool = False,
        full_frozen: bool = False,
        w_lora: bool = False,
        activate_token_embeds: bool = False,
        **kwargs
    ):
        """
        配置类，用于 MotionLLM 模型，整合 LLM 和 VQ-VAE 的参数。

        参数:
            llm_backbone (str): LLM 主干模型的标识符（例如 "gpt2"）。
            w_flash_attention (bool): 是否为 LLM 使用 flash attention。
            nb_code (int): VQ-VAE 码本条目数。
            code_dim (int): VQ-VAE 码本嵌入的维度。
            output_emb_width (int): VQ-VAE 输出嵌入的宽度。
            down_t (int): VQ-VAE 的时间下采样因子。
            stride_t (int): VQ-VAE 的时间步幅。
            width (int): VQ-VAE 网络的宽度。
            depth (int): VQ-VAE 网络的深度。
            dilation_growth_rate (int): VQ-VAE 卷积的膨胀增长率。
            vq_act (str): VQ-VAE 的激活函数（例如 "relu"）。
            vq_norm (Optional[str]): VQ-VAE 的归一化类型（例如 "layer_norm", None）。
            nb_joints (int): 动作数据的关节数。
            dataname (str): 数据集名称（例如 "t2m"）。
            vq_path (str): 预训练 VQ-VAE 检查点的路径。
            mean_path (str): 动作数据的均值和标准差文件路径。
            lora_r_t2m (int): 文本到动作任务的 LoRA 秩。
            lora_alpha_t2m (int): 文本到动作任务的 LoRA alpha。
            lora_r_m2t (int): 动作到文本任务的 LoRA 秩。
            lora_alpha_m2t (int): 动作到文本任务的 LoRA alpha。
            lora_dropout (float): LoRA 层的丢弃率。
            lora_target_modules (List[str]): LoRA 适配的目标模块列表。
            first_lora (str): 第一个 LoRA 适配器的名称（例如 "shared"）。
            need_normalize (bool): 是否需要归一化动作数据。
            eval_task (str): 评估任务的类型（例如 "motion_generation"）。
            context_style (str): 提示的上下文风格（例如 "qwen" 或 "default"）。
            prompt_w_response (bool): 是否在提示中包含响应标记。
            generation_mode (str): 生成模式（例如 "motion_generation_direct"）。
            device (str): 计算设备（例如 "cuda" 或 "cpu"）。
            version (str): 配置版本号。
            **kwargs: 传递给父类 PretrainedConfig 的额外参数。
        """
        # 验证 llm_backbone 并加载 llm_config
        try:
            self.llm_config = AutoConfig.from_pretrained(llm_backbone)
        except Exception as e:
            raise ValueError(f"无效的 llm_backbone: {llm_backbone}。错误: {str(e)}")

        # 验证可选参数
        if vq_norm is not None and vq_norm not in ["layer_norm", "batch_norm"]:
            raise ValueError(f"vq_norm 必须是 ['layer_norm', 'batch_norm', None] 中的一个，得到 {vq_norm}")
        if lora_dropout < 0 or lora_dropout > 1:
            raise ValueError(f"lora_dropout 必须在 [0, 1] 范围内，得到 {lora_dropout}")
        if generation_mode not in [
            "direct",
            "think_w_analysis",
            "think_w_analysis_multi_round_gen",
            "think_w_analysis_multi_round_gen_v2",
            "think_w_analysis_multi_round_gen_v3"
        ]:
            raise ValueError(f"generation_mode 必须是 ['direct', 'think_w_analysis', "
                           f"'think_w_analysis_multi_round_gen'] 中的一个，得到 {generation_mode}")
        if device not in ["cuda", "cpu"]:
            raise ValueError(f"device 必须是 ['cuda', 'cpu'] 中的一个，得到 {device}")

        # 设置属性
        self.llm_backbone = llm_backbone
        self.llm_checkpoint = llm_backbone
        self.w_flash_attention = w_flash_attention
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t
        self.width = width
        self.depth = depth
        self.dilation_growth_rate = dilation_growth_rate
        self.vq_act = vq_act
        self.vq_norm = vq_norm
        self.nb_joints = nb_joints
        self.dataname = dataname
        self.vq_path = vq_path
        self.mean_path = mean_path
        self.lora_r_t2m = lora_r_t2m
        self.lora_alpha_t2m = lora_alpha_t2m
        self.lora_r_m2t = lora_r_m2t
        self.lora_alpha_m2t = lora_alpha_m2t
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.first_lora = first_lora
        self.need_normalize = need_normalize
        self.eval_task = eval_task
        self.context_style = context_style
        self.prompt_w_response = prompt_w_response
        self.generation_mode = generation_mode
        self.device = device
        self.version = version
        self.full_finetuned = full_finetuned
        self.full_frozen = full_frozen
        self.w_lora = w_lora
        self.activate_token_embeds = activate_token_embeds

        # 明确设置 hidden_size，确保 DeepSpeed 能访问
        self.hidden_size = getattr(self.llm_config, 'hidden_size', getattr(self.llm_config, 'n_embd', None))
        if self.hidden_size is None:
            raise ValueError(f"llm_config 中缺少 hidden_size 或 n_embd 属性，无法满足 DeepSpeed 的要求")

        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典，包含所有参数。

        返回:
            Dict[str, Any]: 包含所有配置参数的字典。
        """
        output = super().to_dict()
        output.update({
            "llm_backbone": self.llm_backbone,
            "llm_checkpoint": self.llm_checkpoint,
            "w_flash_attention": self.w_flash_attention,
            "nb_code": self.nb_code,
            "code_dim": self.code_dim,
            "output_emb_width": self.output_emb_width,
            "down_t": self.down_t,
            "stride_t": self.stride_t,
            "width": self.width,
            "depth": self.depth,
            "dilation_growth_rate": self.dilation_growth_rate,
            "vq_act": self.vq_act,
            "vq_norm": self.vq_norm,
            "nb_joints": self.nb_joints,
            "dataname": self.dataname,
            "vq_path": self.vq_path,
            "mean_path": self.mean_path,
            "lora_r_t2m": self.lora_r_t2m,
            "lora_alpha_t2m": self.lora_alpha_t2m,
            "lora_r_m2t": self.lora_r_m2t,
            "lora_alpha_m2t": self.lora_alpha_m2t,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "first_lora": self.first_lora,
            "need_normalize": self.need_normalize,
            "eval_task": self.eval_task,
            "context_style": self.context_style,
            "prompt_w_response": self.prompt_w_response,
            "generation_mode": self.generation_mode,
            "device": self.device,
            "version": self.version,
            "hidden_size": self.hidden_size,
            "llm_config": self.llm_config.to_dict(),
            "full_finetuned": self.full_finetuned,
            "full_frozen": self.full_frozen,
            'w_lora': self.w_lora,
            "activate_token_embeds": self.activate_token_embeds
        })
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        从预训练模型路径或名称加载配置。

        参数:
            pretrained_model_name_or_path (str): 预训练模型的路径或名称。
            **kwargs: 额外的关键字参数。

        返回:
            MotionLLMConfig: 加载的配置对象。
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        llm_config_dict = config_dict.pop("llm_config", {})
        config_dict["llm_config"] = AutoConfig.from_pretrained(config_dict["llm_backbone"], **llm_config_dict)
        return cls(**config_dict, **kwargs)


class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        # print(f'Load tokenizer and LLM from {self.args.llm_backbone}')
        if self.args.llm_tokenizer == '':
            self.args.llm_tokenizer = self.args.llm_backbone
        print(f'Load LLM from {self.args.llm_backbone}')
        print(f'Load tokenizer from {self.args.llm_tokenizer}')

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_tokenizer)
        # import pdb; pdb.set_trace()
        if hasattr(args, 'w_flash_attention') and args.w_flash_attention:
            print('Attention: flash_attention_2')
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.args.llm_backbone,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        else:
            print('Attention: default')
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.args.llm_backbone,
                torch_dtype=torch.bfloat16,
            )
        # self.llm.to('cpu')
        self.nb_text_tokens = len(self.tokenizer)
        self.mean = torch.from_numpy(np.load(args.mean_path))
        self.std = torch.from_numpy(np.load(args.std_path))
        self.need_normalize = True   # 用来表示在生成的时候是否需要对motion 做normalization
        # self.device = args.device
        self.training_task = None # t2m or m2t for training
        self.with_unified_forward = False
        self.with_cot_forward = False
        self.dual_codebook = False
        self.gt_forcing = False
        self.ignore_incorrect = False
        self.prompt_w_response = False


        self.lora_config_shared = LoraConfig(
            r=self.args.lora_r_t2m,
            lora_alpha=self.args.lora_alpha_t2m,
            target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        ) 
        self.w_lora = True
        if hasattr(args, 'wo_lora') and args.wo_lora:
            self.w_lora = False
        if self.w_lora:           # ⚠️临时去掉lora
            self.llm = get_peft_model(self.llm, self.lora_config_shared, adapter_name='shared')
        
        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = "ckpt/vqvae.pth"
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        # self.net.to(self.device)

        # freeze vqvae
        for param in self.net.parameters():
            param.requires_grad = False
        # activate token embeds
        # for name, param in self.llm.named_parameters():
        #     if 'embed' in name.lower() or 'lm_head' in name.lower():
        #         param.requires_grad = True
        
        # import pdb; pdb.set_trace()
        self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
        self.motion_token_indices = np.arange(self.args.nb_code) 
        self.motion_token_indices = len(self.tokenizer) + self.motion_token_indices
        for i in range(self.args.nb_code):
            self.tokenizer.add_tokens([f'<Motion_{i}>'])
        self.llm.resize_token_embeddings(len(self.tokenizer))           # 这个版本以后模型会将token embed 设置为可以训练

        self.eval_task = args.eval_task
        self.use_chat_template = (args.context_style == 'qwen')
        self.prompt_w_response = args.prompt_w_response
        # self.llm.eval()

        self.training_task = 'multitask_shared'
        self.generation_mode = args.generation_mode
        self.llm_checkpoint = args.llm_ckpt

        # 初始化配置
        self.config = MotionLLMConfig(
            llm_backbone=args.llm_backbone,
            llm_checkpoint=args.llm_ckpt,
            w_flash_attention=getattr(args, 'w_flash_attention', False),
            nb_code=args.nb_code,
            code_dim=args.code_dim,
            output_emb_width=args.output_emb_width,
            down_t=args.down_t,
            stride_t=args.stride_t,
            width=args.width,
            depth=args.depth,
            dilation_growth_rate=args.dilation_growth_rate,
            vq_act=args.vq_act,
            vq_norm=args.vq_norm,
            nb_joints=args.nb_joints,
            dataname=args.dataname,
            vq_path=args.vq_path,
            lora_r_t2m=args.lora_r_t2m,
            lora_alpha_t2m=args.lora_alpha_t2m,
            lora_r_m2t=args.lora_r_m2t,
            lora_alpha_m2t=args.lora_alpha_m2t,
            lora_dropout=args.lora_dropout,
            first_lora=getattr(args, 'first_lora', 'shared'),
            full_finetuned=args.full_finetuned,
            full_frozen=args.full_frozen,
            w_lora=self.w_lora,
            activate_token_embeds=args.activate_token_embeds,
            generation_mode=args.generation_mode,
        )
        # Load_LLM_checkpoint
        if self.llm_checkpoint is not None and self.llm_checkpoint != "":
            _ = self.load_model(self.llm_checkpoint, verbose=False)

    
    def activate_token_embeds(self):
        """
        Activate the token embeddings to make them trainable.
        """
        for name, param in self.llm.named_parameters():
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                param.requires_grad = True
                # param.data[self.nb_text_tokens:].requires_grad = True


    def activate_new_token_embeds(self):
        """
        Activate the new token embeddings to make them trainable.
        """
        embedding_weight = self.llm.get_input_embeddings().weight
        embedding_weight.requires_grad = True

        def mask_gradients(grad):
            mask = torch.zeros_like(grad)
            mask[self.nb_text_tokens:] = 1.0
            return grad * mask

        embedding_weight.register_hook(mask_gradients)

    def activate_full_model(self):
        if self.w_lora:
            self.llm = self.llm.merge_and_unload()
        self.llm.to(dtype=torch.bfloat16)
        for name, param in self.llm.named_parameters():
            param.requires_grad = True
        self.w_lora = False


    def forward(self, data, return_detailed_acc=False):
        if self.w_lora:
            self.llm.set_adapter('shared')

        """
            data = {
                "input_ids": 
                "targets":
                "attention_mask"
            }
        """
        input_ids_unified = data['input_ids']
        attention_mask_unified = data['attention_mask']
        targets_unified = data['targets']


        outputs = self.llm(
            input_ids=input_ids_unified,
            attention_mask=attention_mask_unified,
            return_dict=True,
            output_hidden_states=True,
            labels=targets_unified,
        )
        return outputs
    
    def forward_multitask(self, caption, motion):
        """
        Forward pass for multitask training where both adapters are active
        """
        # In multitask mode, both adapters are active
        inputs_ids_t2m, targets_t2m, attention_mask_t2m = process_batch(
            tokenizer=self.tokenizer, 
            batch_of_captions=caption, 
            max_tgt_len=200, 
            batch_of_motions=motion,
            training_task='t2m')
        
        inputs_ids_m2t, targets_m2t, attention_mask_m2t = process_batch(
            tokenizer=self.tokenizer, 
            batch_of_captions=caption, 
            max_tgt_len=200, 
            batch_of_motions=motion,
            training_task='m2t')
        
        # Process both tasks
        inputs_ids_t2m = inputs_ids_t2m.to(self.device)
        attention_mask_t2m = attention_mask_t2m.to(self.device)
        targets_t2m = targets_t2m.to(self.device)
        
        inputs_ids_m2t = inputs_ids_m2t.to(self.device)
        attention_mask_m2t = attention_mask_m2t.to(self.device)
        targets_m2t = targets_m2t.to(self.device)

        # Forward for T2M
        self.llm.set_adapter('t2m')
        outputs_t2m = self.llm(
            input_ids=inputs_ids_t2m,
            attention_mask=attention_mask_t2m,
            return_dict=True,
            output_hidden_states=True,
            labels=targets_t2m,
        )
        loss_t2m = outputs_t2m.loss
        
        # Forward for M2T
        self.llm.set_adapter('m2t')
        outputs_m2t = self.llm(
            input_ids=inputs_ids_m2t,
            attention_mask=attention_mask_m2t,
            return_dict=True,
            output_hidden_states=True,
            labels=targets_m2t,
        )
        loss_m2t = outputs_m2t.loss
        
        # Combined loss
        loss = (loss_t2m + loss_m2t) / 2
        
        # Calculate accuracy for one of the tasks (T2M)
        chosen_tokens = torch.max(outputs_t2m.logits, dim=-1)[1][:, 1:-1]
        labels = targets_t2m[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        
        return loss, gen_acc, chosen_tokens, labels


    def forward_cot(self, data, skip_forward=False, return_detailed_acc=False):
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('t2m')

        inputs_ids_unified, targets_unified, attention_mask_unified = process_cot_batch(
            data, tokenizer=self.tokenizer, max_tgt_len=1024, gt_forcing=self.gt_forcing, ignore_incorrect=self.ignore_incorrect)

        if skip_forward:
            return None

        inputs_ids_unified = inputs_ids_unified.to(self.device)
        attention_mask_unified = attention_mask_unified.to(self.device)
        targets_unified = targets_unified.to(self.device)

        outputs_unified = self.llm(
            input_ids=inputs_ids_unified,
            attention_mask=attention_mask_unified,
            return_dict=True,
            output_hidden_states=True,
            labels=targets_unified,
        )
        if not self.dual_codebook:
            loss = outputs_unified.loss
        else:
            loss = None

        # Calculate accuracy
        chosen_tokens = torch.max(outputs_unified.logits, dim=-1)[1][:, 1:-1]
        labels = targets_unified[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        motion_token_mask = (labels >= self.tokenizer.vocab_size).reshape(-1) & valid_mask
        valid_motion_tokens = valid_tokens & motion_token_mask
        motion_gen_acc = valid_motion_tokens.sum().item() / (motion_token_mask.sum().item() + 1.0)
        
        text_token_mask = (labels < self.tokenizer.vocab_size).reshape(-1) & valid_mask
        valid_text_tokens = valid_tokens & text_token_mask
        text_gen_acc = valid_text_tokens.sum().item() / (text_token_mask.sum().item() + 1.0)
        if return_detailed_acc:
            return loss, gen_acc, motion_gen_acc, text_gen_acc, chosen_tokens, labels
        return loss, gen_acc, chosen_tokens, labels

    def generate(self, caption):
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('t2m')
        self.llm.eval()
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        input_text = '### Input:\n' + caption + '\n\nResponse: <Motion>'
        input = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2, 
            early_stopping=True, 
            return_dict_in_generate=True, 
            output_scores=True,
            do_sample=True,
        )

        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]
        motion_logits = best_beam_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        
        # print(motion_tokens)
        return motion_tokens
    def get_gen_motions_from_logit_scores(self, logit_scores, mode='last'):
        # 这个函数的目的是从completions中获取所有生成的motions
        motion_tag_st = self.tokenizer.encode('<Motion>', add_special_tokens=False)
        motion_tag_ed = self.tokenizer.encode('</Motion>', add_special_tokens=False)

        tokens = torch.argmax(logit_scores, dim=-1)
        tokens_list = tokens.tolist()

        def find_token_sequence(self, tokens_list, target_sequence):
            """Find the starting index of a target sequence in tokens_list."""
            all_valid_ids = []
            for i in range(len(tokens_list) - len(target_sequence) + 1):
                if tokens_list[i:i+len(target_sequence)] == target_sequence:
                    all_valid_ids.append(i)
            return all_valid_ids
        
        all_st_locations = find_token_sequence(tokens_list, motion_tag_st)
        all_ed_locations = find_token_sequence(tokens_list, motion_tag_ed)

        if mode == 'last_long':
            st_idx = all_st_locations[-1]
            ed_idx = all_ed_locations[-1]
        elif mode == 'first_short':
            st_idx = all_st_locations[0]
            ed_idx = -1
            for idx in all_ed_locations:
                if idx > st_idx:
                    ed_idx = idx
                    break
        elif mode == 'longest':
            st_idx = all_st_locations[0]
            ed_idx = all_ed_locations[-1]
    
    def extract_answer_ids(self, best_beam_ids, tokenizer, return_position=False, return_prefix_scores=False):
        """Extracts scores between <answer> and </answer> tags from best_beam_scores, handling multi-token tags.
        Returns original best_beam_scores if either tag is not found."""
        # TODO
        return 

    def extract_answer_scores(self, best_beam_scores, tokenizer, return_position=False, return_prefix_scores=False):
        """Extracts scores between <answer> and </answer> tags from best_beam_scores, handling multi-token tags.
        Returns original best_beam_scores if either tag is not found."""
        # Convert scores to tokens
        tokens = torch.argmax(best_beam_scores, dim=-1)
        tokens_list = tokens.tolist()
        
        # Get token IDs for <answer> and </answer>, which may be multiple tokens
        # 注意: <answer>不一定是独立的，所以要考虑另一个可能的形式: ><answer>
        answer_end_tokens = self.tokenizer.encode('</answer>', add_special_tokens=False)
        
        def find_token_sequence(tokens_list, target_sequence):
            """Find the starting index of a target sequence in tokens_list."""
            for i in range(len(tokens_list) - len(target_sequence) + 1):
                if tokens_list[i:i+len(target_sequence)] == target_sequence:
                    return i
            return -1

        # Find start and end indices of the tags
        answer_start_tokens_1 = self.tokenizer.encode('<answer>', add_special_tokens=False)
        start_idx_1 = find_token_sequence(tokens_list, answer_start_tokens_1)
        answer_start_tokens_2 = self.tokenizer.encode('><answer>', add_special_tokens=False)
        start_idx_2 = find_token_sequence(tokens_list, answer_start_tokens_2)
        if start_idx_1 == -1:
            if start_idx_2 == -1:
                if return_position:
                    if return_prefix_scores:
                        return best_beam_scores, best_beam_scores, 0, -1
                    return best_beam_scores, 0, -1
                return best_beam_scores  # Return original scores if <answer> not found
            else:
                start_idx = start_idx_2
                answer_start_tokens = answer_start_tokens_2
        else:
            start_idx = start_idx_1
            answer_start_tokens = answer_start_tokens_1

        # Look for end tag after the start tag
        end_idx = find_token_sequence(tokens_list[start_idx + len(answer_start_tokens):], answer_end_tokens)
        if end_idx == -1:
            if return_position:
                if return_prefix_scores:
                    return best_beam_scores, best_beam_scores, start_idx + len(answer_start_tokens), -1
                return best_beam_scores, start_idx + len(answer_start_tokens), -1
            return best_beam_scores  # Return original scores if </answer> not found
        
        # Adjust end_idx to account for the offset
        end_idx += start_idx + len(answer_start_tokens)
        
        # Extract scores between the tags (excluding the tags themselves)
        answer_scores = best_beam_scores[start_idx + len(answer_start_tokens):end_idx]
        prefix_scores = best_beam_scores[:start_idx + len(answer_start_tokens)]
        if return_position:
            if return_prefix_scores:
                return answer_scores, prefix_scores, start_idx+len(answer_start_tokens), end_idx
            return answer_scores, start_idx+len(answer_start_tokens), end_idx
        return answer_scores

    def get_motion_from_scores(self, answer_scores):
        motion_logits = answer_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # 因为CoT输出包含<Motion> tag，所以需要把第一位的token 去掉
        if 0 in motion_tokens:
            motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        return motion_tokens

    def get_motion_from_scores_w_ids(self, answer_scores, answer_ids=None):
        def in_motion_id_range(idx):
            motion_ids_range = [len(self.tokenizer)-(self.args.nb_code+2), len(self.tokenizer)] # 左闭右开区间
            return idx >= motion_ids_range[0] and idx < motion_ids_range[1]
        
        motion_logits = answer_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)

        max_motion_ids = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # 如果原本的completion id 处于 motion_id_range内，,则直接使用原本的id，否则才使用logits  
        motion_tokens = torch.tensor([idx.item()-(len(self.tokenizer)-(self.args.nb_code+2)) if in_motion_id_range(idx) else max_motion_ids[i].item() for i, idx in enumerate(answer_ids) ]).to(answer_scores.device)
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # 因为CoT输出包含<Motion> tag，所以需要把第一位的token 去掉
        if 0 in motion_tokens:
            motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        return motion_tokens 

    def get_motion_from_ids(self, answer_ids):
        def in_motion_id_range(idx):
            motion_ids_range = [len(self.tokenizer)-(self.args.nb_code+2), len(self.tokenizer)] # 左闭右开区间
            return idx >= motion_ids_range[0] and idx < motion_ids_range[1]
        
        # motion_logits = answer_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)

        # max_motion_ids = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # 如果原本的completion id 处于 motion_id_range内，,则直接使用原本的id，否则才使用logits  
        motion_tokens = torch.tensor([idx.item()-(len(self.tokenizer)-(self.args.nb_code+2))  for i, idx in enumerate(answer_ids) if in_motion_id_range(idx)]).to(answer_ids.device)
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # 因为CoT输出包含<Motion> tag，所以需要把第一位的token 去掉
        if 0 in motion_tokens:
            motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        return motion_tokens

    def extract_motions_from_scores_w_ids(self, answer_ids):
        # 从response 中找到所有motion
        def in_motion_id_range(idx):
            motion_ids_range = [len(self.tokenizer)-(self.args.nb_code+2), len(self.tokenizer)] # 左闭右开区间
            return idx >= motion_ids_range[0] and idx < motion_ids_range[1]

        # 找到所有 <Motion> token 所在的位置
        mo_st_tag_id = self.tokenizer.encode('<Motion>', add_special_tokens=False)
        mo_st_tag_pos_list = [i for i, idx in enumerate(answer_ids) if idx in mo_st_tag_id]

        # 找到所有 </Motion> token 所在的位置
        mo_ed_tag_id = self.tokenizer.encode('</Motion>', add_special_tokens=False)
        mo_ed_tag_pos_list = [i for i, idx in enumerate(answer_ids) if idx in mo_ed_tag_id]

        # Motion匹配，原则是最短匹配
        all_mo_pos = []
        i, j = 0, 0 # 设计一个双指针，遍历匹配 <Motion> 和 </Motion> position_list
        last_mo = None # 用这个变量来记录上一次匹配的结果，用来避免重复匹配

        while True:
            if i >= len(mo_st_tag_pos_list) or j >= len(mo_ed_tag_pos_list):
                break
            mo_st_tag_pos = mo_st_tag_pos_list[i]
            mo_ed_tag_pos = mo_ed_tag_pos_list[j]

            if mo_st_tag_pos < mo_ed_tag_pos:       # <Motion> 出现在 </Motion> 之前
                if last_mo is not None and mo_st_tag_pos > last_mo[0] and mo_st_tag_pos < last_mo[1]:  
                    # 如果出现 <M> <M> </M> 这样的情况, 取现在这个更短的motion
                    all_mo_pos[-1] = [mo_st_tag_pos, mo_ed_tag_pos]
                    last_mo = [mo_st_tag_pos, mo_ed_tag_pos]
                all_mo_pos.append([mo_st_tag_pos, mo_ed_tag_pos])
                i += 1
            else:
                j += 1
        if all_mo_pos.__len__() == 0:
            return [[0, -1]]
        # motion_ids = [answer_ids[mo_st_tag_pos[i]:mo_ed_tag_pos[i]] for i in range(len(mo_st_tag_pos))]
        return all_mo_pos

    def generate_cot(self, caption, return_dict = False):
        # 基本上跟原始的generate 函数一直，但是要改成unified task对应的prompt
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('t2m')

        self.llm.eval()

        system_prompt_dict = {
            'direct': "You are an assistant who helps users generate 3D human motion representations. The users begin by describing the motion they envision. You need to generate a 3D human motion based on the user's description.\n\n",
            'think_w_analysis': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, your job is to (1) first reason with the given prompt and provide clear description of the reasoning step that identifies the key elements related to the prompt. Show your reasoning inside <think> </think>; (2) output motion in <Motion> </Motion> tags based on your reasoning process.\n\n",
            'think_w_analysis_multi_round_gen': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, and your job includes the following parts: (1) analyzing the input prompt and clearly describing the reasoning process to identify key elements; (2) performing multiple rounds of generation and self-assessment until a satisfactory motion is achieved; (3) outputting the final motion based on the reasoning process.\n Response format: Show initial analysis and generation-assessment cycles indide <think> and </think> tags; Place the final motion inside <answer> and </answer> tags.\n\n"
        }
        prompt = system_prompt_dict[self.generation_mode]
        instruction = "User: " + caption
        input_text = '\n\n'
        if self.prompt_w_response:
            input_text += " Response:"
        input = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids, 
            max_length=1024, 
            num_beams=2, 
            early_stopping=True, 
            return_dict_in_generate=True, 
            output_scores=True,
            use_cache=True,
        )
        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]        # 这里是所有的预测结果，torch.argmax(best_beam_scores, dim=-1)可以获取输出的logits
        # 获取<answer> </answer> 内的回答
        # import pdb; pdb.set_trace()

        def recover_sequence_from_beam_search(outputs, input_length):
            """
            使用outputs.scores和outputs.beam_indices恢复生成的序列。
            """
            # 获取基本参数
            num_beams = outputs.beam_indices.shape[0] # batch_size * num_beams
            sequence_length = outputs.sequences.shape[1]
            new_tokens_length = sequence_length - input_length
            
            # 初始化一个张量来存储恢复的token IDs
            recovered_tokens = torch.zeros((num_beams, sequence_length), dtype=torch.long)
            
            # 前input_length个token就是输入本身（对所有beam都一样）
            # 这里我们只恢复第一个beam（index 0）作为演示
            beam_idx = 0
            recovered_tokens[beam_idx, :input_length] = outputs.sequences[beam_idx, :input_length]
            
            # 关键：使用beam_indices来回溯路径
            current_beam = outputs.beam_indices[beam_idx, -1].item() # 最后一步选择的beam
            
            # 逆向回溯：从最后一个生成token回溯到第一个
            for step in range(new_tokens_length - 1, -1, -1): # 从后往前
                # 获取在step时刻，current_beam这个beam的分数
                scores_at_step = outputs.scores[step]
                
                # 找到这个beam在该时间步实际选择的token ID
                # beam_indices的shape为 (batch_size * num_return_sequences, max_new_tokens)
                # 我们需要找到在step时间步，是哪个beam被选中了
                # 对于当前序列，它在step+1时间步选择的beam索引存储在beam_indices[beam_idx, step]
                # 但更直接的方法是：当前时间步的token就是sequences中对应位置的token
                token_id = outputs.sequences[beam_idx, input_length + step].item()
                recovered_tokens[beam_idx, input_length + step] = token_id
                
                # 回溯到上一步：找出是哪个parent beam产生了当前beam
                if step > 0:
                    current_beam = outputs.beam_indices[beam_idx, step - 1].item()
            
            return recovered_tokens
        # import pdb; pdb.set_trace()
        input_length = input_ids.shape[1]
        recovered_sequences = recover_sequence_from_beam_search(outputs, input_length)

        answer_scores = self.extract_answer_scores(best_beam_scores, self.tokenizer)




        motion_logits = answer_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # 因为CoT输出包含<Motion> tag，所以需要把第一位的token 去掉
        if 0 in motion_tokens:
            motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        # print(motion_tokens)
        if return_dict:
            response_dict = {
                'motion_tokens': motion_tokens,
                'input_text': input,
                'best_beam_scores': best_beam_scores,
                'answer_scores': answer_scores,
                'best_beam_text': self.tokenizer.decode(torch.argmax(best_beam_scores, dim=-1))
            }
            return response_dict
        return motion_tokens

    def generate_cot_batched(self, captions, return_dict=False, do_sample=False, do_beam_search=False, early_stopping=False):
        # Set the appropriate adapter based on training task
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            pass

        self.llm.eval()

        if do_beam_search:
            return self.generate_cot_batched_beam_search(captions, return_dict, early_stopping=early_stopping)

        # System prompt definitions
        system_prompt_dict = {
            'direct': "You are an assistant who helps users generate 3D human motion representations. The users begin by describing the motion they envision. You need to generate a 3D human motion based on the user's description.\n\n",
            'think_w_analysis': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, your job is to (1) first reason with the given prompt and provide clear description of the reasoning step that identifies the key elements related to the prompt. Show your reasoning inside <think> </think>; (2) output motion in <Motion> </Motion> tags based on your reasoning process.\n\n",
            'think_w_analysis_multi_round_gen': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, and your job includes the following parts: (1) analyzing the input prompt and clearly describing the reasoning process to identify key elements; (2) performing multiple rounds of generation and self-assessment until a satisfactory motion is achieved; (3) outputting the final motion based on the reasoning process.\n Response format: Show initial analysis and generation-assessment cycles indide <think> and </think> tags; Place the final motion inside <answer> and </answer> tags.\n\n"
        }
        prompt = system_prompt_dict[self.generation_mode]

        # Prepare batch inputs
        batch_inputs = []
        for caption in captions:
            instruction = "User: " + caption
            input_text = '\n\n'
            if self.prompt_w_response:
                input_text += " Response:"
            batch_inputs.append(prompt + instruction + input_text)

        # Tokenize batch input
        input_ids = self.tokenizer(batch_inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate outputs for the batch
        outputs = self.llm.generate(
            input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_length=1024,
            # num_beams=2,
            do_sample=do_sample,
            # early_stopping=early_stopping,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

        # Initialize result lists
        result = []
        batch_size = len(captions)

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract generated token IDs for the i-th sample (best beam)
            generated_ids = outputs.sequences[i, input_ids.input_ids.shape[1]:]  # Skip input tokens
            # Find the index of the first <eos> token
            eos_index = (generated_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_index) > 0:
                eos_index = eos_index[0].item() + 1  # Include the <eos> token
                generated_ids = generated_ids[:eos_index]
            else:
                eos_index = len(generated_ids)  # Use full sequence if no <eos> found

            # Extract scores for the i-th sample, truncated to eos_index
            scores = torch.stack(outputs.scores)[:eos_index, i, :]  # [num_generated_tokens, vocab_size]

            # Extract answer scores (assuming extract_answer_scores handles single sample)
            answer_scores = self.extract_answer_scores(scores, self.tokenizer)

            # Extract motion logits and tokens
            motion_logits = answer_scores[:, -(self.args.nb_code+2):]
            motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]

            # Remove end_of_motion token (index=1) if present
            if 1 in motion_tokens:
                motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
            # Remove <Motion> tag token (index=0) if present
            if 0 in motion_tokens:
                motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
            # Adjust for special tokens
            motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # Remove first two special tokens

            # Prepare output based on return_dict flag
            if return_dict:
                response_dict = {
                    'motion_tokens': motion_tokens,
                    'input_text': batch_inputs[i],
                    'best_beam_scores': scores,
                    'answer_scores': answer_scores,
                    'best_beam_text': self.tokenizer.decode(generated_ids)
                }
                result.append(response_dict)
            else:
                result.append(motion_tokens)

        return result

    def generate_cot_batched_beam_search(self, captions, return_dict=False, early_stopping=False):
        # import pdb; pdb.set_trace()
        # Set the appropriate adapter based on training task
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('t2m')

        self.llm.eval()

        # Beam search configuration
        num_beams = 2

        # System prompt definitions
        system_prompt_dict = {
            'direct': "You are an assistant who helps users generate 3D human motion representations. The users begin by describing the motion they envision. You need to generate a 3D human motion based on the user's description.\n\n",
            'think_w_analysis': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, your job is to (1) first reason with the given prompt and provide clear description of the reasoning step that identifies the key elements related to the prompt. Show your reasoning inside <think> </think>; (2) output motion in <Motion> </Motion> tags based on your reasoning process.\n\n",
            'think_w_analysis_multi_round_gen': "You are an assistant who helps users generate 3D human motion representations. The users will describe a motion, and your job includes the following parts: (1) analyzing the input prompt and clearly describing the reasoning process to identify key elements; (2) performing multiple rounds of generation and self-assessment until a satisfactory motion is achieved; (3) outputting the final motion based on the reasoning process.\n Response format: Show initial analysis and generation-assessment cycles indide <think> and </think> tags; Place the final motion inside <answer> and </answer> tags.\n\n"
        }
        prompt = system_prompt_dict[self.generation_mode]

        # Prepare batch inputs
        batch_inputs = []
        for caption in captions:
            instruction = "User: " + caption
            input_text = '\n\n'
            if self.prompt_w_response:
                input_text += " Response:"
            batch_inputs.append(prompt + instruction + input_text)

        # Tokenize batch input
        input_ids = self.tokenizer(batch_inputs, return_tensors="pt", padding=True).to(self.device)

        # Generate outputs for the batch
        outputs = self.llm.generate(
            input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_length=1024,
            num_beams=num_beams,
            num_return_sequences=1,
            early_stopping=early_stopping,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

        # Initialize result lists
        result = []
        batch_size = len(captions)

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract generated token IDs for the i-th sample (best beam)
            generated_ids = outputs.sequences[i, input_ids.input_ids.shape[1]:]  # Skip input tokens
            # Find the index of the first <eos> token
            eos_index = (generated_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_index) > 0:
                eos_index = eos_index[0].item() + 1  # Include the <eos> token
                generated_ids = generated_ids[:eos_index]
            else:
                eos_index = len(generated_ids)  # Use full sequence if no <eos> found

            # Extract scores for the i-th sample, using the first beam as approximation for the best
            beam_idx = i * num_beams
            scores = torch.stack(outputs.scores)[:eos_index, beam_idx, :]  # [num_generated_tokens, vocab_size]

            # Extract answer scores (assuming extract_answer_scores handles single sample)
            answer_scores = self.extract_answer_scores(scores, self.tokenizer)

            # Extract motion logits and tokens
            motion_logits = answer_scores[:, -(self.args.nb_code+2):]
            motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]

            # Remove end_of_motion token (index=1) if present
            if 1 in motion_tokens:
                motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
            # Remove <Motion> tag token (index=0) if present
            if 0 in motion_tokens:
                motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
            # Adjust for special tokens
            motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # Remove first two special tokens

            # Prepare output based on return_dict flag
            if return_dict:
                response_dict = {
                    'motion_tokens': motion_tokens,
                    'input_text': batch_inputs[i],
                    'best_beam_scores': scores,
                    'answer_scores': answer_scores,
                    'best_beam_text': self.tokenizer.decode(generated_ids)
                }
                result.append(response_dict)
            else:
                result.append(motion_tokens)

        return result

    def post_process_final_motion_output(self, outputs, input_ids, batch_size, return_dict=False):
        result = []
        batch_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # Process each sample in the batch
        for i in range(batch_size):
            # Extract generated token IDs for the i-th sample (best beam)
            generated_ids = outputs.sequences[i, input_ids.shape[1]:]  # Skip input tokens
            # Find the index of the first <eos> token
            eos_index = (generated_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_index) > 0:
                eos_index = eos_index[0].item() + 1  # Include the <eos> token
                generated_ids = generated_ids[:eos_index]
            else:
                eos_index = len(generated_ids)  # Use full sequence if no <eos> found

            # Extract scores for the i-th sample, truncated to eos_index
            scores = torch.stack(outputs.scores)[:eos_index, i, :]  # [num_generated_tokens, vocab_size]

            # Extract answer scores (assuming extract_answer_scores handles single sample)
            answer_scores = self.extract_answer_scores(scores, self.tokenizer)

            # Extract motion logits and tokens
            motion_logits = answer_scores[:, -(self.args.nb_code+2):]
            motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]

            # Remove end_of_motion token (index=1) if present
            if 1 in motion_tokens:
                motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
            # Remove <Motion> tag token (index=0) if present
            if 0 in motion_tokens:
                motion_tokens = motion_tokens[motion_tokens.tolist().index(0)+1:]
            # Adjust for special tokens
            motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # Remove first two special tokens

            # Prepare output based on return_dict flag
            if return_dict:
                response_dict = {
                    'motion_tokens': motion_tokens,
                    'input_text': batch_inputs[i],
                    'best_beam_scores': scores,
                    'answer_scores': answer_scores,
                    'best_beam_text': self.tokenizer.decode(generated_ids)
                }
                result.append(response_dict)
            else:
                result.append(motion_tokens)

        return result



    def generate_unified(self, caption):
        # 基本上跟原始的generate 函数一直，但是要改成unified task对应的prompt
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('t2m')

        self.llm.eval()
        # prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        # instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
        # input_text = '### Input:\n' + caption + '\n\nResponse: <Motion>'
        prompt = "Below is an description of a task togeter with the inputs that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "Generate a motion that conveys the sentiment of " + caption
        input_text = '\n\n Response:'

        input = prompt + instruction + input_text
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2, 
            early_stopping=True, 
            return_dict_in_generate=True, 
            output_scores=True
        )

        scores = torch.stack(outputs.scores)  # [num_generated_tokens, num_beams, vocab_size]
        # print(scores.shape)
        # Take only the best beam (beam 0)
        best_beam_scores = scores[:, 0, :]  # [num_generated_tokens, vocab_size]
        motion_logits = best_beam_scores[:, -(self.args.nb_code+2):]
        # print(motion_logits.shape)
        motion_tokens = torch.argmax(motion_logits, dim=-1)  # [num_generated_tokens]
        # print(motion_tokens)
        # Remove end_of_motion token (index=1) if present
        if 1 in motion_tokens:
            motion_tokens = motion_tokens[:motion_tokens.tolist().index(1)]
        # Ensure tokens don't go below 0 when adjusting for special tokens
        motion_tokens = torch.clamp(motion_tokens - 2, min=0)  # remove the first two special tokens while preventing negative values
        
        # print(motion_tokens)
        return motion_tokens
    
    def caption_unified(self, motion):
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('m2t')
        self.llm.eval()
        if self.need_normalize:
            motion = self.normalize(motion)
        # 修复：不强制转换为float，让数据类型与模型保持一致
        motion = torch.from_numpy(motion).to(self.device).unsqueeze(0)
        # 确保数据类型与网络权重一致
        motion = motion.to(next(self.net.parameters()).dtype)
        motion_tokens = self.net.encode(motion).squeeze(0)
        motion_tokens = motion_tokens + self.nb_text_tokens + 2 # reindex the motion tokens
        # print(motion_tokens)

        prompt = "Below is an description of a task togeter with the inputs that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "Describe the motion represented by " + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + " using plain English"
        input_text = '\n\n Response:'


        # prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        # instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
        # input_text = '### Input:\n' + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + '\n\nResponse: '
        input_texts = prompt + instruction + input_text
        # print(input_texts)
        input_ids = self.tokenizer.encode(input_texts, return_tensors="pt").to(self.device)
        pred = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2
        )
        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred)
        caption = pred.split('<eos>')[0]

        return caption

    def caption(self, motion):
        if self.training_task == 'multitask_shared':
            self.llm.set_adapter('shared')
        elif self.training_task == 't2m':
            self.llm.set_adapter('t2m')
        elif self.training_task == 'm2t':
            self.llm.set_adapter('m2t')
        else:
            self.llm.set_adapter('m2t')
        self.llm.eval()
        if self.need_normalize:
            motion = self.normalize(motion)
        # 修复：不强制转换为float，让数据类型与模型保持一致
        motion = torch.from_numpy(motion).to(self.device).unsqueeze(0)
        # 确保数据类型与网络权重一致
        motion = motion.to(next(self.net.parameters()).dtype)
        motion_tokens = self.net.encode(motion).squeeze(0)
        motion_tokens = motion_tokens + self.nb_text_tokens + 2 # reindex the motion tokens
        # print(motion_tokens)

        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        instruction = "### Instruction:\nGenerate a caption matching the following input human motion token sequence.\n\n"
        input_text = '### Input:\n' + "<Motion>" + self.tokenizer.decode(motion_tokens) + '</Motion>' + '\n\nResponse: '
        input_texts = prompt + instruction + input_text
        # print(input_texts)
        input_ids = self.tokenizer.encode(input_texts, return_tensors="pt").to(self.device)
        pred = self.llm.generate(
            input_ids, 
            max_length=200, 
            num_beams=2
        )
        pred = pred[0, len(input_ids[0]):]
        pred = self.tokenizer.decode(pred)
        caption = pred.split('<eos>')[0]

        return caption
    
    def save_model(self, path, save_full_model=False):
        # only save the lora weights of the model
        save_dict = {}
        for name, param in self.llm.named_parameters():
            if save_full_model:
                save_dict[name] = param
            elif 'lora' in name:
                save_dict[name] = param
            if not self.w_lora:     # 如果不是lora训练，那么将所有参数都存下来
                save_dict[name] = param

        # save the additional token embeddings
        embeddings = self.llm.get_input_embeddings().weight       # 修改：存下整个embedding
        save_dict['embeddings'] = embeddings

        # save the lm_head of the additional tokens
        # lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        lm_head = self.llm.lm_head.weight       # 修改：存下整个embedding
        save_dict['lm_head'] = lm_head

        torch.save(save_dict, path)

    def save_model_for_resume(self, path, resume_info):
        # only save the lora weights of the model
        save_dict = {}
        for name, param in self.llm.named_parameters():
            if 'lora' in name and self.w_lora:
                save_dict[name] = param
            if not self.w_lora:     # 如果不是lora训练，那么将所有参数都存下来
                save_dict[name] = param

        # save the additional token embeddings
        # embeddings = self.llm.get_input_embeddings().weight[self.nb_text_tokens:]
        embeddings = self.llm.get_input_embeddings().weight       # 修改：存下整个embedding
        save_dict['embeddings'] = embeddings

        # save the lm_head of the additional tokens
        # lm_head = self.llm.lm_head.weight[self.nb_text_tokens:]
        lm_head = self.llm.lm_head.weight       # 修改：存下整个embedding
        save_dict['lm_head'] = lm_head

        ckpt_dict = {}
        for key, value in resume_info.items():
            ckpt_dict[key] = value
        ckpt_dict['state_dict'] = save_dict

        torch.save(ckpt_dict, path)

    def load_model_old(self, path, verbose=False):
        import deepspeed
        print(f"Loading model from {path}")
        
        # 1. 加载 checkpoint，映射到 CPU 以避免设备冲突
        checkpoint_info = torch.load(path, map_location='cpu')
        resume_info = {}
        save_dict = {}
        
        # 2. 处理 checkpoint 格式
        if 'state_dict' in checkpoint_info:
            for key in checkpoint_info:
                if key != 'state_dict':
                    resume_info[key] = checkpoint_info[key]
            save_dict = checkpoint_info['state_dict']
        else:
            save_dict = checkpoint_info

        # 3. 适配可能的 'llm.' 前缀
        state_dict = {}
        for name, param in self.llm.named_parameters():
            if name in save_dict:
                state_dict[name] = save_dict[name]
            elif 'llm.' + name in save_dict:
                state_dict[name] = save_dict['llm.' + name]
            else:
                print(f"Warning: Parameter {name} not found in checkpoint, keeping original value")
        
        # 4. 使用 load_state_dict 加载参数
        self.llm.load_state_dict(state_dict, strict=False)
        
        # 5. 如果使用 DeepSpeed，同步参数到所有 GPU 分区
        if hasattr(self.llm, 'module'):  # 检查是否已初始化为 DeepSpeed 引擎
            torch.distributed.barrier()  # 确保所有进程同步
            deepspeed.utils.broadcast_tensors(self.llm.parameters())  # 广播参数到所有 GPU
            print("Parameters synchronized across all GPUs")
        
        # 6. 记录加载情况
        loaded_names = set(state_dict.keys())
        all_names = set(name for name, _ in self.llm.named_parameters())
        not_loaded_names = all_names - loaded_names
        
        if verbose:
            print(f"Loaded parameters: {loaded_names}")
            print(f"Not loaded parameters: {not_loaded_names}")
        
            return resume_info, loaded_names, not_loaded_names
        return resume_info

    def load_model(self, path, verbose=False):
        print(f"Loading model from {path}")
        checkpoint_info = torch.load(path, map_location=self.llm.device)
        resume_info = {}
        if 'state_dict' in checkpoint_info:
            for key in checkpoint_info:
                if key != 'state_dict':
                    resume_info[key] = checkpoint_info[key]
            save_dict = checkpoint_info['state_dict']
        else:
            save_dict = checkpoint_info
        loaded_names = set()
        not_loaded_names = set()

        for name, param in self.llm.named_parameters():
            # print(name)
            if name in save_dict:
                param.data = save_dict[name]
                loaded_names.add(name)
            elif 'llm.' + name in save_dict:      # 这是为了另一种存储方式：整个模型存下来
                param.data = save_dict['llm.' + name]
                loaded_names.add(name)
            elif name.replace('base_model.model.','').replace('base_layer.', '') in save_dict:
                param.data = save_dict[name.replace('base_model.model.','').replace('base_layer.', '')]
                loaded_names.add(name)
            else:
                not_loaded_names.add(name)
        
        if 'embeddings' in save_dict:
            if len(save_dict['embeddings']) == self.args.nb_code + 2:
                self.llm.get_input_embeddings().weight.data[self.nb_text_tokens:] = save_dict['embeddings']
            else:
                self.llm.get_input_embeddings().weight.data = save_dict['embeddings']
            loaded_names.add('embeddings')
        if 'lm_head' in save_dict:
            if len(save_dict['lm_head']) == self.args.nb_code + 2:
                self.llm.lm_head.weight.data[self.nb_text_tokens:] = save_dict['lm_head']
            else:
                self.llm.lm_head.weight.data = save_dict['lm_head']
            loaded_names.add('lm_head')
        if verbose:
            return resume_info, loaded_names, not_loaded_names

        return resume_info
    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std
    
    def motion_denorm(self, motion):
        mean = self.mean.to(motion.device)
        std = self.std.to(motion.device)

        return motion * std + mean