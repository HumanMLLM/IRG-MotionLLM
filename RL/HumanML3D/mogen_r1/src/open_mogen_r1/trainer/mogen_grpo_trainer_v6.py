# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
from PIL import Image
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
torch._dynamo.config.suppress_errors = True
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from tqdm import tqdm
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from transformers import StoppingCriteria, StoppingCriteriaList
from trl.trainer.grpo_config import GRPOConfig
from trl import ModelConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import re
# from janus.models import MultiModalityCausalLM, VLChatProcessor
# from utils.reward_hps import HPSv2
# from utils.reward_git import GIT
# from utils.reward_gdino import GDino
# from utils.reward_orm import ORM
import shutil
import sys
import traceback
import copy

# from models.mllm import MotionLLM
from models.mllm_single_lora import MotionLLM   # 这个版本最多只会使用一个lora 
from models.tmr_eval_wrapper import EvaluatorModelWrapper_TMR as TMR_Evaluator
from models.motionpatch_eval_wrapper import EvaluatorModelWrapper as MotionPatch_Evaluator
from models.evaluator_wrapper import EvaluatorModelWrapper as Guo_Evaluator
from models.evaluator_wrapper import EvaluatorModelWrapper as Official_Evaluator
import deepspeed
# from options.get_eval_option import get_opt
from accelerate.utils import is_peft_model, set_seed
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from data import dataset_TM_eval
from utils.word_vectorizer import WordVectorizer
from utils.metrics import *


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class MogenR1Trainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        sub_trace_reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_names: list[str] = None,
        reward_weights: list[float] = None,
        sub_trace_reward_names: list[str] = None,
        args: GRPOConfig = None,
        model_args: ModelConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        sub_trace_reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
        eval_task='direct',
        context_mode="think_w_analysis_multi_round_gen",
        grad_ckpt=True,
        reward_norm_before_add=False,
    ):
    
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        self.grad_ckpt = grad_ckpt

        # Models
        # Trained model
        model_args.eval_task = eval_task
        model_args.generation_mode = context_mode
        model = MotionLLM(model_args)

        if model.w_lora:
            model.llm.set_adapter('shared')     # 专门设置用特定的lora
        
        
        # model.llm.config._attn_implementation == "flash_attention_2"

        # try gradient checkpointing
        if self.grad_ckpt:
            model.llm.config.use_cache = False
            model.llm.gradient_checkpointing_enable()
        # import pdb; pdb.set_trace()

        # Reference model
        if is_deepspeed_zero3_enabled() and args.beta != 0:
            self.ref_model = MotionLLM(model_args)
            # if llm_ckpt_path is not None and llm_ckpt_path != "":
            #     if hasattr(self.ref_model, 'module'):
            #         _ = self.ref_model.module.load_model(llm_ckpt_path)
            #     else:
            #         _ = self.ref_model.load_model(llm_ckpt_path)
        elif args.beta != 0:
        #     # If PEFT configuration is not provided, create a reference model based on the initial model.
        #     self.ref_model = create_reference_model(model)
            self.ref_model = MotionLLM(model_args)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # TODO: Processing class

        # NOTE: Reward Functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'tmr' in reward_func:
                reward_funcs[i] = TMR_Evaluator(reward_mode=reward_func, device=torch.device('cpu'))
            elif isinstance(reward_func, str) and 'guo' in reward_func:
                from utils.get_eval_option import get_opt as get_opt_guo
                dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
                wrapper_opt = get_opt_guo(dataset_opt_path, torch.device('cpu'))
                reward_funcs[i] = Guo_Evaluator(wrapper_opt, reward_func)
            elif isinstance(reward_func, str) and 'motionpatch' in reward_func:
                reward_funcs[i] = MotionPatch_Evaluator(reward_mode=reward_func, device=torch.device('cpu'))
            elif isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        self.reward_names = reward_names
        self.reward_weights = reward_weights
        self.reward_norm_before_add = reward_norm_before_add
        # NOTE: Sub-Trace Reward Functions
        if not isinstance(sub_trace_reward_funcs, list):
            sub_trace_reward_funcs = [sub_trace_reward_funcs]
        for i, sub_trace_reward_func in enumerate(sub_trace_reward_funcs):
            if isinstance(sub_trace_reward_func, str) and 'tmr' in sub_trace_reward_func:
                sub_trace_reward_funcs[i] = TMR_Evaluator(reward_mode=sub_trace_reward_func, device=torch.device('cpu'))
            elif isinstance(sub_trace_reward_func, str) and 'guo' in sub_trace_reward_func:
                from utils.get_eval_option import get_opt as get_opt_guo
                dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
                wrapper_opt = get_opt_guo(dataset_opt_path, torch.device('cpu'))
                sub_trace_reward_funcs[i] = Guo_Evaluator(wrapper_opt, sub_trace_reward_func)
            elif isinstance(sub_trace_reward_func, str) and 'motionpatch' in sub_trace_reward_func:
                sub_trace_reward_funcs[i] = MotionPatch_Evaluator(reward_mode=sub_trace_reward_func, device=torch.device('cpu'))
            elif isinstance(sub_trace_reward_func, str):
                sub_trace_reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    sub_trace_reward_func, num_labels=1, **model_init_kwargs
                )
        self.sub_trace_reward_funcs = sub_trace_reward_funcs
        self.sub_trace_reward_names = sub_trace_reward_names

        # Reward processing class
        # NOTE: 我感觉这一步基本可以跳过
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")
        if sub_trace_reward_processing_classes is None:
            sub_trace_reward_processing_classes = [None] * len(sub_trace_reward_funcs)
        elif not isinstance(sub_trace_reward_processing_classes, list):
            sub_trace_reward_processing_classes = [sub_trace_reward_processing_classes]
        else:
            if len(sub_trace_reward_processing_classes) != len(sub_trace_reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        for i, (sub_trace_reward_processing_class, sub_trace_reward_func) in enumerate(zip(sub_trace_reward_processing_classes, sub_trace_reward_funcs)):
            if isinstance(sub_trace_reward_func, PreTrainedModel):
                if sub_trace_reward_processing_class is None:
                    sub_trace_reward_processing_class = AutoTokenizer.from_pretrained(sub_trace_reward_func.config._name_or_path)
                if sub_trace_reward_processing_class.pad_token_id is None:
                    sub_trace_reward_processing_class.pad_token = sub_trace_reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                sub_trace_reward_func.config.pad_token_id = sub_trace_reward_processing_class.pad_token_id
                sub_trace_reward_processing_classes[i] = sub_trace_reward_processing_class
        self.sub_trace_reward_processing_classes = sub_trace_reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = None
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.beta = args.beta
        # import pdb; pdb.set_trace()
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # import pdb; pdb.set_trace()

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # 这里是为了模型用accelerate / deepspeed 封装
        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, TMR_Evaluator):
                reward_func.load_to_device(self.accelerator.device)
                reward_func.accelerator = self.accelerator
                if self.is_deepspeed_enabled:   
                    reward_func.text_encoder = prepare_deepspeed(reward_func.text_encoder, self.accelerator)
                else:
                    reward_func.text_encoder = self.accelerator.prepare_model(reward_func.text_encoder, evaluation_mode=True)
            elif isinstance(reward_func, Guo_Evaluator):
                reward_func.load_to_device(self.accelerator.device)
            elif isinstance(reward_func, MotionPatch_Evaluator):
                reward_func.load_to_device(self.accelerator.device)
        for i, sub_trace_reward_func in enumerate(self.sub_trace_reward_funcs):
            if isinstance(sub_trace_reward_func, PreTrainedModel):
                self.sub_trace_reward_funcs[i] = self.accelerator.prepare_model(sub_trace_reward_func, evaluation_mode=True)
            elif isinstance(sub_trace_reward_func, TMR_Evaluator):
                sub_trace_reward_func.load_to_device(self.accelerator.device)
                sub_trace_reward_func.accelerator = self.accelerator
                if self.is_deepspeed_enabled:   
                    sub_trace_reward_func.text_encoder = prepare_deepspeed(sub_trace_reward_func.text_encoder, self.accelerator)
                else:
                    sub_trace_reward_func.text_encoder = self.accelerator.prepare_model(sub_trace_reward_func.text_encoder, evaluation_mode=True)
            elif isinstance(sub_trace_reward_func, Guo_Evaluator):
                sub_trace_reward_func.load_to_device(self.accelerator.device)

        # Other parameters
        self.prompt_w_response = model_args.prompt_w_response

        # ### MODIFIED: Initialize eval-specific buffers for collecting rewards
        self._eval_rewards_per_func = None  # Will be a tensor to accumulate rewards across eval batches

        # self.t2m_eval_dataloader = self.create_eval_dataloader()
        # Evaluator
        self.official_eval_wrapper = self.load_evaluator()

        self.save_output_dir = args.output_dir

        self.all_trace_types = self.sub_trace_reward_names + ['whole_trace']

    def load_evaluator(self):
        from utils.get_eval_option import get_opt
        dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        wrapper_opt = get_opt(dataset_opt_path, self.accelerator.device)
        eval_wrapper = Official_Evaluator(wrapper_opt)
        return eval_wrapper


    def create_eval_dataloader(self):
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        dataset = dataset_TM_eval.Text2MotionDataset('t2m', 'val', w_vectorizer, unit_length=4, is_debug=False, return_all_captions=True)

        sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.accelerator.process_index,
                shuffle=False
        )
        t2m_eval_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset_TM_eval.collate_fn
        )
        return t2m_eval_dataloader
    
    
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_embeds, text_ids, img_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []

            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        # here, we only compute either text or image loss, so ids of other one could be omitted
        if img_ids is not None:
            # compute logits for image tokens
            hidden_states = model.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            # (text input id, image start token, image input id)
            # text_ids: text input id + image start token
            # img_ids: img_id (image token)
            image_logits = model.gen_head(last_hidden_states[:, -(img_ids.size(1)+1):, :]) # image prediction
            
            img_input_ids = torch.cat([img_ids.new_zeros(img_ids.size(0), 1), img_ids], dim=1) # cat a random one here, since it is not used in the loss calculation
            per_token_logps_img = _get_per_token_logps_part(image_logits, img_input_ids) # only calculate image loss
            return torch.cat([
                per_token_logps_img.new_zeros(
                    (per_token_logps_img.size(0), input_embeds.size(1) - per_token_logps_img.size(1) - 1)
                ), # the return length should be the input length minus 1 (the last token does not need predict)
                per_token_logps_img
            ], 
            dim=1)
        else: # only calculate text ids
            if hasattr(model, 'module'):
                text_logits = model.module.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=False).logits
            else:
                text_logits = model.llm(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=False).logits

            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids) 
            return per_token_logps_text
    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model, target_trace_type='whole_trace') -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # prompts preparation
        
        # prompts = [x["all_prompt"] + ' Response:' for x in inputs]   
        prompts = ["<bos>" + x["all_prompt"] for x in inputs]        
        if self.prompt_w_response:
            prompts = [p + ' Response:' for p in prompts]

        if hasattr(model, 'module'):
            prompt_inputs = model.module.tokenizer(
                prompts, 
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False)
        else:
            prompt_inputs = model.tokenizer(
                prompts, 
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.accelerator.is_main_process:
            print('=' * 25 + 'prompts' + '=' * 25)
            for p in prompts:
                print(' - ' + p)
            print('=' * 50)
        # import pdb; pdb.set_trace()
        # Generate completions
        # st_completion_gen = time.time()
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.llm.config.use_cache = False
            unwrapped_model.llm.gradient_checkpointing_disable()
            input_embeds = unwrapped_model.llm.get_input_embeddings()(prompt_ids)
            total_answer_motion_tokens = []
            total_answer_motion_263 = []
            total_answer_motion_denormed_263 = []
            total_think_motion_tokens = []
            total_think_motion_263 = []
            total_think_motion_denormed_263 = []
            total_think_motion_pos = []   # 用这个list来保存每一段sub-trace 的position
            # ============================================================================================
            current_prompt_ids = prompt_ids
            current_attn_mask = prompt_mask
            if unwrapped_model.w_lora:
                unwrapped_model.llm.set_adapter('shared')
            prompt_completion_outputs = unwrapped_model.llm.generate(
                current_prompt_ids, 
                attention_mask=current_attn_mask,
                pad_token_id=unwrapped_model.tokenizer.pad_token_id,
                bos_token_id=unwrapped_model.tokenizer.bos_token_id,
                eos_token_id=unwrapped_model.tokenizer.eos_token_id,
                max_new_tokens=self.max_completion_length,
                do_sample=True,   # 保持True，与num_beams=1配合使用随机采样
                use_cache=True,
                # return_dict_in_generate=True, 
                # output_hidden_states=False,
                # output_scores=True,
            )
            prompt_completion_ids = prompt_completion_outputs
            # completion_only_scores = torch.stack(prompt_completion_outputs.scores).transpose(0,1)      # completion_only_scores -> [self.num_generations, out_len, vocab_size]
            
            # NOTE: completion_only_scores.shape[1] + current_attn_mask.shape[1] == prompt_completion_ids.sequence.shape[1]
            num_gens_in_think = []
            for j in range(len(prompt_completion_ids)):
                # _, think_scores_j, st_ans_pos_j, ed_ans_pos_j = unwrapped_model.extract_answer_scores(score, unwrapped_model.tokenizer, return_position=True, return_prefix_scores=True)
                # answer_ids_j = prompt_completion_ids[j][prompt_ids.shape[1]:][st_ans_pos_j: ed_ans_pos_j]
                think_ids_j = prompt_completion_ids[j][prompt_ids.shape[1]:]
                # answer_motion_tokens_j = unwrapped_model.get_motion_from_scores_w_ids(answer_scores_j, answer_ids_j) 

                all_think_motion_pos_j = unwrapped_model.extract_motions_from_scores_w_ids(think_ids_j)
                all_think_motion_ids_j = []
                all_think_motion_tokens_j = []
                num_gens_in_think.append(len(all_think_motion_pos_j))
                for g, pos in enumerate(all_think_motion_pos_j):
                    think_motion_ids_g = think_ids_j[pos[0]: pos[1]+1]
                    # think_motion_scores_g = think_scores_j[pos[0]: pos[1]+1]
                    think_motion_tokens_g = unwrapped_model.get_motion_from_ids(think_motion_ids_g) 
                    all_think_motion_ids_j.append(think_motion_ids_g)
                    all_think_motion_tokens_j.append(think_motion_tokens_g)

                # answer_ids_j = all_think_motion_ids_j[-1]
                answer_motion_tokens_j = all_think_motion_tokens_j[-1]

                
                total_answer_motion_tokens.append(answer_motion_tokens_j)
                total_think_motion_tokens.append(all_think_motion_tokens_j)
                total_think_motion_pos.append(all_think_motion_pos_j)
                # import pdb; pdb.set_trace()

            
            if self.accelerator.is_main_process:
                print('='*50)
                print(unwrapped_model.tokenizer.decode(prompt_completion_ids[0], skip_special_tokens=True))
                # print('-'*50)
                # print(unwrapped_model.tokenizer.decode(prompt_completion_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True))
                print('-'*50)
                print(total_answer_motion_tokens[0])
                print('-'*50)
                print(num_gens_in_think)
                print('='*50)

            # ============================================================================================
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_ids
                
            completion_ids = prompt_completion_ids

            # 获取所有生成的motions
            assert len(total_answer_motion_tokens) == len(prompt_completion_ids), f"{len(total_answer_motion_tokens)} != {len(prompt_completion_ids)}"
            for a_i, answer_motion_tokens in enumerate(total_answer_motion_tokens):
                try:
                    pred_motion = unwrapped_model.net.forward_decoder(answer_motion_tokens)
                except:
                    # import pdb; pdb.set_trace()
                    pred_motion = torch.zeros(1, 196, 263).to(device)

                total_answer_motion_263.append(pred_motion)
                total_answer_motion_denormed_263.append(unwrapped_model.motion_denorm(pred_motion))
            for t_i, think_motions in enumerate(total_think_motion_tokens):
                # 获取一个think 过程里所有的motions
                think_motions_263 = []
                think_motions_denormed_263 = []
                for t_j, think_motion_tokens in enumerate(think_motions):
                    # 获取think过程里的每一个motion
                    try:
                        pred_motion_think = unwrapped_model.net.forward_decoder(think_motion_tokens)
                    except:
                        # import pdb; pdb.set_trace()
                        pred_motion_think = torch.zeros(1, 196, 263).to(device)
                    think_motions_263.append(pred_motion_think)
                    think_motions_denormed_263.append(unwrapped_model.motion_denorm(pred_motion_think))
                total_think_motion_263.append(think_motions_263)
                total_think_motion_denormed_263.append(think_motions_denormed_263)

        if hasattr(model, 'module'):
            is_eos = completion_ids == model.module.tokenizer.eos_token_id
        else:
            is_eos = completion_ids == model.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_pad_length = prompt_completion_ids.size(1) - prompt_mask.size(1)
        prompt_padding = torch.full((prompt_completion_ids.size(0), prompt_pad_length), 1, dtype=torch.long, device=prompt_mask.device)
        prompt_mask_padded = torch.cat([prompt_mask, prompt_padding], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        attention_mask = prompt_mask_padded * completion_mask  # 逐元素相乘
        # attention_mask[:, :prompt_mask.shape[1]] = 0           # mask 掉 prompt的部分
        # prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_all_ids = completion_ids    # 这里应该包含了生成的内容和prompt       # model.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        if hasattr(model, 'module'):
            input_embeds = model.module.llm.get_input_embeddings()(prompt_all_ids)
        else:
            input_embeds = model.llm.get_input_embeddings()(prompt_all_ids)

        with torch.inference_mode():
        # with torch.no_grad():
            if self.num_iterations > 1:
                # model.llm.gradient_checkpointing_enable()
                old_per_token_logps = self._get_per_token_logps(
                    model=model, 
                    input_embeds=input_embeds,
                    text_ids=prompt_all_ids, 
                    img_ids=None, 
                    attention_mask=attention_mask)
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                # self.ref_model.llm.gradient_checkpointing_enable()
                if self.ref_model.w_lora:
                    self.ref_model.llm.set_adapter('shared')    # 专门设置用特定的lora
                self.ref_model.llm.eval()
                # self.ref_model.lllm.gradient_checkpointing_enable()
                ref_per_token_logps = self._get_per_token_logps(
                    model=self.ref_model, 
                    input_embeds=input_embeds,
                    text_ids=prompt_all_ids, 
                    img_ids=None, 
                    attention_mask=attention_mask)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                # dummy ref_per_token_logps
                ref_per_token_logps = torch.zeros_like(old_per_token_logps)
        # get completions
        # completions = model.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        if hasattr(model, 'module'):
            text_completions = [completion.split(model.module.tokenizer.eos_token)[0] for completion in model.module.tokenizer.batch_decode(completion_ids[:, prompt_length:])]   # len(completions) = batch_size * num_generations
        else:
            text_completions = [completion.split(model.tokenizer.eos_token)[0] for completion in model.tokenizer.batch_decode(completion_ids[:, prompt_length:])]
        assert len(text_completions) == len(total_answer_motion_263)
        completions = [{'text_content': text_comp, 'answer_motion_263': ans_mo_263, 'answer_motion_denormed_263':ans_mo_denormed_263, 'think_motion_263': tnk_mo_263, 'think_motion_denormed_263': tnk_mo_denormed_263, 'completion_id': comp_id, 'think_motion_pos':tnk_mo_pos} for text_comp, ans_mo_263, ans_mo_denormed_263, tnk_mo_263, tnk_mo_denormed_263, tnk_mo_pos, comp_id in zip(text_completions, total_answer_motion_263, total_answer_motion_denormed_263, total_think_motion_263, total_think_motion_denormed_263, total_think_motion_pos, completion_ids[:, prompt_length:])]
        
        # 这里为每一个中间生成的motion获取一段sub-completions
        # Compute the global-sequence rewards
        # import pdb; pdb.set_trace()
        assert len(completions) == len(prompts)
        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device) # [batch_size, num_funcs]
        for i, (reward_func, reward_processing_class, reward_weight) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_weights)):    # 后者看起来是None，前面的是reward grpo.py 里定义的reward function
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # No need to duplicate prompts as we're not generating multiple completions per prompt
                    # reward_kwargs[key].extend([example[key]] * self.num_generations)
                    reward_kwargs[key].extend([example[key]])
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            output_reward_func = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            if self.reward_norm_before_add and (not 'format' in str(reward_func)):
                output_reward_func = (output_reward_func - min(output_reward_func)) / (max(output_reward_func) - min(output_reward_func) + 1e-4)
            rewards_per_func[:, i] = output_reward_func * reward_weight
        
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)        
        # import pdb; pdb.set_trace()
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        if self.accelerator.is_main_process:
            print('='*25 + f"RANK {self.accelerator.process_index}" + '='*25)
            print(rewards_per_func)
            print('-'*50)
            print(rewards)
            print('-'*50)
            print(advantages)
            print('='*50)
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        print(f'RANK {self.accelerator.process_index} - Logging Metrics.')
        print(f'RANK {self.accelerator.process_index} - completion_length: {completion_mask[:, prompt_ids.shape[1]:].sum(1)}')
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask[:, prompt_ids.shape[1]:].sum(1)).float().mean().item()  # 因为我们返回的结果里包含了prompt，所以这里要减去prompt的长度
        print(f'RANK {self.accelerator.process_index} - completion_length: Gathering Finished.')

        self._metrics["completion_length"].append(completion_length)
        print(f'RANK {self.accelerator.process_index} - Logging Completion Length Finished.')

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, (reward_func, reward_func_name) in enumerate(zip(self.reward_funcs, self.reward_names)):
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        print(f'RANK {self.accelerator.process_index} - Logging sub-rewards Finished.')
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        self._metrics['num_gens_in_think'].append(self.accelerator.gather_for_metrics(torch.tensor(num_gens_in_think).to(device, dtype=float)).mean().item())
        print(f'RANK {self.accelerator.process_index} - Logging Metrics Finished.')

        # Return all traces and rewards
        whole_trace_reward_dict = {
            "reward_name": "whole_trace_reward",
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids[:, prompt_ids.shape[1]:],
            "completion_mask": completion_mask[:, prompt_ids.shape[1]:],    # 因为我们返回的结果里包含了prompt，所以这里要减去prompt的长度
            "old_per_token_logps": old_per_token_logps,     # old-model 每个输出token的对数似然
            "ref_per_token_logps": ref_per_token_logps,     # ref-model 每个输出token的对数似然
            "advantages": advantages,
        }
        if target_trace_type == "whole_trace":
            all_reward_dicts = [whole_trace_reward_dict]
            print(f'RANK {self.accelerator.process_index} - Return all_reward_dicts.')
            del prompt_completion_outputs
            torch.cuda.empty_cache()
            return all_reward_dicts

        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        selected_id = self.state.global_step % len(self.all_trace_types)
        target_trace_type = self.all_trace_types[selected_id]
        # import pdb; pdb.set_trace()
        print(f'RANK {self.accelerator.process_index} - global step: {self.state.global_step} / {self.state.max_steps}')
        print(f'RANK {self.accelerator.process_index} - target trace: {target_trace_type}')


        all_inputs_ = self._generate_and_score_completions(inputs, model, target_trace_type)
        
        self._step += 1

        device = self.accelerator.device
        total_loss = 0
        # if self.accelerator.is_main_process:
        print(f'RANK {self.accelerator.process_index} - global step: {self.state.global_step} / {self.state.max_steps}')
        # Get the current policy's log probabilities
        # model.llm.gradient_checkpointing_enable()
        if self.grad_ckpt:
            if hasattr(model, 'module'):
                model.module.llm.config.use_cache = False
                model.module.llm.gradient_checkpointing_enable()
            else:
                model.llm.config.use_cache = False
                model.llm.gradient_checkpointing_enable()
        torch.set_grad_enabled(True)

        # 按照顺序优化其中某一类轨迹
        selected_id = self.state.global_step % len(self.all_trace_types)

        for inputs in all_inputs_:
            type_ = inputs["reward_name"]
            # recorded_type_.append(type_)
            # import pdb; pdb.set_trace()

            # Get the prepared inputs
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            # multimodal_inputs = inputs["multimodal_inputs"]   # 我们没有multi-modal input

            # Concatenate for full sequence
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            if hasattr(model, 'module'):
                input_embeds = model.module.llm.get_input_embeddings()(input_ids)
            else:
                input_embeds = model.llm.get_input_embeddings()(input_ids)

            
            per_token_logps_ = self._get_per_token_logps(model=model, input_embeds=input_embeds,text_ids=input_ids, img_ids=None, attention_mask=attention_mask)
            per_token_logps = per_token_logps_[:, prompt_ids.size(1) - 1:]
            # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
            # per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]
            # Get the advantages from inputs
            advantages = inputs["advantages"]
            old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
            ref_per_token_logps = inputs["ref_per_token_logps"]
        
            # ================================================================================================================
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
            # and use per_token_logps.detach() instead
            # Compute the policy ratio and clipped version
            coef_1 = torch.exp(per_token_logps - old_per_token_logps) # Pi_theta / Pi_theta_old
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)   # clip operation
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            # Add KL penalty if beta > 0
            if self.beta > 0:
                ref_per_token_logps = inputs["ref_per_token_logps"]
                per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                per_token_loss = per_token_loss + self.beta * per_token_kl
                # Log KL divergence
                mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                # self._metrics[f"kl/{type_}"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

            # Compute final loss
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            total_loss += loss
            # ================================================================================================================
            # Log clip ratio
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            # self._metrics[f"clip_ratio/{type_}"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

 
            # self.accelerator.wait_for_everyone()  # 等待所有进程完成推理
        print(f'RANK {self.accelerator.process_index} - Loss Computation Finished. Optimization Starts.')

        return total_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            # mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count
