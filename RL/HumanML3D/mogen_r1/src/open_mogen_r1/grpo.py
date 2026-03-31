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
import sys
sys.path.insert(0, '/PATH/to/YOUR/CODEBASE/src/mogen_r1/src')
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pathlib
from datasets import load_dataset, load_from_disk
from datasets import Dataset
# from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, get_peft_config
from trl.scripts.utils import TrlParser
# from open_r1.trainer import JanusT2IR1Trainer

from models.format_rewards import format_reward_mogen_basic, format_reward_mogen_strict
from data.data_utils import make_conversation
import json
import shutil
import datetime
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},     # data_paths="/mnt/data1/yuanming/datasets/VLM-R1/rec_jsons_processed/refcoco_train.jsonl:...."
    )
    motion_folders: str = field(
        default=None,
        metadata={"help": "Paths to motion folders, separated by ':'"},
    )
    context_modes: str = field(
        default=None,
        metadata={"help": "The context mode. separated by ':'"},
    )
    generation_round: int = field(
        default=-1,
        metadata={"help": "The generation round"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "tmr_similarity"],                                # 在这里注册 reward functions
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    sub_trace_reward_funcs: list[str] = field(
        default_factory=lambda: [],                                # 在这里注册 sub-trace process reward functions
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    sub_trace_reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    eval_task: str = field(
        default="motion_generation_direct",
        metadata={"help": "task for generation"},
    )
    trainer_version: str = field(
        default="v1",
        metadata={"help": "The version of the trainer"},
    )
    grad_ckpt: bool = field(
        default=True,
        metadata={"help": "Usding gradient checkpointing"},
    )
    reward_norm_before_add: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the reward before adding it to the loss"},
    )

# add motion_generation_prompt in the GRPOConfig
@dataclass
class GRPOConfig(GRPOConfig):
    """
    Configuration class for the GRPO training script.
    """
    cfg_weight: float = field(default=3.0, metadata={"help": "The cfg weight for motion generation"})
    reasoning_prompt_path: Optional[str] = field(
        default='',
    )
    save_safetensors: bool = field(
        default=False,
        metadata={"help": "Whether to save safetensors"},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    indepandent_tokens: bool = False
    llm_backbone: str = ''
    llm_tokenizer: str = ''
    llm_ckpt: str = ''
    w_flash_attention: bool = True
    # LoRa config
    lora_r_t2m: int = 64            # lora_r for t2m
    lora_alpha_t2m: int = 64        # lora_alpha for t2m
    lora_r_m2t: int = 32            # lora_r for m2t
    lora_alpha_m2t: int = 32        # lora_alpha for m2t
    lora_dropout: float = 0.00       # lora_dropout
    # vavae
    code_dim: int = 512             # "embedding dimension")
    nb_code: int = 512              # "nb of embedding")
    mu: float = 0.99                # "exponential moving average to update the codebook")
    down_t: int = 2                 # "downsampling rate")
    stride_t: int = 2               # "stride size")
    width: int = 512                # "width of the network")
    depth: int = 3                  # "depth of the network")
    dilation_growth_rate: int = 3               # "dilation growth rate")
    output_emb_width: int = 512                 # "output embedding width")
    vq_act: str = 'relu'                 
    vq_norm: str = None                         
    ## quantizer
    quantizer: str = 'ema_reset'                #"eps for optimal transport")
    vq_beta: float = 1.0                           #'commitment loss in standard VQ')
    # text preprocess
    context_style: str = 'gemma'
    prompt_w_response: bool = False
    # dataname
    dataname: str = 't2m'
    nb_joints: int = 22
    # mean & std
    mean_path: str = '/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy'
    std_path: str = '/mnt/data1/yuanming/Code/Motion_Gen/Motion-Agent/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy'
    full_finetuned: bool = False     # whether we finetune the whole model
    full_frozen: bool = False
    wo_lora: bool = False
    model_cls: str = field(
        default="motionllm",
        metadata={"help": "The model class"},
    )
    activate_token_embeds: bool = field(
        default=False,
        metadata={"help": "Whether to activate token embeddings"},
    )

reward_funcs_registry = {
    "tmr_tm_similarity": 'tmr_tm_similarity',
    "guo_tm_distance": 'guo_tm_distance',
    "guo_tm_similarity": 'guo_tm_similarity',
    "tmr_mm_similarity": 'tmr_mm_similarity',
    "guo_mm_distance": 'guo_mm_distance',
    "guo_mm_similarity": 'guo_mm_similarity',
    "format": format_reward_mogen_basic,
    "strict_format": format_reward_mogen_strict,
    "tmr_process_motion_best_similarity": "tmr_process_motion_best_similarity",
    "tmr_process_motion_avg_similarity": "tmr_process_motion_avg_similarity",
    "tmr_process_motion_last_similarity": "tmr_process_motion_last_similarity",
    "tmr_process_motion_first_similarity": "tmr_process_motion_first_similarity",
    "tmr_process_motion_similarity_rank": "tmr_process_motion_similarity_rank",
    "guo_process_motion_best_similarity": "guo_process_motion_best_similarity",
    "guo_process_motion_avg_similarity": "guo_process_motion_avg_similarity",
    "guo_process_motion_first_similarity": "guo_process_motion_first_similarity",
    "guo_process_motion_last_similarity": "guo_process_motion_last_similarity",
    "guo_process_motion_last_distance_best": "guo_process_motion_last_distance_best",
    "guo_process_motion_similarity_rank": "guo_process_motion_similarity_rank",
    "guo_process_motion_best_distance": "guo_process_motion_best_distance",
    "guo_process_motion_avg_distance": "guo_process_motion_avg_distance",
    "guo_process_motion_last_distance": "guo_process_motion_last_distance",
    "guo_process_motion_first_distance": "guo_process_motion_first_distance",
    "guo_process_init_gen_distance": "guo_process_init_gen_distance",
    "guo_process_refinement_single_step_tm_distance": "guo_process_refinement_single_step_tm_distance",
    "guo_process_refinement_single_step_mm_distance": "guo_process_refinement_single_step_mm_distance",
    "guo_process_refinement_single_step_tm_mm_distance": "guo_process_refinement_single_step_tm_mm_distance",
    "guo_process_refinement_single_step_tm_mm_distance_improvement_rate": "guo_process_refinement_single_step_tm_mm_distance_improvement_rate",
    "guo_process_motion_distance_rank": "guo_process_motion_distance_rank",
    "guo_sub_trace_mm_distance": "guo_sub_trace_mm_distance",
    "guo_sub_trace_mm_similarity": "guo_sub_trace_mm_similarity",
    "guo_sub_trace_tm_distance": "guo_sub_trace_tm_distance",
    "guo_sub_trace_tm_similarity": "guo_sub_trace_tm_similarity",

    "motionpatch_tm_similarity": "motionpatch_tm_similarity"
}
def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func.split(':')[0]] for func in script_args.reward_funcs]
    reward_weights = [float(func.split(':')[1]) if len(func.split(':')) == 2 else 1 for func in script_args.reward_funcs]
    reward_names = [func for func in script_args.reward_funcs]
    sub_trace_reward_funcs = [reward_funcs_registry[func] for func in script_args.sub_trace_reward_funcs]
    sub_trace_reward_names = [func for func in script_args.sub_trace_reward_funcs]

    # Load the JSONL datasets
    data_files = script_args.data_file_paths.split(":")
    motion_folders = script_args.motion_folders.split(":")
    context_modes = script_args.context_modes.split(":")

    if len(data_files) != len(motion_folders):
        raise ValueError("Number of data files must match number of motion folders")
    
    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"
    if script_args.sub_trace_reward_method is None:
        sub_trace_reward_methods = ["default"] * len(data_files)
    else:
        sub_trace_reward_methods = script_args.sub_trace_reward_method.split(":")
        assert len(sub_trace_reward_methods) == len(data_files), f"Number of sub-trace reward methods must match number of data files: {len(sub_trace_reward_methods)} != {len(data_files)}"

    # 下面是构造数据，需要针对不同的任务做特殊设计
    all_data = []
    for data_file, motion_folder, accu_reward_method, context_mode in zip(data_files, motion_folders, accu_reward_methods, context_modes):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['context_mode'] = context_mode
                item['generation_round'] = script_args.generation_round
                if "motions" in item:
                    if isinstance(item['motions'], dict):
                        item['motions'] = {key: os.path.join(motion_folder, value) for key, value in item['motions'].items()}
                    else:
                        raise ValueError(f"Unsupported motion type: {type(item['motions'])}")
                all_data.append(item)
    
    dataset = Dataset.from_list(all_data)
    # Map the conversations
    dataset = dataset.map(make_conversation, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    # import pdb; pdb.set_trace()
    # if script_args.trainer_version == "v3":
    #     from open_mogen_r1.trainer import MogenR1TrainerV3
    #     trainer_cls = MogenR1TrainerV3
    # elif script_args.trainer_version == "v4":
    #     from open_mogen_r1.trainer import MogenR1TrainerV4
    #     trainer_cls = MogenR1TrainerV4
    # elif script_args.trainer_version == "v5":
    #     from open_mogen_r1.trainer import MogenR1TrainerV5
    #     trainer_cls = MogenR1TrainerV5
    # elif script_args.trainer_version == "v7":
    #     from open_mogen_r1.trainer import MogenR1TrainerV7
    #     trainer_cls = MogenR1TrainerV7
    elif script_args.trainer_version == "v6":
        from open_mogen_r1.trainer import MogenR1TrainerV6
        trainer_cls = MogenR1TrainerV6
    # elif script_args.trainer_version == "v2":
    #     from open_mogen_r1.trainer import MogenR1TrainerV2
    #     trainer_cls = MogenR1TrainerV2
    # else:
    #     from open_mogen_r1.trainer import MogenR1Trainer
    #     trainer_cls = MogenR1Trainer
    print("using trainer:", trainer_cls.__name__)
    # initialize_tokenizer(model_args.model_name_or_path)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_names=reward_names,
        reward_weights=reward_weights,
        sub_trace_reward_funcs=sub_trace_reward_funcs,
        sub_trace_reward_names=sub_trace_reward_names,
        args=training_args,
        model_args=model_args,
        # vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        # freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        eval_task=script_args.eval_task,
        context_mode=context_mode,
        grad_ckpt=script_args.grad_ckpt,
        reward_norm_before_add=script_args.reward_norm_before_add,
        # max_pixels=script_args.max_pixels,
        # min_pixels=script_args.min_pixels,
        # max_anyres_num=script_args.max_anyres_num,    # for InternVL only
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


def codebase_backup(out_dir):
    backup_dir = os.path.join(out_dir, "code_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # List of directories and files to backup
    backup_items = [
        "PATH/to/YOUR/CODEBASE/src"
    ]
    ignore_folders = [
        "PATH/to/YOUR/CODEBASE/src/mogen_r1/src/checkpoints",
        "PATH/to/YOUR/CODEBASE/src/mogen_r1/src/ckpt",
        "PATH/to/YOUR/CODEBASE/src/mogen_r1/src/glove",
        "PATH/to/YOUR/CODEBASE/src/mogen_r1/src/wandb"
    ]
    
    # Extract base folder names for ignore patterns
    ignore_patterns = [os.path.basename(folder) for folder in ignore_folders]
    
    # Save the command used to execute the program
    command = " ".join(sys.argv)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    command_file = os.path.join(backup_dir, f"backup_command_{timestamp}.txt")
    try:
        with open(command_file, "w") as f:
            f.write(f"Command executed: {command}\n")
            f.write(f"Timestamp: {timestamp}\n")
        print(f"Command saved to: {command_file}")
    except Exception as e:
        print(f"Error saving command: {str(e)}")
    
    # Copy each item to backup directory
    for item in backup_items:
        try:
            # Check if item exists
            if not os.path.exists(item):
                print(f"Warning: Source path {item} does not exist")
                continue
                
            # Get the base name of the item
            item_name = os.path.basename(item)
            # Create destination path
            dest_path = os.path.join(backup_dir, item_name)
            
            # If it's a directory, use copytree with ignore patterns
            if os.path.isdir(item):
                ignore = shutil.ignore_patterns(*ignore_patterns)
                shutil.copytree(item, dest_path, dirs_exist_ok=True, ignore=ignore)
                print(f"Successfully backed up directory: {item} to {dest_path}")
            # If it's a file, use copy2 to preserve metadata
            elif os.path.isfile(item):
                shutil.copy2(item, dest_path)
                print(f"Successfully backed up file: {item} to {dest_path}")
            else:
                print(f"Warning: {item} is neither a file nor directory")
                
        except Exception as e:
            print(f"Error backing up {item}: {str(e)}")
    
    print(f"Backup completed to: {backup_dir}")
    return backup_dir

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    codebase_backup(training_args.output_dir)

    main(script_args, training_args, model_args)
