ARG_WORLD_SIZE=${1:-4}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=23555
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

export HF_ENDPOINT=https://hf-mirror.com
export REPO_HOME="/mnt/data1/yuanming/Code/Cot-Mogen-V4"
echo "REPO_HOME: $REPO_HOME"

# Change the data_paths and image_folders to your own data
data_paths="/PATH/to/Your/dataset/HumanML3D/stage-3/combined_rl_data_clean.jsonl"
motion_folders="/mnt/data1/yuanming/datasets/HumanML3D_guo/new_joint_vecs"
context_modes="think_w_analysis_multi_round_gen_v3"
model_path="/Path/to/gemma-2-2b-it"
pretrained_model_path="/Path/to/Stage2/Checkpoint"


# pretrained_model_path=""
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "motion_folders: $motion_folders"

export EXP_NAME="debug" # 
# TASK_TYPE="rec"
# export EXP_NAME="debug" # TODO: change this to your own experiment name
cd ${REPO_HOME}/src/mogen_r1/src

EXP_PATH="${REPO_HOME}/runs/grpo/${EXP_NAME}"
mkdir -p ${EXP_PATH}/log


export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
# mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${EXP_PATH}/log/log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps
export WANDB_API_KEY=""


ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    open_mogen_r1/grpo.py \
    --use_vllm False \
    --output_dir $EXP_PATH \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --llm_backbone $model_path \
    --llm_ckpt $pretrained_model_path \
    --prompt_w_response True \
    --data_file_paths $data_paths \
    --context_modes $context_modes \
    --motion_folders $motion_folders \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --num_train_epochs 4 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --trainer_version v6 \
    --reward_funcs strict_format guo_tm_distance guo_mm_distance\
    --beta 0.01 \
    --val_split_ratio 0.05 \
    --eval_strategy no \
    --eval_on_start False \
    --eval_steps 50 \
    --report_to tensorboard \
    --dataset-name this_is_not_used \
    --learning_rate 1e-6 \
    --deepspeed "${REPO_HOME}/src/mogen_r1/local_scripts/zero2.json" \
    --lr_scheduler_type "cosine" \
    --wo_lora True \
    --reward_norm_before_add True
            
echo "Training completed for ${EXP_NAME}"