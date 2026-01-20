# IRG-MotionLLM: Interleaving Motion Generation, Assessment and Refinement for Text-to-Motion Generation

[paper link](https://arxiv.org/abs/2512.10730)


## 👀Overview

<p align="center">
    <img src="./assets/irg-motionllm.jpg" width="100%" height="100%">
</p>

Recent advances in motion-aware large language models have shown remarkable promise for unifying motion understanding and generation tasks. However, these models typically treat understanding and generation separately, limiting the mutual benefits that could arise from interactive feedback between tasks. In this work, we reveal that motion assessment and refinement tasks act as crucial bridges to enable bidirectional knowledge flow between understanding and generation. Leveraging this insight, we propose Interleaved Reasoning for Motion Generation (IRMoGen), a novel paradigm that tightly couples motion generation with assessment and refinement through iterative text-motion dialogue. To realize this, we introduce IRG-MotionLLM, the first model that seamlessly interleaves motion generation, assessment, and refinement to improve generation performance.

## 🏠Model Zoo
| Stages | Ckpt on HumanML3D | Ckpts on KIT-ML |
|---------|----------|----------|
| Base Model | [Motion-Agent Offical Link](https://github.com/szqwu/Motion-Agent/tree/main) |[Coming Soon]() |
| Stage-1 | [Coming Soon]() | [Coming Soon]() |
| Stage-2 | [Coming Soon]() |[Coming Soon]() |
| Stage-3 | [Coming Soon]() |[Coming Soon]() |




## ⛰️Get Started
### Build Environment
```
conda create -n irg_motionllm python=3.10
conda activate irg_motionllm
pip install -r requirements.txt
```

### Download Glove and extractor
Download evaluation models and gloves for evaluation.
```
bash prepare/download_glove.sh
bash prepare/download_extractor.sh
```

### Prepare LLM and Base Model
We build our base model based on [Motion-Agent](https://github.com/szqwu/Motion-Agent/tree/main). For a quick start, please download the foundational LLM (Gemma2-2b-it) from [HuggingFace](https://huggingface.co/google/gemma-2-2b) and use the following script to download the pre-trained base model. 

```
bash prepare/download_motionllm_ckpt.sh
```

Note that you can also train the base model on your own.

### Prepare the Motion Datasets
In our work, we mainly conduct experiments on the HumanML3D and KIT-ML datasets. Please refer to [this link](https://github.com/EricGuo5513/HumanML3D) for motion dataset preparations.

### Prepare the Annotations of SFT and RL Data 
Coming soon.

## 🔥Training
You can train your own IRG-MotionLLM by using the following scripts. Do remember to replace ambiguous paths to the exact paths to your dataset and pre-trained models. 

### Stage-1
On stage-1, we train our model on eight related tasks to endow the model with the meta abilities of motion understanding, motion generation, motion assessment and motion refinement.
```
cd SFT/HumanML3D
bash train_stage1.sh

cd SFT/KIT-ML
bash train_stage1.sh
```

### Stage-2
On stage-2, we train our model on IRMoGen data to explicit interleave motion generation, asssessment and refinement.
```
cd SFT/HumanML3D
bash train_stage2.sh

cd SFT/KIT-ML
bash train_stage2.sh
```

### Stage-3
On stage-3, we further enhance the IRMoGen ability via GRPO.
```
cd RL/HumanML3D/src/mogen_r1/src
bash train_stage3.sh

cd RL/KIT-ML/src/mogen_r1/src
bash train_stage3.sh
```

## 📏Evaluation
You can evaluate the IRG-MotionLLM by using the following scripts. Do remember to replace ambiguous paths to the exact paths to your dataset and pre-trained models. 

### Stage-1 Model
```
cd SFT/HumanML3D

# 1️⃣ Direct Text-to-Motion Generation
torchrun --nproc_per_node 1 test_unified.py --llm-ckpt /PATH/TO/YOUR/STAGE-1/MODEL  --eval-task direct_generation-instructed_refinement --eval-batch-size 8 --eval-repeat-times 20 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora

# 2️⃣ Direct Text-to-Motion Generation + Refinement Instructing + Instructed Refinement
torchrun --nproc_per_node 1 test_unified.py --llm-ckpt /PATH/TO/YOUR/STAGE-1/MODEL  --eval-task direct_generation-instructed_refinement --eval-batch-size 8 --eval-repeat-times 20 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora

# 3️⃣ Motion-to-Text Captioning
torchrun --nproc_per_node 1 test_unified_m2t.py --llm-ckpt /PATH/TO/YOUR/STAGE-1/MODEL --eval-task m2t --eval-repeat-times 1 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora --eval-batch-size 32
```

### Stage-2 and Stage-3 Models
```
cd SFT/HumanML3D

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

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# 1️⃣ MAIN EVALUATION: Interleaved Reasoning for Text-to-Motion Generation
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_unified.py --llm-ckpt PATH/TO/YOUR/MODEL --eval-task unified_mogen_cot_v3 --eval-repeat-times 20 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora --eval-tag REPEAT_20

# 2️⃣ ROBUSTNESS EVALUATION: Randomly Replace the first generated motion into a random one to evaluate the robustness of the model
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_unified.py --llm-ckpt PATH/TO/YOUR/MODEL  --eval-task unified_mogen_cot_v3_random_flip --eval-repeat-times 20 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora --eval-tag RandomFlip_REPEAT_20  --dataset-return-neg-motion

```

### 🧠 Serving as a Reward Model
IRG-MotionLLM can also serve as a reward model to evaluate the text-motion alignment. You can calculate the alignment score by using the following scripts.

```
cd SFT/HumanML3D

# Stage-1 Model
torchrun --nproc_per_node=1 test_alignment_score_calculation_s1.py --llm-ckpt PATH/TO/YOUR/STAGE-1-MODEL  --eval-task unified_mogen_cot_v3 --eval-repeat-times 1 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora --eval-tag debug

# Stage-2 and Stage-3 Model
torchrun --nproc_per_node=1 test_alignment_score_calculation_s2_s3.py --llm-ckpt PATH/TO/YOUR/STAGE-2_3-MODEL  --eval-task unified_mogen_cot_v3 --eval-repeat-times 1 --generation-mode think_w_analysis_multi_round_gen --w-flash-attention --prompt-w-response --eval-set test --merge-lora --eval-tag debug
```

### ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{li2025irg-motionllm,
  title={IRG-MotionLLM: Interleaving Motion Generation, Assessment and Refinement for Text-to-Motion Generation},
  author={Li, Yuan-Ming and Yang, Qize and Lei, Nan and Fu, Shenghao and Zeng, Ling-An and Hu, Jian-Fang and Wei, Xihan and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2512.10730},
  year={2025}
}
```

## 📜 License

- Our models and code are under the Apache License 2.0. Our data is under MIT License.

## Acknowledgement
We sincerely acknowledge and appreciate the exceptional open-source contributions that form the foundation of our work: [Motion-Agent](https://github.com/szqwu/Motion-Agent/tree/main), [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [AToM](https://github.com/VincentHancoder/AToM/blob/main), [MARDM](https://github.com/neu-vi/MARDM/tree/main), [Text-to-Motion](https://github.com/EricGuo5513/text-to-motion), [VLM-R1](https://github.com/om-ai-lab/VLM-R1).
