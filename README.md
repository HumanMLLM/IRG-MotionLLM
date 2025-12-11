# IRG-MotionLLM: Interleaving Motion Generation, Assessment and Refinement for Text-to-Motion Generation

[paper link]()


## 👀Overview

<p align="center">
    <img src="./assets/irg-morionllm.jpg" width="100%" height="100%">
</p>

Recent advances in motion-aware large language models have shown remarkable promise for unifying motion understanding and generation tasks. However, these models typically treat understanding and generation separately, limiting the mutual benefits that could arise from interactive feedback between tasks. In this work, we reveal that motion assessment and refinement tasks act as crucial bridges to enable bidirectional knowledge flow between understanding and generation. Leveraging this insight, we propose Interleaved Reasoning for Motion Generation (IRMoGen), a novel paradigm that tightly couples motion generation with assessment and refinement through iterative text-motion dialogue. To realize this, we introduce IRG-MotionLLM, the first model that seamlessly interleaves motion generation, assessment, and refinement to improve generation performance. IRG-MotionLLM is developed progressively with a novel three-stage training scheme, initializing and subsequently enhancing native IRMoGen capabilities. To facilitate this development, we construct an automated data engine to synthesize interleaved reasoning annotations from existing text-motion datasets. Extensive experiments demonstrate that: (i) Assessment and refinement tasks significantly improve text-motion alignment; (ii) Interleaving motion generation, assessment, and refinement steps yields consistent performance gains across training stages; and (iii) IRG-MotionLLM clearly outperforms the baseline model and achieves advanced performance on standard text-to-motion generation benchmarks. Cross-evaluator testing further validates its effectiveness.
## 📈Experimental Results

## 🏠Model Zoo

## ⛰️Get Ready

## 🔥Training

#### Base Model

#### 📍 Stage 1

#### 📍 Stage 2

#### 📍 Stage 3


## 📏Evaluation

#### 📍 Stage 1

#### 📍 Stage 2

#### 📍 Stage 3



### ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
```

## 📜 License

- Our models and code are under the Apache License 2.0. Our data is under MIT License.
