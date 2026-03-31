import os
import torch.nn as nn
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
import json
# from utils.word_vectorizer import WordVectorizer
# w_vectorizer = WordVectorizer('./glove', 'our_vab')
# import torch.distributed as dist
# from utils.motion_utils import plot_3d_motion, recover_from_ric
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def calculate_rouge_l(gt_texts, pred_texts):
    """
    计算 ROUGE-L 分数，支持多参考文本。
    
    参数:
        gt_texts: List[List[str]]，每个数据点包含多个参考文本
        pred_texts: List[str]，预测文本列表
    
    返回:
        List[dict]，包含每个数据点的 ROUGE-L 分数
        dict，包含所有数据点的平均 ROUGE-L 分数
    """
    # 初始化 ROUGE 评分器
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    # 存储每个数据点的 ROUGE-L 分数
    all_scores = []
    
    # 确保输入长度匹配
    if len(gt_texts) != len(pred_texts):
        raise ValueError("gt_texts 和 pred_texts 的长度必须相同")
    
    # 遍历每个数据点
    for refs, pred in zip(gt_texts, pred_texts):
        # 对预测文本进行分词（中文）
        pred_seg = " ".join(jieba.cut(pred.strip()))
        
        # 对每个参考文本进行分词并计算 ROUGE-L
        scores = []
        for ref in refs:
            ref_seg = " ".join(jieba.cut(ref.strip()))
            score = scorer.score(ref_seg, pred_seg)['rougeL']
            scores.append({
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            })
        
        # 取多个参考文本中的最大 fmeasure 作为该数据点的分数
        max_score = max(scores, key=lambda x: x['fmeasure'])
        all_scores.append(max_score)

    # 计算所有数据点的平均分数
    avg_scores = {
        'precision': sum([score['precision'] for score in all_scores]) / len(all_scores),
        'recall': sum([score['recall'] for score in all_scores]) / len(all_scores),
        'fmeasure': sum([score['fmeasure'] for score in all_scores]) / len(all_scores)
    }
    
    return all_scores, avg_scores



def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        # print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    try:
        assert activation.shape[0] > diversity_times, f"Not enough samples: {activation.shape[0]} vs {diversity_times}"
        num_samples = activation.shape[0]

        first_indices = np.random.choice(num_samples, diversity_times, replace=False)
        second_indices = np.random.choice(num_samples, diversity_times, replace=False)
        dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
        return dist.mean()
    except:
        print('Set Diversity to 0')
        return 0


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # return 1000
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
