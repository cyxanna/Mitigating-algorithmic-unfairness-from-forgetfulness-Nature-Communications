import numpy as np
from sklearn.metrics import auc, recall_score, precision_score, roc_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import accuracy_score
import torch
import fairlearn.metrics as fairness_funcs
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations


def compute_auroc(true, pred):
    return roc_auc_score(y_true=true, y_score=pred)

# Define utility functions for metrics
def compute_best_threshold(true, pred):
    precision, recall, thresholds = precision_recall_curve(y_true=true, y_score=pred,)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore_replace = np.nan_to_num(fscore, nan=0.0, posinf=0.0, neginf=0.0)  # sanity check
    best_index = np.argmax(fscore_replace)
    best_threshold = thresholds[best_index]
    return best_threshold

def cutoff_youdens_j(true, pred):
    # find optimal threshold based on Youden's J-Score
    fpr, tpr, thresholds = roc_curve(true, pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def demographic_parity_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    return fairness_funcs.demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight
    )

def decomposed_equalized_odds_difference(y_true, y_pred, *, sensitive_features):
    """
    Decompose equalized odds into per-group TPR/FPR, overall SDs, and pairwise SDs.

    Returns:
        dict: {
            'group_metrics': {group: {'tpr': ..., 'fpr': ...}, ...},
            'tpr_sd': float,                 
            'fpr_sd': float,                 
            'pairwise_tpr_sd': { 'a_b': sd}, 
            'pairwise_fpr_sd': { 'a_b': sd}, 
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    unique_groups = np.unique(sensitive_features)

    group_metrics = {}
    tpr_values = []
    fpr_values = []

    # TPR / FPR
    for group in unique_groups:
        group_key = int(group) if isinstance(group, (np.integer,)) else float(group) if isinstance(group, (np.floating,)) else group
        group_mask = (sensitive_features == group)

        # True Positive Rate (TPR)
        tpr_mask = (y_true == 1) & group_mask
        if np.sum(tpr_mask) > 0:
            tpr = float(np.sum((y_pred == 1) & tpr_mask) / np.sum(tpr_mask))
        else:
            tpr = float('nan')

        # False Positive Rate (FPR)
        fpr_mask = (y_true == 0) & group_mask
        if np.sum(fpr_mask) > 0:
            fpr = float(np.sum((y_pred == 1) & fpr_mask) / np.sum(fpr_mask))
        else:
            fpr = float('nan')

        group_metrics[group_key] = {'tpr': tpr, 'fpr': fpr}

        if not np.isnan(tpr):
            tpr_values.append(tpr)
        if not np.isnan(fpr):
            fpr_values.append(fpr)

    #  overall standard deviation
    tpr_sd = np.std(tpr_values) if len(tpr_values) > 1 else float('nan')
    fpr_sd = np.std(fpr_values) if len(fpr_values) > 1 else float('nan')

    # pairwise standard deviation of two groups
    pairwise_tpr_sd = {}
    pairwise_fpr_sd = {}

    norm_groups = [int(g) if isinstance(g, (np.integer,)) else float(g) if isinstance(g, (np.floating,)) else g for g in unique_groups]
    for a, b in combinations(norm_groups, 2):
        ga, gb = a, b
        tpr_a = group_metrics[ga]['tpr'] if ga in group_metrics else float('nan')
        tpr_b = group_metrics[gb]['tpr'] if gb in group_metrics else float('nan')
        fpr_a = group_metrics[ga]['fpr'] if ga in group_metrics else float('nan')
        fpr_b = group_metrics[gb]['fpr'] if gb in group_metrics else float('nan')

        key = f"{ga}_{gb}"

        if not (np.isnan(tpr_a) or np.isnan(tpr_b)):
            pairwise_tpr_sd[key] = float(np.std([tpr_a, tpr_b], ddof=0))
        if not (np.isnan(fpr_a) or np.isnan(fpr_b)):
            pairwise_fpr_sd[key] = float(np.std([fpr_a, fpr_b], ddof=0))

    return {
        'group_metrics': group_metrics,
        'tpr_sd': tpr_sd,
        'fpr_sd': fpr_sd,
        'pairwise_tpr_sd': pairwise_tpr_sd,
        'pairwise_fpr_sd': pairwise_fpr_sd,
    }

def compute_dp_sd(y_pred, sensitive_features, return_pairwise=False, pairs=None):
    """
    Compute the overall standard deviation and optional pairwise standard deviation of Demographic Parity.

    Args:
        y_pred: binary prediction (0/1) or probability (if probability, threshold to get 0/1)
        sensitive_features: sensitive attribute (same length as y_pred)
        return_pairwise: if True, return dictionary containing overall and pairwise SD; otherwise, return only overall SD float
        pairs: optional, only compute given group pairs (e.g. [(0,2), (0,3)]); default compute all pairwise combinations

    Returns:
        - when return_pairwise=False: float dp_sd
        - when return_pairwise=True: {
            'dp_sd': float,
            'group_dp': {group: dp_rate},
            'pairwise_dp_sd': { 'a_b': sd }
          }
    """
    y_protected = np.asarray(sensitive_features)
    y_pred_class = np.asarray(y_pred)
    unique_groups = np.unique(y_protected)

    # DP of each group = P(ŷ=1 | A=g)
    group_dp = {}
    dp_values = []
    for group in unique_groups:
        group_key = int(group) if isinstance(group, (np.integer,)) else float(group) if isinstance(group, (np.floating,)) else group
        group_mask = (y_protected == group)
        if np.sum(group_mask) > 0:
            dp_rate = float(np.mean(y_pred_class[group_mask]))
            group_dp[group_key] = dp_rate
            dp_values.append(dp_rate)

    dp_sd = float(np.std(dp_values)) if len(dp_values) > 1 else float('nan')

    if not return_pairwise:
        return dp_sd

    # pairwise standard deviation of two groups
    pairwise_dp_sd = {}
    norm_groups = list(group_dp.keys())
    pair_iter = pairs if pairs is not None else list(combinations(norm_groups, 2))
    for a, b in pair_iter:
        ga, gb = a, b
        if ga in group_dp and gb in group_dp:
            pairwise_dp_sd[f"{ga}_{gb}"] = float(np.std([group_dp[ga], group_dp[gb]], ddof=0))

    return {
        'dp_sd': dp_sd,
        'group_dp': group_dp,
        'pairwise_dp_sd': pairwise_dp_sd,
    }

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)
