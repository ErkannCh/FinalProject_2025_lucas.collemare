import numpy as np
import pandas as pd

def precision_at_k(ranked_list, ground_truth, k):
    """
    ranked_list: liste d'item_id proposés
    ground_truth: set d'item_id réellement consommés
    """
    hit = sum([1 for item in ranked_list[:k] if item in ground_truth])
    return hit / k

def recall_at_k(ranked_list, ground_truth, k):
    if not ground_truth:
        return 0.0
    hit = sum([1 for item in ranked_list[:k] if item in ground_truth])
    return hit / len(ground_truth)

def dcg_at_k(ranked_list, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)  # i starts at 0
    return dcg

def idcg_at_k(ground_truth, k):
    # meilleur DCG possible : tous les relevant placés au top
    ideal_hits = min(len(ground_truth), k)
    return sum((1.0 / np.log2(i + 2) for i in range(ideal_hits)))

def ndcg_at_k(ranked_list, ground_truth, k):
    idcg = idcg_at_k(ground_truth, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_list, ground_truth, k) / idcg

def evaluate(recs_df, truth_df, k=10):
    """
    recs_df: DataFrame avec colonnes [user_id, item_id, rank]
    truth_df: DataFrame [user_id, item_id] des interactions réelles sur la période de test
    """
    users = recs_df['user_id'].unique()
    metrics = {'precision': [], 'recall': [], 'ndcg': []}
    for u in users:
        preds = recs_df[recs_df.user_id==u].sort_values('rank')['item_id'].tolist()
        actual = set(truth_df[truth_df.user_id==u]['item_id'])
        metrics['precision'].append(precision_at_k(preds, actual, k))
        metrics['recall'].append(recall_at_k(preds, actual, k))
        metrics['ndcg'].append(ndcg_at_k(preds, actual, k))
    # Moyenne globale
    return {m: np.mean(scores) for m, scores in metrics.items()}
