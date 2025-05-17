#!/usr/bin/env python3
import os
import sys

# ─── Permettre à joblib de retrouver cf_model.py et content_model.py ───
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = script_dir
sys.path.insert(0, os.path.join(repo_root, "src", "models"))

import itertools
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from src.evaluate.metrics import evaluate

# 1) Charger CF / CB
cf = joblib.load("models/cf_model.pkl")
cb = joblib.load("models/content_model.pkl")
user_items_cf = cf.user_items
user_map_cf   = joblib.load("features/user_map.pkl")
video_map_cf  = joblib.load("features/video_map.pkl")
inv_map       = {c: v for v, c in video_map_cf.items()}

# 2) Charger la vérité small_matrix
test  = pd.read_csv("data/small_matrix.csv")
truth = test.rename(columns={"video_id":"item_id"})[["user_id","item_id"]]
users = truth["user_id"].unique()

# 3) Fonction de recommandation hybride ultra‐simple
def hybrid_precision(alpha, CF_K, CB_K):
    recs = []
    for u in users:
        scores = {}
        # CF part
        if u in user_map_cf:
            uidx    = user_map_cf[u]
            uvec    = cf.model.item_factors[uidx]
            sc_cf   = cf.model.user_factors.dot(uvec)
            seen    = user_items_cf[uidx].indices
            sc_cf[seen] = -np.inf
            top_cf  = np.argpartition(-sc_cf, CF_K)[:CF_K]
            for idx in top_cf:
                scores[idx] = scores.get(idx,0) + alpha*sc_cf[idx]
        # CB part
        if u in cb.user_map:
            uidx_cb = cb.user_map[u]
            up      = cb.user_profiles[uidx_cb].reshape(1,-1)
            sc_cb   = cosine_similarity(up, cb.tfidf_matrix).flatten()
            top_cb  = np.argpartition(-sc_cb, CB_K)[:CB_K]
            for idx in top_cb:
                scores[idx] = scores.get(idx,0) + (1-alpha)*sc_cb[idx]
        # build recs
        ranked = sorted(scores, key=lambda i:-scores[i])[:10]
        for rank, idx in enumerate(ranked,1):
            recs.append({"user_id":u, "item_id":inv_map[idx], "rank":rank})
    df = pd.DataFrame(recs)
    return evaluate(df, truth, k=10)["precision"]

# 4) Grid search
best = (None,None,None, -1)
for alpha, CF_K, CB_K in itertools.product(
        np.linspace(0,1,11), [10,30,50], [10,30,50]
):
    prec = hybrid_precision(alpha, CF_K, CB_K)
    if prec > best[3]:
        best = (alpha, CF_K, CB_K, prec)
    print(f"α={alpha:.1f} CF_K={CF_K} CB_K={CB_K} → {prec:.4f}")

print(">>> Best config:", best)
