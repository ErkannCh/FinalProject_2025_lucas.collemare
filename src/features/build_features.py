import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import joblib
import os
import numpy as np

def build_interaction_matrix(train_df, min_interactions=5, alpha=1.0, decay=1e-7):
    """
    Construit une CSR sparse matrix user×video à partir de train_df (big_matrix uniquement).
    - min_interactions : nombre minimal d’interactions pour filtrer users/items.
    - alpha            : poids constant appliqué à toutes les interactions.
    - decay            : coefficient d’exponentiel pour pondérer l’ancienneté.
    """
    df = train_df.copy()

    # 0) Appliquer un poids constant à chaque interaction
    df["source_weight"] = alpha

    # 1) Filtrer utilisateurs et vidéos peu actifs
    active_users  = df["user_id"].value_counts()[lambda x: x >= min_interactions].index
    active_videos = df["video_id"].value_counts()[lambda x: x >= min_interactions].index
    df = df[df["user_id"].isin(active_users) & df["video_id"].isin(active_videos)]

    # 2) Remappage en indices utilisateurs/vidéos
    user_map  = {u: i for i, u in enumerate(df["user_id"].unique())}
    video_map = {v: k for k, v in enumerate(df["video_id"].unique())}
    df["uidx"] = df["user_id"].map(user_map)
    df["vidx"] = df["video_id"].map(video_map)

    # 3) Calcul des pondérations temporelles et de force
    df["play_duration_s"] = df["play_duration"] / 1000.0
    max_ts = df["timestamp"].astype(int).max() / 1e9
    df["age"]        = (max_ts - df["timestamp"].astype(int) / 1e9)
    df["time_weight"] = np.exp(-decay * df["age"])
    df["strength"]    = df["play_duration_s"] * df["source_weight"] * df["time_weight"]

    # 4) Construction de la matrice CSR normalisée
    rows = df["uidx"].to_numpy()
    cols = df["vidx"].to_numpy()
    data = df["strength"].to_numpy()
    mat = sp.csr_matrix((data, (rows, cols)),
                        shape=(len(user_map), len(video_map)))
    mat = normalize(mat, norm="l1", axis=1)

    return mat, user_map, video_map

def build_user_item_features(preprocessed_dir="preprocessed", out_dir="features"):
    """
    Exporte dans out_dir/ :
    - interaction_matrix.npz (CSR user×video)
    - user_map.pkl, video_map.pkl
    - user_features.parquet (total_interactions)
    - video_features.parquet (avg_daily_plays)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Charger le train set
    big = pd.read_parquet(f"{preprocessed_dir}/big_matrix.parquet")
    daily = pd.read_parquet(f"{preprocessed_dir}/item_daily_features.parquet")

    # 2) Construire matrice & mappings (uniquement sur big)
    mat, user_map, video_map = build_interaction_matrix(big, alpha=1.0)
    sp.save_npz(f"{out_dir}/interaction_matrix.npz", mat)
    joblib.dump(user_map, f"{out_dir}/user_map.pkl")
    joblib.dump(video_map, f"{out_dir}/video_map.pkl")

    # 3) Features utilisteur : nombre total d’interactions
    df_users = big.groupby("user_id").size().rename("total_interactions").reset_index()
    df_users.to_parquet(f"{out_dir}/user_features.parquet", index=False)

    # 4) Features vidéo : moyenne des vues quotidiennes
    df_videos = daily.groupby("video_id")["play_cnt"].mean().rename("avg_daily_plays").reset_index()
    df_videos.to_parquet(f"{out_dir}/video_features.parquet", index=False)

    print(f"Features saved under {out_dir}/")

if __name__ == "__main__": 
    build_user_item_features()
