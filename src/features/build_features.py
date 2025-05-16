import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import joblib
import os
import numpy as np

def build_interaction_matrix(big_df, small_df, min_interactions=5, alpha_big=2.0, alpha_small=1.0, decay=1e-7):
    # Concatène toutes les interactions
    df = pd.concat([big_df, small_df], ignore_index=True)
    big = big_df.copy()
    small = small_df.copy()
    # 1. Tag des sources
    big["source_weight"]   = alpha_big
    small["source_weight"] = alpha_small

    # 2. Concatène toutes les interactions
    df = pd.concat([big, small], ignore_index=True)

    # Filtre utilisateurs et vidéos peu actifs
    active_users = df["user_id"].value_counts()[lambda x: x >= min_interactions].index
    active_videos = df["video_id"].value_counts()[lambda x: x >= min_interactions].index
    df = df[df["user_id"].isin(active_users) & df["video_id"].isin(active_videos)]

    # Remappage en indices
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    video_map = {v: k for k, v in enumerate(df["video_id"].unique())}
    df["uidx"] = df["user_id"].map(user_map)
    df["vidx"] = df["video_id"].map(video_map)
    df["play_duration_s"] = df["play_duration"] / 1000.0
    max_ts = df["timestamp"].astype(int).max() / 1e9
    df["age"] = (max_ts - df["timestamp"].astype(int) / 1e9)
    df["time_weight"] = np.exp(-decay * df["age"])

    df["strength"] = df["play_duration_s"] * df["source_weight"] * df["time_weight"]

    # Construire la matrice CSR
    row = df["uidx"].to_numpy()
    col = df["vidx"].to_numpy()
    data = df["strength"].to_numpy()
    mat = sp.csr_matrix((data, (row, col)),
                        shape=(len(user_map), len(video_map)))
    mat = normalize(mat, norm="l1", axis=1)
    return mat, user_map, video_map

def build_user_item_features(preprocessed_dir="preprocessed", out_dir="features"):
    os.makedirs(out_dir, exist_ok=True)

    big = pd.read_parquet(f"{preprocessed_dir}/big_matrix.parquet")
    small = pd.read_parquet(f"{preprocessed_dir}/small_matrix.parquet")
    daily = pd.read_parquet(f"{preprocessed_dir}/item_daily_features.parquet")

    # Matrice et mappings
    mat, user_map, video_map = build_interaction_matrix(big, small)
    sp.save_npz(f"{out_dir}/interaction_matrix.npz", mat)
    joblib.dump(user_map, f"{out_dir}/user_map.pkl")
    joblib.dump(video_map, f"{out_dir}/video_map.pkl")

    # Feature utilisateur : nombre total d’interactions
    df_users = big.groupby("user_id").size().rename("total_interactions").reset_index()
    df_users.to_parquet(f"{out_dir}/user_features.parquet", index=False)

    # Feature vidéo : moyenne des vues quotidiennes
    df_videos = daily.groupby("video_id")["play_cnt"].mean().rename("avg_daily_plays").reset_index()
    df_videos.to_parquet(f"{out_dir}/video_features.parquet", index=False)

    print(f"Features saved under {out_dir}/")

if __name__ == "__main__":
    build_user_item_features()
