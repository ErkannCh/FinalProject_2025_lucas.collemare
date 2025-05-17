#!/usr/bin/env python3
import os
import sys

# Permet d'importer src.models et src.data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import pandas as pd
import scipy.sparse as sp

from src.models.content_model import ContentModel
from src.data.load_data import load_data

def build_mappings_and_matrix(preproc_dir="preprocessed"):
    """
    Construit :
      - mat CSR user×video
      - user_map, video_map
      - user_hist: dict user_id→set(video_id)
    """
    big = pd.read_parquet(f"{preproc_dir}/big_matrix.parquet")
    small = pd.read_parquet(f"{preproc_dir}/small_matrix.parquet")
    df = pd.concat([big, small], ignore_index=True).dropna(subset=["user_id", "video_id"])

    users = df["user_id"].unique()
    videos = df["video_id"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    video_map = {v: i for i, v in enumerate(videos)}

    rows = df["user_id"].map(user_map)
    cols = df["video_id"].map(video_map)
    data = [1.0] * len(df)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(videos)))

    user_hist = df.groupby("user_id")["video_id"].apply(set).to_dict()
    return mat, user_map, video_map, user_hist

def main():
    parser = argparse.ArgumentParser(
        description="Génère des recommandations content-based filtrées"
    )
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--submission", default="submission_content.csv")
    args = parser.parse_args()

    # 1) Prépare dossier de sortie
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)

    # 2) Charge métadonnées
    dfs = load_data("data")
    metadata = dfs["item_categories"]

    # 3) Construit matrice + mappings + historique
    interaction_matrix, user_map, video_map, user_hist = build_mappings_and_matrix()

    # 4) Entraîne le modèle
    model = ContentModel(max_features=10000, ngram_range=(1,2), stop_words="english")
    model.fit(
        metadata_df=metadata,
        interaction_matrix=interaction_matrix,
        user_map=user_map,
        video_map=video_map,
        text_field="feat"
    )

    # 5) Génère et filtre les recommandations
    recs = []
    for user_id in user_map:
        raw = model.recommend(user_id, N=args.N*3)  # extraire un peu plus pour filtrer
        filtered = [vid for vid in raw if vid not in user_hist.get(user_id, set())][:args.N]
        for rank, vid in enumerate(filtered, start=1):
            recs.append({"user_id": user_id, "video_id": vid, "rank": rank})

    # 6) Sauvegarde
    pd.DataFrame(recs).to_csv(args.submission, index=False)
    print(f"✔ Content-based recommendations saved to {args.submission}")

if __name__ == "__main__":
    main()
