#!/usr/bin/env python3
import os
import sys

# ────────────────────────────────────────────────────────────────
# Ajout de la racine du projet dans sys.path pour pouvoir faire
# des imports "from src.models..." et "from src.data..."
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────

import argparse
import pandas as pd
import scipy.sparse as sp
import joblib

from src.models.content_model import ContentModel
from src.data.load_data import load_data

def build_mappings_and_matrix(preproc_dir="preprocessed"):
    """
    Reconstruit la matrice user×video et les mappings pour ContentModel.fit.
    """
    # 1) Charger les interactions pré-traitées
    big = pd.read_parquet(f"{preproc_dir}/big_matrix.parquet")
    small = pd.read_parquet(f"{preproc_dir}/small_matrix.parquet")
    df = pd.concat([big, small], ignore_index=True).dropna(subset=["user_id", "video_id"])

    # 2) Générer les mappings user↔idx et video↔idx
    users = df["user_id"].unique()
    videos = df["video_id"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    video_map = {v: i for i, v in enumerate(videos)}

    # 3) Construire la CSR user×video
    rows = df["user_id"].map(user_map)
    cols = df["video_id"].map(video_map)
    data = [1.0] * len(df)
    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(users), len(videos)))

    return mat, user_map, video_map

def main():
    parser = argparse.ArgumentParser(
        description="Génère des recommandations content-based (profil utilisateur)."
    )
    parser.add_argument("--N", type=int, default=10,
                        help="Nombre de recommandations par utilisateur.")
    parser.add_argument("--submission", default="submission_content.csv",
                        help="Chemin du fichier de sortie CSV.")
    args = parser.parse_args()

    # 1) Charger les métadonnées
    dfs = load_data("data")
    metadata = dfs["item_categories"]

    # 2) Construire la matrice d’interaction et les mappings
    interaction_matrix, user_map, video_map = build_mappings_and_matrix()

    # 3) Entraîner le modèle content-based
    model = ContentModel(max_features=5000)
    model.fit(
        metadata_df=metadata,
        interaction_matrix=interaction_matrix,
        user_map=user_map,
        video_map=video_map,
        text_field="feat"
    )

    # 4) Générer les recommandations pour chaque utilisateur
    recs = []
    for user_id in user_map:
        top_videos = model.recommend(user_id, N=args.N)
        for rank, vid in enumerate(top_videos, start=1):
            recs.append({
                "user_id": user_id,
                "video_id": vid,
                "rank": rank
            })

    # 5) Sauvegarder la soumission
    os.makedirs(os.path.dirname(args.submission) or ".", exist_ok=True)
    pd.DataFrame(recs).to_csv(args.submission, index=False)
    print(f"✔ Content-based recommendations saved to {args.submission}")

if __name__ == "__main__":
    main()
