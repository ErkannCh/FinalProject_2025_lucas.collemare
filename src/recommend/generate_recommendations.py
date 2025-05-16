import argparse
import pandas as pd
import scipy.sparse as sp
import joblib
import os
from src.models.content_model import ContentModel
from src.data.load_data import load_data

def build_mappings_and_matrix(preproc_dir="preprocessed", feat_dir="features"):
    """
    Reconstruit la matrice user×video et les mappings pour ContentModel.fit.
    """
    # 1) Charger préprocessed
    big = pd.read_parquet(f"{preproc_dir}/big_matrix.parquet")
    small = pd.read_parquet(f"{preproc_dir}/small_matrix.parquet")
    # On fusionne big+small interactions
    df = pd.concat([big, small], ignore_index=True)

    # 2) Filtre basique
    df = df.dropna(subset=["user_id", "video_id"])
    # ID → indices
    users = df["user_id"].unique()
    videos = df["video_id"].unique()
    user_map = {u: i for i, u in enumerate(users)}
    video_map = {v: i for i, v in enumerate(videos)}

    # 3) Construction matrice CSR
    rows = df["user_id"].map(user_map)
    cols = df["video_id"].map(video_map)
    data = 1.0  # binaire
    mat = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(videos))
    )

    return mat, user_map, video_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10,
                        help="Nombre de recommandations par utilisateur")
    parser.add_argument("--submission", default="submission_content.csv")
    args = parser.parse_args()

    # 1) Chargement métadonnées
    dfs = load_data("data")
    metadata = dfs["item_categories"]

    # 2) Construire matrice et mappings
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

    # 4) Génération recommendations
    recs = []
    for user_id in user_map:
        top_v = model.recommend(user_id, N=args.N)
        for rank, vid in enumerate(top_v, start=1):
            recs.append({
                "user_id": user_id,
                "video_id": vid,
                "rank": rank
            })

    # 5) Sauvegarde
    df_sub = pd.DataFrame(recs)
    df_sub.to_csv(args.submission, index=False)
    print(f"✔ Content-based recommendations saved to {args.submission}")

if __name__ == "__main__":
    main()
