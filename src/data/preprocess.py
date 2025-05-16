import pandas as pd
import os
from load_data import load_data

def preprocess_and_export(out_dir="preprocessed"):
    dfs = load_data()
    big = dfs["big_matrix"]
    small = dfs["small_matrix"]
    items = dfs["item_categories"]
    users = dfs["user_features"]
    daily = dfs["item_daily_feat"]
    social = dfs["social_network"]

    # 1. Nettoyage : on ne garde que les lignes complètes
    big = big.dropna(subset=["user_id", "video_id", "timestamp"])
    small = small.dropna(subset=["user_id", "video_id", "timestamp"]).copy()

    # 2. Conversion timestamp (timestamp est déjà en secondes UNIX)
    big["timestamp"] = pd.to_datetime(big["timestamp"], unit="s")
    small.loc[:, "timestamp"] = pd.to_datetime(small["timestamp"], unit="s")

    # 3. Fusion contenu : on ajoute la liste de features 'feat' à chaque interaction
    big = big.merge(items[["video_id", "feat"]], on="video_id", how="left")
    small = small.merge(items[["video_id", "feat"]], on="video_id", how="left")

    # 4. Export au format Parquet
    os.makedirs(out_dir, exist_ok=True)
    big.to_parquet(f"{out_dir}/big_matrix.parquet", index=False)
    small.to_parquet(f"{out_dir}/small_matrix.parquet", index=False)
    users.to_parquet(f"{out_dir}/user_features.parquet", index=False)
    items.to_parquet(f"{out_dir}/item_categories.parquet", index=False)
    daily.to_parquet(f"{out_dir}/item_daily_features.parquet", index=False)
    social.to_parquet(f"{out_dir}/social_network.parquet", index=False)

    print(f"Preprocessed files exported to {out_dir}/")

if __name__ == "__main__":
    preprocess_and_export()
