import argparse
import joblib
import scipy.sparse as sp
import pandas as pd
import os
from cf_model import CFModel
from content_model import ContentModel

def load_features():
    mat = sp.load_npz("features/interaction_matrix.npz")
    user_map = joblib.load("features/user_map.pkl")
    video_map = joblib.load("features/video_map.pkl")
    metadata = pd.read_parquet("preprocessed/item_categories.parquet")  # contient 'video_id','feat'
    return mat, user_map, video_map, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cf", "content"], required=True)
    parser.add_argument("--out", default="models/")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)

    mat, user_map, video_map, metadata = load_features()

    if args.model == "cf":
        model = CFModel(factors=64, regularization=0.05, iterations=20, alpha=40.0)
        model.fit(mat)
    else:
        model = ContentModel()
        model.fit(metadata, text_field="feat")

    joblib.dump(model, f"{args.out}{args.model}_model.pkl")
    print(f"Model '{args.model}' saved to {args.out}")

if __name__ == "__main__":
    main()
