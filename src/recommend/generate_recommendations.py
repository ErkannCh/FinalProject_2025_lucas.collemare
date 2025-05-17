import argparse
import joblib
import pandas as pd
import scipy.sparse as sp
import os
import sys
import inspect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--test-file",
        default="data/small_matrix.csv",
        help="Fichier small_matrix.csv à scorer"
    )
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--N", type=int, default=10)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Assure le bon import des modules cf_model.py et content_model.py
    repo_root = os.path.abspath(os.path.join(script_dir, "../../"))
    src_models = os.path.join(repo_root, "src", "models")
    sys.path.insert(0, src_models)
    # 2. Prépare le dossier de sortie
    submission_dir = os.path.dirname(os.path.abspath(args.submission))
    if submission_dir and not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    # 3. Charge test set
    test = pd.read_csv(args.test_file)
    users = test['user_id'].unique()

    # 4. Charge le modèle picklé
    model = joblib.load(args.model_path)

    # 5. Charge mappings et matrice
    user_map = joblib.load("features/user_map.pkl")
    video_map = joblib.load("features/video_map.pkl")
    interaction_matrix = sp.load_npz("features/interaction_matrix.npz")

    # print("len(user_map) =", len(user_map), "— len(video_map) =", len(video_map))
    # for u in list(users)[:10] + list(users)[-10:]:
    #     print(u, "→", user_map.get(u))
    # valid_uidx = [user_map[u] for u in users if u in user_map]
    # print("min uid =", min(valid_uidx), "— max uid =", max(valid_uidx))
    # print(test[['user_id']].drop_duplicates().head(10))
    # print(test[['user_id']].drop_duplicates().tail(10))



    
    # 6. Prépare la matrice user×item pour recommend
    # (on peut passer directement interaction_matrix CSR pour filtrer les vues existantes)
    user_items = interaction_matrix.tocsr()

    # 7. Boucle de recommandation
    inv_video_map = {v:k for k,v in video_map.items()}
    recs = []
    als = getattr(model, "model", model)  # l'objet AlternatingLeastSquares
    #print(inspect.signature(als.recommend))
    #print(als.recommend.__doc__)

    for u in users:
        uidx = user_map.get(u)
        if uidx is None:
            continue
        # on passe user_items (shape n_users×n_items)
        pairs = als.recommend(uidx, user_items, N=args.N, filter_already_liked_items=False)
        ids, scores = pairs
        for rank, (vidx, score) in enumerate(zip(ids, scores), start=1):
            recs.append({
                'user_id': u,
                'video_id': inv_video_map[vidx],
                'rank': rank,
                'score': score
            })


    # 8. Écrit la soumission
    pd.DataFrame(recs).to_csv(args.submission, index=False)
    print(f"Submission saved to {args.submission}")

if __name__ == "__main__":
    main()