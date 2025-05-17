#!/usr/bin/env bash
set -e

# echo "1/5 Prétraitement des données"
# python src/data/preprocess.py

# echo "2/5 Construction des features"
# python src/features/build_features.py

# echo "3/5 Entraînement"
# python src/models/train.py --model cf --out models/
# python src/models/train.py --model content --out models/

echo "4/5 Génération recommandations CF"
python src/recommend/generate_recommendations.py \
  --model-path models/cf_model.pkl \
  --test-file data/small_matrix.csv \
  --submission submission_cf.csv

echo "4/5 Génération recommandations CB"
python src/recommend/generate_recommendations_content.py \
  --model-path models/content_model.pkl \
  --test-file data/small_matrix.csv \
  --submission submission_content.csv

python src/recommend/generate_recommendations_hybrid.py \
  --cf-model-path models/cf_model.pkl \
  --cb-model-path models/content_model.pkl \
  --test-file data/small_matrix.csv \
  --submission submission_hybrid.csv \
  --N 10 \
  --alpha 0.7 \
  --cf-k 10 \
  --cb-k 10

python - << 'PYCODE'
from src.evaluate.metrics import evaluate
import pandas as pd

truth = pd.read_csv("data/small_matrix.csv")\
            .rename(columns={"video_id":"item_id"})
for name in ["cf","content","hybrid"]:
    recs = pd.read_csv(f"submission_{name}.csv")\
                .rename(columns={"video_id":"item_id"})
    res  = evaluate(recs, truth, k=10)
    print(f"📊 Résultats du modèle {name}")
    print(f"  - Precision@10 : {res['precision']:.4f}")
    print(f"  - NDCG@10      : {res['ndcg']:.4f}")
PYCODE


echo "Pipeline terminé !"
