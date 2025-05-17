!/usr/bin/env bash
set -e

# echo "1/5 Prétraitement des données"
# python src/data/preprocess.py

# echo "2/5 Construction des features"
# python src/features/build_features.py

echo "3/5 Entraînement CF"
python src/models/train.py --model cf --out models/

echo "4/5 Génération recommandations CF"
python src/recommend/generate_recommendations.py \
  --model-path models/cf_model.pkl \
  --test-file data/small_matrix.csv \
  --submission submission_cf.csv


echo "5/5 Évaluation"
python - << 'PYCODE'
from src.evaluate.metrics import evaluate
import pandas as pd

# Charge la vérité terrain
truth = pd.read_csv("data/small_matrix.csv")\
          .rename(columns={"video_id":"item_id"})

# Charge la soumission et renomme video_id → item_id
recs = pd.read_csv("submission_cf.csv")\
           .rename(columns={"video_id":"item_id"})

# Calcule et affiche
results = evaluate(recs, truth, k=10)
print("📊 Résultats du modèle CF")
print(f"  - Precision@10 : {results['precision']:.4f}")
print(f"  - Recall@10    : {results['recall']:.4f}")
print(f"  - NDCG@10      : {results['ndcg']:.4f}")
PYCODE


echo "Pipeline terminé !"
