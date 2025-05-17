# FinalProject\_2025\_lucas.collemare

## 🚀 Projet : Système de recommandation de courtes vidéos (KuaiRec)

**Objectif :** Développer un moteur de recommandations personnalisées pour des vidéos courtes, à l’instar de TikTok ou Kuaishou, en tirant parti des historiques d’interactions, des métadonnées et du réseau social des utilisateurs.

### 📂 Structure du dépôt

* `01_Datas.ipynb` : prétraitement et export au format Parquet
* `02_Build_Features.ipynb` : construction de la matrice interactions et features agrégées
* `03_Model_Developpement.ipynb` : implémentation et entraînement des modèles CF (ALS) et Content-Based (TF-IDF)
* `04_Generate_Recommandations.ipynb` : génération des recommandations CF-only, CB-only, et hybride+pop
* `05_Evaluate.ipynb` : calcul et affichage des métriques (Precision\@10, NDCG\@10)
* `preprocessed/` : fichiers Parquet prêts à l’usage
* `features/` : matrices et mappings serialisés
* `models/` : modèles entraînés (.pkl)
* `submission_*.csv` : exemples de fichiers de soumission
* `requirements.txt` : dépendances Python

## 🧠 Choix du Modèle et Justifications

1. 🎯 **Objectif principal**

   * Proposer des suggestions de vidéos courtes pertinentes pour chaque utilisateur, même en cas de cold-start.

2. ⚙️ **Approches retenues**

   * **Collaborative Filtering (CF)** via **ALS** pondéré par **BM25** : capte les similarités implicites entre utilisateurs et items.
   * **Content-Based (CB)** via **TF-IDF** sur les métadonnées (`feat`) : s’appuie sur les caractéristiques intrinsèques des vidéos.
   * **Hybrid + Popularité** : combinaison pondérée CF/CB (paramètre `alpha`) avec fallback sur popularité pour garantir un top-N complet.

3. 👥 **Gestion des cold users**

   * Les utilisateurs sans historique héritent d’un profil via propagation sur le **réseau social**, avec décroissance exponentielle du poids selon la profondeur des amis.

## 🛠️ Installation

```bash
git clone [<lien-vers-votre-repo>](https://github.com/ErkannCh/FinalProject_2025_lucas.collemare.git)
cd FinalProject_2025_lucas.collemare
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 📱 Datas
Ajouter les datas à la racine :
notebooks/
datas/


## 📈 Pipeline

1. **Prétraitement** :

   ```bash
   01_Datas.ipynb
   ```

2. **Construction des features** :

   ```bash
   02_Build_Features.ipynb
   ```

3. **Entraînement des modèles** :

   ```bash
   03_Model_Developpement.ipynb
   ```

4. **Génération des recommandations** :

   ```bash
   04_Generate_Recommandations.ipynb
   ```

5. **Évaluation** :

   ```bash
   05_Evaluate.ipynb
   ```

## 📊 Résultats et interprétation

* **Precision\@10** et **NDCG\@10** sont calculés pour chaque pipeline (CF, CB, Hybrid).
* Le modèle **Hybrid + Popularité** tend à offrir un bon compromis entre pertinence et couverture.

## 🔧 Difficultés et perspectives

Malgré plusieurs tentatives pour améliorer les métriques (notamment **NDCG** et **Recall**), je n’ai pas réussi à obtenir de résultats significativement meilleurs que ceux présentés :

* Ajustements des hyperparamètres (nombre de facteurs, régularisation, pondération BM25).
* Variation du paramètre `alpha` pour l’hybridation.
* Renforcement du mécanisme cold-start via exploration plus profonde du graphe social.

> **Bilan :** les approches explorées montrent leur potentiel, mais des optimisations supplémentaires (par ex. model tuning plus fin, incorporation de modèles séquentiels ou deep learning two-tower) seraient nécessaires pour franchir un palier de performance supplémentaire.
