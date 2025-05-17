import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from scipy import sparse
import joblib

class ContentModel:
    """
    Content-based recommender using TF-IDF (uni+bi-grammes) and user profiles.
    """

    def __init__(self,
                 max_features: int = 10000,
                 ngram_range=(1, 2),
                 stop_words="english"):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words

        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words
        )
        self.video_ids = None
        self.tfidf_matrix = None
        self.user_profiles = None
        self.user_map = None
        self.vid_map = None

    def fit(self,
            metadata_df: pd.DataFrame,
            interaction_matrix: sp.csr_matrix,
            user_map: dict,
            video_map: dict,
            text_field: str = "feat"):
        """
        - metadata_df: DataFrame avec ['video_id', text_field]
        - interaction_matrix: CSR user×video
        - user_map, video_map: dicts pour convertir IDs ↔ indices
        """
        # 1) Prépare dossier de sortie
        os.makedirs("models", exist_ok=True)

        # 2) Conversion safe de chaque token en str
        def to_text(x):
            if isinstance(x, (list, tuple)):
                return " ".join(str(tok) for tok in x)
            elif pd.notnull(x):
                return str(x)
            else:
                return ""

        corpus = metadata_df[text_field].apply(to_text).tolist()
        self.video_ids = metadata_df["video_id"].tolist()

        # 3) TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)  # (n_videos, n_terms)

        # 4) Profil utilisateur = moyenne pondérée des embeddings
        n_users, n_videos = interaction_matrix.shape
        um = interaction_matrix.astype("float32")
        row_sums = np.array(um.sum(axis=1)).flatten() + 1e-9
        um = um.multiply(1.0 / row_sums[:, None])
        self.user_profiles = um.dot(self.tfidf_matrix).toarray()  # (n_users, n_terms)

        # 5) Sauvegarde modèles & données
        sparse.save_npz("models/tfidf_matrix.npz", self.tfidf_matrix)
        joblib.dump(self.tfidf, "models/tfidf_vectorizer.pkl")
        joblib.dump(self.user_profiles, "models/user_profiles.npy")
        joblib.dump(user_map, "models/user_map_content.pkl")
        joblib.dump(video_map, "models/video_map_content.pkl")

        # 6) Stocke en mémoire pour recommend()
        self.user_map = user_map
        self.vid_map = video_map

        print("ContentModel: modèles et profils enregistrés sous models/")

    def recommend(self, user_id, N: int = 10) -> list:
        """
        Retourne top-N video_id par similarité cosinus,
        en rechargeant si besoin les matrices depuis models/.
        """
        # Si on est hors session, recharge tout
        if self.user_profiles is None:
            self.tfidf_matrix = sparse.load_npz("models/tfidf_matrix.npz")
            self.user_profiles = joblib.load("models/user_profiles.npy")
            self.tfidf = joblib.load("models/tfidf_vectorizer.pkl")
            self.user_map = joblib.load("models/user_map_content.pkl")
            self.vid_map = joblib.load("models/video_map_content.pkl")

        inv_vid_map = {v: k for k, v in self.vid_map.items()}
        uidx = self.user_map.get(user_id, None)
        if uidx is None:
            return []

        profile = self.user_profiles[uidx].reshape(1, -1)  # (1, n_terms)
        sims = cosine_similarity(profile, self.tfidf_matrix).flatten()  # (n_videos,)

        # Top-N indices
        best = np.argpartition(-sims, N)[:N]
        best = best[np.argsort(-sims[best])]
        return [inv_vid_map[i] for i in best]
