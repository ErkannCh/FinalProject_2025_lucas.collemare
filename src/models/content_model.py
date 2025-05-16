import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import joblib

class ContentModel:
    """
    Content-based recommender using TF-IDF embeddings and user profiles.
    """

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=self.max_features)
        self.video_ids = []
        self.tfidf_matrix = None
        # user_profiles[user_idx] = np.array embedding
        self.user_profiles = None

    def fit(self,
            metadata_df: pd.DataFrame,
            interaction_matrix: sp.csr_matrix,
            user_map: dict,
            video_map: dict,
            text_field: str = 'feat'):
        """
        metadata_df: DataFrame with columns ['video_id', text_field]
        interaction_matrix: sparse CSR user×video (binary or weighted)
        user_map: map from user_id → uidx
        video_map: map from video_id → vidx
        """
        # 1) TF-IDF sur les textes
        self.video_ids = metadata_df['video_id'].tolist()
        corpus = metadata_df[text_field].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        ).tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)  # shape (n_videos, n_terms)

        # 2) Construction des profils utilisateurs : 
        #    profil(u) = moyenne pondérée des embeddings TF-IDF des vidéos qu’il a vues
        n_users = interaction_matrix.shape[0]
        n_videos = interaction_matrix.shape[1]
        # on normalise par lignes pour convertir en probabilités
        umatrix = interaction_matrix.astype('float32')
        umatrix = umatrix.multiply(1.0 / (umatrix.sum(axis=1) + 1e-9))
        # Profil = umatrix × tfidf_matrix
        # umatrix (n_users×n_videos) × tfidf_matrix (n_videos×n_terms)
        self.user_profiles = umatrix.dot(self.tfidf_matrix).toarray()  # (n_users, n_terms)

        # 3) Sauvegarde des mappages et vecteurs
        joblib.dump(self.tfidf, "models/tfidf_vectorizer.pkl")
        joblib.dump(self.user_profiles, "models/user_profiles.npy")
        joblib.dump(user_map, "models/user_map_content.pkl")
        joblib.dump(video_map, "models/video_map_content.pkl")
        print("ContentModel: TF-IDF and user profiles saved.")

    def recommend(self,
                  user_id,
                  N: int = 10) -> list:
        """
        Retourne top-N video_ids pour user_id en similarité cosinus entre
        le profil utilisateur et chaque embedding vidéo.
        """
        # 1) Charger mappings si besoin
        if self.user_profiles is None:
            self.user_profiles = joblib.load("models/user_profiles.npy")
            self.tfidf = joblib.load("models/tfidf_vectorizer.pkl")
            self.user_map = joblib.load("models/user_map_content.pkl")
            self.vid_map = joblib.load("models/video_map_content.pkl")
        else:
            self.user_map = joblib.load("models/user_map_content.pkl")
            self.vid_map = joblib.load("models/video_map_content.pkl")

        inv_vid_map = {v: k for k, v in self.vid_map.items()}

        uidx = self.user_map.get(user_id, None)
        if uidx is None:
            return []

        # 2) Calcul des similarités
        profile = self.user_profiles[uidx].reshape(1, -1)  # (1, n_terms)
        sims = cosine_similarity(profile, self.tfidf_matrix).flatten()  # (n_videos,)

        # 3) Top-N indices
        best_idx = np.argpartition(-sims, N)[:N]
        best_idx = best_idx[np.argsort(-sims[best_idx])]  # tri final

        return [inv_vid_map[i] for i in best_idx]
