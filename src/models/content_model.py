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
        - metadata_df: DataFrame with ['video_id', text_field]
        - interaction_matrix: CSR user×video
        - user_map, video_map: dicts to convert IDs ↔ indices
        """
        # 1) Prepare output directory
        os.makedirs("models", exist_ok=True)

        # 2) Safe conversion of each token to text
        def to_text(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return " ".join(str(tok) for tok in x)
            if pd.isna(x):
                return ""
            return str(x)

        # Build corpus and full TF-IDF
        corpus = metadata_df[text_field].apply(to_text).tolist()
        tfidf_full = self.tfidf.fit_transform(corpus)  # (n_meta_videos, n_terms)

        # Map of all metadata video_ids
        all_video_ids = metadata_df["video_id"].tolist()

        # 3) Align TF-IDF to only those videos in the interaction matrix
        #    - ordered_videos[i] = video_id whose column index is i
        ordered_videos = [None] * len(video_map)
        for vid, idx in video_map.items():
            ordered_videos[idx] = vid

        #    - map each video_id to its row in tfidf_full
        id2row = {v: i for i, v in enumerate(all_video_ids)}

        #    - extract and reorder rows
        rows = [id2row[vid] for vid in ordered_videos]
        tfidf_aligned = tfidf_full[rows, :]  # (n_videos_int, n_terms)

        # Store aligned TF-IDF and the corresponding video_ids
        self.tfidf_matrix = tfidf_aligned
        self.video_ids = ordered_videos

        # 4) Build user profiles as weighted average of their video embeddings
        um = interaction_matrix.astype("float32")
        row_sums = np.array(um.sum(axis=1)).flatten() + 1e-9
        um = um.multiply(1.0 / row_sums[:, None])
        # um (n_users×n_videos_int) dot tfidf_aligned (n_videos_int×n_terms)
        self.user_profiles = um.dot(self.tfidf_matrix).toarray()  # (n_users, n_terms)

        # 5) Save models & data
        sparse.save_npz("models/tfidf_matrix.npz", self.tfidf_matrix)
        joblib.dump(self.tfidf, "models/tfidf_vectorizer.pkl")
        joblib.dump(self.user_profiles, "models/user_profiles.npy")
        joblib.dump(user_map, "models/user_map_content.pkl")
        joblib.dump(video_map, "models/video_map_content.pkl")

        # 6) Keep maps in memory for recommend()
        self.user_map = user_map
        self.vid_map = video_map

        print("ContentModel: models and profiles saved under models/")

    def recommend(self, user_id, N: int = 10) -> list:
        """
        Return top-N video_id by cosine similarity,
        reloading artifacts from models/ if necessary.
        """
        # Reload if needed
        if self.user_profiles is None:
            self.tfidf_matrix = sparse.load_npz("models/tfidf_matrix.npz")
            self.user_profiles = joblib.load("models/user_profiles.npy")
            self.tfidf = joblib.load("models/tfidf_vectorizer.pkl")
            self.user_map = joblib.load("models/user_map_content.pkl")
            self.vid_map = joblib.load("models/video_map_content.pkl")

        inv_vid_map = {v: k for k, v in self.vid_map.items()}
        uidx = self.user_map.get(user_id)
        if uidx is None:
            return []

        profile = self.user_profiles[uidx].reshape(1, -1)  # (1, n_terms)
        sims = cosine_similarity(profile, self.tfidf_matrix).flatten()  # (n_videos_int,)

        # Get top-N indices
        best = np.argpartition(-sims, N)[:N]
        best = best[np.argsort(-sims[best])]
        return [inv_vid_map[i] for i in best]
