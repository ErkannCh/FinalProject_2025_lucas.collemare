import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentModel:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.video_ids = None
        self.tfidf_matrix = None

    def fit(self, metadata_df: pd.DataFrame, text_field='feat'):
        self.video_ids = metadata_df['video_id'].tolist()
        corpus = metadata_df[text_field].apply(lambda x: ' '.join(x) if isinstance(x, list) else '').tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)

    def recommend(self, video_id, N=10):
        if video_id not in self.video_ids:
            return []
        idx = self.video_ids.index(video_id)
        sims = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        sim_scores = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[1:N+1]
        return [self.video_ids[i] for i, _ in sim_scores]
