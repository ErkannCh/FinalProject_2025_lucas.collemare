import inspect
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from typing import Any, Dict, List, Optional

class CFModel:
    """
    Collaborative Filtering avec ALS + BM25.
    Conserve self.user_items pour recommend sans repasser la matrice.
    """
    def __init__(self,
                 factors: int = 128,
                 regularization: float = 0.01,
                 iterations: int = 40,
                 alpha: float = 40.0,
                 K1: float = 100,
                 B: float = 0.8):
        self.factors = factors
        self.reg = regularization
        self.iter = iterations
        self.alpha = alpha
        self.K1 = K1
        self.B = B
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iter
        )
        self.user_items: Optional[sp.csr_matrix] = None

    def fit(self, interaction_matrix: sp.csr_matrix):
        # On stocke user×item pour recommend()
        self.user_items = interaction_matrix.tocsr()
        # Transpose + BM25 pour l'entraînement
        item_user = self.user_items.T
        weighted = bm25_weight(item_user, K1=self.K1, B=self.B)
        self.model.fit(weighted)

    def recommend(self,
                  user_id: Any,
                  user_map: Dict[Any, int],
                  video_map: Dict[Any, int],
                  interaction_matrix: Optional[sp.csr_matrix] = None,
                  N: int = 10
                 ) -> List[Any]:
        """
        Renvoie top-N video_id pour user_id.
        Si interaction_matrix n'est pas fourni (None), utilise self.user_items.
        """
        # 1) Récupère l’indice de l’utilisateur
        uidx = user_map.get(user_id)
        if uidx is None:
            return []

        # 2) Choisit explicitement la matrice à utiliser
        if interaction_matrix is not None:
            user_items = interaction_matrix.tocsr()
        elif self.user_items is not None:
            user_items = self.user_items
        else:
            # pas de matrice disponible
            return []

        # 3) Appel à l’ALS
        ids, scores = self.model.recommend(
            uidx,
            user_items,
            N=N,
            filter_already_liked_items=True
        )

        # 4) Reconstruction des video_id originaux
        inv_video_map = {v: k for k, v in video_map.items()}
        return [inv_video_map[i] for i in ids]
