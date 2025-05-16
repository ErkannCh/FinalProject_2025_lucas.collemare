import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from typing import Any, Dict, List

class CFModel:
    def __init__(self,
					factors: int = 50,
					regularization: float = 0.01,
					iterations: int = 20,
					alpha: float = 40.0,
                	epochs=50,
                    lr=0.01):
        self.factors = factors
        self.reg = regularization
        self.iter = iterations
        self.alpha = alpha
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=lr,
            regularization=regularization,
            iterations=epochs
        )

    def fit(self, interaction_matrix: sp.csr_matrix):
        # implicit attend un item×user matrix pondérée
        confidence = (interaction_matrix * self.alpha).astype('double')
        self.model.fit(confidence.T)

def recommend(self,
                  user_id: Any,
                  user_map: Dict[Any,int],
                  video_map: Dict[Any,int],
                  interaction_matrix: sp.csr_matrix,
                  N: int = 10
                 ) -> List[Any]:
        """
        Retourne le top-N video_id pour user_id.
        On construit d’abord confidence = interaction_matrix * alpha,
        puis on transpose en item×user, comme pour le fit.
        """
        # 1) on récupère l’indice user interne
        uidx = user_map.get(user_id)
        if uidx is None:
            return []

        # 2) on reconstruit exactement la même matrice que pour le fit
        #    fit(confidence.T) → recommend(uidx, confidence.T, N)
        #    donc ici :
        confidence = (interaction_matrix * self.alpha).astype('double')
        item_user = confidence.T.tocsr()

        # 3) appel à implicit
        recs = self.model.recommend(uidx, item_user, N=N)

        # 4) on récupère les video_id originaux
        inv_video_map = {v: k for k, v in video_map.items()}
        return [inv_video_map[idx] for idx, _ in recs]

