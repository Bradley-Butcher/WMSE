import numpy as np
import pandas as pd

from wmse.scores.score_utils import get_pygx_px, regret

def weighted_mse(data: pd.DataFrame, child: int, parents: np.ndarray, regret_penalty: bool = False, split: bool = False):

    data = data.values

    score = 0

    N = data.shape[0]

    if len(parents) <= 0:
        return score

    p_ygxs, p_xs = get_pygx_px(data, child, parents)
    
    p_y = np.dot(p_ygxs, p_xs)

    mu = np.mean(p_ygxs, axis=1)

    score = np.sum(p_xs * np.sqrt(np.sum((p_ygxs.T - mu) ** 2, axis=1)))

    if regret_penalty:
        penalty = regret(N, p_ygxs.size) - regret(N, p_ygxs.shape[0]) / np.sqrt(N * 2) 
    else:
        params = (p_ygxs.shape[1] * (p_ygxs.shape[0] - 1))
        penalty = params * np.log(data.shape[0]) / (data.shape[0] / np.log(data.shape[0]))
    
    return score - penalty