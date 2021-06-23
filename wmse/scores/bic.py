import pandas as pd
import numpy as np

from wmse.scores.score_utils import get_pygx_px


def bic(data: pd.DataFrame, child: int, parents: np.ndarray):

    data = data.values

    if len(parents) <= 0:
        _, f_x = np.unique(data[:, child], return_counts=True)
        p_x = f_x / data.shape[0]
        entropy = -np.sum(p_x * np.log2(p_x))
        penalty = -1 * (p_x.shape[0] - 1) * np.log2(data.shape[0])
        return penalty - data.shape[0] * entropy
    
    p_ygxs, p_xs = get_pygx_px(data, child, parents)

    params = (p_ygxs.shape[1] * (p_ygxs.shape[0] - 1))

    cond_entropy = np.sum(p_xs * np.sum(p_ygxs * np.log2(p_ygxs), axis=0))
    penalty = -1 * params * np.log2(data.shape[0])
    return penalty + cond_entropy * data.shape[0]