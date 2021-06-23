import pandas as pd
import numpy as np

from wmse.scores.score_utils import get_pygx_px, regret

def qnml(data: pd.DataFrame, child: int, parents: np.ndarray) -> float:

    data = data.values

    N = data.shape[0]

    if len(parents) <= 0:
        _, f_x = np.unique(data[:, child], return_counts=True)
        p_x = f_x / N
        entropy = (-np.sum(p_x * np.log2(p_x))) * N
        regrets = regret(N, len(p_x)) - regret(N, 1)
        return -(entropy + regrets)

    p_ygxs, p_xs = get_pygx_px(data, child, parents)

    cond_entropy = (-np.sum(p_xs * np.sum(p_ygxs * np.log2(p_ygxs), axis=0))) * N

    regrets = regret(N, p_ygxs.size) - regret(N, p_ygxs.shape[0])
    return -(regrets + cond_entropy) - len(parents) * np.log(2)