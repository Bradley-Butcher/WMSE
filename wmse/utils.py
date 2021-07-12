import numpy as np
import pandas as pd

from pathlib import Path
from typing import List

from wmse.scores import weighted_mse

def score_file(data: pd.DataFrame, savepath: Path, penalty_mult: float = 0.1, use_bic: bool = False):
    lines = []
    lines += [str(len(data.columns))]
    for variable in data.columns:
        scores = weighted_mse(data.columns.get_loc(variable), data.values, penalty_mult, use_bic=use_bic)
        lines += [f"{variable} {len(scores)}"]
        for p_set, score in scores:
            if len(p_set) > 0:
                lines += [f"{score - 1000000} {len(p_set)} {' '.join([str(data.columns[p]) for p in p_set])}"]
            else:
                lines += [f"{score - 1000000} 0"]
    with open(savepath, "w") as f:
        f.write('\n'.join(lines) + '\n')

def get_distributions(data: pd.DataFrame, child: str, parents: List[str]):
    child_idx = data.columns.get_loc(child)
    parent_idxs = np.array([data.columns.get_loc(parent) for parent in parents])
    data = data.values
    levels = np.array([np.unique(data[:, i]).astype(int).tolist() for i in range(data.shape[1])])
    parent_combinations = np.array(np.meshgrid(*levels[parent_idxs].tolist())).T.reshape(-1, len(parent_idxs))
    p_ygxs = np.zeros([len(parent_combinations), len(levels[child_idx])])
    p_xs = np.zeros(len(parent_combinations))
    for j, parent_combination in enumerate(parent_combinations):
        rows = np.argwhere(np.all(data[:, parent_idxs] == parent_combination, axis=1)).flatten()
        x_count = len(rows)
        p_xs[j] = x_count / data.shape[0]
        for i in range(len(levels[child_idx])):
            p_ygxs[j, i] = np.sum(data[rows, child_idx] == levels[child_idx][i]) / x_count
    p_ygxs = np.nan_to_num(p_ygxs)
    mu = np.mean(p_ygxs, axis=0)
    return p_ygxs, mu, p_xs