import numpy as np
from numba import njit, float64, int64
from baynet import DAG
import pandas as pd


@njit
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n_i = [x.size for x in arrays]
    n = 1
    for item in n_i:
        n *= item
    if out is None:
        out = np.zeros((n, len(arrays)), dtype=np.int64)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out

@njit
def nb_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def nb_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)

@njit
def nb_all(array, axis):
    return np_apply_along_axis(np.all, axis, array)

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result
    
@njit
def weighted_mse(data: np.ndarray, child: int, parents: np.ndarray):
    if len(parents) <= 0:
        return 0

    parent_levels = [np.unique(data[:, parents[i]]) for i in range(len(parents))]
    child_levels = np.unique(data[:, child])
    parent_combinations = cartesian(parent_levels)
    score = 0

    params = (len(parent_combinations) * (len(child_levels) - 1))
    penalty = (params * np.sqrt(data.shape[0])) / (data.shape[0] * 2)

    p_ykgxjs = np.zeros((len(child_levels), len(parent_combinations)), dtype=np.float64)
    p_xjs = np.zeros(len(parent_combinations), dtype=np.float64)

    for a, j in enumerate(parent_combinations):
        parent_rows = np.argwhere(nb_all(data[:, parents] == j, axis=1)).flatten()
        p_xjs[a] = len(parent_rows) / len(data)
        for b, k in enumerate(child_levels):
            child_rows = np.argwhere(data[parent_rows, child] == k).flatten()
            p_ykgxjs[b, a] = len(child_rows) / len(parent_rows)

    mu = nb_mean(p_ykgxjs, axis=1)
    score = 0

    for i in range(len(p_xjs)):
        score += p_xjs[i] * np.sqrt(np.sum((p_ykgxjs[:, i] - mu) ** 2))

    return score - penalty

@njit
def bic(data: np.ndarray, child: int, parents: np.ndarray):
    pass
        
def dag_score(dag: DAG, data: pd.DataFrame, score: str = "wmse") -> float:
    scores = {
        "wsme": weighted_mse,
        "bic": bic
    }
    score = 0
    for node in dag.vs:
        parents = dag.get_ancestors(node, only_parents=True)
        child_idx = data.columns.get_loc(node["name"])
        parent_idxs = np.array([data.columns.get_loc(parent["name"]) for parent in parents])
        if len(parent_idxs) <= 0:
            score += 0
        else:
            score += scores[score](data.values, child_idx, parent_idxs)
    return score
