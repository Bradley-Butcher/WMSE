from itertools import chain, combinations
from pathlib import Path
from typing import Callable, List

import cartesian as cartesian
import numpy as np
import pandas as pd
from numba import jit, njit

from tqdm import tqdm

from scipy.spatial import distance



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
def nb_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@njit
def nb_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)


@njit
def _drce_l1(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    min_distance = np.sum(np.abs(p_y - mu))
    distance = np.sum(p_xs * np.sum(np.abs(p_ygxs - mu), axis=1))
    return distance - min_distance


def _drce_l2(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    eucl = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    distance = np.sum(p_xs * np.array([(eucl(p_ygxs[i, :], mu) - eucl(p_y, mu)) for i in range(p_ygxs.shape[0])]))
    return distance


@njit
def _drce(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    min_distance = np.log(p_y.shape[0]) - np.sum(p_y * np.log(p_y / mu))
    distance = np.sum(p_xs * (np.log(p_ygxs.shape[1]) - np.sum(p_ygxs * np.log(p_ygxs / mu), axis=1)))
    return -(distance - min_distance)


def _drce_cos(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = np.mean(p_ygxs, axis=0)
    cos_dist = lambda x, y: 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    min_distance = cos_dist(p_y, mu)
    distance = np.sum(p_xs * np.array([cos_dist(p_ygxs[i, :], mu) for i in range(p_ygxs.shape[0])]))
    return distance - min_distance


def _drce_js(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = np.mean(p_ygxs, axis=0)
    cos_dist = lambda x, y: 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    min_distance = distance.braycurtis(p_y, mu)
    dist = np.sum(p_xs * np.array([distance.braycurtis(p_ygxs[i, :], mu) for i in range(p_ygxs.shape[0])]))
    return dist - min_distance


def _bic(p_ygxs: np.ndarray, p_xs: np.ndarray) -> float:
    return -np.sum(p_xs * np.sum(p_ygxs * np.log(p_ygxs), axis=1))


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(5))


def _all_parent_sets(child_idx: int, data: np.ndarray) -> List[int]:
    parent_idxs = list(set(range(data.shape[1])) - {child_idx})
    return powerset(parent_idxs)


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n_i = [x.size for x in arrays]
    n = 1
    for item in n_i:
        n *= item
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=int)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def drce(child_idx: int, data: np.ndarray, penalty_mult: float, use_bic: bool = False) -> List[tuple]:
    parent_sets = _all_parent_sets(child_idx, data)
    levels = np.array([np.unique(data[:, i]).astype(int).tolist() for i in range(data.shape[1])])

    def _marginal_dist(idx):
        _, py_counts = np.unique(data[:, idx], return_counts=True)
        return py_counts / data.shape[0]

    p_y = _marginal_dist(child_idx)

    def _score_sets():
        scores = []
        for p_set in tqdm(parent_sets):
            if len(p_set) <= 0:
                if use_bic:
                    e_pen = -(1 * penalty_mult) * (len(p_y) - 1) * np.log(len(data))
                    scores += [(p_set, e_pen + (np.sum(p_y * np.log(p_y)) * data.shape[0]))]
                else:
                    scores += [(p_set, 0)]
                continue
            p_set = list(p_set)

            parent_combinations = cartesian(levels[p_set].tolist())
            score = 0
            p_ygxs = np.zeros([len(parent_combinations), len(levels[child_idx])])
            p_xs = np.zeros(len(parent_combinations))

            for j, parent_combination in enumerate(parent_combinations):
                rows = np.argwhere(np.all(data[:, p_set] == parent_combination, axis=1)).flatten()
                x_count = len(rows)
                p_xs[j] = x_count / data.shape[0]
                for i in range(len(levels[child_idx])):
                    p_ygxs[j, i] = np.sum(data[rows, child_idx] == levels[child_idx][i]) / x_count
            p_ygxs = np.nan_to_num(p_ygxs) + 1e-05
            p_xs = np.nan_to_num(p_xs) + 1e-05
            n_params = (p_ygxs.shape[0] * (p_ygxs.shape[1] - 1))
            penalty = -(1 * penalty_mult) * n_params * np.log(data.shape[0])
            if not use_bic:
                score += penalty / data.shape[0] + _drce_l2(p_ygxs, p_y, p_xs)
            else:
                score += penalty - _bic(p_ygxs, p_xs) * data.shape[0]
            scores += [(p_set, score)]
        return scores

    return _score_sets()


def score_file(data: pd.DataFrame, savepath: Path, penalty_mult: float = 0.1, use_bic: bool = False):
    lines = []
    lines += [str(len(data.columns))]
    for variable in data.columns:
        scores = drce(data.columns.get_loc(variable), data.values, penalty_mult, use_bic=use_bic)
        lines += [f"{variable} {len(scores)}"]
        for p_set, score in scores:
            if len(p_set) > 0:
                lines += [f"{score - 1000000} {len(p_set)} {' '.join([str(data.columns[p]) for p in p_set])}"]
            else:
                lines += [f"{score - 1000000} 0"]
    with open(savepath, "w") as f:
        f.write('\n'.join(lines) + '\n')
