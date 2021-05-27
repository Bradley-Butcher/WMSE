from typing import List

import numpy as np
from numba import njit, jit

from baynet import DAG
import pandas as pd


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


def score(data: np.ndarray, child_idx: int, parent_idxs: np.ndarray, scaled_kl: bool = True,
          dynamic_norm: bool = True) -> float:
    levels = np.array([np.unique(data[:, i]).astype(int).tolist() for i in range(data.shape[1])])
    parent_combinations = np.array(np.meshgrid(*levels[parent_idxs].tolist())).T.reshape(-1, len(parent_idxs))
    return _score(data, child_idx, parent_idxs, levels, parent_combinations, scaled_kl, dynamic_norm)


def _score(data: np.ndarray,
           child_idx: int,
           parent_idxs: np.ndarray,
           levels: np.ndarray,
           parent_combinations: np.ndarray,
           scaled_kl: bool,
           dynamic_norm: bool) -> float:
    p_ygxs = np.zeros([len(parent_combinations), len(levels[child_idx])])
    p_xs = np.zeros(len(parent_combinations))

    def _inner_score(p_ygxs, p_xs):
        for j, parent_combination in enumerate(parent_combinations):
            rows = np.argwhere(np.all(data[:, parent_idxs] == parent_combination, axis=1)).flatten()
            x_count = len(rows)
            p_xs[j] = x_count / data.shape[0]
            for i in range(len(levels[child_idx])):
                p_ygxs[j, i] = np.sum(data[rows, child_idx] == levels[child_idx][i]) / x_count
        p_ygxs = np.nan_to_num(p_ygxs) + 1e-05
        p_xs = np.nan_to_num(p_xs) + 1e-05
        _, py_counts = np.unique(data[:, child_idx], return_counts=True)
        p_y = py_counts / data.shape[0]
        if scaled_kl:
            return _drce_l2(p_ygxs, p_y, p_xs), p_y
        else:
            return np.sum(p_xs * np.sum(p_ygxs * np.log(p_ygxs), axis=1)), p_y

    n_params = (p_ygxs.shape[0] * (p_ygxs.shape[1] - 1))
    if scaled_kl:
        score, p_y = _inner_score(p_ygxs, p_xs)
        if dynamic_norm:
            penalty = uni_pen_2(n_params, p_y, data.shape[0])
        else:
            penalty = sqrt_pen(n_params, data.shape[0])
        return score - penalty, penalty
    else:
        score, p_y = _inner_score(p_ygxs, p_xs)
        penalty = -1 * n_params * np.log(data.shape[0])
        return penalty + score * data.shape[0], penalty


def sqrt_pen(n_params, N):
    return (n_params * np.sqrt(N)) / (N * 2)


def uni_pen_1(n_params, p_y, N):
    uni = np.ones(p_y.shape[0]) / p_y.shape[0]
    distance = np.sqrt(np.sum((p_y - uni) ** 2))
    return (n_params * np.sqrt(N)) / (N * (1 + distance) * 2)


def uni_pen_2(n_params, p_y, N):
    uni = np.ones(p_y.shape[0]) / p_y.shape[0]
    distance = np.sqrt(np.sum((p_y - uni) ** 2))
    return (n_params ** 2) / (N * (1 / 1 - (distance / p_y.shape[0])))


@njit
def _drce_l1(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    min_distance = np.sum(np.abs(p_y - mu))
    distance = np.sum(p_xs * np.sum(np.abs(p_ygxs - mu), axis=1))
    return distance - min_distance


def _drce_l2(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    eucl = lambda x, y: np.sqrt(np.sum((x - y) ** 2))
    # distance = np.sum(p_xs * np.array([(eucl(p_ygxs[i, :], mu) - eucl(p_y, mu)) for i in range(p_ygxs.shape[0])]))
    distance = np.sum(p_xs * np.array([eucl(p_ygxs[i, :], mu) for i in range(p_ygxs.shape[0])]))
    return distance


@njit
def _drce(p_ygxs: np.ndarray, p_y: np.ndarray, p_xs: np.ndarray) -> float:
    mu = nb_mean(p_ygxs, axis=0)
    min_distance = np.log(p_y.shape[0]) - np.sum(p_y * np.log(p_y / mu))
    distance = np.sum(p_xs * (np.log(p_ygxs.shape[1]) - np.sum(p_ygxs * np.log(p_ygxs / mu), axis=1)))
    return -(distance - min_distance)


def score_from_data(data: pd.DataFrame, child: str, parents: List[str], scaled_kl: bool):
    child_idx = data.columns.get_loc(child)
    parent_idxs = np.array([data.columns.get_loc(parent) for parent in parents])
    return score(data.values, child_idx, parent_idxs, scaled_kl, False)


def dag_score(dag: DAG, data: pd.DataFrame, scaled_kl: bool, dynamic_norm: bool) -> float:
    score_total = 0
    for node in dag.vs:
        parents = dag.get_ancestors(node, only_parents=True)
        child_idx = data.columns.get_loc(node["name"])
        parent_idxs = np.array([data.columns.get_loc(parent["name"]) for parent in parents])
        # all_other = set(range(len(data.columns))) - {child_idx}
        # base = np.mean([score(data.values, other, np.array([child_idx])) for other in all_other])
        if len(parent_idxs) <= 0:
            if scaled_kl:
                # all_other = set(range(len(data.columns))) - {child_idx}
                # base = np.max([score(data.values, child_idx, np.array([other])) for other in all_other])
                score_total += 0
            else:
                p_x = data.groupby(node["name"]).size().div(len(data)).values
                p_x = np.nan_to_num(p_x)
                n_params = p_x.shape[0]
                penalty = -1 * n_params * np.log(len(data))
                entropy = -np.sum(p_x * np.log(p_x))
                base = penalty - len(data) * entropy
                score_total += base
        else:
            s, _ = score(data.values, child_idx, parent_idxs, scaled_kl, dynamic_norm)
            score_total += s
    return score_total
