import numpy as np
from numba import jit, njit, float64, int64

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
def nb_abs(array, axis):
    return np_apply_along_axis(np.abs, axis, array)

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
def nb_argwhere_all(data, value):
    indicies = np.zeros(data.shape[0], dtype=np.bool_)
    for i in range(data.shape[0]):
        eval = data[i] == value
        prod = 1
        if data.ndim > 1:
            for e in eval:
                prod *= e
        else:
            prod *= eval
        indicies[i] = prod
    return np.where(indicies)[0]

@njit
def get_pygx_px(data: np.ndarray, y_idx: int, x_idxs: np.ndarray):
    parent_levels = [np.unique(data[:, x_idxs[i]]) for i in range(len(x_idxs))]
    child_levels = np.unique(data[:, y_idx])
    parent_combinations = cartesian(parent_levels)

    p_ykgxjs = np.zeros((len(child_levels), len(parent_combinations)), dtype=np.float64)
    p_xjs = np.zeros(len(parent_combinations), dtype=np.float64)

    for a, j in enumerate(parent_combinations):
        parent_rows = nb_argwhere_all(data[:, x_idxs], j)
        parent_data = data[parent_rows, :]
        if parent_data.shape[0] == 0:
            p_xjs[a] = 0
            p_ykgxjs[:, a] = 0
            continue
        p_xjs[a] = len(parent_rows) / len(data)
        for b, k in enumerate(child_levels):
            child_rows = nb_argwhere_all(parent_data[:, y_idx], k)
            p_ykgxjs[b, a] = len(child_rows) / len(parent_rows)

    return p_ykgxjs, p_xjs

def _binary_regret_precal(num_categories: int) -> float:
    some_var_p = 10
    if num_categories < 1:
        return 0.0
    cur_sum = 1.0
    i_bound = 1.0
    bound = int(np.ceil(2.0 + np.sqrt(2.0 * num_categories * some_var_p * np.log(10))))
    for i in range(bound):
        i_bound = (num_categories - i) * (i_bound / num_categories)
        cur_sum += i_bound
    return cur_sum


def _regret_precal(num_categories: int, unique_vals: int) -> float:
    if unique_vals < 1:
        return 0.0
    if unique_vals == 1:
        return 1.0
    cur_sum = _binary_regret_precal(num_categories)
    old_sum = 1.0
    if unique_vals > 2:
        for j in range(1, unique_vals - 1):
            new_sum = cur_sum + (num_categories * old_sum) / float(j)
            old_sum = cur_sum
            cur_sum = new_sum
    return cur_sum


def regret(num_categories: int, unique_vals: int) -> float:
    """
    Compute the Szpankowski and Weinberger approximation for regret.

    (normalization term for NML) for a vector.

    :param num_categories: The length of the categorical vector which (q)NML is being calculated for
    :type num_categories: integer
    :param unique_vals: The number of unique categorical values in a vector
    :type unique_vals: integer
    :return: Regret approximation
    :type: numpy.float64
    """
    if unique_vals > 100:
        return _regret_eqn7(num_categories, unique_vals)
    return _regret_eqn6(num_categories, unique_vals)

def _regret_eqn7(num_categories: int, unique_vals: int, test: bool = False) -> float:
    alpha = unique_vals / num_categories
    cappa = 0.5 + 0.5 * np.sqrt(1 + 4 / alpha)
    log_reg = num_categories * (
        np.log(alpha) + (alpha + 2) * np.log(cappa) - 1 / cappa
    ) - 0.5 * np.log(cappa + 2 / alpha)
    if not test:
        log_reg /= np.log(2)  ### matches the table in the paper WITHOUT this log(2)
    return log_reg

def _regret_eqn6(num_categories: int, unique_vals: int) -> float:
    costs = _regret_precal(num_categories, unique_vals)
    if costs <= 0.0:
        return 0.0
    return np.log2(costs)