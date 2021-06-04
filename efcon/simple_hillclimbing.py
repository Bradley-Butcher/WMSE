import pandas as pd
import numpy as np
from baynet import DAG

from efcon.main import dag_score

import tqdm

def hill_climbing(data: pd.DataFrame, iterations: int = 10, drdc: bool = True, dynamic_norm: bool = True):
    n_var = len(data.columns)
    current_amat = np.zeros([n_var, n_var])
    current_score = dag_score(DAG.from_amat(current_amat, colnames=data.columns), data, scaled_kl=drdc, dynamic_norm=dynamic_norm)
    for i in tqdm.tqdm(range(iterations)):
        best_improvement = 0
        best_amat = current_amat
        for j in tqdm.tqdm(range(current_amat.size)):
            mod_amat = current_amat.copy()
            mod_amat[int(j / n_var), j % n_var] = 1

            if np.array_equal(mod_amat, current_amat):
                continue
            try:
                temp_dag = DAG.from_amat(mod_amat, colnames=data.columns)
            except AssertionError:
                continue
            temp_score = dag_score(temp_dag, data, scaled_kl=drdc, dynamic_norm=dynamic_norm)
            delta = temp_score - current_score
            if delta > best_improvement:
                best_improvement = delta
                best_amat = mod_amat
        current_amat = best_amat
        if best_improvement == 0:
            return DAG.from_amat(current_amat, colnames=data.columns)
    return DAG.from_amat(current_amat, colnames=data.columns)
