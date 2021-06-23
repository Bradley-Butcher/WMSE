import pandas as pd
import numpy as np
from baynet import DAG

from efcon.score_function import Scorer
from efcon.fges.fges_scores import WMSEScore, BICScore, qNMLScore, BDeuScore
from efcon.fges.fges import FGES

import tqdm

def _step(data: pd.DataFrame, amat: np.ndarray, score: str, j: int, scorer: Scorer):
    n_var = len(data.columns)

    if amat[int(j / n_var), j % n_var] == 1:
        amat_c1 = amat.copy()
        amat_c1[int(j / n_var), j % n_var] = 0

        amat_c2 = amat.copy()
        amat_c2[int(j / n_var), j % n_var] = 0
        amat_c2[j % n_var, int(j / n_var)] = 1

        try:
            temp_dag = DAG.from_amat(amat_c1, colnames=data.columns)
            score_c1 = scorer.score(temp_dag, data, score)
        except AssertionError:
            score_c1 = None
        
        try:
            temp_dag = DAG.from_amat(amat_c2, colnames=data.columns)
            score_c2 = scorer.score(temp_dag, data, score)
        except AssertionError:
            score_c2 = None

        if score_c1:
            if score_c2:
                if score_c1 > score_c2:
                    return amat_c1, score_c1
                else:
                    return amat_c2, score_c2
            else:
                return amat_c1, score_c1
        else:
            if score_c2:
                return amat_c2, score_c2
            else:
                return None, None
    else:
        amat_c3 = amat.copy()
        amat_c3[int(j / n_var), j % n_var] = 1
        score_c3 = None
        try:
            temp_dag = DAG.from_amat(amat_c3, colnames=data.columns)
            score_c3 = scorer.score(temp_dag, data, score)
        except AssertionError:
            score_c3 = None
        if score_c3 is not None:
            return amat_c3, score_c3
        else:
            return None, None



def maximal_hill_climbing(data: pd.DataFrame, iterations: int = 10, score: str = "wmse"):
    n_var = len(data.columns)
    current_amat = np.zeros([n_var, n_var])
    scorer = Scorer()
    current_score = scorer.score(DAG.from_amat(current_amat, colnames=data.columns), data, score)
    for i in tqdm.tqdm(range(iterations)):
        best_improvement = 0
        best_amat = current_amat
        for j in range(current_amat.size):
            mod_amat, temp_score = _step(data, current_amat, score, j, scorer)
            if mod_amat is None:
                continue
            delta = temp_score - current_score
            if delta > best_improvement:
                best_improvement = delta
                best_amat = mod_amat
        current_amat = best_amat
        current_dag = DAG.from_amat(current_amat, colnames=data.columns)
        current_score = scorer.score(current_dag, data, score)
        if best_improvement == 0:
            return DAG.from_amat(current_amat, colnames=data.columns)
    return DAG.from_amat(best_amat, colnames=data.columns)


def hill_climbing(data: pd.DataFrame, iterations: int = 10, score: str = "wmse"):
    n_var = len(data.columns)
    current_amat = np.zeros([n_var, n_var])
    scorer = Scorer()
    current_score = scorer.score(DAG.from_amat(current_amat, colnames=data.columns), data, score)
    for i in tqdm.tqdm(range(iterations)):
        break_condition = True
        for j in np.random.permutation(current_amat.size):
            mod_amat = current_amat.copy()
            if mod_amat[int(j / n_var), j % n_var] == 1:
                b = np.random.choice(a=[False, True])
                if b:
                    mod_amat[int(j / n_var), j % n_var] = 0
                else:
                    mod_amat[int(j / n_var), j % n_var] = 0
                    mod_amat[j % n_var, int(j / n_var)] = 1
            else:
                mod_amat[int(j / n_var), j % n_var] = 1
            try:
                temp_dag = DAG.from_amat(mod_amat, colnames=data.columns)
            except AssertionError:
                continue
            temp_score = scorer.score(temp_dag, data, score)
            if temp_score > current_score:
                current_amat = mod_amat
                current_score = temp_score
                break_condition = False
                break
        if break_condition:
            break
    return DAG.from_amat(current_amat, colnames=data.columns)


def safe_dag(nodes, edges):
    dag = DAG()
    dag.add_vertices(nodes)
    for source, target in edges:
        try:
            dag.add_edge(source, target)
        except AssertionError:
            continue
    return dag


def run_fges(data: pd.DataFrame, score: str = "wmse") -> DAG:
    scores = {
            "wmse": WMSEScore,
            "bic": BICScore,
            "qnml": qNMLScore,
            "bdeu": BDeuScore
    }
    variables = list(range(data.shape[1]))

    score_obj = scores[score](data)
    fges_obj = FGES(variables, score_obj)
    graph = fges_obj.search()["graph"]
    dag = safe_dag(graph.nodes, graph.edges)
    for i in variables:
        dag.get_node(i)["name"] = data.columns[i]
    return dag