from baynet import DAG

import pandas as pd
import numpy as np

from efcon.simple_hillclimbing import hill_climbing
from baynet import metrics

import time

def test_hc():
    seed = 12
    true_dag = DAG.generate("ide_cozman", 10, seed=seed)
    true_dag.generate_discrete_parameters(alpha=[30, 10, 10], min_levels=3, max_levels=3, seed=seed)
    with open("modelstring.txt", "w") as f:
        f.write(true_dag.get_modelstring())
    data = true_dag.sample(5000, seed=seed).astype(int)
    data.to_csv("data.csv", index_label=False)

    start = time.time()
    wmse_dag = hill_climbing(data, iterations=50, score="wmse")
    duration_1 = time.time() - start

    start = time.time()
    bic_dag = hill_climbing(data, iterations=50, score="bic")
    duration_2 = time.time() - start


    wmse_f1 = metrics.f1_score(true_dag, wmse_dag, True)
    wmse_prec = metrics.precision(true_dag, wmse_dag, True)
    wmse_recall = metrics.recall(true_dag, wmse_dag, True)

    bic_f1 = metrics.f1_score(true_dag, bic_dag, True)
    bic_prec = metrics.precision(true_dag, bic_dag, True)
    bic_recall = metrics.recall(true_dag, bic_dag, True)

    print("RESULTS:")
    print(f"WMSE: -- F1: {wmse_f1}; P: {wmse_prec}; R: {wmse_recall}")
    print(f"WMSE Duration: {duration_1}")
    print(f"BIC: -- F1: {bic_f1}; P: {bic_prec}; R: {bic_recall}")
    print(f"BIC Duration: {duration_2}")
