from baynet import DAG

from efcon.fast import drce, score_file
import pandas as pd
import numpy as np

from efcon.simple_hillclimbing import hill_climbing
from baynet import metrics


def test_hc():
    seed = 12
    true_dag = DAG.generate("ide_cozman", 20, seed=seed)
    true_dag.generate_discrete_parameters(alpha=[30, 10, 10], min_levels=3, max_levels=3, seed=seed)
    with open("modelstring.txt", "w") as f:
        f.write(true_dag.get_modelstring())
    data = true_dag.sample(5000, seed=seed).astype(int)
    data.to_csv("data.csv", index_label=False)
    drdc_dag = hill_climbing(data, iterations=50, dynamic_norm=False)
    # bic_dag = hill_climbing(data, iterations=50, drdc=False)

    drdc_f1 = metrics.f1_score(true_dag, drdc_dag, True)
    drdc_prec = metrics.precision(true_dag, drdc_dag, True)
    drdc_recall = metrics.recall(true_dag, drdc_dag, True)

    # bic_f1 = metrics.f1_score(true_dag, bic_dag, True)
    # bic_prec = metrics.precision(true_dag, bic_dag, True)
    # bic_recall = metrics.recall(true_dag, bic_dag, True)

    print("RESULTS:")
    print(f"L2: -- F1: {drdc_f1}; P: {drdc_prec}; R: {drdc_recall}")
    # print(f"BIC: -- F1: {bic_f1}; P: {bic_prec}; R: {bic_recall}")