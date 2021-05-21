from baynet import DAG

from efcon.fast import drce, score_file
import pandas as pd
import numpy as np

from efcon.main import score
from efcon.simple_hillclimbing import hill_climbing
from baynet import metrics


def test_hc():
    seed = 11
    true_dag = DAG.generate("ide_cozman", 10, seed=seed)
    true_dag.generate_discrete_parameters(alpha=[10, 10, 10], min_levels=3, max_levels=3, seed=seed)
    with open("modelstring.txt", "w") as f:
        f.write(true_dag.get_modelstring())
    data = true_dag.sample(5000, seed=seed).astype(int)
    data.to_csv("data.csv", index_label=False)
    drdc_dag = hill_climbing(data, iterations=50)
    bic_dag = hill_climbing(data, iterations=50, drdc=False)
    drdc_score = metrics.f1_score(true_dag, drdc_dag)
    bic_score = metrics.f1_score(true_dag, bic_dag)
    drdc_skel_score = metrics.f1_score(true_dag, drdc_dag, True)
    bic_skel_score = metrics.f1_score(true_dag, bic_dag, True)
    breakpoint()
