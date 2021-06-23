from baynet import DAG

from wmse.fast import drce, score_file
import pandas as pd
import numpy as np

from wmse.main import score


def test_score_computes():
    data = pd.read_csv("data.csv")
    scores = drce(0, data.values, 0.1)


def test_score_file():
    seed = 5
    true_dag = DAG.generate("ide_cozman", 10, seed=seed)
    true_dag.generate_discrete_parameters(alpha=[10, 10, 10], min_levels=3, max_levels=3, seed=seed)
    with open("modelstring.txt", "w") as f:
        f.write(true_dag.get_modelstring())
    data = true_dag.sample(5000, seed=seed).astype(int)
    data.to_csv("data.csv", index_label=False)
    score_file(data, "drcd_test.txt", 1, use_bic=False)
    score_file(data, "bic_test.txt", 1, use_bic=True)

