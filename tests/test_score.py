import time
from pathlib import Path
from typing import List

import pytest
import numpy as np
import pandas as pd
from baynet import DAG
from baynet.metrics import f1_score
import itertools
import seaborn as sns
from matplotlib import pyplot as plt

from efcon.main import score, dag_score, score_from_data
from efcon.plotting import plot_simplex_from_data, plot_distribution, plots_from_dag, score_plot


def test_score_equal():
    additive = np.array(
        [[0.01, -0.02, 0.01],
         [-0.02, 0.01, 0.01],
         [-0.06, 0.02, 0.04]])
    marg1 = np.array([0.95, 0.02, 0.03])
    marg2 = np.array([0.33, 0.3, 0.37])

    p_y1gx = additive + marg1
    p_y2gx = additive + marg2

    p_x = np.array([0.333, 0.333, 0.334])

    x = np.random.choice(3, 10_000, p=p_x)

    def cond_sample(p_ygx):
        samples = np.zeros(len(x))
        for i in range(x.shape[0]):
            samples[i] = np.random.choice(3, 1, p=p_ygx[x[i], :])[0]
        return samples

    y1gx = cond_sample(p_y1gx)
    y2gx = cond_sample(p_y2gx)

    data1 = np.c_[y1gx, x]
    data2 = np.c_[y2gx, x]

    score1 = score(data1, 0, np.array([1]))
    score2 = score(data2, 0, np.array([1]))

    np.testing.assert_almost_equal(score1, score2, decimal=3)


def test_score_multidim():
    data = pd.read_csv("data.csv")
    start = time.time()
    print(score(data.values, 0, np.array([1, 2, 3, 4, 5, 6])))
    end = time.time()
    print(end - start)


def all_dags(nodes: int = 5, colnames: List[str] = []):
    n = np.sum(range(nodes))
    combs = list(itertools.product([0, 1], repeat=n))
    dags = []
    if len(colnames) <= 0:
        colnames = [f"X{i}" for i in range(nodes)]
    for comb in combs:
        mat = np.zeros([nodes, nodes])
        idxs = np.triu_indices(nodes, k=1)
        mat[idxs] = comb
        dags.append(DAG.from_amat(mat, colnames))
    return dags


def test_all_dags():
    data = pd.read_csv("data.csv")
    dags = all_dags(5, data.columns)
    dag = DAG.from_modelstring("[LT|GD][GD][GH][VD|LT:PF][PF|GH]")
    true_score = dag_score(dag, data)
    scores = [dag_score(dag_i, data) for dag_i in dags]
    sns.distplot(scores)
    plt.axvline(true_score)
    plt.show()


def test_dag_score():
    dag = DAG.from_modelstring("[LT|GD][GD][GH][VD|LT:PF][PF|GH]")
    data = pd.read_csv("data.csv")
    true_score = dag_score(dag, data)
    for i in range(1000):
        random_dag = DAG.generate("forest_fire", n_nodes=len(data.columns))
        for j in range(len(data.columns)):
            random_dag.vs[j]["name"] = data.columns[j]
        random_score = dag_score(random_dag, data)
        print(f"Random: {random_score}; True: {true_score}")
        assert random_score <= true_score


def test_plot():
    dag = DAG.from_modelstring("[LT|GD][GD][GH][VD|LT:PF][PF|GH]")
    data = pd.read_csv("data.csv")
    # plot_simplex_from_data(data, "VD", ["LT", "PF"])
    # #plot_simplex_from_data(data, "VD", ["LT"])
    # plot_simplex_from_data(data, "VD", ["LT", "GD", "PF", "GH"])
    # plot_simplex_from_data(data, "VD", ["LT", "GH"])
    plot_distribution(data, "VD", ["LT", "PF"])
    plot_distribution(data, "PF", ["GH"])


def test_highest():
    seed = 333
    true_dag = DAG.generate("forest fire", 5, seed=seed)
    true_dag.generate_discrete_parameters(alpha=[1, 10, 10], min_levels=3, max_levels=3, seed=seed)
    with open("modelstring.txt", "w") as f:
        f.write(true_dag.get_modelstring())
    data = true_dag.sample(5_000, seed=seed).astype(int)
    data.to_csv("data.csv", index_label=False)
    plots_dir = Path(__file__).parent / "plots"
    (plots_dir / "bic").mkdir(parents=True, exist_ok=True)
    (plots_dir / "skl").mkdir(parents=True, exist_ok=True)

    dags = all_dags(5, data.columns)

    # BIC

    true_score = dag_score(true_dag, data, scaled_kl=False, dynamic_norm=False)
    scores = [dag_score(dag_i, data, scaled_kl=False, dynamic_norm=False) for dag_i in dags]
    highest_dag = dags[np.argmax(scores)]

    sns.distplot(scores)
    plt.axvline(true_score)

    plt.savefig(plots_dir / "bic" / "distribution.png")
    plt.close()

    plots_from_dag(data, true_dag, plots_dir / "bic" / "true", scaled_kl=False)
    plots_from_dag(data, highest_dag, plots_dir / "bic" / "highest", scaled_kl=False)

    true_dag.graph['label'] = f"True DAG, Score: {dag_score(true_dag, data, scaled_kl=False, dynamic_norm=False):.5f}"
    highest_dag.graph['label'] = f"Highest DAG, Score: {dag_score(highest_dag, data, scaled_kl=False, dynamic_norm=False):.5f}"

    true_dag.plot(plots_dir / "bic" / "true" / "dag.png")
    highest_dag.plot(plots_dir / "bic" / "highest" / "dag.png")

    score_plot(scores, dags, true_dag, plots_dir / "bic")

    # Scaled KL

    true_score = dag_score(true_dag, data, scaled_kl=True, dynamic_norm=False)
    scores = [dag_score(dag_i, data, scaled_kl=True, dynamic_norm=False) for dag_i in dags]
    highest_dag = dags[np.argmax(scores)]

    plt.figure()

    ax = sns.distplot(scores)
    plt.axvline(true_score)

    ax.get_figure().savefig(plots_dir / "skl" / "distribution.png")

    plots_from_dag(data, true_dag, plots_dir / "skl" / "true", scaled_kl=True)
    plots_from_dag(data, highest_dag, plots_dir / "skl" / "highest", scaled_kl=True)

    true_dag.graph['label'] = f"True DAG, Score: {dag_score(true_dag, data, scaled_kl=True, dynamic_norm=False):.5f}"
    highest_dag.graph['label'] = f"Highest DAG, Score: {dag_score(highest_dag, data, scaled_kl=True, dynamic_norm=False):.5f}"

    true_dag.plot(plots_dir / "skl" / "true" / "dag.png")
    highest_dag.plot(plots_dir / "skl" / "highest" / "dag.png")

    score_plot(scores, dags, true_dag, plots_dir / "skl")


def conditional_kl_scaled(p_x, p_ygx, uniform_mu: bool, p_y):
    if uniform_mu:
        mu = np.ones(p_ygx.shape[1]) / p_ygx.shape[1]
    else:
        mu = np.mean(p_ygx, axis=0)
    return -(np.sum(p_x * (np.log(p_ygx.shape[1]) - np.sum(p_ygx * np.log(p_ygx / mu), axis=1))) - (
                np.log(p_y.shape[0]) - np.sum(p_y * np.log(p_y / mu))))


def conditional_entropy(p_x, p_ygx, p_y):
    return -np.sum(p_x * np.sum(p_ygx * np.log(p_ygx), axis=1)) - np.sum(p_y * np.log(p_y))


def _drce_l2(p_xs: np.ndarray, p_ygxs: np.ndarray, p_y: np.ndarray) -> float:
    mu = np.mean(p_ygxs, axis=0)
    min_distance = np.sqrt(np.sum((p_y - mu) ** 2))
    distance = np.sum(p_xs * np.sqrt(np.sum((p_ygxs - mu) ** 2, axis=1)))
    return distance - min_distance


def test_conditional_kl():
    additive = np.array(
        [[-0.09, -0.02, 0.11],
         [0.15, -0.11, 0.04]])

    marg1 = np.array([0.4, 0.2, 0.4])
    marg2 = np.array([0.33, 0.3, 0.37])

    p_y1gx = additive + marg1
    p_y2gx = additive + marg2

    p_x = np.array([0.2, 0.8])

    print("")

    print(f"scaled kl: {conditional_kl_scaled(p_x, p_y1gx, False, marg1)}")
    print(f"scaled kl: {conditional_kl_scaled(p_x, p_y2gx, False, marg2)}")

    print(f"l2: {_drce_l2(p_x, p_y1gx, marg1)}")
    print(f"l2: {_drce_l2(p_x, p_y2gx, marg2)}")

    print(f"conditional entropy: {conditional_entropy(p_x, p_y1gx, marg1)}")
