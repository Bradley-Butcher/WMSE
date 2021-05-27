from typing import List

from efcon.simple_hillclimbing import hill_climbing
from baynet import metrics, DAG
from tqdm import tqdm
import pandas as pd


def scores(true_dag, comp_dag):
    return {
        f"PRECISION": metrics.precision(true_dag, comp_dag, True),
        f"RECALL": metrics.recall(true_dag, comp_dag, True),
        f"V PRECISION": metrics.v_precision(true_dag, comp_dag),
        f"V RECALL": metrics.v_recall(true_dag, comp_dag),
        f"DAG PRECISION": metrics.precision(true_dag, comp_dag, False),
        f"DAG RECALL": metrics.recall(true_dag, comp_dag, False),
    }


def experiment(samples: List[int], alpha: List[float], n_exp: int, filename: str, max_iter: int = 50, struc_type: str = ):
    results = []
    with tqdm(total=n_exp * len(samples)) as pbar:
        for i in range(n_exp):
            seed = i
            true_dag = DAG.generate("ide_cozman", 10, seed=seed)
            true_dag.generate_discrete_parameters(alpha=alpha, min_levels=3, max_levels=3, seed=seed)
            for n_sample in samples:
                data = true_dag.sample(n_sample, seed=n_sample).astype(int)
                l2_dn_dag = hill_climbing(data, iterations=max_iter)
                l2_dag = hill_climbing(data, iterations=max_iter, dynamic_norm=False)
                bic_dag = hill_climbing(data, iterations=max_iter, drdc=False, dynamic_norm=False)
                dags = [l2_dn_dag, l2_dag, bic_dag]
                names = ["L2-DN", "L2", "BIC"]
                for dag, name in zip(dags, names):
                    results.append({**{"Seed": seed, "Samples": n_sample, "Score": name}, **scores(true_dag, dag)})
                pbar.update(1)
                results_df = pd.DataFrame(results)
                results_df.to_csv(f"../results/{filename}.csv")


if __name__ == "__main__":
    n_exp = 10
    low_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    samples = [1000, 2000, 3000, 4000, 5000]

    # Low Sample Experiments

    # experiment(samples=low_samples, alpha=[10, 10, 10], n_exp=n_exp, filename="balanced_low")
    # experiment(samples=low_samples, alpha=[30, 10, 10], n_exp=n_exp, filename="mid_imb_low")
    # experiment(samples=low_samples, alpha=[10, 50, 10], n_exp=n_exp, filename="extreme_imb_low")

    experiment(samples=low_samples, alpha=10, n_exp=n_exp, filename="dynamic_low")


    # Normal Sample Experiments

    # experiment(samples=samples, alpha=[10, 10, 10], n_exp=n_exp, filename="balanced")
    # experiment(samples=samples, alpha=[30, 10, 10], n_exp=n_exp, filename="mid_imb")
    # experiment(samples=samples, alpha=[10, 50, 10], n_exp=n_exp, filename="extreme_imb")

