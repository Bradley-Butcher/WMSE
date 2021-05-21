from efcon.simple_hillclimbing import hill_climbing
from baynet import metrics, DAG
from tqdm import tqdm
import pandas as pd


def scores(true_dag, comp_dag, name):
    return {
        f"{name} PRECISION": metrics.precision(true_dag, comp_dag, True),
        f"{name} RECALL": metrics.recall(true_dag, comp_dag, True),
        f"{name} V PRECISION": metrics.v_precision(true_dag, comp_dag),
        f"{name} V RECALL": metrics.v_recall(true_dag, comp_dag),
    }


if __name__ == "__main__":
    n_exp = 10
    samples = [100, 200, 400, 800, 1600, 3200, 6400]
    alpha = [10, 10, 10]
    results = []
    with tqdm(total=n_exp * len(samples)) as pbar:
        for i in range(n_exp):
            seed = i
            true_dag = DAG.generate("ide_cozman", 10, seed=seed)
            true_dag.generate_discrete_parameters(alpha=alpha, min_levels=3, max_levels=3, seed=seed)
            for n_sample in samples:
                data = true_dag.sample(n_sample, seed=seed).astype(int)
                n_edges = len(true_dag.edges)
                drdc_dag = hill_climbing(data, iterations=50)
                bic_dag = hill_climbing(data, iterations=50, drdc=False)
                results.append(
                    {**{"Seed": seed, "Samples": n_sample}, **scores(true_dag, drdc_dag, "L2"), **scores(true_dag, bic_dag, "BIC")})
                pbar.update(1)
    results_df = pd.DataFrame(results)
    results_df.to_csv("../results/l2_results_precrecall_imb.csv")
