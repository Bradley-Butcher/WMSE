from typing import List, Optional

from wmse.optimization import run_fges
from baynet import metrics, DAG
from tqdm import tqdm
import pandas as pd

from pathlib import Path

def scores(true_dag: DAG, comp_dag: DAG):
    return {
        f"PRECISION": metrics.precision(true_dag, comp_dag, True),
        f"RECALL": metrics.recall(true_dag, comp_dag, True),
        f"V PRECISION": metrics.v_precision(true_dag, comp_dag),
        f"V RECALL": metrics.v_recall(true_dag, comp_dag),
        f"DAG PRECISION": metrics.precision(true_dag, comp_dag, False),
        f"DAG RECALL": metrics.recall(true_dag, comp_dag, False),
        f"HAMMING": metrics.shd(true_dag, comp_dag, True),
        f"SCALED HAMMING": metrics.shd(true_dag, comp_dag, True) / len(true_dag.edges),
        f"SHD": metrics.shd(true_dag, comp_dag, False),
        f"SCALED SHD": metrics.shd(true_dag, comp_dag, False) / len(true_dag.edges)
    }

def experiment(
    filename: str, 
    samples: List[int], 
    alpha: Optional[List[float]] = None,
    bif_dataset: Optional[str] = None, 
    n_exp: int = 10,
    max_iter: int = 1000,
    struc_type: str = "ide_cozman"
) -> None:
    bif_dir = Path(__file__).parent.parent / "data" / "bifs"
    results = []
    with tqdm(total=n_exp * len(samples)) as pbar:
        for i in range(n_exp):
            seed = i
            if alpha and bif_dataset:
                raise ValueError("Only EITHER alpha or bif_dataset should be set.")
            if alpha:
                true_dag = DAG.generate(struc_type, 40, seed=seed)
                true_dag.generate_discrete_parameters(alpha=alpha, min_levels=3, max_levels=3, seed=seed)
            elif bif_dataset:
                true_dag = DAG.from_bif(bif_dir / f"{bif_dataset}.bif")
            for n_sample in samples:
                data = true_dag.sample(n_sample, seed=i)
                data = data.apply(lambda col: pd.Categorical(col).codes)
                try:
                    bdeu_dag = run_fges(data,score="bdeu")
                    wmse_dag = run_fges(data, score="wmse")
                    bic_dag = run_fges(data, score="bic")
                    qnml_dag = run_fges(data,score="qnml")
                except TypeError:
                    pbar.update(1)
                    continue
                dags = [bdeu_dag, wmse_dag, bic_dag, qnml_dag]
                names = ["BDeu", "WMSE", "BIC", "qNML"]
                for dag, name in zip(dags, names):
                    results.append({**{"Seed": seed, "Samples": n_sample, "Score": name}, **scores(true_dag, dag)})
                pbar.update(1)
                results_df = pd.DataFrame(results)
                results_df.to_csv(f"../results/{filename}.csv")