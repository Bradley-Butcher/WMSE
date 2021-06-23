# %%
import sys
sys.path.append('/mnt/c/Users/Bradley/Desktop/EfCon')
from efcon.experiment import experiment
from pathlib import Path
# ----------------------------------- BASE EXPERIMENT CONFIGURATION ----------------------------------------
result_path = Path(__file__).parent.parent / "results"

n_exp = 10

small_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

samples = [1600, 3200, 6400, 12800, 25600, 51200, 102400]

alphas = [[2, 2, 2], [6, 2, 2], [10, 2, 2]]
alpha_names = ["balanced", "imbalanced", "extremely_imbalanced"]

# %%

# ----------------------------------- RUN EXPERIMENTS ----------------------------------------

# RUN ALL SMALL EXPERIMENTS FIRST

for name, alpha in zip(alpha_names, alphas):
    fn = f"{name}_small.csv"
    if not (result_path / fn).exists():
        experiment(filename=f"{name}_small", samples=small_samples, alpha=alpha, n_exp=n_exp)

# THEN LARGE SAMPLE EXPERIMENTS

for name, alpha in zip(alpha_names, alphas):
    fn = "{name}_large"
    if not (result_path / fn).exists():
        experiment(filename=f"{name}_large", samples=samples, alpha=alpha, n_exp=n_exp)
# %%
