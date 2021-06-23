# %%
import sys
sys.path.append('/mnt/c/Users/Bradley/Desktop/EfCon')
from wmse.experiment import experiment
from pathlib import Path
# ----------------------------------- BASE EXPERIMENT CONFIGURATION ----------------------------------------
result_path = Path(__file__).parent.parent / "results"

n_exp = 10

small_samples = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

samples = [1600, 3200, 6400, 12800, 25600]

bifs = ["alarm", "child", "hailfinder", "hepar2", "insurance", "pathfinder", "water", "win95pts"]


# %%

# ----------------------------------- RUN EXPERIMENTS ----------------------------------------

# RUN ALL SMALL EXPERIMENTS FIRST

for bif in bifs:
    fn = f"{bif}_small.csv"
    if not (result_path / fn).exists():
        experiment(filename=f"{bif}_small", samples=small_samples, bif_dataset=bif, n_exp=n_exp)

# THEN LARGE SAMPLE EXPERIMENTS

for bif in bifs:
    fn = "{bif}_large"
    if not (result_path / fn).exists():
        experiment(filename=f"{bif}_large", samples=samples, bif_dataset=bif, n_exp=n_exp)
# %%
