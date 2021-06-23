# %%
import sys
sys.path.append('/mnt/c/Users/Bradley/Desktop/EfCon')

from wmse.experiment import experiment
from pathlib import Path
# ----------------------------------- BASE EXPERIMENT CONFIGURATION ----------------------------------------
result_path = Path(__file__).parent.parent / "results"

n_exp = 10

small_samples = [100, 500, 1000, 10000]

bifs = ["alarm"]


# %%

# ----------------------------------- RUN EXPERIMENTS ----------------------------------------

# RUN ALL SMALL EXPERIMENTS FIRST

for bif in bifs:
    experiment(filename=f"{bif}_test", samples=small_samples, bif_dataset=bif, n_exp=n_exp)

# %%
