# %% 
import pandas as pd
from pathlib import Path
import numpy as np

result_path = Path(__file__).parent.parent.parent / "results"

bifs = ["alarm", "child", "hailfinder", "hepar2", "insurance", "munin1", "pathfinder", "water", "win95pts"]

combination_df = None

for bif in bifs:
    try: 
        df = pd.read_csv(result_path / f"{bif}_small.csv")
        df["dataset"] = bif.title()
        if combination_df is not None:
            combination_df = pd.concat([combination_df, df])
        else:
            combination_df = df
    except FileNotFoundError:
        continue

# %%

grouped = combination_df.groupby("Score")["SCALED SHD"]

results = {}

for name, group in grouped:
    results[name] = group.values

result_df = pd.DataFrame(results)

func = lambda x: np.abs(x - np.max(x))

result_df = result_df.apply(func, axis=1)

result_df.to_csv(result_path / "combination_data_shd.csv")

# %%
