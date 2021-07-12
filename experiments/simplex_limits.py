"""The aim of this script is to plot correct / incorrect parent sets at large sample sizes."""
# %%
# Imports

import numpy as np
import ternary
import pandas as pd
import imageio
from matplotlib import pyplot as plt

from baynet import DAG

from typing import List, Optional

import sys
sys.path.append('/home/dev/Thesis/WMSE')

from wmse.utils import get_distributions

# %%
# Generate Data
seed = 1

bn = DAG.generate("forest fire", 10, seed=seed)
bn.generate_discrete_parameters(alpha=10, min_levels=3, max_levels=3, seed=seed)
data = bn.sample(n_samples=1_000, seed=1).astype(int)
# bn.plot()
# %%
#Plotting functions

def mle(data: pd.DataFrame, child, parents):
    n_parent_levels = np.array([data[p].nunique() for p in parents])
    array = np.zeros([*n_parent_levels, data[child].nunique()])
    matches = data.groupby([*parents, child]).size()
    for indexer, count in matches.iteritems():
        array[indexer] = count
    array[array.sum(axis=-1) == 0] = 1.0
    array = np.nan_to_num(array, nan=1e-8, posinf=1.0 - 1e-8)
    array /= np.expand_dims(array.sum(axis=-1), axis=-1)
    return array


def plot_conditions(data: pd.DataFrame,
                    child: str,
                    true_parents: List[str],
                    false_parents: Optional[List[str]]) -> None:
    
    # Plot Setup
    fig, tax = ternary.figure(scale=100)
    fig.set_size_inches(10, 8)
    tax.boundary(linewidth=2)
    tax.gridlines(color="black", multiple=20, linewidth=0.5)
    tax.gridlines(color="blue", multiple=5, linewidth=0.25)
    tax.ticks(axis='lbr', linewidth=1, multiple=20)
    tax.get_axes().axis('off')
    
    tax.set_title(f"Sample Size: {len(data)}", fontsize=20)

    n_child = data[child].nunique()
    
    p_ygx = mle(data, child, true_parents).reshape(-1, n_child)

    if false_parents:
        n_false_parent = np.prod([data[fp].nunique() for fp in false_parents])
        p_ygfx = mle(data, child, true_parents + false_parents).reshape(-1, n_false_parent, n_child)
        for i in range(p_ygx.shape[0]):
            tax.scatter([p_ygx[i, :] * 100], marker='*', c="black", s=200, label="True Conditions")
            for j in range(n_false_parent):
                tax.scatter([p_ygfx[i, j, :] * 100], s=50, c="red", label="True Conditions + False Conditions")
                tax.line(p_ygfx[i, j, :] * 100, p_ygx[i, :] * 100, linewidth=2, color='blue', alpha=0.5, linestyle="solid")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    filename = f"../gifs/{len(data)}.png"
    fig.savefig(filename)
    return filename
# %%

#Plot

child = "G"
true_parents = ["H", "J"]
false_parents = ["F"]

plot_conditions(data, child, true_parents, false_parents)
# %%

n_frames = 8
resolution = 100
pause = 100

with imageio.get_writer("simplex_limits.gif", mode='I') as writer:
    for i, N in enumerate(np.logspace(1, 6, resolution, base=10)):
        N = int(N)
        data = bn.sample(n_samples=N, seed=1).astype(int)
        filename = plot_conditions(data, child, true_parents, false_parents)
        image = imageio.imread(filename)
        if i == resolution:
            for frame in range(n_frames + pause):
                writer.append_data(image)
        else:
            for frame in range(n_frames):
                writer.append_data(image)
            


# %%
