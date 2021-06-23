# %%
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

result_path = Path(__file__).parent.parent / "results"

def plot(results: pd.DataFrame, name: str):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True)
    fig.suptitle(f'Performance on synthetic BN: {name}')
    fig.set_size_inches(14.5, 10.5)
    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=results,
        x="Samples", y="PRECISION", hue="Score", style="Score",
        markers=True, dashes=False,markersize=10,ax=ax1
    )

    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=results,
        x="Samples", y="RECALL", hue="Score", style="Score",
        markers=True, dashes=False,markersize=10,ax=ax2
    )
    sns.lineplot(
        data=results,
        x="Samples", y="DAG PRECISION", hue="Score", style="Score",
        markers=True, dashes=False,markersize=10,ax=ax3
    )

    sns.set_theme(style="whitegrid")
    sns.lineplot(
        data=results,
        x="Samples", y="DAG RECALL", hue="Score", style="Score",
        markers=True, dashes=False,markersize=10,ax=ax4
    )

    ax1.set_ylabel("Skeleton Precision")
    ax2.set_ylabel("Skeleton Recall")
    ax3.set_ylabel("DAG Precision")
    ax4.set_ylabel("DAG Recall")


    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, 1])
    ax4.set_ylim([0, 1])
# %%
## ALARM
a_df = pd.read_csv(result_path / "balanced_low.csv")

plot(a_df, "ALARM")

# %%
