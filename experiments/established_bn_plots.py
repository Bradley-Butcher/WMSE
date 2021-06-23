# %%
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

result_path = Path(__file__).parent.parent / "results"
save_path = Path(__file__).parent.parent / "plots" / "performance"
bifs = ["alarm", "child", "hailfinder", "hepar2", "insurance", "munin1", "pathfinder", "water", "win95pts"]

def prec_recall_plot(results: pd.DataFrame, name: str, filename: str):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True)
    fig.suptitle(f'FGES Performance on {name} dataset.')
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

    fig.savefig(f"{save_path / filename}.png", dpi=300)

def multibn_plot(df: pd.DataFrame, column: str, filename: str):
    g = sns.relplot(
        data=df, x="Samples", y=column, col="dataset",
         hue="Score", style="Score", kind="line",
          markers=True, dashes=False,markersize=10, col_wrap=3
    )
    (g.map(plt.axhline)
        .set_axis_labels("Samples", column.title())
        .set_titles("Dataset: {col_name}")
        .tight_layout(w_pad=0, h_pad=0))



# %%

# --------------------------------------------------- SMALL EXPERIMENT PLOTS ----------------------------------------------------
for bif in bifs:
    try: 
        df = pd.read_csv(result_path / f"{bif}_small.csv")
    except FileNotFoundError:
        continue
    prec_recall_plot(df, f"small (100-1000 sample) {bif}", f"{bif}_small_plot")
    

# %%
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

multibn_plot(combination_df, "SCALED HAMMING", "")

# %%
