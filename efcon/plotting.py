from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

import ternary
from baynet import DAG, metrics
from matplotlib import pyplot as plt

from efcon.main import score_from_data


def get_distributions(data: pd.DataFrame, child: str, parents: List[str]):
    child_idx = data.columns.get_loc(child)
    parent_idxs = np.array([data.columns.get_loc(parent) for parent in parents])
    data = data.values
    levels = np.array([np.unique(data[:, i]).astype(int).tolist() for i in range(data.shape[1])])
    parent_combinations = np.array(np.meshgrid(*levels[parent_idxs].tolist())).T.reshape(-1, len(parent_idxs))
    p_ygxs = np.zeros([len(parent_combinations), len(levels[child_idx])])
    p_xs = np.zeros(len(parent_combinations))
    for j, parent_combination in enumerate(parent_combinations):
        rows = np.argwhere(np.all(data[:, parent_idxs] == parent_combination, axis=1)).flatten()
        x_count = len(rows)
        p_xs[j] = x_count / data.shape[0]
        for i in range(len(levels[child_idx])):
            p_ygxs[j, i] = np.sum(data[rows, child_idx] == levels[child_idx][i]) / x_count
    p_ygxs = np.nan_to_num(p_ygxs)
    mu = np.mean(p_ygxs, axis=0)
    return p_ygxs, mu, p_xs


def ternary_circle(center, radius, n=100, scale=1):
    xy = ternary.helpers.project_point(center)
    points = np.array(
        [(np.cos(2 * np.pi / n * x) * radius, np.sin(2 * np.pi / n * x) * radius) for x in range(0, n + 1)])
    points += xy
    return [ternary.helpers.planar_to_coordinates((x, y), scale) for x, y in points]


def plots_from_dag(data: pd.DataFrame, dag: DAG, savedir: Path, scaled_kl: bool):
    savedir.mkdir(parents=True, exist_ok=True)
    for node in dag.vs:
        parents = [p["name"] for p in dag.get_ancestors(node, only_parents=True)]
        child = node["name"]
        if len(parents) > 0:
            plot_distribution(data, child, parents, savedir, scaled_kl)


def plot_distribution(data: pd.DataFrame, child: str, parents: List[str], savedir: Optional[Path] = None, scaled_kl: bool = True):
    if data[child].nunique() > 2:
        plot_simplex_from_data(data, child, parents, savedir, scaled_kl)
    else:
        line_plot_from_data(data, child, parents, savedir, scaled_kl)


def line_plot_from_data(data: pd.DataFrame, child: str, parents: List[str], savedir: Optional[Path] = None, scaled_kl: bool = True):
    p_ygxs, mu, p_xs = get_distributions(data, child, parents)

    fig = plt.figure()
    fig.set_size_inches(10, 8)

    ax = fig.add_subplot(111)
    ax.set_xlim(-0.25, 1.5)
    ax.set_ylim(0, 10)

    ax.axis('off')

    score = score_from_data(data, child, parents, scaled_kl)
    ax.set_title(f"P({child}|{','.join(parents)}), Score: {score:.5f}", fontsize=20)

    # draw lines
    xmin = 0
    xmax = 1.2
    y = 5
    height = 1
    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)

    plt.annotate("D = ", [-0.1, 2.5], fontsize=15)
    plt.annotate("P(Y = 0|X) = ", [-0.17, 7.5], fontsize=15)

    plt.plot(mu[0], y, '*', ms=10, mfc='yellow')

    for i in range(p_ygxs.shape[0]):
        plt.plot(p_ygxs[i, 0], y, 'ro', ms=5, mfc='r')
        plt.annotate(f"{np.sum([np.abs(p_ygxs[i, :] - mu)]):.5f}",
                     (p_ygxs[i, 0], y),
                     xytext=(p_ygxs[i, 0] + 0.1, y - i - 1),
                     arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=5),
                     horizontalalignment='right')
        plt.annotate(f"{p_ygxs[i, 0]:.2f}",
                     (p_ygxs[i, 0], y),
                     xytext=(p_ygxs[i, 0] + 0.1, y + i + 1),
                     arrowprops=dict(facecolor='black', shrink=0.1, width=1, headwidth=5),
                     horizontalalignment='right')

    plt.text(xmin - 0.1, y, '0', horizontalalignment='right', fontsize=15)
    plt.text(xmax + 0.1, y, '1', horizontalalignment='left', fontsize=15)

    if savedir:
        fig.savefig(savedir / f"{child}.png")
    else:
        plt.show()


def plot_simplex_from_data(data: pd.DataFrame, child: str, parents: List[str], savedir: Optional[Path] = None, scaled_kl: bool = True):
    p_ygxs, mu, p_xs = get_distributions(data, child, parents)
    if not scaled_kl:
        mu = np.ones(p_ygxs.shape[1]) / p_ygxs.shape[1]
    fig, tax = ternary.figure(scale=100)
    fig.set_size_inches(10, 8)

    tax.boundary(linewidth=2)
    tax.gridlines(color="black", multiple=20, linewidth=0.5)
    tax.gridlines(color="blue", multiple=5, linewidth=0.25)
    tax.ticks(axis='lbr', linewidth=1, multiple=20)
    tax.get_axes().axis('off')
    tax.scatter([mu * 100], marker='*', c="yellow", s=200, label="mu")

    score = score_from_data(data, child, parents, scaled_kl)
    tax.set_title(f"P({child}|{','.join(parents)}), Score: {score:.5f}", fontsize=20)

    # distance = np.sum([np.abs(p_ygxs[i, :] - mu)])
    # points = ternary_circle(mu * 100, distance * 10)
    # tax.plot(points, linewidth=3.0, label="Average Distance")

    tax.scatter(p_ygxs * 100, s=100, label="p(y|x = xi)")

    text_offset = ternary.helpers.planar_to_coordinates([-0.04, 0.02], scale=100)

    for i in range(p_ygxs.shape[0]):
        p_ygxs[i, :] = p_ygxs[i, :] + 1e-5
        distance = np.sqrt(np.sum((p_ygxs[i, :] - mu)**2))
        tax.annotate(f"l2 = {distance:.2f}, p(x) = {p_xs[i]:.2f}", (p_ygxs[i, :] + text_offset) * 100)
        tax.line(p_ygxs[i, :] * 100, mu * 100, linewidth=3, color='k', alpha=0.35, linestyle="--")

    p_y = np.dot(p_xs, p_ygxs)
    tax.scatter([p_y * 100], marker='+', c="blue", s=200, label="p(y)")

    points = ternary_circle(mu * 100, np.sqrt(np.sum((p_y - mu)**2)) * 70)
    tax.plot(points, linewidth=2.0, label="No Parent Threshold", c="black")
    tax.line(mu * 100, p_y * 100, linewidth=1, color='red', alpha=0.35, linestyle="-.")

    fontsize = 20
    offset = 0.1

    # tax.top_corner_label("1", fontsize=fontsize, offset=0.2)
    # tax.left_corner_label("2", fontsize=fontsize, offset=offset)
    # tax.right_corner_label("3", fontsize=fontsize, offset=offset)

    tax.legend()
    if savedir:
        fig.savefig(savedir / f"{child}.png")
    else:
        fig.show()


def score_plot(scores: List[float], dags: List[DAG], true_dag: DAG, savedir: Path):
    dag_scores = []
    skel_scores = []
    for i in np.argsort(np.array(scores)):
        dag_scores += [metrics.f1_score(true_dag, dags[i])]
        skel_scores += [metrics.f1_score(true_dag, dags[i], True)]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax2.scatter(x=list(range(len(dag_scores))), y=dag_scores)
    ax1.scatter(x=list(range(len(dag_scores))), y=skel_scores)
    fig.savefig(savedir / "score_plot.png")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax2.scatter(x=np.sort(scores), y=dag_scores)
    skel_order = np.array(np.argsort(skel_scores))
    ax1.scatter(x=np.array(scores)[skel_order], y=np.array(skel_scores)[skel_order])
    fig.savefig(savedir / "score_plot_alt.png")







