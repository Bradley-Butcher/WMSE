from pathlib import Path

import numpy as np
import pandas as pd

from efcon.plotting import plot_simplex_from_data


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    additive = np.array(
        [[0.12, -0.06, -0.06],
         [-0.22, 0.11, 0.11],
         [-0.06, -0.11, 0.17]])

    additive2 = np.array(
        [[0.05, -0.06, 0.01],
         [-0.01, -0.2, 0.21],
         [-0.01, -0.01, 0.02]])

    marg1 = np.array([0.66, 0.22, 0.12])
    marg2 = np.array([0.33, 0.3, 0.37])

    p_y1gx1 = additive + marg1
    p_y2gx1 = additive + marg2

    p_y1gx2 = additive2 + marg1
    p_y2gx2 = additive2 + marg2

    p_x1 = np.array([0.1, 0.4, 0.5])
    p_x2 = np.array([0.2, 0.5, 0.3])

    x1 = np.random.choice(3, 100000, p=p_x1)
    x2 = np.random.choice(3, 100000, p=p_x2)

    def cond_sample(p_ygx, x):
        samples = np.zeros(len(x))
        for i in range(x.shape[0]):
            samples[i] = np.random.choice(3, 1, p=p_ygx[x[i], :])[0]
        return samples

    y1gx1 = cond_sample(p_y1gx1, x1)
    y2gx1 = cond_sample(p_y2gx1, x1)

    y1gx2 = cond_sample(p_y1gx2, x2)
    y2gx2 = cond_sample(p_y2gx2, x2)

    data1 = pd.DataFrame(np.c_[y1gx1, x1], columns=["X0", "X1"])
    data2 = pd.DataFrame(np.c_[y2gx1, x1], columns=["X0", "X1"])
    data3 = pd.DataFrame(np.c_[y1gx2, x2], columns=["X0", "X2"])
    data4 = pd.DataFrame(np.c_[y2gx2, x2], columns=["X0", "X2"])

    return data1, data2, data3, data4


def generate_max_data() -> tuple[np.ndarray, np.ndarray]:
    p_x = np.ones(3) / 3
    x = np.random.choice(3, 100000, p=p_x)

    pygx = np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]])

    def cond_sample(p_ygx, x):
        samples = np.zeros(len(x))
        for i in range(x.shape[0]):
            samples[i] = np.random.choice(3, 1, p=p_ygx[x[i], :])[0]
        return samples

    y1gx = cond_sample(pygx, x)

    data = pd.DataFrame(np.c_[y1gx, x], columns=["X0", "X1"])
    return data


if __name__ == "__main__":
    data_1, data_2, data_3, data_4 = generate_data()
    savedir = Path(__file__).parent.parent / "plots" / "examples"

    # (savedir / "dcrd_1").mkdir(exist_ok=True)
    # (savedir / "dcrd_2").mkdir(exist_ok=True)
    #
    # (savedir / "dcrd_3").mkdir(exist_ok=True)
    # (savedir / "dcrd_4").mkdir(exist_ok=True)

    plot_simplex_from_data(data_1, "X0", ["X1"])
    plot_simplex_from_data(data_2, "X0", ["X1"])
    plot_simplex_from_data(data_3, "X0", ["X2"])
    plot_simplex_from_data(data_4, "X0", ["X2"])

    max_data = generate_max_data()

    plot_simplex_from_data(max_data, "X0", ["X1"])

    # plot_simplex_from_data(data_1, "X0", ["X1"], scaled_kl=False)
    # plot_simplex_from_data(data_2, "X0", ["X1"], scaled_kl=False)
