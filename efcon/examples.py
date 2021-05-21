from pathlib import Path

import numpy as np
import pandas as pd

from efcon.plotting import plot_simplex_from_data


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    additive = np.array(
        [[0.12, -0.06, -0.06],
         [-0.22, 0.11, 0.11],
         [-0.06, -0.11, 0.17]])
    marg1 = np.array([0.66, 0.22, 0.12])
    marg2 = np.array([0.33, 0.3, 0.37])

    p_y1gx = additive + marg1
    p_y2gx = additive + marg2

    p_x = np.array([0.1, 0.4, 0.5])

    x = np.random.choice(3, 100000, p=p_x)

    def cond_sample(p_ygx):
        samples = np.zeros(len(x))
        for i in range(x.shape[0]):
            samples[i] = np.random.choice(3, 1, p=p_ygx[x[i], :])[0]
        return samples

    y1gx = cond_sample(p_y1gx)
    y2gx = cond_sample(p_y2gx)

    data1 = np.c_[y1gx, x]
    data2 = np.c_[y2gx, x]

    return pd.DataFrame(data1, columns=["X0", "X1"]), pd.DataFrame(data2, columns=["X0", "X1"])


if __name__ == "__main__":
    data_1, data_2 = generate_data()
    savedir = Path(__file__).parent.parent / "plots" / "examples"
    (savedir / "dcrd_1").mkdir(exist_ok=True)
    (savedir / "dcrd_2").mkdir(exist_ok=True)

    (savedir / "dcrd_3").mkdir(exist_ok=True)
    (savedir / "dcrd_4").mkdir(exist_ok=True)

    plot_simplex_from_data(data_1, "X0", ["X1"])
    plot_simplex_from_data(data_2, "X0", ["X1"])

    plot_simplex_from_data(data_1, "X0", ["X1"], scaled_kl=False)
    plot_simplex_from_data(data_2, "X0", ["X1"], scaled_kl=False)

