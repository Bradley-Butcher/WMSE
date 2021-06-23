# %%
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

x = np.linspace(1, 10_000, num=100)

params = lambda x, y: (x - 1) * (x ** y)

# %%

n_levels = 3
n_parents = 2

bic_pen = lambda x: (params(n_levels, n_parents) * np.log(x)) / x
wmse_pen = lambda x: (params(n_levels, n_parents) * np.sqrt(x)) / x
# %%

sns.lineplot(x, list(map(bic_pen, x)))

sns.lineplot(x, list(map(wmse_pen, x)))


# %%

n_levels = 3
n_parents = 4

sns.lineplot(x, list(map(bic_pen, x)))

sns.lineplot(x, list(map(wmse_pen, x)))

plt.ylim(0)

# %%
