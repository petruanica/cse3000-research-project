import numpy as np
from toolz import curry


@curry
def elast(data, y, t):
    d = np.sum((data[t] - data[t].mean()) ** 2)
    if d == 0:
        return 0
    return np.sum((data[t] - data[t].mean()) * (data[y] - data[y].mean())) / d


def elast_ci(df, y, t, z=1.96):
    n = df.shape[0]
    t_bar = df[t].mean()
    beta1 = elast(df, y, t)
    beta0 = df[y].mean() - beta1 * t_bar
    e = df[y] - (beta0 + beta1 * df[t])
    d = np.sum((df[t] - t_bar) ** 2)
    if d == 0:
        se = 0
    else:
        se = np.sqrt(((1 / (n - 2)) * np.sum(e ** 2)) / d)
    return np.array([beta1 - z * se, beta1 + z * se])


def cumulative_elast_curve_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]

    return np.array([elast_ci(ordered_df.head(rows), y, t) for rows in n_rows])


def cumulative_gain_ci(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    steps = min(steps, size)
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast_ci(ordered_df.head(rows), y, t) * (rows / size) for rows in n_rows])


def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    steps = min(steps, size)
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows), y, t) * (rows / size) for rows in n_rows])


def get_curve_with_ci(test, pred):
    gain_curve = cumulative_gain(test, pred, y="mort_28", t="peep_regime", min_periods=0)
    gain_curve_ci = cumulative_gain_ci(test, pred, y="mort_28", t="peep_regime", min_periods=0)
    return gain_curve, gain_curve_ci
