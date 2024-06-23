import numpy as np
import pandas as pd
from scipy.stats import bernoulli, uniform, beta


def simulate_data_without_confounding(n_samples, d, e, mu0, mu1):
    # Generate the feature vectors
    sigma = np.eye(d)  # Identity matrix as a placeholder for Σ
    X = np.random.multivariate_normal(np.zeros(d), sigma, n_samples)

    # Generate the potential outcomes
    Y0 = np.array([mu0(x) + np.random.normal() for x in X])
    Y1 = np.array([mu1(x) + np.random.normal() for x in X])

    # Generate the treatment assignments
    T = bernoulli.rvs(e, size=n_samples)

    # Generate the observed outcomes
    Y = T * Y1 + (1 - T) * Y0

    # Calculate ITE
    ITE = Y1 - Y0

    # Calculate ATE
    ate = np.mean(ITE)

    # Combine everything into a dataset
    data = nparray_to_df(X, T, Y, ITE)
    return data, ate


def simulate_data_with_confounding(n_samples, d, mu0, mu1):
    X = uniform.rvs(size=(n_samples, d))

    # Generate the potential outcomes
    Y0 = np.array([mu0(x) + np.random.normal() for x in X])
    Y1 = np.array([mu1(x) + np.random.normal() for x in X])

    # Define the propensity score
    e = lambda x: 0.25 * (1 + beta.pdf(x[0], 2, 4))

    # Generate the treatment assignments
    T = bernoulli.rvs(np.array([e(x) for x in X]))

    # Generate the observed outcomes
    Y = T * Y1 + (1 - T) * Y0

    # Calculate ITE
    ITE = Y1 - Y0

    # Calculate ATE
    ate = np.mean(ITE)

    # Combine everything into a dataset
    data = nparray_to_df(X, T, Y, ITE)
    return data, ate


def nparray_to_df(X, T, Y, ITE):
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)
    T = pd.DataFrame(T, columns=["T"])
    Y = pd.DataFrame(Y, columns=["Y"])
    ITE = pd.DataFrame(ITE, columns=["ITE"])

    data = X.join(T).join(Y).join(ITE)
    return data

