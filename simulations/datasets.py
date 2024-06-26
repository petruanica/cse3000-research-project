from simulation import *


def sim1(n_samples):
    d = 10
    e = 0.1
    beta = uniform.rvs(-5, 10, size=d)
    mu0 = lambda x: x @ beta + 5 * (x[0] > 0.5)
    mu1 = lambda x: mu0(x) + 8 * (x[1] > 0.1)
    return simulate_data_without_confounding(n_samples, d, e, mu0, mu1)


def sim2(n_samples):
    d = 10
    e = 0.5
    beta0 = uniform.rvs(1, 29, size=d)
    mu0 = lambda x: x @ beta0
    beta1 = uniform.rvs(1, 29, size=d)
    mu1 = lambda x: x @ beta1
    return simulate_data_without_confounding(n_samples, d, e, mu0, mu1)


def sim3(n_samples):
    d = 10
    e = 0.5
    c = lambda x: 2 / (1 + np.exp(-12 * (x - 0.5)))
    mu0 = lambda x: c(x[0]) * c(x[1]) / 2
    mu1 = lambda x: c(x[0]) * c(x[1]) / (-2)
    return simulate_data_without_confounding(n_samples, d, e, mu0, mu1)


def sim4(n_samples):
    d = 10
    e = 0.5
    beta = uniform.rvs(1, 29, size=d)
    mu0 = lambda x: x @ beta
    mu1 = lambda x: mu0(x)
    return simulate_data_without_confounding(n_samples, d, e, mu0, mu1)


def sim5(n_samples):
    d = 10
    e = 0.5
    beta = uniform.rvs(-15, 30, size=d)
    beta_l = np.array([beta[i] * (i <= 2) for i in range(beta.size)])
    beta_m = np.array([beta[i] * (3 <= i <= 6) for i in range(beta.size)])
    beta_u = np.array([beta[i] * (7 <= i <= 9) for i in range(beta.size)])
    mu0 = lambda x: (x[9] < -0.4) * x @ beta_l + (-0.4 <= x[9] <= 0.4) * x @ beta_m + (0.4 < x[9]) * x @ beta_u
    mu1 = lambda x: mu0(x)
    return simulate_data_without_confounding(n_samples, d, e, mu0, mu1)


def sim6(n_samples):
    d = 10
    mu0 = lambda x: 2 * x[0] - 1
    mu1 = lambda x: mu0(x)
    return simulate_data_with_confounding(n_samples, d, mu0, mu1)
