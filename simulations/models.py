from sklearn.metrics import mean_squared_error

from base_models import *


def s_nn(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out, learning_rate, batch_size, binary):
    # Create and train the S-learner
    model = fit_s_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                     n_layers_out, binary)

    ite_train = model.predict(train_x.assign(**{"T": 1.0})) - model.predict(train_x.assign(**{"T": 0.0}))
    ite_test = model.predict(test_x.assign(**{"T": 1.0})) - model.predict(test_x.assign(**{"T": 0.0}))

    mse_train = mean_squared_error(train_ite, ite_train)
    mse_test = mean_squared_error(test_ite, ite_test)

    return ite_train, ite_test, mse_train, mse_test


def t_nn(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out, learning_rate, batch_size, binary):
    # Create and train the T-learners
    m0, m1 = fit_t_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                      n_layers_out, binary)

    # Estimate the CATE
    ite_train = m1.predict(train_x) - m0.predict(train_x)
    ite_test = m1.predict(test_x) - m0.predict(test_x)

    mse_train = mean_squared_error(train_ite, ite_train)
    mse_test = mean_squared_error(test_ite, ite_test)

    return ite_train, ite_test, mse_train, mse_test


def cfr(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out, learning_rate, batch_size, binary, alpha):
    # Instantiate and train a CFR instance
    model = fit_cfr(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                    n_layers_out, alpha, binary)

    # Evaluate the model
    ite_train = model.predict(train_x).detach().cpu().numpy()
    ite_test = model.predict(test_x).detach().cpu().numpy()

    mse_train = mean_squared_error(train_ite, ite_train)
    mse_test = mean_squared_error(test_ite, ite_test)

    return ite_train, ite_test, mse_train, mse_test


def ate_estimator(ate, train_ite, test_ite):
    ate_train = np.full(train_ite.shape, ate)
    ate_test = np.full(test_ite.shape, ate)
    mse_train = mean_squared_error(train_ite, ate_train)
    mse_test = mean_squared_error(test_ite, ate_test)
    return mse_train, mse_test
