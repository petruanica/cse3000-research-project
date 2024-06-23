import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from base_models import *


def s_nn(train_x, test_x, train_y, test_y, train_t, test_t, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out,
         learning_rate, batch_size, binary):
    model = fit_s_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                     n_layers_out, binary)

    pred_train = model.predict(train_x.join(train_t), verbose=0)
    pred_test = model.predict(test_x.join(test_t), verbose=0)

    loss_train = mean_squared_error(train_y.to_numpy(), pred_train)
    loss_test = mean_squared_error(test_y.to_numpy(), pred_test)

    ite_train = model.predict(train_x.assign(**{"peep_regime": 1.0})) - model.predict(
        train_x.assign(**{"peep_regime": 0.0}))
    ite_test = model.predict(test_x.assign(**{"peep_regime": 1.0})) - model.predict(
        test_x.assign(**{"peep_regime": 0.0}))

    ite_train = ite_train.reshape((ite_train.shape[0],))
    ite_test = ite_test.reshape((ite_test.shape[0],))

    return ite_train, ite_test, loss_train, loss_test


def t_nn(train_x, test_x, train_y, test_y, train_t, test_t, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out,
         learning_rate, batch_size, binary):
    train_x0 = train_x[train_t == 0]
    train_y0 = train_y[train_t == 0]
    train_x1 = train_x[train_t == 1]
    train_y1 = train_y[train_t == 1]

    test_x0 = test_x[test_t == 0]
    test_y0 = test_y[test_t == 0]
    test_x1 = test_x[test_t == 1]
    test_y1 = test_y[test_t == 1]

    m0, m1 = fit_t_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                      n_layers_out, binary)

    pred_train0 = m0.predict(train_x0, verbose=0)
    pred_train1 = m1.predict(train_x1, verbose=0)
    pred_test0 = m0.predict(test_x0, verbose=0)
    pred_test1 = m1.predict(test_x1, verbose=0)

    pred_train = np.concatenate((pred_train0, pred_train1), axis=0)
    pred_test = np.concatenate((pred_test0, pred_test1), axis=0)

    loss_train = mean_squared_error(pd.concat([train_y0, train_y1], ignore_index=True).to_numpy(), pred_train)
    loss_test = mean_squared_error(pd.concat([test_y0, test_y1], ignore_index=True).to_numpy(), pred_test)

    ite_train = m1.predict(train_x) - m0.predict(train_x)
    ite_test = m1.predict(test_x) - m0.predict(test_x)

    ite_train = ite_train.reshape((ite_train.shape[0],))
    ite_test = ite_test.reshape((ite_test.shape[0],))

    return ite_train, ite_test, loss_train, loss_test


def cfr(train_x, test_x, train_y, test_y, train_t, test_t, epochs, n_layers_r, n_layers_out, neurons_r, neurons_out,
        learning_rate, batch_size, binary, alpha):
    model = fit_cfr(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
                    n_layers_out, alpha, binary)

    ite_train, y0_pred_train, y1_pred_train = model.predict(train_x.to_numpy(), return_po=True)
    ite_test, y0_pred_test, y1_pred_test = model.predict(test_x.to_numpy(), return_po=True)

    y0_pred_train, y1_pred_train = y0_pred_train.detach().cpu().numpy(), y1_pred_train.detach().cpu().numpy()
    y0_pred_test, y1_pred_test = y0_pred_test.detach().cpu().numpy(), y1_pred_test.detach().cpu().numpy()

    pred_train = np.where(train_t.to_numpy() == 0, y0_pred_train, y1_pred_train)
    pred_test = np.where(test_t.to_numpy() == 0, y0_pred_test, y1_pred_test)

    loss_train = mean_squared_error(train_y.to_numpy(), pred_train)
    loss_test = mean_squared_error(test_y.to_numpy(), pred_test)

    return ite_train.detach().cpu().numpy(), ite_test.detach().cpu().numpy(), loss_train, loss_test


def random_predictor(test_y):
    pred_test = np.random.uniform(0.0, 1.0, test_y.shape[0])
    loss_test = mean_squared_error(test_y.to_numpy(), pred_test)
    return loss_test


def ate_predictor(test_y, test_t):
    treated_avg_outcome = test_y[test_t == 1].mean()
    control_avg_outcome = test_y[test_t == 0].mean()
    pred_test = np.where(test_t.to_numpy() == 0, control_avg_outcome, treated_avg_outcome)
    loss_test = mean_squared_error(test_y.to_numpy(), pred_test)
    return loss_test


def s_linear(train_x, test_x, train_y, test_y, train_t, test_t):
    model = LinearRegression()

    model.fit(train_x.join(train_t), train_y)

    pred_train = model.predict(train_x.join(train_t))
    pred_test = model.predict(test_x.join(test_t))

    loss_train = mean_squared_error(train_y.to_numpy(), pred_train)
    loss_test = mean_squared_error(test_y.to_numpy(), pred_test)

    ite_train = model.predict(train_x.assign(**{"peep_regime": 1})) - model.predict(
        train_x.assign(**{"peep_regime": 0}))
    ite_test = model.predict(test_x.assign(**{"peep_regime": 1})) - model.predict(test_x.assign(**{"peep_regime": 0}))

    return ite_train, ite_test, loss_train, loss_test
