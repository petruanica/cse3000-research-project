from copy import deepcopy

from catenets.models.torch import TARNet
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from scipy import integrate

from nb21 import *


def calculate_area(test, ite_test):
    new_test = test.assign(ite=ite_test)
    gain_curve_test = cumulative_gain(new_test, "ite", y="mort_28", t="peep_regime")

    x_axis = np.array(range(len(gain_curve_test)))
    x_axis = x_axis / x_axis.max() * 100
    rnd = np.linspace(0, elast(test, "mort_28", "peep_regime"), gain_curve_test.size)
    area_diff = integrate.simpson(gain_curve_test - rnd, x=x_axis)
    return area_diff


def create_model(binary, neurons_r, neurons_out, learning_rate, layers_r, layers_out):
    loss = "binary_crossentropy" if binary else "mean_squared_error"
    output_activation = "sigmoid" if binary else None

    model = Sequential()
    for i in range(layers_r):
        model.add(Dense(neurons_r, activation='elu'))
    for i in range(layers_out):
        model.add(Dense(neurons_out, activation='elu'))

    model.add(Dense(1, activation=output_activation))
    model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate))

    return model


def select_best_t_learner(x, y, t, configuration, iterations):
    max_models = None
    max_area = -1000

    epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha = configuration
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}"

    for iteration in range(iterations):
        print(f"Running T-learner - {config} --- Iteration {iteration + 1}")
        train_x, test_x, train_y, test_y, train_t, test_t = \
            train_test_split(x, y, t, test_size=0.2, stratify=t)
        test = test_x.join(test_y).join(test_t)
        try:
            train_x0 = train_x[train_t == 0]
            train_y0 = train_y[train_t == 0]
            train_x1 = train_x[train_t == 1]
            train_y1 = train_y[train_t == 1]

            m0 = create_model(True, neurons_r, neurons_out, lr, layers_r, layers_out)
            m1 = create_model(True, neurons_r, neurons_out, lr, layers_r, layers_out)

            callbacks = [EarlyStopping(monitor="val_loss", patience=10)]
            m0.fit(train_x0, train_y0, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
                   callbacks=callbacks, verbose=0)
            m1.fit(train_x1, train_y1, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
                   callbacks=callbacks, verbose=0)

            ite_test = m1.predict(test_x) - m0.predict(test_x)
            ite_test = ite_test.reshape((ite_test.shape[0],))
            print(f"T-learner done")

            area_diff = calculate_area(test, ite_test)

            if area_diff > max_area:
                max_area = area_diff
                max_models = [m0, m1]

            print(f"Area: {area_diff}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"Max area: {max_area}")
    return max_models


def select_best_s_learner(x, y, t, configuration, iterations):
    max_model = None
    max_area = -1000

    epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha = configuration
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}"

    for iteration in range(iterations):
        print(f"Running S-learner - {config} --- Iteration {iteration + 1}")
        train_x, test_x, train_y, test_y, train_t, test_t = \
            train_test_split(x, y, t, test_size=0.2, stratify=t)
        test = test_x.join(test_y).join(test_t)
        try:
            model = create_model(True, neurons_r, neurons_out, lr, layers_r, layers_out)

            callbacks = [EarlyStopping(monitor="val_loss", patience=10)]
            model.fit(train_x.join(train_t), train_y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                      shuffle=True, callbacks=callbacks, verbose=0)

            ite_test = model.predict(test_x.assign(**{"peep_regime": 1.0})) - model.predict(
                test_x.assign(**{"peep_regime": 0.0}))
            ite_test = ite_test.reshape((ite_test.shape[0],))
            print(f"S-learner done")

            area_diff = calculate_area(test, ite_test)

            if area_diff > max_area:
                max_area = area_diff
                max_model = model

            print(f"Area: {area_diff}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"Max area: {max_area}")
    return max_model


def select_best_cfr(x, y, t, configuration, iterations):
    max_model = None
    max_area = -1000

    epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha = configuration
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}"
    name = "TARNet" if alpha == 0.0 else "CFR"

    for iteration in range(iterations):
        print(f"Running {name} - {config} --- Iteration {iteration + 1}")
        train_x, test_x, train_y, test_y, train_t, test_t = \
            train_test_split(x, y, t, test_size=0.2, stratify=t)
        test = test_x.join(test_y).join(test_t)
        try:
            model = TARNet(n_unit_in=train_x.shape[1], nonlin="elu", n_iter=epochs, lr=lr, batch_size=batch_size,
                           val_split_prop=0.2, n_layers_out=layers_out, n_layers_r=layers_r,
                           n_units_out=neurons_out, n_units_r=neurons_r, binary_y=True, penalty_disc=alpha)

            model.fit(train_x.to_numpy(), train_y.to_numpy(), train_t.to_numpy())

            ite_test = model.predict(test_x.to_numpy()).detach().cpu().numpy()
            print(f"{name} done")

            area_diff = calculate_area(test, ite_test)

            if area_diff > max_area:
                max_area = area_diff
                max_model = deepcopy(model)

            print(f"Area: {area_diff}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"Max area: {max_area}")
    return max_model
