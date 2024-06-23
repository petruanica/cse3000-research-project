import itertools

import numpy as np
from catenets.models.torch import TARNet
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

n_epochs = [1000]
learning_rates = [0.005]
n_layers_r = [1, 2, 3]
n_layers_out = [1, 2, 3]
n_neurons_r = [20, 50, 100, 200]
n_neurons_out = [20, 50, 100, 200]
batch_sizes = [100, 200, 500, 700]
alphas = [np.power(10.0, k / 2) for k in range(-6, 7, 2)]


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


def fit_s_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, layers_r,
             layers_out, binary):
    model = create_model(binary, neurons_r, neurons_out, learning_rate, layers_r, layers_out)

    callbacks = [EarlyStopping(monitor="val_loss", patience=10)]
    model.fit(train_x.join(train_t), train_y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
              callbacks=callbacks, verbose=0)

    return model


def fit_t_nn(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, layers_r,
             layers_out, binary):
    train_x0 = train_x[train_t == 0]
    train_y0 = train_y[train_t == 0]
    train_x1 = train_x[train_t == 1]
    train_y1 = train_y[train_t == 1]

    m0 = create_model(binary, neurons_r, neurons_out, learning_rate, layers_r, layers_out)
    m1 = create_model(binary, neurons_r, neurons_out, learning_rate, layers_r, layers_out)

    callbacks = [EarlyStopping(monitor="val_loss", patience=10)]
    m0.fit(train_x0, train_y0, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
           callbacks=callbacks, verbose=0)
    m1.fit(train_x1, train_y1, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True,
           callbacks=callbacks, verbose=0)

    return m0, m1


def fit_cfr(train_x, train_y, train_t, neurons_r, neurons_out, epochs, learning_rate, batch_size, n_layers_r,
            n_layers_out, alpha, binary):
    model = TARNet(n_unit_in=train_x.shape[1], nonlin="elu", n_iter=epochs, lr=learning_rate, batch_size=batch_size,
                   val_split_prop=0.2, n_layers_out=n_layers_out, n_layers_r=n_layers_r, n_units_out=neurons_out,
                   n_units_r=neurons_r, binary_y=binary, penalty_disc=alpha)

    model.fit(train_x.to_numpy(), train_y.to_numpy(), train_t.to_numpy())

    return model


def get_configurations():
    configurations = itertools.product(n_epochs, n_layers_r, n_layers_out, n_neurons_r, n_neurons_out, learning_rates,
                                       batch_sizes)
    return configurations
