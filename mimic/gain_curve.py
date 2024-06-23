import os
from datetime import datetime

import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from models import *
from nb21 import *


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
now = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("gain/curves", exist_ok=True)

configurations = [(1000, 3, 1, 200, 100, 0.001, 200, 0.001)]

print("Importing dataset...")
df = pd.read_csv(f"dataset/knn_imputed_norm.csv")
x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "bilirubin", "urea", "fio2", "plateau_pressure"]]
y = df["mort_28"]
t = df["peep_regime"]
print("Done importing")

train_x, test_x, train_y, test_y, train_t, test_t = \
    train_test_split(x, y, t, test_size=0.2, stratify=t, shuffle=True)
train = train_x.join(train_y).join(train_t)
test = test_x.join(test_y).join(test_t)

for epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha in configurations:
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}"
    print(f"Config: {config}")


    def plot_gain_curve(df, s_nn_cate, t_nn_cate, tarnet_cate, cfr_cate, s_linear_cate, train_or_test):
        gain_curve_s_nn, gain_curve_s_nn_ci = get_curve_with_ci(df, s_nn_cate)
        gain_curve_t_nn, gain_curve_t_nn_ci = get_curve_with_ci(df, t_nn_cate)
        gain_curve_tarnet, gain_curve_tarnet_ci = get_curve_with_ci(df, tarnet_cate)
        gain_curve_cfr, gain_curve_cfr_ci = get_curve_with_ci(df, cfr_cate)
        gain_curve_s_linear, gain_curve_s_linear_ci = get_curve_with_ci(df, s_linear_cate)

        xs = np.array(range(len(gain_curve_s_nn)))
        xs = xs / xs.max() * 100

        plt.plot(xs, gain_curve_s_nn, label="S-learner")
        # plt.fill_between(x, gain_curve_s_nn_ci[:, 0], gain_curve_s_nn_ci[:, 1], alpha=0.2)

        plt.plot(xs, gain_curve_t_nn, label="T-learner")
        # plt.fill_between(x, gain_curve_t_nn_ci[:, 0], gain_curve_t_nn_ci[:, 1], alpha=0.2)

        plt.plot(xs, gain_curve_tarnet, label="TARNet")
        # plt.fill_between(x, gain_curve_tarnet_ci[:, 0], gain_curve_tarnet_ci[:, 1], alpha=0.2)

        plt.plot(xs, gain_curve_cfr, label=f"CFR (alpha={alpha})")
        # plt.fill_between(x, gain_curve_cfr_ci[:, 0], gain_curve_cfr_ci[:, 1], alpha=0.2)

        # plt.plot(xs, gain_curve_s_linear, label="S-linear")
        # plt.fill_between(x, gain_curve_s_linear_ci[:, 0], gain_curve_s_linear_ci[:, 1], alpha=0.2)

        plt.plot([0, 100], [0, elast(df, "mort_28", "peep_regime")], linestyle="--", color="black", label="Random")
        plt.title(f"Cumulative gain curve ({train_or_test})")
        plt.legend()
        plt.xlabel("Percentage of patients targeted")
        plt.ylabel("Cumulative gain")
        plt.savefig(f"gain/curves/gain_{train_or_test}_{config}_{now}.png")
        plt.show()

    print("Running S-learner")
    s_nn_cate_train, s_nn_cate_test, s_loss_train, s_loss_test = \
        s_nn(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
            neurons_r, neurons_out, lr, batch_size, True)
    print("S-learner done")
    print("Running T-learner")
    t_nn_cate_train, t_nn_cate_test, t_loss_train, t_loss_test = \
        t_nn(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
            neurons_r, neurons_out, lr, batch_size, True)
    print("T-learner done")
    print(f"Running TARNet")
    tarnet_cate_train, tarnet_cate_test, tarnet_loss_train, tarnet_loss_test = \
        cfr(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
            neurons_r, neurons_out, lr, batch_size, True, 0.0)
    print("TARNet done")
    print("Running CFR")
    cfr_cate_train, cfr_cate_test, cfr_loss_train, cfr_loss_test = \
        cfr(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
            neurons_r, neurons_out, lr, batch_size, True, alpha)
    print("CFR done")
    print("Running S-learner (linear model)")
    s_linear_cate_train, s_linear_cate_test, s_linear_loss_train, s_linear_loss_test = \
        s_linear(train_x, test_x, train_y, test_y, train_t, test_t)
    print("S-learner (linear model) done")

    plot_gain_curve(train, s_nn_cate_train, t_nn_cate_train, tarnet_cate_train, cfr_cate_train, s_linear_cate_train, "train")
    plot_gain_curve(test, s_nn_cate_test, t_nn_cate_test, tarnet_cate_test, cfr_cate_test, s_linear_cate_test, "test")




