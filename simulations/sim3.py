import os
import time
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import train_test_split

from datasets import *
from models import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

sample_sizes = [3000, 5000, 7500, 10000, 12500, 15000]
simulations = [sim3]


def run_simulation(simulation, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha, iterations):
    s_mse = []
    t_mse = []
    tarnet_mse = []
    cfr_mse = []
    ate_mse = []
    s_mse_training = []
    t_mse_training = []
    tarnet_mse_training = []
    cfr_mse_training = []
    ate_mse_training = []

    for n_samples in sample_sizes:
        s_errors = []
        t_errors = []
        tarnet_errors = []
        cfr_errors = []
        ate_errors = []

        s_errors_train = []
        t_errors_train = []
        tarnet_errors_train = []
        cfr_errors_train = []
        ate_errors_train = []

        for iteration in range(iterations):
            print(f"Simulation {simulation.__name__} --- Size {n_samples} --- Iteration {iteration + 1}")
            dataset, ate = simulation(n_samples)
            x = dataset.drop(columns=["Y", "T", "ITE"])
            y = dataset["Y"]
            t = dataset["T"]
            ite = dataset["ITE"]

            train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite = \
                train_test_split(x, y, t, ite, test_size=0.20, stratify=t)

            print("Running S-learner")
            s_learner_cate_train, s_learner_cate_test, s_mse_train, s_mse_test = \
                s_nn(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, False)
            print("S-learner done")
            print("Running T-learner")
            t_learner_cate_train, t_learner_cate_test, t_mse_train, t_mse_test = \
                t_nn(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, False)
            print("T-learner done")
            print("Running TARNet")
            tarnet_cate_train, tarnet_cate_test, tarnet_mse_train, tarnet_mse_test = \
                cfr(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, False, 0.0)
            print("TARNet done")
            print("Running CFR")
            cfr_cate_train, cfr_cate_test, cfr_mse_train, cfr_mse_test = \
                cfr(train_x, test_x, train_y, test_y, train_t, test_t, train_ite, test_ite, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, False, alpha)
            print("CFR done")
            ate_mse_train, ate_mse_test = ate_estimator(ate, train_ite, test_ite)
            print("-" * 100)

            s_errors.append(s_mse_test)
            t_errors.append(t_mse_test)
            tarnet_errors.append(tarnet_mse_test)
            cfr_errors.append(cfr_mse_test)
            ate_errors.append(ate_mse_test)


            s_errors_train.append(s_mse_test)
            t_errors_train.append(t_mse_test)
            tarnet_errors_train.append(tarnet_mse_test)
            cfr_errors_train.append(cfr_mse_test)
            ate_errors_train.append(ate_mse_test)

        s_mse.append(np.mean(s_errors))
        t_mse.append(np.mean(t_errors))
        tarnet_mse.append(np.mean(tarnet_errors))
        cfr_mse.append(np.mean(cfr_errors))
        ate_mse.append(np.mean(ate_errors))

        s_mse_training.append(np.std(s_errors_train))
        t_mse_training.append(np.std(t_errors_train))
        tarnet_mse_training.append(np.std(tarnet_errors_train))
        cfr_mse_training.append(np.std(cfr_errors_train))
        ate_mse_training.append(np.std(ate_errors_train))

    return s_mse, s_mse_training, t_mse, t_mse_training, tarnet_mse, tarnet_mse_training, cfr_mse, cfr_mse_training, ate_mse, ate_mse_training


def plot_results(epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha, iterations):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}_i{iterations}"
    path = f"plots/plots_{config}_{now}"
    os.makedirs(path, exist_ok=True)

    execution_times = []

    for sim in simulations:
        print(f"Configuration: {config}")
        start = time.time()
        s_mse, s_mse_train, t_mse, t_mse_train, tarnet_mse, tarnet_mse_train, cfr_mse, cfr_mse_train, ate_mse, ate_mse_train = \
            run_simulation(sim, epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha, iterations)
        end = time.time()
        elapsed = format(end - start, '.4f')
        execution_times.append(f"{config}: {elapsed} s")
        print(f"Execution time: {elapsed}")
        print()

        with open(f"{path}/{sim.__name__}.txt", "w") as f:
            f.write(f"{config}, {sample_sizes}\n")
            f.write("Model, Test STD, Test MSE\n")

            f.write(f"S-learner, {s_mse_train}, {s_mse}\n")
            f.write(f"T-learner, {t_mse_train}, {t_mse}\n")
            f.write(f"TARNet, {tarnet_mse_train}, {tarnet_mse}\n")
            f.write(f"CFR, {cfr_mse_train}, {cfr_mse}\n")
            f.write(f"ATE estimator, {ate_mse_train}, {ate_mse}\n")

    return execution_times


def main():
    n_iterations = 50

    times_per_sim = {}
    for sim in simulations:
        times_per_sim[sim.__name__] = []

    configurations = [(1000, 3, 1, 200, 100, 0.001, 200, 0.001)]
    for epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha in configurations:
        ts = plot_results(epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size, alpha, n_iterations)
        for i, sim in enumerate(simulations):
            times_per_sim[sim.__name__].append(ts[i])

    os.makedirs("times", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    f = open(f"times/simulations_{now}.txt", "w")
    for key, value in times_per_sim.items():
        print(f"{key}")
        f.write(f"{key}\n")
        for item in value:
            print(f"\t{item}")
            f.write(f"\t{item}\n")
    f.close()


if __name__ == "__main__":
    main()
