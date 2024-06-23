import os
from datetime import datetime

import tensorflow as tf
from scipy import integrate
from sklearn.model_selection import train_test_split

from models import *
from nb21 import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
now = datetime.now().strftime("%Y%m%d_%H%M%S")

iterations = 200

configurations = [(1000, 2, 1, 200, 20, 0.001, 200), (1000, 2, 1, 200, 100, 0.001, 200), (1000, 1, 2, 100, 100, 0.001, 700), (1000, 2, 1, 100, 100, 0.001, 500), (1000, 2, 1, 100, 50, 0.001, 700), (1000, 1, 1, 200, 50, 0.001, 200), (1000, 1, 3, 200, 100, 0.001, 500), (1000, 2, 2, 50, 200, 0.001, 100), (1000, 2, 1, 200, 20, 0.001, 700), (1000, 1, 2, 100, 200, 0.001, 500)]
configurations += [(1000, 2, 3, 20, 100, 0.001, 200), (1000, 3, 1, 200, 50, 0.001, 100), (1000, 3, 1, 200, 50, 0.001, 200), (1000, 2, 1, 200, 20, 0.001, 100), (1000, 1, 1, 50, 200, 0.001, 700), (1000, 3, 2, 100, 50, 0.001, 100), (1000, 3, 1, 100, 20, 0.001, 100), (1000, 2, 1, 200, 50, 0.001, 200), (1000, 1, 2, 50, 100, 0.001, 500), (1000, 1, 2, 50, 100, 0.001, 700)]
configurations += [(1000, 1, 3, 50, 100, 0.001, 700), (1000, 3, 2, 100, 20, 0.001, 100), (1000, 3, 1, 50, 20, 0.001, 700), (1000, 1, 3, 100, 50, 0.001, 200), (1000, 3, 1, 200, 100, 0.001, 200), (1000, 1, 1, 200, 20, 0.001, 500), (1000, 1, 1, 200, 100, 0.001, 500), (1000, 3, 1, 200, 200, 0.001, 200), (1000, 3, 3, 50, 50, 0.001, 100), (1000, 2, 2, 50, 50, 0.001, 200)]
configurations += [(1000, 1, 2, 200, 100, 0.001, 500), (1000, 2, 1, 200, 100, 0.001, 700), (1000, 2, 1, 100, 50, 0.001, 100), (1000, 2, 2, 100, 200, 0.001, 100), (1000, 3, 3, 200, 200, 0.001, 200), (1000, 3, 1, 50, 200, 0.001, 200), (1000, 1, 2, 200, 20, 0.001, 500), (1000, 3, 3, 100, 200, 0.001, 500), (1000, 1, 3, 100, 50, 0.001, 500), (1000, 2, 1, 200, 50, 0.001, 500)]
configurations += [(1000, 2, 1, 100, 20, 0.001, 200), (1000, 3, 1, 100, 50, 0.001, 700), (1000, 2, 2, 200, 100, 0.001, 500), (1000, 1, 3, 20, 50, 0.001, 700), (1000, 3, 1, 200, 20, 0.001, 200), (1000, 2, 1, 100, 20, 0.001, 500), (1000, 1, 3, 50, 200, 0.001, 500), (1000, 1, 3, 100, 50, 0.001, 700), (1000, 1, 3, 50, 200, 0.001, 100), (1000, 2, 1, 100, 20, 0.001, 700)]
configurations += [(1000, 2, 2, 100, 200, 0.001, 200), (1000, 2, 1, 100, 50, 0.001, 200), (1000, 1, 2, 200, 100, 0.001, 200), (1000, 2, 2, 100, 200, 0.001, 500), (1000, 2, 1, 100, 50, 0.001, 500), (1000, 3, 1, 50, 100, 0.001, 500), (1000, 1, 2, 20, 100, 0.001, 500), (1000, 3, 2, 200, 50, 0.001, 200), (1000, 3, 3, 50, 50, 0.001, 200), (1000, 2, 2, 200, 200, 0.001, 500)]
configurations = itertools.product(configurations, alphas)

print("Importing dataset...")
df = pd.read_csv(f"dataset/knn_imputed_norm.csv")
x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "bilirubin", "urea", "fio2", "plateau_pressure"]]
y = df["mort_28"]
t = df["peep_regime"]
print("Done importing")

errors = 0
areas = {}
for (epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size), alpha in configurations:
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}_alpha{alpha}"
    area_list = []

    for iteration in range(iterations):
        train_x, test_x, train_y, test_y, train_t, test_t = \
            train_test_split(x, y, t, test_size=0.2, stratify=t, shuffle=True)
        test = test_x.join(test_y).join(test_t)

        print(f"Running CFR - {config} --- Iteration {iteration + 1}")
        try:
            cfr_cate_train, cfr_cate_test, cfr_loss_train, cfr_loss_test = \
                cfr(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
                    neurons_r, neurons_out, lr, batch_size, True, alpha)
            print("CFR done")

            test_cfr = test.assign(ite=cfr_cate_test)
            gain_curve_test_cfr = cumulative_gain(test_cfr, "ite", y="mort_28", t="peep_regime")

            x_axis = np.array(range(len(gain_curve_test_cfr)))
            x_axis = x_axis / x_axis.max() * 100
            rnd = np.linspace(0, elast(test, "mort_28", "peep_regime"), gain_curve_test_cfr.size)
            area_diff = integrate.simpson(gain_curve_test_cfr - rnd, x=x_axis)

            area_list.append(area_diff)
            print(f"Area: {area_diff}")
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    areas_mean = np.mean(area_list)
    areas_std = np.std(area_list)
    areas[f"{config}"] = {"mean": areas_mean, "std": areas_std}
    print(f"Area --- mean: {areas_mean}, std: {areas_std} ({len(area_list)} iterations)\n")

areas_sorted = sorted(areas.items(), key=lambda item: item[1]["mean"], reverse=True)
os.makedirs("gain/alpha", exist_ok=True)
f = open(f"gain/alpha/areas_i{iterations}_{now}.txt", "w")
f.write("Sorted areas under the curve (mean and std):\n")
for conf, area in areas_sorted:
    f.write(f"{conf} - mean: {area['mean']}, std: {area['std']}\n")
f.close()

print(f"Errors: {errors}")
