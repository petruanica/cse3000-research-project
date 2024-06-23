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

configurations = get_configurations()

print("Importing dataset...")
df = pd.read_csv(f"dataset/knn_imputed_norm.csv")
x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "bilirubin", "urea", "fio2", "plateau_pressure"]]
y = df["mort_28"]
t = df["peep_regime"]
print("Done importing")

errors = 0
areas = {}
for epochs, layers_r, layers_out, neurons_r, neurons_out, lr, batch_size in configurations:
    config = f"e{epochs}_lyr{layers_r}_lyo{layers_out}_nr{neurons_r}_no{neurons_out}_lr{lr}_bs{batch_size}"
    area_list = []

    for iteration in range(iterations):
        train_x, test_x, train_y, test_y, train_t, test_t = \
            train_test_split(x, y, t, test_size=0.2, stratify=t, shuffle=True)
        test = test_x.join(test_y).join(test_t)

        print(f"Running TARNet - {config} --- Iteration {iteration + 1}")
        try:
            tarnet_cate_train, tarnet_cate_test, tarnet_loss_train, tarnet_loss_test = \
                cfr(train_x, test_x, train_y, test_y, train_t, test_t, epochs, layers_r, layers_out,
                    neurons_r, neurons_out, lr, batch_size, True, 0.0)
            print("TARNet done")

            test_tarnet = test.assign(ite=tarnet_cate_test)
            gain_curve_test_tarnet = cumulative_gain(test_tarnet, "ite", y="mort_28", t="peep_regime")

            x_axis = np.array(range(len(gain_curve_test_tarnet)))
            x_axis = x_axis / x_axis.max() * 100
            rnd = np.linspace(0, elast(test, "mort_28", "peep_regime"), gain_curve_test_tarnet.size)
            area_diff = integrate.simpson(gain_curve_test_tarnet - rnd, x=x_axis)

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
os.makedirs("gain", exist_ok=True)
f = open(f"gain/areas_i{iterations}_{now}.txt", "w")
f.write("Sorted areas under the curve (mean and std):\n")
for conf, area in areas_sorted:
    f.write(f"{conf} - mean: {area['mean']}, std: {area['std']}\n")
f.close()

print(f"Errors: {errors}")
