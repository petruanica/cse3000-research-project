from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate

from nb21 import *

now = datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_area(test, pred):
    gain_curve_test = cumulative_gain(test, pred, y="mort_28", t="peep_regime")
    x_axis = np.array(range(len(gain_curve_test)))
    x_axis = x_axis / x_axis.max() * 100
    rnd = np.linspace(0, elast(test, "mort_28", "peep_regime"), gain_curve_test.size)
    area_diff = integrate.simpson(gain_curve_test - rnd, x=x_axis)
    return area_diff


df = pd.read_csv(f"rct/s_learner.csv")
df["mort_28"] = df["mort_28"].apply(lambda x: 1 - x)
test = df[["mort_28", "peep_regime", "ite"]]
test = test.rename(columns={"ite": "s_learner_ite"})
df = pd.read_csv(f"rct/t_learner.csv")
test = test.join(df["ite"])
test = test.rename(columns={"ite": "t_learner_ite"})
df = pd.read_csv(f"rct/tarnet.csv")
test = test.join(df["ite"])
test = test.rename(columns={"ite": "tarnet_ite"})
df = pd.read_csv(f"rct/cfr.csv")
test = test.join(df["ite"])
test = test.rename(columns={"ite": "cfr_ite"})

print(test["peep_regime"].sum())
print(test["peep_regime"].shape[0])
print(test["peep_regime"].sum() / test["peep_regime"].shape[0])

gain_curve_s_nn, gain_curve_s_nn_ci = get_curve_with_ci(test, "s_learner_ite")
gain_curve_t_nn, gain_curve_t_nn_ci = get_curve_with_ci(test, "t_learner_ite")
gain_curve_tarnet, gain_curve_tarnet_ci = get_curve_with_ci(test, "tarnet_ite")
gain_curve_cfr, gain_curve_cfr_ci = get_curve_with_ci(test, "cfr_ite")

xs = np.array(range(len(gain_curve_s_nn)))
xs = xs / xs.max() * 100

s_nn_area = calculate_area(test, "s_learner_ite")
plt.plot(xs, gain_curve_s_nn, label=f"S-learner (AUC={s_nn_area:.2f})", color="C0")
plt.fill_between(xs, gain_curve_s_nn_ci[:, 0], gain_curve_s_nn_ci[:, 1], alpha=0.2, color="C0")
rnd = np.linspace(0, elast(test, "mort_28", "peep_regime"), gain_curve_s_nn.shape[0])
print("Confidence interval - S-learner")
print(f"[{integrate.simpson(gain_curve_s_nn_ci[:, 0] - rnd, x=xs)}, {integrate.simpson(gain_curve_s_nn_ci[:, 1] - rnd, x=xs)}]")


t_nn_area = calculate_area(test, "t_learner_ite")
plt.plot(xs, gain_curve_t_nn, label=f"T-learner (AUC={t_nn_area:.2f})", color="C1")
plt.fill_between(xs, gain_curve_t_nn_ci[:, 0], gain_curve_t_nn_ci[:, 1], alpha=0.2, color="C1")
print("Confidence interval - T-learner")
print(f"[{integrate.simpson(gain_curve_t_nn_ci[:, 0] - rnd, x=xs)}, {integrate.simpson(gain_curve_t_nn_ci[:, 1] - rnd, x=xs)}]")


tarnet_area = calculate_area(test, "tarnet_ite")
plt.plot(xs, gain_curve_tarnet, label=f"TARNet (AUC={tarnet_area:.2f})", color="C2")
plt.fill_between(xs, gain_curve_tarnet_ci[:, 0], gain_curve_tarnet_ci[:, 1], alpha=0.2, color="C2")
print("Confidence interval - TARNet")
print(f"[{integrate.simpson(gain_curve_tarnet_ci[:, 0] - rnd, x=xs)}, {integrate.simpson(gain_curve_tarnet_ci[:, 1] - rnd, x=xs)}]")


cfr_area = calculate_area(test, "cfr_ite")
plt.plot(xs, gain_curve_cfr, label=f"CFR (AUC={cfr_area:.2f})", color="C3")
plt.fill_between(xs, gain_curve_cfr_ci[:, 0], gain_curve_cfr_ci[:, 1], alpha=0.2, color="C3")
print("Confidence interval - CFR")
print(f"[{integrate.simpson(gain_curve_cfr_ci[:, 0] - rnd, x=xs)}, {integrate.simpson(gain_curve_cfr_ci[:, 1] - rnd, x=xs)}]")

plt.plot([0, 100], [0, elast(test, "mort_28", "peep_regime")], linestyle="--", color="black", label="Random")
plt.title(f"Cumulative gain curve for RCT dataset")
plt.legend()
plt.xlabel("Percentage of patients targeted")
plt.ylabel("Cumulative gain")
plt.savefig(f"rct/gain_{now}.png")
plt.show()
