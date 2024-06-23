from datetime import datetime

import numpy as np
import pandas as pd

from imputer import impute_values, train_imputer
from models import select_best_t_learner, calculate_area
from variables import *

now = datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    print("Importing training dataset...")
    df = pd.read_csv(training_data)
    imputer = train_imputer(df)
    df = impute_values(df, imputer)
    x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "plateau_pressure"]]
    y = df["mort_28"]
    t = df["peep_regime"]
    print("Done importing")

    best_t_learner = select_best_t_learner(x, y, t, configuration, iterations)

    print("Importing RCT dataset...")
    df = pd.read_csv(rct_data)
    df = impute_values(df, imputer)
    x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "plateau_pressure"]]
    y = df["mort_28"]
    t = df["peep_regime"]
    print("Done importing")
    test = x.join(y).join(t)

    ite_rct_t_learner = best_t_learner[1].predict(x) - best_t_learner[0].predict(x)
    ite_rct_t_learner = ite_rct_t_learner.reshape((ite_rct_t_learner.shape[0],))
    print(calculate_area(test, ite_rct_t_learner))

    np.savetxt(f"ite_t_learner_{now}.txt", ite_rct_t_learner)

    return ite_rct_t_learner


if __name__ == "__main__":
    main()
