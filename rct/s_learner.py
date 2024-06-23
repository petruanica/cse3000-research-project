from datetime import datetime

import numpy as np
import pandas as pd

from imputer import impute_values, train_imputer
from models import select_best_s_learner, calculate_area
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

    best_s_learner = select_best_s_learner(x, y, t, configuration, iterations)

    print("Importing RCT dataset...")
    df = pd.read_csv(rct_data)
    df = impute_values(df, imputer)
    x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "plateau_pressure"]]
    y = df["mort_28"]
    t = df["peep_regime"]
    print("Done importing")
    test = x.join(y).join(t)

    ite_rct_s_learner = (best_s_learner.predict(x.assign(**{"peep_regime": 1.0}))
                         - best_s_learner.predict(x.assign(**{"peep_regime": 0.0})))
    ite_rct_s_learner = ite_rct_s_learner.reshape((ite_rct_s_learner.shape[0],))
    print(calculate_area(test, ite_rct_s_learner))

    np.savetxt(f"ite_s_learner_{now}.txt", ite_rct_s_learner)

    return ite_rct_s_learner


if __name__ == "__main__":
    main()
