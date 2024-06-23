from datetime import datetime

import numpy as np
import pandas as pd

from imputer import impute_values, train_imputer
from models import select_best_cfr, calculate_area
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

    best_cfr = select_best_cfr(x, y, t, configuration, iterations)

    print("Importing RCT dataset...")
    df = pd.read_csv(rct_data)
    df = impute_values(df, imputer)
    x = df[["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "plateau_pressure"]]
    y = df["mort_28"]
    t = df["peep_regime"]
    print("Done importing")
    test = x.join(y).join(t)

    ite_test_cfr = best_cfr.predict(x.to_numpy()).detach().cpu().numpy()
    print(calculate_area(test, ite_test_cfr))

    np.savetxt(f"ite_cfr_{now}.txt", ite_test_cfr)

    return ite_test_cfr


if __name__ == "__main__":
    main()
