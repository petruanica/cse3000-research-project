from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)


def train_imputer(df):
    df = deepcopy(df)
    df["sex"] = df["sex"].map({"M": 0.0, "F": 1.0})
    df["mort_28"] = df["mort_28"].astype(float)
    df["peep_regime"] = df["peep_regime"].map({"low": 0.0, "high": 1.0})

    # Remove unwanted columns
    df = df.drop(columns=["Unnamed: 0", "id"])

    df_norm = df.apply(lambda x: normalize(x))

    # Create the imputers
    k = int(np.round(np.sqrt(df.shape[0])))
    k = k + (k % 2 == 0)
    knn_imputer = KNNImputer(n_neighbors=k)

    # Perform imputation
    knn_imputer.fit(df_norm)
    return knn_imputer


def impute_values(df, imputer):
    df = deepcopy(df)
    df["sex"] = df["sex"].map({"M": 0.0, "F": 1.0})
    df["mort_28"] = df["mort_28"].astype(float)
    df["peep_regime"] = df["peep_regime"].map({"low": 0.0, "high": 1.0})

    # Remove unwanted columns
    df = df.drop(columns=["Unnamed: 0", "id"])

    df_norm = df.apply(lambda x: normalize(x))

    df_knn = imputer.transform(df_norm)

    df_knn = pd.DataFrame(df_knn, columns=df_norm.columns)

    return df_knn
