import pandas as pd
import numpy as np
from data_split import stratified_split

def import_heart_disease_data():
    X = pd.read_csv("../data/features.csv", index_col=0)
    y = pd.read_csv("../data/target_binary.csv", index_col=0)
    X_train, X_test, y_train, y_test = map(lambda x: x.reset_index(drop=True), stratified_split(X, y))
    return X_train, X_test, y_train["disease_degree"], y_test["disease_degree"]