import pandas as pd
from split_data import stratified_split

def import_heart_disease_data():
    X = pd.read_csv("./data/features.csv", index_col=0)
    y = pd.read_csv("./data/target.csv", index_col=0)
    X_train, X_test, y_train, y_test = map(lambda x: x.reset_index(drop=True), stratified_split(X, y))
    return X_train, X_test, y_train, y_test