import pandas as pd
from data_split import stratified_split
from hdp_data_pipeline import HdpDataPipeline

def import_heart_disease_data():
    data = pd.read_csv("../data/data.csv", index_col=0).reset_index(drop=True)
    y = data[["disease_degree"]]
    X = data.drop(["disease_degree"], axis=1)
    X, y = HdpDataPipeline.filter_nan(X, y)
    X_train, X_test, y_train, y_test = map(lambda x: x.reset_index(drop=True), stratified_split(X, y))
    return X_train, X_test, y_train["disease_degree"], y_test["disease_degree"]