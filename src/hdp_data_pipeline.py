import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

class HdpDataPipeline():
    def __init__(self):
        self.__numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.__categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.__numerical_imputer = SimpleImputer(strategy="mean")
        self.__categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.__scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
        
    def fit(self, X_train):
        """
        Fit for production.
        """
        X_train_copy = X_train.copy() 
        X_train_copy.loc[:, self.__numerical] = self.__numerical_imputer.fit_transform(X_train_copy.loc[:, self.__numerical])
        X_train_copy.loc[:, self.__categorical] = self.__categorical_imputer.fit_transform(X_train_copy.loc[:, self.__categorical])
        X_train_copy = pd.DataFrame(self.__scaler.fit_transform(X_train_copy), columns=X_train_copy.columns)
        return self
    
    def transform(self, X):
        """
        Transform for production.
        """
        X_nan_less_50 = X[~(X.isna().sum(axis=1) > len(X.columns)/2)].reset_index(drop=True)
        if len(X) - len(X_nan_less_50) > 0:
            raise Exception("Please fill at least half of the parameters for the model's prediction.")
        X.loc[:, self.__numerical] = self.__numerical_imputer.transform(X.loc[:, self.__numerical])
        X.loc[:, self.__categorical] = self.__categorical_imputer.transform(X.loc[:, self.__categorical])
        X = pd.DataFrame(self.__scaler.transform(X), columns=X.columns)
        return X
    
    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
    
    @staticmethod
    def filter_nan(X, y):
        X_nan_less_50 = X[~(X.isna().sum(axis=1) > len(X.columns)/2)].copy()
        X = X_nan_less_50
        y = y.loc[X.index]
        return X.reset_index(drop=True), y.reset_index(drop=True)