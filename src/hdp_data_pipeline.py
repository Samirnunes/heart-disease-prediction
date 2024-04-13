import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class HdpDataPipeline():
    def __init__(self):
        self.__numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.__categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.__numerical_imputer = SimpleImputer(strategy="mean")
        self.__categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.__scaler = StandardScaler()
        
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
        X.loc[:, self.__numerical] = self.__numerical_imputer.transform(X.loc[:, self.__numerical])
        X.loc[:, self.__categorical] = self.__categorical_imputer.transform(X.loc[:, self.__categorical])
        X = pd.DataFrame(self.__scaler.transform(X), columns=X.columns)
        return X
    
    def fit_tranform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
        