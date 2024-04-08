import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class HdpDataTransformer():
    
    def __init__(self):
        self.__numerical = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.__categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        self.__numerical_imputer = SimpleImputer(strategy="mean")
        self.__categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.__scaler = StandardScaler()
    
    def fit_transform(self, X_train, X_test, y_train, y_test):
        X_train, X_test = self.fit_impute(X_train, X_test)
        X_train, X_test = self.fit_scale(X_train, X_test)
        return X_train, X_test, y_train, y_test
    
    def fit_impute(self, X_train, X_test):
        X_train.loc[:, self.__numerical] = self.__numerical_imputer.fit_transform(X_train.loc[:, self.__numerical])
        X_test.loc[:, self.__numerical] = self.__numerical_imputer.transform(X_test.loc[:, self.__numerical])
        
        X_train.loc[:, self.__categorical] = self.__categorical_imputer.fit_transform(X_train.loc[:, self.__categorical])
        X_test.loc[:, self.__categorical] = self.__categorical_imputer.transform(X_test.loc[:, self.__categorical])
        
        return X_train, X_test
    
    def fit_scale(self, X_train, X_test):
        X_train = pd.DataFrame(self.__scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(self.__scaler.transform(X_test), columns=X_test.columns)
        return X_train, X_test