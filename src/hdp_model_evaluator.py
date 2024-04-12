from copy import deepcopy
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, accuracy_score
from sklearn.base import clone

class HdpModelEvaluator():
    def __init__(self, model, pipeline):
        self.__model = model
        self.__pipeline = pipeline
        
    def kfold_cross_val(self, X_train, y_train):
        kf = KFold(n_splits=5)
        accuracies = []
        recalls = []
        for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
            X_train_fold = X_train.iloc[train_index].copy()
            y_train_fold = y_train.iloc[train_index].copy()
            X_val_fold = X_train.iloc[val_index].copy()
            y_val_fold = y_train.iloc[val_index].copy() 
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = \
                deepcopy(self.__pipeline).fit_transform(X_train_fold, X_val_fold, y_train_fold, y_val_fold)
            model = clone(self.__model)
            model.fit(X_train_fold, y_train_fold)  
            y_pred = model.predict(X_val_fold)
            accuracies.append(accuracy_score(y_val_fold, y_pred))
            recalls.append(recall_score(y_val_fold, y_pred, average="macro"))
            
        return {"mean_accuracy": sum(accuracies)/len(accuracies), 
                "mean_recall": sum(recalls)/len(recalls)}

    def test_scores(self, X_train, y_train, X_test, y_test):
        model = clone(self.__model)
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()
        y_train_copy = y_train.copy()
        y_test_copy = y_test.copy()
        X_train_copy, X_test_copy, y_train_copy, y_test_copy = \
            deepcopy(self.__pipeline).fit_transform(X_train_copy, X_test_copy, y_train_copy, y_test_copy)
        model.fit(X_train_copy, y_train_copy) 
        y_pred = model.predict(X_test_copy)
        
        return {"test_accuracy": accuracy_score(y_test_copy, y_pred), 
                "test_recall": recall_score(y_test_copy, y_pred, average="macro")}
    