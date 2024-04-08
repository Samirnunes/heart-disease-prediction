from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, accuracy_score
from sklearn.base import clone

class HdpModelEvaluator():
    def __init__(self, model, random_state=100):
        self.__model = model
        self.__random_state = random_state
        
    def kfold_cross_val(self, X_train, y_train):
        kf = KFold(n_splits=5)
        accuracies = []
        recalls = []

        for fold, (train_index, val_index) in enumerate(kf.split(X_train), 1):
            X_train_fold = X_train.iloc[train_index]
            y_train_fold = y_train.iloc[train_index] 
            X_val_fold = X_train.iloc[val_index]
            y_val_fold = y_train.iloc[val_index] 
            X_train_fold_oversampled, y_train_fold_oversampled = \
                SMOTE(k_neighbors=5, random_state=self.__random_state).fit_resample(X_train_fold, y_train_fold)
            model = clone(self.__model)
            model.fit(X_train_fold_oversampled, y_train_fold_oversampled)  
            y_pred = model.predict(X_val_fold)
            accuracies.append(accuracy_score(y_val_fold, y_pred))
            recalls.append(recall_score(y_val_fold, y_pred, average="macro"))
            
        return {"mean_accuracy": sum(accuracies)/len(accuracies), 
                "mean_recall": sum(recalls)/len(recalls)}

    def test_scores(self, X_train, y_train, X_test, y_test):
        model = clone(self.__model)
        X_train_oversampled, y_train_oversampled = \
            SMOTE(k_neighbors=5, random_state=self.__random_state).fit_resample(X_train, y_train)
        model.fit(X_train_oversampled, y_train_oversampled) 
        y_pred = model.predict(X_test)
        
        return {"test_accuracy": accuracy_score(y_test, y_pred), 
                "test_recall": recall_score(y_test, y_pred, average="macro")}
    