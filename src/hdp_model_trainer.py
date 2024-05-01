from imblearn.over_sampling import SMOTE
from joblib import dump, load
import os

class HdpModelTrainer():
    def __init__(self, model, random_state=100):
        self.__random_state = random_state
        self.__model = model
        self.trained = False
        
    def save_model(self, folder_path="../deploy-models/", filename="model.joblib"):
        if self.trained:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            dump(self.__model, folder_path + filename)
            
    def load_model(self, path="model.joblib"):
        self.__model = load(path)
        self.trained = True       
        
    def get_model(self):
        return self.__model
    
    def train(self, X_train, y_train):
        X_train_over, y_train_over = self.__fit_resample(X_train, y_train)
        self.__model.fit(X_train_over, y_train_over)
        self.trained = True           
        return self
    
    def __fit_resample(self, X_train, y_train):
        X_train_oversampled, y_train_oversampled = \
            SMOTE(k_neighbors=5, random_state=self.__random_state).fit_resample(X_train, y_train)
        return X_train_oversampled, y_train_oversampled