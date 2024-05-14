from imblearn.over_sampling import SMOTE
import pickle
import os

class HdpModelTrainer():
    def __init__(self, model=None, random_state=100):
        self.__random_state = random_state
        self.__model = model
        self.trained = False
        
    def save_trainer(self, folder_path="../deploy-models/", filename="trainer.pkl"):
        if self.trained:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(folder_path + filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)    
    
    @staticmethod
    def load_trainer(folder_path="../deploy-models/", filename="trainer.pkl"):
        with open(folder_path + filename, "rb") as f:
            return pickle.load(f)    
        
    def get_model(self):
        return self.__model
    
    def set_model(self, model):
        self.__model = model
    
    def train(self, X_train, y_train):
        X_train_over, y_train_over = self.__fit_resample(X_train, y_train)
        self.__model.fit(X_train_over, y_train_over)
        self.trained = True         
        return self
    
    def predict(self, X):
        if(self.trained):
            return self.__model.predict(X)
        return None
    
    def __fit_resample(self, X_train, y_train):
        X_train_oversampled, y_train_oversampled = \
            SMOTE(k_neighbors=5, random_state=self.__random_state).fit_resample(X_train, y_train)
        return X_train_oversampled, y_train_oversampled