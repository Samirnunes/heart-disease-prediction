from imblearn.over_sampling import SMOTE

class HdpModelTrainer():
    def __init__(self, model, random_state=100):
        self.__random_state = random_state
        self.__model = model
        self.is_trained = False
        
    def get_model(self):
        return self.__model
    
    def train(self, X_train, y_train):
        X_train_over, y_train_over = self.__fit_resample(X_train, y_train)
        self.__model.fit(X_train_over, y_train_over)
        self.is_trained = True           
        return self
    
    def __fit_resample(self, X_train, y_train):
        X_train_oversampled, y_train_oversampled = \
            SMOTE(k_neighbors=5, random_state=self.__random_state).fit_resample(X_train, y_train)
        return X_train_oversampled, y_train_oversampled