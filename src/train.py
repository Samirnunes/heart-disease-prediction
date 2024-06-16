from hdp_data_import import import_heart_disease_data
from hdp_data_pipeline import HdpDataPipeline
from hdp_model_trainer import HdpModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score
import pandas as pd
import json
from models import selected_model

def train():
    random_state = 100
    model = selected_model
    name = "LogisticRegressionBase"
    
    X_train, X_test, y_train, y_test = import_heart_disease_data()
    X_all = pd.concat([X_train, X_test], axis=0)
    pipeline = HdpDataPipeline()
    X_all = pipeline.fit_transform(X_all)
    y_all = pd.concat([y_train, y_test], axis=0)
    trainer = HdpModelTrainer(model, random_state)
    trainer.train(X_all, y_all)
    
    folder_path = f"../deploy-models/{name}/"
    pipeline.save_pipeline(folder_path, "pipeline.pkl")
    trainer.save_trainer(folder_path, "trainer.pkl")
    
    y_pred = trainer.predict(X_all)
    metrics = {}
    metrics["name"] = name
    metrics["train_precision"] = precision_score(y_all, y_pred, average="binary")
    metrics["train_recall"] = recall_score(y_all, y_pred, average="binary")
    with open(f"{folder_path}/metrics.txt", 'w') as f:
        f.write(json.dumps(metrics))
    
train()