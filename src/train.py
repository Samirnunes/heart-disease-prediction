from hdp_data_import import import_heart_disease_data
from hdp_data_pipeline import HdpDataPipeline
from hdp_model_trainer import HdpModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score
import pandas as pd
import json

def train():
    random_state = 100
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    name = "RandomForestBaseDeploy"
    threshold = 0.5
    
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
    
    model = trainer.get_model()
    probs = model.predict_proba(X_all)
    y_pred = (probs[:, 1] >= threshold).astype(int)
    metrics = {}
    metrics["name"] = name
    metrics["threshold"] = threshold
    metrics["train_accuracy"] = accuracy_score(y_all, y_pred)
    metrics["train_precision"] = precision_score(y_all, y_pred, average="binary")
    metrics["train_recall"] = recall_score(y_all, y_pred, average="binary")
    with open(f"{folder_path}/metrics.txt", 'w') as f:
        f.write(json.dumps(metrics))
    
train()