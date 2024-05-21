import pandas as pd

from hdp_data_pipeline import HdpDataPipeline
from hdp_model_trainer import HdpModelTrainer

def predict(unprocessed_df):
    deploy_model = "LogisticRegressionBase"
    folder_path = "../deploy-models/" + deploy_model + "/"
    pipeline = HdpDataPipeline.load_pipeline(folder_path, filename="pipeline.pkl")
    trainer = HdpModelTrainer.load_trainer(folder_path, filename="trainer.pkl")
    data = pipeline.transform(unprocessed_df)
    return str(trainer.predict(data))

rows = [
    {"age": 63.0, "sex": 1.0, "cp": 1.0, "trestbps": 145.0, "chol": 233.0, "fbs": 1.0, "restecg": 2.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3, "slope": 3.0, "ca": 0.0, "thal": 6.0},
    {"age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0, "chol": 286.0, "fbs": 0.0, "restecg": 2.0, "thalach": 108.0, "exang": 1.0, "oldpeak": 1.5, "slope": 2.0, "ca": 3.0, "thal": 3.0},
    {"age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 120.0, "chol": 229.0, "fbs": 0.0, "restecg": 2.0, "thalach": 129.0, "exang": 1.0, "oldpeak": 2.6, "slope": 2.0, "ca": 2.0, "thal": 7.0},
    {"age": 37.0, "sex": 1.0, "cp": 3.0, "trestbps": 130.0, "chol": 250.0, "fbs": 0.0, "restecg": 0.0, "thalach": 187.0, "exang": 0.0, "oldpeak": 3.5, "slope": 3.0, "ca": 0.0, "thal": 3.0},
    {"age": 41.0, "sex": 0.0, "cp": 2.0, "trestbps": 130.0, "chol": 204.0, "fbs": 0.0, "restecg": 2.0, "thalach": 172.0, "exang": 0.0, "oldpeak": 1.4, "slope": 1.0, "ca": 0.0, "thal": 3.0},
    {"age": 56.0, "sex": 1.0, "cp": 2.0, "trestbps": 120.0, "chol": 236.0, "fbs": 0.0, "restecg": 0.0, "thalach": 178.0, "exang": 0.0, "oldpeak": 0.8, "slope": 1.0, "ca": 0.0, "thal": 3.0},
    {"age": 62.0, "sex": 0.0, "cp": 4.0, "trestbps": 140.0, "chol": 268.0, "fbs": 0.0, "restecg": 2.0, "thalach": 160.0, "exang": 0.0, "oldpeak": 3.6, "slope": 3.0, "ca": 2.0, "thal": 3.0},
    {"age": 57.0, "sex": 0.0, "cp": 4.0, "trestbps": 120.0, "chol": 354.0, "fbs": 0.0, "restecg": 0.0, "thalach": 163.0, "exang": 1.0, "oldpeak": 0.6, "slope": 1.0, "ca": 0.0, "thal": 3.0},
    {"age": 63.0, "sex": 1.0, "cp": 4.0, "trestbps": 130.0, "chol": 254.0, "fbs": 0.0, "restecg": 2.0, "thalach": 147.0, "exang": 0.0, "oldpeak": 1.4, "slope": 2.0, "ca": 1.0, "thal": 7.0},
    {"age": 53.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 203.0, "fbs": 1.0, "restecg": 2.0, "thalach": 155.0, "exang": 1.0, "oldpeak": 3.1, "slope": 3.0, "ca": 0.0, "thal": 7.0},
    {"age": 57.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 192.0, "fbs": 0.0, "restecg": 0.0, "thalach": 148.0, "exang": 0.0, "oldpeak": 0.4, "slope": 2.0, "ca": 0.0, "thal": 6.0}
]

predict(pd.DataFrame.from_dict(rows))