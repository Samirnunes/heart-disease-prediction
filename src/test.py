from hdp_data_import import import_heart_disease_data
from hdp_data_pipeline import HdpDataPipeline
from hdp_many_model_evaluator import HdpManyModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
import json

def test():
    random_state = 100
    models = [
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        LogisticRegression(),
        SVC(probability=True, random_state=random_state),
        XGBClassifier(random_state=random_state)
    ]
    names = [
        "RandomForestBase",
        "DecisionTreeBase",
        "LogisticRegressionBase",
        "SVCBase",
        "XGBClassifierBase"
    ]
      
    X_train, X_test, y_train, y_test = import_heart_disease_data()
    pipeline = HdpDataPipeline()
    many_evaluator = HdpManyModelEvaluator(models, pipeline)
    for score, name in zip(many_evaluator.test_scores(X_train, y_train, X_test, y_test), names):
        folder = f"../test_metrics/{name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/score.txt", 'w') as f:
            f.write(json.dumps(score))
        
test()