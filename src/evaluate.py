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
import matplotlib.pyplot as plt
import numpy as np

def evaluate():
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
    for metrics, name in zip(many_evaluator.kfold_cross_val(X_train, y_train, threshold=0.5), names):
        metrics["name"] = name
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(metrics["recalls"])
        axs[0].vlines(metrics["mean_recall"], 0, max(axs[0].get_yticks()), linestyles="--", color="red")
        axs[0].set_title("Recall distribution")
        axs[0].set_ylabel("Count")
        axs[0].set_xlabel("Recall")
        axs[0].vlines(0.75, 0, max(axs[0].get_yticks()), linestyles="--", color="green")
        axs[0].legend(["Mean", "Goal", "Histogram"])
        axs[1].boxplot(metrics["recalls"])
        axs[1].set_title("Recall boxplot")
        metrics.pop("recalls")
        folder = f"../evaluation_metrics/{name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/metrics.txt", 'w') as f:
            f.write(json.dumps(metrics))
        plt.savefig(f"{folder}/recalls_hist.png")
        plt.close()

evaluate()    