from hdp_data_import import import_heart_disease_data
from hdp_data_pipeline import HdpDataPipeline
from hdp_many_model_evaluator import HdpManyModelEvaluator
import os
import json
import matplotlib.pyplot as plt
from models import *

def evaluate():      
    X_train, X_test, y_train, y_test = import_heart_disease_data()
    pipeline = HdpDataPipeline()
    many_evaluator = HdpManyModelEvaluator(models, pipeline)
    for metrics, name in zip(many_evaluator.kfold_cross_val(X_train, y_train), names):
        metrics["name"] = name
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(metrics["recalls"])
        axs[0].vlines(metrics["mean_recall"], 0, max(axs[0].get_yticks()), linestyles="--", color="red")
        axs[0].errorbar(metrics["mean_recall"], max(axs[0].get_yticks())/2, xerr=metrics["std_recall"], fmt='o', color='red', capsize=6)
        axs[0].set_title("Recall distribution")
        axs[0].set_ylabel("Count")
        axs[0].set_xlabel("Recall")
        axs[0].vlines(0.75, 0, max(axs[0].get_yticks()), linestyles="--", color="green")
        axs[0].legend(["Mean", "Goal", "Histogram"])
        
        axs[1].hist(metrics["precisions"])
        axs[1].vlines(metrics["mean_precision"], 0, max(axs[1].get_yticks()), linestyles="--", color="red")
        axs[1].errorbar(metrics["mean_precision"], max(axs[1].get_yticks())/2, xerr=metrics["std_precision"], fmt='o', color='red', capsize=6)
        axs[1].set_title("Precision distribution")
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("Precision")
        axs[1].vlines(0.7, 0, max(axs[1].get_yticks()), linestyles="--", color="green")
        axs[1].legend(["Mean", "Goal", "Histogram"])
        metrics.pop("recalls")
        metrics.pop("precisions")
        folder = f"../evaluation_metrics/{name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/metrics.txt", 'w') as f:
            f.write(json.dumps(metrics))
        plt.savefig(f"{folder}/recalls_precisions_hist.png")
        plt.close()

evaluate()    