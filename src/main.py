from hdp_data_import import import_heart_disease_data
from hdp_data_pipeline import HdpDataPipeline
from hdp_many_model_evaluator import HdpManyModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def main():
    X_train, X_test, y_train, y_test = import_heart_disease_data()
    random_state = 100
    pipeline = HdpDataPipeline()
    models = [
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        LogisticRegression(),
        SVC(probability=True),
        XGBClassifier()
    ]
    many_evaluator = HdpManyModelEvaluator(models, pipeline)
    for metrics in many_evaluator.kfold_cross_val(X_train, y_train, threshold=0.5):
        print(metrics)

if __name__ == "__main__":
    main()    