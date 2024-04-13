from hdp_model_evaluator import HdpModelEvaluator

class HdpManyModelEvaluator():
    def __init__(self, models, pipeline):
        self.__models = models
        self.__pipeline = pipeline
        
    def kfold_cross_val(self, X_train, y_train, threshold=0.5):
        metrics = []
        for model in self.__models:
            evaluator = HdpModelEvaluator(model, self.__pipeline)
            metrics.append(evaluator.kfold_cross_val(X_train, y_train, threshold))
        return metrics

    def test_scores(self, X_train, y_train, X_test, y_test, threshold=0.5):
        scores = []
        for model in self.__models:
            evaluator = HdpModelEvaluator(model, self.__pipeline)
            scores.append(evaluator.test_scores(X_train, y_train, X_test, y_test, threshold))
        return scores
    