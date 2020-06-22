from sklearn.metrics import f1_score

class Evaluator:
    def calc_f1_score(self, y_true, y_pred, average):
        return f1_score(y_true, y_pred, average=average)
