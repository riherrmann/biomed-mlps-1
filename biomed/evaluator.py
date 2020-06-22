from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, y_true, y_pred, average):
        self.y_true = y_true
        self.y_pred = y_pred
        self.average = average

    def calc_f1_score(self):
        return f1_score(self.y_true, self.y_pred, self.average)
