import numpy as np
from biomed.evaluator import Evaluator


def test_calc_f1_score():
    Y_true = [0, 1, 2, 0, 1, 2]
    Y_pred = [0, 2, 1, 0, 0, 1]
    average = 'macro'
    evalu = Evaluator()
    r = evalu.calc_f1_score(Y_true, Y_pred, average)
    assert r == 0.26666666666666666
