import numpy as np


def prediction_accuracy(y_pred, y_target):
    """
    Prediction Accuracy
    """
    assert len(y_pred) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_pred, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_pred)
