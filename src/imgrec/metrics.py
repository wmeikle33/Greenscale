import numpy as np
def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0
