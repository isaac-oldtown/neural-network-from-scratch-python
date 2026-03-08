import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def gradient(self, x):
        s = self.forward(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def gradient(self, x):
        return (x > 0).astype(int)

class Logloss:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.log(np.sum(y_true*y_pred))

    def gradient(self, y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -y_true / np.sum(y_true * y_pred)